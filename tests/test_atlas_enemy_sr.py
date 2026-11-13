from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from turkeyworld.atlas_enemy_sr import AtlasEnemyEDSRUpscaler,ASSET_SHA256,UNUSED_PAINT_PREFIXES
from turkeyworld.enemy_decoder_sr import EnemyDecoderEDSRUpscaler
from test_enemy_decoder_visual_first import force_visual,hypothesis


def small():
    return AtlasEnemyEDSRUpscaler(channels=4,blocks=1,semantic_channels=4,
        semantic_blocks=1,enemy_decoder_channels=4)


class AtlasEnemySRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_old_parameter_prefix_and_moments_layout_survive_with_only49_appended_values(self):
        old=EnemyDecoderEDSRUpscaler(enemy_decoder_stable=True,enemy_decoder_visual_first=True,
            enemy_decoder_restoration=True,restoration_blend_scale=1.)
        new=AtlasEnemyEDSRUpscaler()
        names=list(dict(new.named_parameters()));old_names=list(dict(old.named_parameters()))
        self.assertEqual(len(old_names),158)
        self.assertEqual(names[:158],old_names)
        self.assertEqual(names[158:],['atlas_player_dead_head.weight','atlas_player_dead_head.bias'])
        self.assertEqual(sum(p.numel()for n,p in new.named_parameters()if n not in old_names),49)
        self.assertTrue(all(not p.requires_grad for n,p in new.named_parameters()
            if not n.startswith(('enemy_decoder_','atlas_player_dead_head.'))))
        self.assertTrue(all(not p.requires_grad for n,p in new.named_parameters()if n.startswith(UNUSED_PAINT_PREFIXES)))
        self.assertTrue(new.atlas_player_dead_head.weight.requires_grad)
        self.assertEqual(new.atlas_frames_rgba.dtype,torch.uint8)
        self.assertEqual(new.config['atlas_asset_sha256'],ASSET_SHA256)

    def test_original_odd_png_tiles_use_Source_round_boundaries_and_keep_all_native_texels(self):
        m=small()
        self.assertEqual(m.config['atlas_buffer_shape'],[7,4,444,444])
        self.assertEqual(m.config['atlas_frame_sizes'],[[444,444],[444,443],[444,443],
            [444,444],[443,444],[443,443],[443,443]])
        with Image.open(Path('assets/turkey_atlas.png'))as source:
            source=source.convert('RGBA');xs=[round(x*source.width/4)for x in range(5)]
            ys=[round(y*source.height/2)for y in range(3)]
            for index in range(7):
                expected=np.asarray(source.crop((xs[index%4],ys[index//4],xs[index%4+1],ys[index//4+1]))).transpose(2,0,1)
                h,w=m.config['atlas_frame_sizes'][index]
                np.testing.assert_array_equal(m.atlas_frames_rgba[index,:,:h,:w].numpy(),expected)

    def test_native_integer_pixel_sampling_matches_Source_formula_for_unclipped_boxes_and_death_tint(self):
        m=small().eval()
        boxes=torch.tensor([[-32.,17.,96.,145.]]*7)/128
        rgba=m.render_label_atlas(boxes,torch.arange(7))
        yy,xx=np.meshgrid(np.arange(128),np.arange(128),indexing='ij')
        inside=(xx>=-32)&(xx<96)&(yy>=17)&(yy<145)
        for index in range(7):
            h,w=m.config['atlas_frame_sizes'][index]
            sx=np.clip(((xx+32)*w/128).astype(int),0,w-1)
            sy=np.clip(((yy-17)*h/128).astype(int),0,h-1)
            expected=m.atlas_frames_rgba[index,:,:h,:w].numpy()[:,sy,sx]*inside[None]
            np.testing.assert_array_equal((rgba[index]*255).round().byte().numpy(),expected)
        dead=m.render_label_atlas(boxes,torch.arange(7),torch.ones(7))
        wanted=((rgba[:,:3]*255).round()+torch.tensor([70,8,5])[None,:,None,None]).div(2).floor().byte()
        np.testing.assert_array_equal((dead[:,:3]*255).round().byte().numpy(),wanted.numpy())
        torch.testing.assert_close(dead[:,3:],rgba[:,3:],rtol=0,atol=0)

    def test_checkpoint_is_self_contained_without_asset_access_and_corrupt_buffers_fail_closed(self):
        m=small().eval();config={k:v for k,v in m.config.items()if k!='architecture'}
        with patch('turkeyworld.atlas_enemy_sr.Image.open',side_effect=FileNotFoundError('asset absent')):
            restored=AtlasEnemyEDSRUpscaler(**config).eval()
            with self.assertRaisesRegex(RuntimeError,'strict checkpoint'):
                restored(torch.full((1,3,32,32),.2),None)
            restored.load_state_dict(m.state_dict(),strict=True)
            rgb=torch.rand(1,3,32,32)
            torch.testing.assert_close(restored(rgb,None),m(rgb,None),rtol=0,atol=0)
        bad=deepcopy(m.state_dict());bad['atlas_frames_rgba'][0,0,0,0]^=1
        with self.assertRaisesRegex(RuntimeError,'pixel SHA256'):
            restored.load_state_dict(bad,strict=True)

    def test_hard_visibility_and_hard_phase_are_crisp_and_high_vision_ignores_wrong_hints(self):
        m=small();force_visual(m)
        with torch.no_grad():
            m.enemy_decoder_visible[-1].bias.fill_(.01)
            m.enemy_decoder_animation.bias.zero_();m.enemy_decoder_animation.bias[4]=2.
        rgb=torch.full((2,3,32,32),.25)
        expected=m.eval().forward_with_aux(rgb,None)
        self.assertEqual(expected['atlas_phase_index'].tolist(),[4,4])
        self.assertTrue((expected['atlas_visibility_hard_hr']==1).all())
        torch.testing.assert_close(expected['atlas_effective_alpha_hr'],expected['atlas_intrinsic_alpha_hr'],rtol=0,atol=0)
        for hint in (hypothesis(),torch.cat((torch.ones(2,20)*.1,torch.ones(2,1)),1)):
            torch.testing.assert_close(m(rgb,hint),expected['image'],rtol=0,atol=0)
        train=m.train().forward_with_aux(rgb,None)
        torch.testing.assert_close(train['image'],expected['image'],rtol=0,atol=0)
        with torch.no_grad():m.enemy_decoder_visible[-1].bias.fill_(-.01)
        hidden=m.eval().forward_with_aux(rgb,None)
        torch.testing.assert_close(hidden['image'],hidden['restoration_base_rgb'],rtol=0,atol=0)
        self.assertEqual(float(hidden['atlas_effective_alpha_hr'].sum()),0.)

    def test_geometry_phase_visibility_and_tint_receive_gradients_but_learned_paint_never_runs(self):
        m=small().train();force_visual(m)
        with torch.no_grad():
            m.enemy_decoder_animation.bias.zero_();m.enemy_decoder_animation.bias[1]=1.
            m.enemy_decoder_visible[-1].bias.fill_(.01)
        rgb=torch.full((2,3,32,32),.2)
        with patch.object(m.enemy_decoder_rgba,'forward',side_effect=AssertionError('paint called')), \
             patch.object(m.enemy_decoder_seed,'forward',side_effect=AssertionError('seed called')), \
             patch.object(m.enemy_decoder_condition,'forward',side_effect=AssertionError('state paint called')):
            output=m.forward_with_aux(rgb,None)
            output['image'].square().mean().backward()
        for name in ('enemy_decoder_box','enemy_decoder_size','enemy_decoder_animation','enemy_decoder_visible','atlas_player_dead_head'):
            gradients=[p.grad for p in getattr(m,name).parameters()if p.grad is not None]
            self.assertTrue(gradients,name)
            self.assertGreater(sum(float(g.abs().sum())for g in gradients),0.,name)
            self.assertTrue(all(bool(torch.isfinite(g).all())for g in gradients),name)
        self.assertTrue(all(p.grad is None for n,p in m.named_parameters()if not p.requires_grad))
        self.assertFalse(output['learned_enemy_paint'])
        self.assertEqual(output['occlusion_queries'],'NN_binary_visibility_only')

    def test_full_box_can_extend_offscreen_and_has_Source_five_frame_size_cap(self):
        m=small().eval()
        with torch.no_grad():
            m.enemy_decoder_box[-1].weight.zero_();m.enemy_decoder_box[-1].bias.fill_(10.)
            m.enemy_decoder_size[-1].weight.zero_();m.enemy_decoder_size[-1].bias.fill_(10.)
        output=m.forward_with_aux(torch.full((1,3,32,32),.2),None)
        box=output['visual_full_bbox']
        self.assertGreater(float(box.max()),1.)
        torch.testing.assert_close(box[:,2:]-box[:,:2],torch.full((1,2),5.))


if __name__=='__main__':unittest.main()
