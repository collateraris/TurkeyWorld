"""Guided SR receives the new causal world prediction; old SR remains RGB-only."""
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from turkeyworld.neural_play import NeuralRuntime


class PredictedWorld(nn.Module):
    def __init__(self):
        super().__init__(); self.anchor=nn.Parameter(torch.zeros(()))
        self.config=SimpleNamespace(architecture='dreamer_rssm', image_size=32,
            simulation_hz=30., transition_mode='event_v1', endless=True, navigation_only=False,
            kind='teacher', renderer_mode='direct')
        self.rgb=torch.full((1,3,32,32),.3)
    def encode_map(self, grid): return grid.float()
    def initialize_from_image(self, image, state, map_features): return state.new_zeros(len(state),4)
    def step(self, state, action, memory, map_features):
        current=state.clone();current[:,0]+=.01;current[:,8]-=.02;current[:,-1]+=1/9000
        return dict(state=current,memory=memory+1,rgb=self.rgb,events=torch.zeros(1,4))


class GuidedSR(nn.Module):
    def __init__(self):
        super().__init__();self.anchor=nn.Parameter(torch.zeros(()))
        self.config={'architecture':'enemy_decoder_edsr'};self.calls=[];self.runtime=None
    def forward(self, image, hypothesis=None):
        self.calls.append(dict(image=image.clone(),hypothesis=None if hypothesis is None else hypothesis.clone(),
            state_before_commit=None if self.runtime is None else self.runtime.state.clone()))
        return F.interpolate(image,scale_factor=4,mode='nearest')


class LegacySR(nn.Module):
    def __init__(self):
        super().__init__();self.anchor=nn.Parameter(torch.zeros(()));self.config={'architecture':'edsr'};self.calls=[]
    def forward(self, image):
        self.calls.append(image.clone());return F.interpolate(image,scale_factor=4,mode='nearest')


class AtlasSR(GuidedSR):
    def __init__(self):
        super().__init__()
        self.config=dict(architecture='atlas_enemy_edsr',atlas_asset_sha256='a'*64,
            atlas_pixels_sha256='b'*64,atlas_buffer_shape=[7,4,444,444],
            atlas_frame_sizes=[[444,444],[444,443],[444,443],[444,444],
                               [443,444],[443,443],[443,443]],
            atlas_layout='round_4x2_Source_frames0_6',
            atlas_sampling='nearest_pixel_index_ST_bilinear_gradient',
            atlas_phase_selection='RGB_hard_argmax_ST_softmax',
            atlas_visibility='hard_forward_ST_sigmoid')


class EnemyDecoderRuntimeTests(unittest.TestCase):
    def runtime(self, sr=None, **changes):
        args=Namespace(**(dict(world=Path('world.pt'),sr=Path('sr.pt'),device='cpu',threads=1,
            fp16=False,seed=1,display_size=128,two_stage_upscale=False,max_catchup_steps=4,
            smooth=True,headless=True,fps=60,warmup=0)|changes))
        world=PredictedWorld();sr=GuidedSR()if sr is None else sr
        with patch('turkeyworld.neural_play.load_world_checkpoint',return_value=(world,{'training_steps':9})), \
             patch('turkeyworld.neural_play.load_sr_checkpoint',return_value=(sr,{'training_steps':7})), \
             patch('turkeyworld.neural_play.Game.render',return_value=np.zeros((32,32,3),np.uint8)):
            runtime=NeuralRuntime(args)
        if isinstance(sr,GuidedSR):sr.runtime=runtime
        return runtime

    def test_current_prediction_not_previous_state_is_passed_before_commit_without_Source_calls(self):
        runtime=self.runtime();old=runtime.state.clone()
        with patch('turkeyworld.neural_play.Game.step',side_effect=AssertionError('No Source step')), \
             patch('turkeyworld.neural_play.Game.render',side_effect=AssertionError('No future Source RGB')):
            frame=runtime.step([0,0,0,0])
        call=runtime.sr.calls[-1];hypothesis=call['hypothesis']
        self.assertEqual(hypothesis.shape,(1,21));self.assertTrue(torch.equal(hypothesis[:,:20],runtime.state))
        self.assertTrue(torch.equal(hypothesis[:,20:],torch.ones(1,1)))
        self.assertTrue(torch.equal(call['state_before_commit'],old))
        self.assertFalse(torch.equal(hypothesis[:,:20],old));self.assertEqual(frame.shape,(128,128,3))
        with patch('turkeyworld.neural_play.parameter_counts',return_value={}):report=runtime.report(.1)
        self.assertEqual(report['sr_enemy_decoder']['hypothesis_source'],'current_world_step_prediction')
        self.assertFalse(report['sr_enemy_decoder']['hypothesis_from_Source'])
        self.assertFalse(report['sr_enemy_decoder']['hypothesis_corrects_world_state'])

    def test_no_argument_is_RGB_only_fallback_and_never_reads_stale_runtime_state(self):
        runtime=self.runtime();runtime.state.fill_(float('nan'))
        output=runtime._resize(runtime.model.rgb)
        self.assertIsNone(runtime.sr.calls[-1]['hypothesis']);self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(runtime.upscale_stages[0]['hypothesis_source'],'none_RGB_only_fallback')

    def test_two_stage_also_passes_same_current_prediction_then_separate_bilinear(self):
        runtime=self.runtime(two_stage_upscale=True,display_size=512)
        runtime.step([0,0,0,0])
        self.assertTrue(torch.equal(runtime.sr.calls[-1]['hypothesis'][:,:20],runtime.state))
        self.assertEqual(runtime.upscale_stages[0]['output_resolution'],[128,128])
        self.assertEqual(runtime.upscale_stages[1],dict(method='bilinear',input_resolution=[128,128],
            output_resolution=[512,512],scale=4,align_corners=False))

    def test_hypothesis_keeps_FP32_state_for_half_RGB_and_invalid_proposals_are_rejected(self):
        runtime=self.runtime();state=runtime.state.detach().clone()
        runtime._resize(runtime.model.rgb.half(),predicted_state=state)
        self.assertEqual(runtime.sr.calls[-1]['hypothesis'].dtype,torch.float32)
        self.assertEqual(runtime.sr.calls[-1]['image'].dtype,torch.float16)
        self.assertTrue(torch.equal(state,runtime.state))
        with self.assertRaisesRegex(ValueError,'current predicted state'):
            runtime._resize(runtime.model.rgb,predicted_state=torch.zeros(1,21))
        state[0,0]=float('nan')
        with self.assertRaisesRegex(FloatingPointError,'Nonfinite current predicted'):
            runtime._resize(runtime.model.rgb,predicted_state=state)

    def test_legacy_SR_receives_only_RGB_and_old_resize_values_and_stage_metadata_remain_exact(self):
        runtime=self.runtime(sr=LegacySR());rgb=runtime.model.rgb
        expected=runtime.sr(rgb);runtime.sr.calls.clear()
        actual=runtime._resize(rgb,predicted_state=runtime.state)
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(len(runtime.sr.calls),1)
        self.assertEqual(runtime.upscale_stages,[dict(method='neural_x4',input_resolution=[32,32],
            output_resolution=[128,128],scale=4)])

    def test_atlas_SR_receives_current_prediction_and_reports_fixed_appearance_without_Source_queries(self):
        runtime=self.runtime(sr=AtlasSR());old=runtime.state.clone()
        with patch('turkeyworld.neural_play.Game.step',side_effect=AssertionError('No Source step')), \
             patch('turkeyworld.neural_play.Game.render',side_effect=AssertionError('No future Source RGB')):
            frame=runtime.step([0,0,0,0])
        call=runtime.sr.calls[-1]
        self.assertTrue(torch.equal(call['hypothesis'][:,:20],runtime.state))
        self.assertTrue(torch.equal(call['state_before_commit'],old));self.assertEqual(frame.shape,(128,128,3))
        with patch('turkeyworld.neural_play.parameter_counts',return_value={}):report=runtime.report(.1)
        value=report['sr_enemy_decoder'];stage=runtime.upscale_stages[0]
        for receipt in(value,stage):
            self.assertEqual(receipt['architecture'],'atlas_enemy_edsr')
            self.assertTrue(receipt['fixed_atlas']);self.assertFalse(receipt['enemy_RGB_pixel_appearance_is_learned'])
            self.assertEqual(receipt['appearance_scope'],'enemy_layer')
            self.assertEqual(receipt['background_source'],'learned_parent_SR')
            self.assertEqual(receipt['appearance_inputs'],['embedded_atlas_RGBA'])
            self.assertEqual(receipt['atlas_asset_sha256'],'a'*64);self.assertEqual(receipt['atlas_pixels_sha256'],'b'*64)
            self.assertEqual(receipt['atlas_buffer_shape'],[7,4,444,444])
            self.assertTrue(receipt['full_bbox_is_unclipped'])
            self.assertTrue(receipt['learned_animation_phase']);self.assertTrue(receipt['learned_occlusion_mask'])
            self.assertTrue(receipt['learned_binary_visibility']);self.assertFalse(receipt['Source_future_inputs'])
            self.assertFalse(receipt['Source_engine_queries_during_SR'])
            self.assertNotIn('visual_geometry_and_appearance_inputs',receipt)
        self.assertFalse(value['hypothesis_from_Source']);self.assertFalse(value['hypothesis_corrects_world_state'])
        self.assertEqual(value['hypothesis_source'],'current_world_step_prediction')

    def test_atlas_RGB_only_initial_display_never_falls_back_to_stale_state(self):
        runtime=self.runtime(sr=AtlasSR());runtime.state.fill_(float('nan'))
        value=runtime._resize(runtime.model.rgb)
        self.assertIsNone(runtime.sr.calls[-1]['hypothesis']);self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(runtime.upscale_stages[0]['hypothesis_source'],'none_independent_RGB')
        self.assertTrue(runtime.upscale_stages[0]['fixed_atlas'])

    def test_atlas_two_stage_keeps_FP32_current_hint_and_distinct_classical_resize(self):
        runtime=self.runtime(sr=AtlasSR(),two_stage_upscale=True,display_size=512)
        state=runtime.state.detach().clone()
        value=runtime._resize(runtime.model.rgb.half(),predicted_state=state)
        self.assertEqual(runtime.sr.calls[-1]['hypothesis'].dtype,torch.float32)
        self.assertTrue(torch.equal(runtime.sr.calls[-1]['hypothesis'][:,:20],state))
        self.assertEqual(value.shape,(1,3,512,512));self.assertTrue(runtime.upscale_stages[0]['fixed_atlas'])
        self.assertEqual(runtime.upscale_stages[1],dict(method='bilinear',input_resolution=[128,128],
            output_resolution=[512,512],scale=4,align_corners=False))

    def test_old_visual_first_report_has_exact_original_RGB_appearance_metadata(self):
        sr=GuidedSR();sr.config['enemy_decoder_visual_first']=True;runtime=self.runtime(sr=sr)
        runtime.step([0,0,0,0])
        with patch('turkeyworld.neural_play.parameter_counts',return_value={}):value=runtime.report(.1)['sr_enemy_decoder']
        self.assertEqual(value,dict(architecture='enemy_decoder_edsr',
            hypothesis_source='current_world_step_prediction',hypothesis_dimensions=21,
            current_predicted_state_plus_validity=True,hypothesis_from_Source=False,
            hypothesis_corrects_world_state=False,learned_neural_confirmation=True,
            gameplay_correctness_evaluated=False,visual_first=True,RSSM_hint_optional=True,
            confident_RGB_cannot_be_vetoed_by_RSSM=True,
            visual_geometry_and_appearance_inputs=['NPR_RGB32']))
        self.assertNotIn('fixed_atlas',runtime.upscale_stages[0])


if __name__=='__main__':unittest.main()
