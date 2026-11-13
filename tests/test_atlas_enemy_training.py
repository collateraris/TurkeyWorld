import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from turkeyworld.atlas_enemy_migration import prepare_atlas_enemy_continuation
from turkeyworld.models import build_upscaler,load_sr_checkpoint
from turkeyworld.train_atlas_enemy_sr import run
from turkeyworld.enemy_animation import enemy_animation_index
from test_decoder_restoration_training import RestorationTrainingTests
from test_sr_patchgan_training import MatchedPairs,equal


class AtlasPairs(MatchedPairs):
    def batch(self,*args,**kwargs):
        result=super().batch(*args,**kwargs)
        result['animation_label']=enemy_animation_index(result['state'])
        return result


def evaluate(model,reference,data,panel,*,step,batch_size=8,baseline_report=None):
    batch=data.batch(8,indices=panel)
    with torch.no_grad():
        model.forward_with_aux(batch['lr_npr'],batch['hypothesis'])
        reference.forward_with_aux(batch['lr_npr'],batch['hypothesis'])
    return dict(kind='atlas_enemy_hybrid_validation_v1',step=step,
        panel_fingerprint=data.panel_fingerprint(panel),data_identity_sha256='data',
        temporal={'panel_fingerprint':'temporal'},models={}),dict(metadata={})


class AtlasTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def initial(self,root):
        parent,_,_=RestorationTrainingTests().initial(root)
        before=parent.read_bytes();rng=torch.get_rng_state().clone()
        initial=root/'atlas_initial.pt'
        receipt=prepare_atlas_enemy_continuation(parent,initial)
        self.assertEqual(before,parent.read_bytes());equal(self,rng,torch.get_rng_state())
        old=torch.load(parent,weights_only=False);new=torch.load(initial,weights_only=False)
        for name,value in old['model'].items():equal(self,value,new['model'][name])
        for index,values in old['optimizer']['state'].items():equal(self,values,new['optimizer']['state'][index])
        for key in ('sampler','rng_numpy','rng_torch','rng_python','rng_cuda'):equal(self,old[key],new[key])
        self.assertEqual(new['optimizer_parameter_names'][:-1],old['optimizer_parameter_names'])
        self.assertEqual(new['optimizer_parameter_names'][-1],['atlas_player_dead_head.weight','atlas_player_dead_head.bias'])
        self.assertEqual(len(new['optimizer']['state']),158)
        self.assertNotIn('restoration_schedule',new)
        self.assertEqual(receipt['added_parameters'],17)
        with patch('PIL.Image.open',side_effect=AssertionError('External atlas read during checkpoint loading')):
            model,_=load_sr_checkpoint(initial,'cpu')
        self.assertEqual(model.config['architecture'],'atlas_enemy_edsr')
        self.assertTrue(model._atlas_ready)
        return initial,parent,old

    def test_full_state_continuation_is_selfcontained_and_exact_after_real_SGD_resume(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);initial,reference,parent=self.initial(root)
            def invoke(path,updates):
                args=argparse.Namespace(run_dir=path,initial_checkpoint=initial,reference_checkpoint=reference,
                    data_root=root/'data',hypotheses_root=root/'hyp',updates=updates,seed=173,device='cpu')
                with patch('turkeyworld.enemy_decoder_data.EnemyDecoderData',AtlasPairs), \
                     patch('turkeyworld.atlas_enemy_validation.evaluate_atlas_enemy_sr',evaluate), \
                     patch('turkeyworld.atlas_enemy_validation.save_atlas_enemy_preview',
                           side_effect=lambda preview,path,every:dict(path=str(path),every=every)):
                    run(args)
            invoke(root/'continuous',2);invoke(root/'resumed',1);invoke(root/'resumed',2)
            a=torch.load(root/'continuous/checkpoints/sr_last.pt',weights_only=False)
            b=torch.load(root/'resumed/checkpoints/sr_last.pt',weights_only=False)
            for key in('model','optimizer','sampler','rng_numpy','rng_torch','rng_python','rng_cuda'):
                equal(self,a[key],b[key])
            self.assertEqual(a['atlas_updates'],2)
            self.assertEqual(a['training_steps'],96202)
            self.assertEqual(a['sampler']['draws'],44016)
            self.assertEqual(len(a['optimizer']['state']),160)
            for name in a['atlas_migration']['frozen_parameter_names']:
                equal(self,parent['model'][name],a['model'][name])
            initial_payload=torch.load(initial,weights_only=False)
            for name in('atlas_frames_rgba','atlas_native_sizes'):
                equal(self,initial_payload['model'][name],a['model'][name])
            for name in('atlas_player_dead_head.weight','atlas_player_dead_head.bias'):
                self.assertFalse(torch.equal(a['model'][name],initial_payload['model'][name]))


if __name__=='__main__':unittest.main()
