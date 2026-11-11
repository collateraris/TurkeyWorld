"""Actual EDSR CPU optimizer/resume equivalence for the future tiny-SR stage."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.models import EDSRUpscaler
from turkeyworld.renderer_student import _hash_tensors
from turkeyworld.train_small_enemy_sr import run
from test_small_enemy_sr import PairedHeldout,TinyFrozenCodec


class Data:
    def __init__(self,root,split,*,seed,device,**kwargs):
        fixture=PairedHeldout()
        self.values={k:v.detach()for k,v in fixture.batch(6,indices=list(range(6))).items()}
        self.split=split;self.rng=np.random.default_rng(seed);self.draws=0;self.case_counts=dict.fromkeys(range(6),0)
        self.scene_keys={split};self.identity={'split':split,'fingerprint':'fixed_'+split,'examples':6}
    def fixed_panel(self,count,seed=173):return list(range(count))
    def panel_fingerprint(self,panel):return self.split+str(panel)
    def batch(self,size,indices=None,rng=None):
        if indices is None:
            indices=(self.rng if rng is None else rng).integers(6,size=size)
            self.draws+=size
            for i in indices:self.case_counts[int(i)]+=1
        return {k:v[indices]for k,v in self.values.items()}
    def sampler_state(self):return {'rng':deepcopy(self.rng.bit_generator.state),'draws':self.draws,'case_counts':self.case_counts.copy()}
    def restore_sampler(self,saved):
        self.rng.bit_generator.state=saved['rng'];self.draws=saved['draws'];self.case_counts=saved['case_counts'].copy()


class SmallEnemySRResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_actual_two_update_resume_is_bitexact_for_model_adam_rng_sampler_and_export_clocks(self):
        with TemporaryDirectory()as directory:
            root=Path(directory);parent=root/'parent.pt';codec_path=root/'codec.pt'
            parent.write_bytes(b'immutable selected-best SR');codec_path.write_bytes(b'immutable selected tiny codec')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(4);template=EDSRUpscaler()
            codecs=[]
            def codec(*args):
                value=TinyFrozenCodec().eval().requires_grad_(False);codecs.append(value)
                return value,{'tensor_sha256':_hash_tensors(value.state_dict())}
            def args(name,steps):return SimpleNamespace(run_dir=root/name,data_root=root/'data',parent=parent,
                codec_checkpoint=codec_path,device='cpu',batch=2,steps=steps,lr=5e-5,seed=173,validation_frames=6)
            with patch('turkeyworld.train_small_enemy_sr.load_sr_checkpoint',side_effect=lambda *a:(deepcopy(template),{'training_steps':38100})), \
                    patch('turkeyworld.train_small_enemy_sr.load_augmentation_codec',side_effect=codec), \
                    patch('turkeyworld.small_enemy_data.SmallEnemyData',Data):
                run(args('direct',2));run(args('resumed',1));run(args('resumed',2))
            left=torch.load(root/'direct/checkpoints/sr_last.pt',weights_only=False)
            right=torch.load(root/'resumed/checkpoints/sr_last.pt',weights_only=False)
            self.assertEqual(left['completed_updates'],2);self.assertEqual(right['completed_updates'],2)
            self.assertEqual(left['training_steps'],38102);self.assertEqual(left['sampler']['draws'],4)
            self.assertTrue(all(torch.equal(v,right['model'][k])for k,v in left['model'].items()))
            self.assertEqual(left['optimizer']['param_groups'],right['optimizer']['param_groups'])
            for i,values in left['optimizer']['state'].items():
                for key,value in values.items():self.assertTrue(torch.equal(value,right['optimizer']['state'][i][key]),key)
            self.assertEqual({int(v['step'])for v in right['optimizer']['state'].values()},{2})
            self.assertEqual(left['rng_numpy'],right['rng_numpy'])
            self.assertTrue(torch.equal(left['rng_torch'],right['rng_torch']))
            self.assertEqual(left['rng_python'],right['rng_python'])
            self.assertEqual(left['sampler'],right['sampler']);self.assertEqual(left['best'],right['best'])
            exported=torch.load(root/'resumed/checkpoints/sr_for_game.pt',weights_only=False)
            self.assertEqual(exported['training_steps'],38100+right['best']['step'])
            self.assertEqual(exported['refinement_updates'],right['best']['step'])
            self.assertFalse(exported['automatic_deployment']);self.assertFalse(exported['gameplay_qualified'])
            self.assertTrue(all(p.grad is None for c in codecs for p in c.parameters()))
            self.assertTrue(all(torch.equal(c.unchanged,torch.arange(3))for c in codecs))
            self.assertEqual(parent.read_bytes(),b'immutable selected-best SR')
            self.assertEqual(codec_path.read_bytes(),b'immutable selected tiny codec')


if __name__=='__main__':unittest.main()
