import unittest
from unittest.mock import patch
import torch
from turkeyworld.weapon_memory_sr import WeaponMemory,initialize_weapon_memory,weapon_labels,weapon_objective
from turkeyworld.engine import Game

class WeaponTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_memory_matches_source_idle_and_needs_no_runtime_engine(self):
        m=WeaponMemory();initialize_weapon_memory(m);m.eval()
        game=Game(seed=0);s=torch.tensor(game.state_vector())[None];phase,mask=weapon_labels(s)
        with patch.object(Game,'render',side_effect=AssertionError('No engine inference')):
            o=m(torch.rand(1,3,32,32))
        self.assertTrue(torch.equal((o['alpha']>0).float(),mask));self.assertEqual(int(o['phase_logits'].argmax()),0)
        self.assertEqual(o['rgb'].shape,(1,3,128,128))
        canvas=game.render(128,overlays=False)
        target=torch.tensor(canvas.copy()).permute(2,0,1)[None].float()/255
        error=((o['rgb']-target).abs()*(o['alpha']>.999)).max()
        self.assertLess(float(error.detach()),1.1e-5)
    def test_flash_labels_and_classifier_gradient(self):
        m=WeaponMemory();initialize_weapon_memory(m);m.train();game=Game(seed=0)
        states=torch.tensor(game.state_vector())[None].repeat(3,1);states[:,7]=torch.tensor([0.,1.,.01])
        phases,masks=weapon_labels(states);self.assertEqual(phases.tolist(),[0,1,2])
        o=m(torch.rand(3,3,32,32));image=o['rgb']*o['alpha']
        loss,metrics=weapon_objective(dict(image=image,weapon_memory=o),dict(state=states,hr_source=torch.rand_like(image)));loss.backward()
        self.assertGreater(float(m.phase_head[-1].weight.grad.abs().sum()),0)
        self.assertTrue(bool(torch.isfinite(m.rgb_logits.grad).all()))
