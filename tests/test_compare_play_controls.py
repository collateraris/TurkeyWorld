"""Scenario keys and initial navigation observation for the paired diagnostic."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_compare_runtime import NeuralStub, SourceStub, PoseRenderer
from test_renderer_enemy_objectives import tiny_model
from test_rssm_runtime import arguments
from turkeyworld.compare_play import keyboard_action, switch_test_environment
from turkeyworld.compare_runtime import PairedRuntime
from turkeyworld.engine import Game
from turkeyworld.neural_play import NeuralRuntime


class ComparePlayControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_t_is_a_scene_key_and_e_is_right_turn_in_the_shared_action_vector(self):
        names=['K_w','K_UP','K_s','K_DOWN','K_d','K_a','K_e','K_t','K_RIGHT','K_q','K_LEFT','K_SPACE']
        pygame=SimpleNamespace(**{name:index for index,name in enumerate(names)},
            mouse=SimpleNamespace(get_pressed=lambda count:(False,False,False)))
        keys={index:False for index in range(len(names))};keys[pygame.K_t]=True
        self.assertEqual(keyboard_action(pygame,keys),[0.,0.,0.,0.])
        keys[pygame.K_e]=True
        self.assertEqual(keyboard_action(pygame,keys),[0.,0.,1.,0.])
        for key in ('K_w','K_d','K_RIGHT','K_SPACE'):keys[getattr(pygame,key)]=True
        self.assertEqual(keyboard_action(pygame,keys),[1.,1.,1.,1.])

    def test_scene_switch_preserves_running_trial_and_restarts_only_an_exhausted_trial(self):
        pair=PairedRuntime(NeuralStub(),source_factory=SourceStub,render_pose=PoseRenderer(),horizon=2)
        pair.step([1,0,0,0]);hidden=pair.neural.memory.clone()
        switch_test_environment(pair)
        self.assertEqual((pair.status()['environment'],pair.steps,pair.resets),('combat',1,0))
        torch.testing.assert_close(pair.neural.memory,hidden,rtol=0,atol=0)
        pair.step([0,0,0,0]);self.assertTrue(pair.finished)
        switch_test_environment(pair)
        self.assertEqual((pair.status()['environment'],pair.steps,pair.resets),('navigation',0,1))
        self.assertFalse(pair.finished)

    def test_navigation_profile_applies_after_asset_checks_before_initial_rgb_and_memory(self):
        model=tiny_model()
        model.config=replace(model.config,navigation_only=False)
        model.vjepa_backend.config=replace(model.vjepa_backend.config,navigation_only=False)
        metadata={'training_steps':4250,'config':asdict(model.config)}
        before={key:value.clone()for key,value in model.state_dict().items()}
        events=[]
        def validated_renderer(candidate,path,**kwargs):
            self.assertFalse(candidate.config.navigation_only)
            self.assertFalse(candidate.vjepa_backend.config.navigation_only)
            events.append('strict_renderer_identity');return {'training_steps':6000}
        args=arguments(initial_navigation_only=True,renderer_checkpoint=Path('renderer.pt'),warmup=0)
        with patch('turkeyworld.neural_play.load_world_checkpoint',return_value=(model,metadata)),\
                patch('turkeyworld.renderer_student.load_renderer_student',side_effect=validated_renderer),\
                patch.object(model,'initialize_from_image',wraps=model.initialize_from_image)as observe,\
                patch('turkeyworld.neural_play.Game.step',side_effect=AssertionError('No Source future')):
            runtime=NeuralRuntime(args)
            self.assertEqual(events,['strict_renderer_identity'])
            self.assertTrue(model.config.navigation_only and model.vjepa_backend.config.navigation_only)
            self.assertFalse(metadata['config']['navigation_only'])
            reference=Game(seed=0,endless=True,enemies_enabled=False)
            np.testing.assert_array_equal(runtime.initial_rgb,reference.render(size=32,overlays=False))
            self.assertEqual(observe.call_count,1)
            torch.testing.assert_close(observe.call_args.args[0],
                torch.from_numpy(runtime.initial_rgb).permute(2,0,1)[None].float()/255,rtol=0,atol=0)
            self.assertFalse(bool(runtime.initial_state[0,13]))
            runtime.step([0,0,0,0])
            self.assertEqual(observe.call_count,1)
        self.assertTrue(all(torch.equal(value,model.state_dict()[key])for key,value in before.items()))


if __name__=='__main__':unittest.main()
