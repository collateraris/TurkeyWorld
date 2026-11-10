"""Explicit no-enemy Source/neural profiles retain ordinary player physics."""
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.engine import Game, STATE_DIM
from turkeyworld.models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
from turkeyworld.structured_state import structured_transition


class NavigationEnvironmentProfileTests(unittest.TestCase):
    def test_old_checkpoint_without_optional_profile_restores_weights_and_optimizer(self):
        from turkeyworld.training_pipeline import Experiment
        original_rng=torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state,original_rng)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'checkpoints').mkdir()
            model=torch.nn.Linear(2,2)
            model.config=WorldModelConfig(image_size=32,architecture='legacy_cnn')
            optimizer=torch.optim.AdamW(model.parameters())
            model(torch.ones(1,2)).sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
            config=asdict(model.config);config.pop('navigation_only')
            generator=np.random.default_rng(17)
            torch.save({'config':config,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'training_steps':7,'rng_numpy':generator.bit_generator.state,'rng_torch':original_rng},
                root/'checkpoints/teacher_world.pt')
            experiment=object.__new__(Experiment)
            experiment.root=root;experiment.device=torch.device('cpu');experiment.state={};experiment.rng=generator
            self.assertEqual(experiment.restore(model,optimizer,'teacher_world'),7)
            self.assertEqual(len(optimizer.state),2)
            model.config.navigation_only=True
            with self.assertRaisesRegex(ValueError,'Configuration mismatch'):
                experiment.restore(model,optimizer,'teacher_world')

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_and_explicit_enabled_games_remain_exact_through_kill_and_respawn(self):
        default, explicit = Game(173, endless=True), Game(173, endless=True, enemies_enabled=True)
        saw_dead = saw_respawn = False
        for index in range(180):
            action = ([0., 0., 0., 1.] if index < 24 else
                      [.35, .15 if index % 40 < 20 else -.15, .1, 0.])
            before = default.enemies[0]['alive']
            first, second = default.step(action), explicit.step(action)
            self.assertEqual(first, second)
            np.testing.assert_array_equal(default.state_vector(), explicit.state_vector())
            saw_dead |= not default.enemies[0]['alive']
            saw_respawn |= not before and default.enemies[0]['alive']
            if index % 19 == 0 or index == 179:
                np.testing.assert_array_equal(default.render(32, False), explicit.render(32, False))
        self.assertTrue(saw_dead and saw_respawn)
        self.assertGreater(default.time, 3.)
        self.assertEqual(default.get_state(), explicit.get_state())
        self.assertNotIn('enemies_enabled', default.get_state())  # old serialization unchanged

    def test_disabled_profile_runs451_steps_without_resets_repairs_respawn_or_visible_corpse(self):
        game = Game(71, endless=True, enemies_enabled=False)
        self.assertEqual(len(game.enemies), 1)
        with patch.object(game, 'reset', side_effect=AssertionError('No reset inside trajectory')), \
                patch.object(game, 'set_state', side_effect=AssertionError('No state correction')):
            previous_time = 0.
            for index in range(451):
                event = game.step([.2, 0., .5, 0.])
                state = game.state_vector()
                self.assertEqual(state.shape, (STATE_DIM,))
                self.assertEqual(float(state[4]), 1.)
                self.assertFalse(state[12:14].any())
                self.assertFalse(Game.event_vector(event).any())
                self.assertAlmostEqual(game.time-previous_time, 1/30, places=12)
                previous_time = game.time
            self.assertFalse(game.done)
        self.assertAlmostEqual(game.time, 451/30, places=10)
        self.assertAlmostEqual(game.enemies[0]['death_timer'], 3.)
        # A front-facing, fully animated corpse is still absent because this
        # is an explicit environment profile, not a near-clip initialization.
        hidden = Game(71, endless=True, enemies_enabled=False)
        state = hidden.get_state(); state['enemies'][0].update(x=4.5, y=2.5, death_timer=.6)
        hidden.set_state(state)
        no_enemy = hidden.render(32, False)
        enabled = Game(71, endless=True)
        state.pop('enemies_enabled')
        enabled.set_state(state)
        self.assertTrue(np.any(no_enemy != enabled.render(32, False)))

    def test_profile_roundtrip_is_persistent_and_refuses_active_reserved_slots(self):
        original = Game(79, endless=True, enemies_enabled=False)
        for _ in range(120): original.step([0., 0., .1, 0.])
        saved = json.loads(json.dumps(original.get_state()))
        restored = Game(999, endless=True)
        restored.set_state(saved)
        self.assertFalse(restored.enemies_enabled)
        self.assertEqual(restored.get_state(), original.get_state())
        np.testing.assert_array_equal(restored.render(32, False), original.render(32, False))
        for _ in range(5):
            self.assertEqual(original.step([.1, 0., 0., 0.]), restored.step([.1, 0., 0., 0.]))
            np.testing.assert_array_equal(original.state_vector(), restored.state_vector())
        for alive, hp in ((True, 60.), (False, 25.), (True, 0.)):
            invalid = copy.deepcopy(saved); invalid['enemies'][0].update(alive=alive, hp=hp)
            before = restored.get_state()
            with self.subTest(alive=alive, hp=hp), self.assertRaisesRegex(ValueError, 'inactive'):
                restored.set_state(invalid)
            self.assertEqual(restored.get_state(), before)

    def test_typed450_step_profile_matches_source_and_enabled_route_genuinely_respawns(self):
        source = Game(41, endless=True, enemies_enabled=False)
        current = enabled = torch.tensor(source.state_vector())[None]
        action = torch.tensor([[.2, 0., .5, 0.]])
        target = torch.tensor([[-20., 20.]])
        attacks = torch.tensor([[-20.]])
        positions, valid = torch.tensor([[[13.5/16, 13.5/16]]]), torch.ones(1, 1, dtype=torch.bool)
        enabled_respawns = 0
        for index in range(450):
            def proposal(previous):
                # Disclosed kinematics in empty floor, calculated only from
                # OWN previous state/action; no future Source state is input.
                yaw = math.atan2(float(previous[0, 2]), float(previous[0, 3]))+.5*2.2/30
                candidate = previous.clone()
                candidate[:, :2] += torch.tensor([[math.cos(yaw), math.sin(yaw)]])*(.2*2.4/30/16)
                return candidate
            actual = structured_transition(current, proposal(current), action, target, attacks,
                endless=True, respawn_enabled=False, spawn_positions=positions, spawn_valid=valid)
            other = structured_transition(enabled, proposal(enabled), action, target, attacks,
                endless=True, respawn_enabled=True, spawn_positions=positions, spawn_valid=valid)
            current, enabled = actual['state'], other['state']
            enabled_respawns += int(other['respawn_flags'].sum())
            source.step(action[0].numpy())
            np.testing.assert_allclose(current[0].numpy(), source.state_vector(), rtol=0, atol=2e-5)
            self.assertFalse(actual['respawn_flags'].any())
            self.assertFalse(current[:, 12:14].any())
        self.assertEqual(enabled_respawns, 1)
        self.assertEqual(float(enabled[0, 13]), 1.)
        self.assertEqual(float(enabled[0, 12]), float(torch.tensor(.6)))
        self.assertAlmostEqual(float(current[0, -1]*300), 15., places=3)

    def test_serialized_bool_profile_and_all_three_backends_forward_respawn_permission(self):
        for wrong in (1, 'true', None):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, 'boolean'):
                WorldModelConfig(navigation_only=wrong)
        self.assertFalse(WorldModelConfig().navigation_only)
        source = Game(73, endless=True, enemies_enabled=False)
        for _ in range(80): source.step([0., 0., 0., 0.])
        state = torch.from_numpy(source.state_vector())[None]
        image = torch.from_numpy(source.render(32, False)).permute(2, 0, 1)[None].float()/255
        grid = torch.tensor(source.map)[None].long()
        for architecture in ('legacy_cnn', 'camera_transformer', 'vjepa2_ac'):
            for profile in (False, True):
                with self.subTest(architecture=architecture, navigation_only=profile):
                    config = WorldModelConfig(kind='student', image_size=32, architecture=architecture,
                        endless=True, navigation_only=profile,
                        vjepa_width=32, vjepa_encoder_depth=1, vjepa_predictor_depth=1,
                        vjepa_heads=4, vjepa_mlp_ratio=1, vjepa_readout_mlp_ratio=1)
                    model = NeuralWorldModel(config).eval()
                    with torch.inference_mode():
                        maps = model.encode_map(grid)
                        memory = model.initialize_from_image(image, state, maps)
                        with patch('turkeyworld.structured_state.structured_transition', wraps=structured_transition) as spy, \
                                patch.object(Game, 'step', side_effect=AssertionError('No runtime engine')), \
                                patch.object(Game, 'render', side_effect=AssertionError('No runtime render')):
                            result = model.transition(state, torch.zeros(1, 4), maps, memory)
                    self.assertEqual(spy.call_count, 1)
                    self.assertEqual(spy.call_args.kwargs['respawn_enabled'], not profile)
                    self.assertEqual(bool(result['respawn_flags'][0, 0]), not profile)
                    if architecture == 'vjepa2_ac' and profile:
                        with tempfile.TemporaryDirectory() as folder:
                            path = Path(folder)/'navigation.pt'
                            torch.save({'stage': 'world', 'config': asdict(config),
                                'model': model.state_dict(), 'training_steps': 1}, path)
                            restored, _ = load_world_checkpoint(path, 'cpu')
                            self.assertIs(restored.config.navigation_only, True)
                            self.assertIs(restored.vjepa_backend.config.navigation_only, True)
                            for name, value in model.state_dict().items():
                                torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
