"""Endless neural bookkeeping, legal learned spawns and checkpoint defaults."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class EndlessModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def state(self):
        from turkeyworld.engine import Game
        game = Game(endless=True)
        return game, torch.from_numpy(game.state_vector())[None]

    def transition(self, state, *, endless=True, valid=True, fire=False, target=ENTITY_SLOTS):
        from turkeyworld.structured_state import structured_transition
        positions = state[:, 8:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)[:, :, :2].clone()
        positions[:, 0] = torch.tensor([10.5 / 16, 2.5 / 16])
        targets = torch.full((1, (ENTITY_SLOTS+1)), -10.)
        targets[:, target] = 10.
        return structured_transition(state, state.clone(),
            torch.tensor([[0., 0., 0., float(fire)]]), targets, torch.full((1, ENTITY_SLOTS), -10.),
            endless=endless, spawn_positions=positions, spawn_valid=torch.full((1, ENTITY_SLOTS), valid))

    def test_ready_slot_revives_with_fresh_health_and_keeps_cumulative_kills(self):
        _, state = self.state()
        state[:, 12:14] = 0  # Slot zero HP and alive flag.
        state[:, 16] = 5.
        state[:, -2] = 21 / ENTITY_SLOTS
        result = self.transition(state)
        enemy = result['state'][:, 8:18]
        self.assertTrue(result['respawn_flags'][0, 0])
        self.assertAlmostEqual(float(enemy[0, 4]), .6, places=6)
        self.assertEqual(float(enemy[0, 5]), 1.)
        self.assertEqual(float(enemy[0, 8]), 0.)
        self.assertEqual(float(result['state'][0, -2]), 21 / ENTITY_SLOTS)
        torch.testing.assert_close(enemy[0, :2], torch.tensor([10.5 / 16, 2.5 / 16]))

    def test_respawn_requires_delay_living_player_and_legal_candidate(self):
        for timer, player_hp, valid in [(4., 1., True), (5., 0., True), (5., 1., False)]:
            with self.subTest(timer=timer, player_hp=player_hp, valid=valid):
                _, state = self.state()
                state[:, 12:14] = 0
                state[:, 16] = timer
                state[:, 4] = player_hp
                result = self.transition(state, valid=valid)
                self.assertFalse(result['respawn_flags'].any())
                self.assertEqual(float(result['state'][0, 13]), 0.)

    def test_finite_mode_keeps_absorbing_death(self):
        _, state = self.state()
        state[:, 12:14] = 0
        state[:, 16] = 1.
        result = self.transition(state, endless=False)
        self.assertFalse(result['respawn_flags'].any())
        self.assertEqual(float(result['state'][0, 13]), 0.)
        self.assertEqual(float(result['state'][0, 16]), 1.)

    def test_zero_ammo_can_fire_and_kill_count_exceeds_eight(self):
        _, state = self.state()
        state[:, 5] = 0.
        state[:, 12] = .35
        state[:, -2] = 40 / ENTITY_SLOTS
        result = self.transition(state, fire=True, target=0)
        self.assertEqual(result['events'][0, :3].tolist(), [1., 1., 1.])
        self.assertEqual(float(result['state'][0, 5]), 0.)
        self.assertEqual(float(result['state'][0, -2]), 41 / ENTITY_SLOTS)

    def movement_result(self, state, candidate, action, *, training=False):
        from turkeyworld.structured_state import structured_transition
        logits = torch.full((1, (ENTITY_SLOTS+1)), -10.)
        logits[:, ENTITY_SLOTS] = 10.
        return structured_transition(state, candidate, torch.tensor([action], dtype=torch.float32),
            logits, torch.full((1, ENTITY_SLOTS), -10.), endless=True, training=training)

    def test_player_motion_respects_commanded_axes_and_both_signs(self):
        _, state = self.state()
        for action, proposal, expected in [([1, 0, 0, 0], [.003, .003], [.003, 0.]),
                                           ([-1, 0, 0, 0], [-.003, .003], [-.003, 0.]),
                                           ([0, 1, 0, 0], [.003, .003], [0., .003]),
                                           ([0, -1, 0, 0], [.003, -.003], [0., -.003]),
                                           ([0, 0, 1, 0], [.003, .003], [0., 0.]),
                                           ([1, 0, 0, 0], [-.003, .003], [0., 0.])]:
            with self.subTest(action=action, proposal=proposal):
                candidate = state.clone()
                candidate[:, :2] += torch.tensor(proposal)
                result = self.movement_result(state, candidate, action)
                torch.testing.assert_close(result['state'][0, :2]-state[0, :2],
                                           torch.tensor(expected), rtol=0, atol=1e-7)

    def test_player_motion_uses_post_turn_heading_and_normalized_diagonal_command(self):
        _, state = self.state()
        candidate = state.clone()
        candidate[:, :2] += .02
        result = self.movement_result(state, candidate, [1, 0, 1, 0])
        angle = 2.2 / 30
        expected = torch.tensor([math.cos(angle), math.sin(angle)]) * (2.4 / 30 / 16)
        torch.testing.assert_close(result['state'][0, :2]-state[0, :2], expected, rtol=0, atol=1e-7)
        result = self.movement_result(state, candidate, [1, 1, 0, 0])
        expected = torch.full((2,), 2.4 / 30 / 16 / math.sqrt(2))
        torch.testing.assert_close(result['state'][0, :2]-state[0, :2], expected, rtol=0, atol=1e-7)

    def test_clipped_player_proposals_retain_training_gradients(self):
        _, state = self.state()
        for delta, desired, gradient_sign in [(-.003, .005, -1), (.02, .0025, 1)]:
            with self.subTest(delta=delta):
                candidate = state.clone()
                candidate[:, 0] += delta
                candidate.requires_grad_()
                result = self.movement_result(state, candidate, [1, 0, 0, 0], training=True)
                loss = (result['state'][0, 0]-state[0, 0]-desired).square()
                loss.backward()
                self.assertTrue(torch.isfinite(candidate.grad).all())
                self.assertGreater(float(candidate.grad[0, 0])*gradient_sign, 0.)

    def test_decoder_rejects_walls_and_near_player_but_not_own_dead_slot(self):
        from turkeyworld.spawn import decode_spawn_positions
        game, state = self.state()
        state[:, 12:14] = 0.
        state[:, 16] = 5.
        state[:, 8:10] = torch.tensor([12.5 / 16, 2.5 / 16])
        logits = torch.zeros(1, ENTITY_SLOTS, 256)
        logits[0, 0, 0] = 1000.  # Wall.
        logits[0, 0, 2 * 16 + 2] = 900.  # Player's tile.
        logits[0, 0, 2 * 16 + 12] = 800.  # Own dead position is a legal respawn.
        logits[0, 0, 1 * 16 + 14] = 700.  # Lower-scoring legal tile.
        positions, valid = decode_spawn_positions(logits, state, torch.tensor(game.map == 0), 1 / 30)
        self.assertTrue(valid[0, 0])
        torch.testing.assert_close(positions[0, 0], torch.tensor([12.5 / 16, 2.5 / 16]))
        self.assertEqual(tuple(valid.shape), (1, ENTITY_SLOTS))
        self.assertGreaterEqual(float(torch.linalg.vector_norm(positions[0, 0]-state[0, :2])*16), 4.)

    def test_ready_slot_uses_available_floor_and_waits_when_no_candidate_is_legal(self):
        from turkeyworld.spawn import decode_spawn_positions
        game, state = self.state()
        entities = state[:, 8:-2].reshape(1, ENTITY_SLOTS, ENTITY_DIM)
        entities[:, :, 4:6] = 0.
        entities[:, :, 8] = 5.
        logits = torch.zeros(1, ENTITY_SLOTS, 256)
        positions, valid = decode_spawn_positions(logits, state, torch.tensor(game.map == 0), 1 / 30)
        self.assertTrue(valid.all())
        x, y = (positions[0, 0]*16).long().tolist()
        self.assertEqual(game.map[y, x], 0)
        self.assertGreaterEqual(float(torch.linalg.vector_norm(positions[0, 0]-state[0, :2])*16), 4.)
        _, valid = decode_spawn_positions(logits, state, torch.zeros(16, 16, dtype=torch.bool), 1 / 30)
        self.assertFalse(valid.any())
        only_near_player = torch.zeros(16, 16, dtype=torch.bool)
        only_near_player[2, 2] = True
        _, valid = decode_spawn_positions(logits, state, only_near_player, 1 / 30)
        self.assertFalse(valid.any())

    def test_both_architectures_predict_trainable_spawn_logits_without_engine_queries(self):
        from turkeyworld.engine import Game
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        for architecture in ['legacy_cnn', 'camera_transformer']:
            with self.subTest(architecture=architecture):
                game, state = self.state()
                state[:, 12:14] = 0.
                state[:, 16] = 5.
                model = NeuralWorldModel(WorldModelConfig(endless=True, image_size=128,
                                                           architecture=architecture))
                with patch.object(Game, 'step', side_effect=AssertionError('No runtime engine step')), \
                     patch.object(Game, 'render', side_effect=AssertionError('No runtime engine render')):
                    features = model.encode_map(torch.tensor(game.map, dtype=torch.long)[None])
                    result = model.transition(state, torch.zeros(1, 4), features)
                self.assertEqual(result['spawn_logits'].shape, (1, ENTITY_SLOTS, 256))
                self.assertTrue(result['respawn_flags'][0, 0])
                target = torch.tensor([1 * 16 + 14])
                loss = torch.nn.functional.cross_entropy(result['spawn_logits'][:, 0], target)
                loss.backward()
                backend = getattr(model, 'camera_backend', model)
                gradient = backend.spawn_head[-1].weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_old_checkpoints_default_to_direct_finite_mode(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
        model = NeuralWorldModel(WorldModelConfig(image_size=128))
        config = asdict(model.config)
        for key in ['endless', 'renderer_mode', 'flow_steps']:
            config.pop(key)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy_defaults.pt'
            torch.save({'stage': 'world', 'config': config, 'model': model.state_dict()}, path)
            restored, _ = load_world_checkpoint(path)
        self.assertFalse(restored.config.endless)
        self.assertEqual(restored.config.renderer_mode, 'direct')
        self.assertFalse(hasattr(restored, 'flow_renderer'))


if __name__ == '__main__':
    unittest.main()
