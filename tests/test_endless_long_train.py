"""Motion and spawn supervision for the recorded-state endless TBPTT stream."""
from dataclasses import asdict
import math
from pathlib import Path
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class EndlessLongTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_motion_is_scaled_to_physical_ticks(self):
        from turkeyworld.engine import Game
        from turkeyworld.long_train import endless_auxiliary_losses
        old = torch.tensor(Game(endless=True).state_vector())[None]
        target = old.clone()
        target[:, 0] += (2.4 / 30) / 16
        target[:, 8] += (1.2 / 30) / 16
        losses = endless_auxiliary_losses({'state': old.clone()}, old, target, 30.)
        self.assertAlmostEqual(float(losses['motion']), .25 + .5/16, places=4)
        self.assertEqual(float(losses['spawn']), 0.)

    def test_teleports_are_excluded_and_only_revived_slots_train_spawn_logits(self):
        from turkeyworld.engine import Game
        from turkeyworld.long_train import endless_auxiliary_losses
        old = torch.tensor(Game(endless=True).state_vector())[None]
        old[:, 12:14] = 0.
        target = old.clone()
        target[:, 8:10] = torch.tensor([14.5/16, 1.5/16])
        target[:, 12:14] = torch.tensor([.6, 1.])
        logits = torch.zeros(1, 8, 256, requires_grad=True)
        losses = endless_auxiliary_losses({'state': old.clone(), 'spawn_logits': logits}, old, target, 30.)
        self.assertEqual(float(losses['motion']), 0.)
        self.assertAlmostEqual(float(losses['spawn'].detach()), math.log(256), places=5)
        losses['spawn'].backward()
        self.assertLess(float(logits.grad[0, 0, 30]), 0.)
        self.assertEqual(float(logits.grad[0, 1:].abs().sum()), 0.)

    def test_endless_tbptt_backpropagates_spawn_loss_across_carried_chunks(self):
        from turkeyworld.engine import Game
        from turkeyworld.long_train import LongEpisode, run_segment
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        game = Game(seed=17, endless=True)
        game.enemies[0].update(alive=False, hp=0., death_timer=2.95)
        game.kills = 12
        states = [game.state_vector()]
        actions, events, hits, attacks = [], [], [], []
        for _ in range(6):
            action = np.array([.5, 0., 0., 0.], dtype=np.float32)
            event = game.step(action)
            states.append(game.state_vector())
            actions.append(action)
            events.append(Game.event_vector(event))
            hits.append(event['hit_id'])
            flags = np.zeros(8, dtype=np.float32)
            flags[event['attacker_ids']] = 1.
            attacks.append(flags)
        episode = LongEpisode(Path('actual_respawn.npz'), np.asarray(states), np.asarray(actions),
                              np.asarray(events), game.map.astype(np.int64), np.asarray(hits),
                              np.asarray(attacks), 'collector_exact', dt=1/30)
        model = NeuralWorldModel(WorldModelConfig(endless=True, architecture='legacy_cnn', image_size=128))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        original = model.spawn_head[-1].weight.detach().clone()
        result = run_segment(model, episode, optimizer=optimizer, burn_in=0, rollout=6,
                             chunk=2, geometry_every=0, collision_weight=0)
        self.assertGreater(result['metrics']['spawn'], 0.)
        self.assertTrue(np.isfinite(result['metrics']['motion']))
        self.assertGreater(float((model.spawn_head[-1].weight.detach()-original).abs().sum()), 0.)
        declaration = result['declaration']
        self.assertEqual(declaration['ground_truth_initializations'], 1)
        self.assertEqual(declaration['optimizer_steps'], 3)
        self.assertTrue(declaration['endless_auxiliary_losses']['npc_motion_excludes_respawn_teleports'])
        for boundary in declaration['chunk_boundaries'][1:]:
            self.assertEqual(boundary['state_carry_max_abs_difference'], 0.)
            self.assertEqual(boundary['memory_carry_max_abs_difference'], 0.)

    def test_config_normalization_preserves_old_resume_defaults(self):
        from turkeyworld.models import WorldModelConfig
        from turkeyworld.train import normalize_world_config
        config = asdict(WorldModelConfig(architecture='legacy_cnn', transition_mode='residual_v0'))
        previous = {key: value for key, value in config.items()
                    if key not in {'endless', 'renderer_mode', 'flow_steps'}}
        self.assertEqual(normalize_world_config(previous), config)
        current = asdict(WorldModelConfig(endless=True, renderer_mode='flow_matching', flow_steps=12))
        self.assertEqual(normalize_world_config(current), current)


if __name__ == '__main__':
    unittest.main()
