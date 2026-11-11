"""HP0 is a genuine player-terminal Source cohort, separate from NPC corpses."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_renderer_combat_data import mixture
from test_renderer_enemy_training import fake_bounds
from turkeyworld.engine import BEHAVIOR_VERSION, Game
from turkeyworld.staged_data import _observe
from turkeyworld.codec_battle_data import CodecBattleFrames, render_condition_key


def _terminal(root, split):
    seed = 701 if split == 'train' else 801
    game = Game(seed=seed, endless=True)
    state = game.get_state()
    state['player'].update(x=2.5, y=2.5, angle=0., hp=8.)
    state['enemies'][0].update(x=3.3, y=2.5, hp=60., alive=True,
                              attack_cooldown=0., vx=0., vy=0.)
    game.set_state(state)
    before = game.state_vector()
    rgb0, mask0, bg0 = _observe(game, 32)
    action = np.zeros(4, np.float32)
    event = game.step(action, dt=1/30)
    assert game.done and game.player['hp'] == 0 and event['player_hit']
    rgb, mask, bg = _observe(game, 32)
    path = root/'full_game/rgb'/split/'terminal_source.npz'
    values = dict(states=np.stack([before, game.state_vector()]), frames=np.stack([rgb0, rgb]),
        object_masks=np.stack([mask0, mask]), background_frames=np.stack([bg0, bg]),
        map=game.map, actions=action[None], events=Game.event_vector(event)[None],
        split=np.asarray(split), behavior_version=np.asarray(BEHAVIOR_VERSION),
        world_transition_compatible=np.asarray(True), seed=np.asarray(seed))
    np.savez_compressed(path, **values)
    return path, values


class CodecBattleDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def _data(self, folder, split='train'):
        root, appearance = Path(folder)/'combat', Path(folder)/'appearance'
        mixture(root, appearance, split)
        path, terminal = _terminal(root, split)
        return CodecBattleFrames(root, appearance, split), path, terminal

    def test_terminal_is_player_hp0_and_not_npc_corpse_and_pixels_are_exact_source(self):
        with TemporaryDirectory() as folder, patch(
                'turkeyworld.renderer_enemy_data.project_sprite_bounds', side_effect=fake_bounds):
            data, path, terminal = self._data(folder)
            source_bytes = path.read_bytes()
            self.assertEqual(len(data.terminal_indices), 1)
            index = data.terminal_indices[0]
            with patch.object(Game, 'step', side_effect=AssertionError('No Source in training loader')), \
                    patch.object(Game, 'render', side_effect=AssertionError('No Source in training loader')):
                batch = data.batch(1, indices=[index])
            self.assertEqual(float(batch['state'][0, 4]), 0.)
            self.assertGreater(float(batch['state'][0, 12]), 0.)
            self.assertTrue(bool(batch['dynamic'][0]))
            np.testing.assert_array_equal((batch['target'][0].permute(1, 2, 0)*255).round().byte().numpy(),
                                          terminal['frames'][1])
            np.testing.assert_array_equal((batch['previous_rgb'][0].permute(1, 2, 0)*255).round().byte().numpy(),
                                          terminal['frames'][0])
            for corpse in data.buckets['corpse']:
                self.assertNotIn(corpse, data.terminal_indices)
            rows = data.cohort_rows([index])
            self.assertEqual(rows['player_hp0'], [0])
            self.assertEqual(rows['player_alive'], [])
            self.assertEqual(path.read_bytes(), source_bytes)

    def test_training_balances_four_cohorts_and_restores_exact_next_draw(self):
        with TemporaryDirectory() as folder, patch(
                'turkeyworld.renderer_enemy_data.project_sprite_bounds', side_effect=fake_bounds):
            data, _, _ = self._data(folder)
            for _ in range(8):
                data.batch(8)
            self.assertEqual(data.battle_counts, dict.fromkeys(('terminal', 'recoil_hurt', 'npc', 'empty'), 16))
            self.assertEqual(data.draws, 64)
            state = copy.deepcopy(data.sampler_state())
            expected = data.batch(8)['row_index']
            data.restore_sampler(state)
            torch.testing.assert_close(expected, data.batch(8)['row_index'], rtol=0, atol=0)
            self.assertEqual(sum(data.battle_counts.values()), data.draws)

    def test_fixed_heldout_panel_contains_each_terminal_once_and_preserves_sampler(self):
        with TemporaryDirectory() as folder, patch(
                'turkeyworld.renderer_enemy_data.project_sprite_bounds', side_effect=fake_bounds):
            data, _, _ = self._data(folder, split='val')
            saved = copy.deepcopy(data.sampler_state())
            panel = data.fixed_panel(12, seed=14)
            self.assertEqual(len(panel), len(set(panel)))
            self.assertEqual(set(data.terminal_indices)&set(panel), set(data.terminal_indices))
            self.assertEqual(sum(i in data.terminal_indices for i in panel), len(data.terminal_indices))
            self.assertEqual(panel, data.fixed_panel(12, seed=14))
            self.assertEqual(data.sampler_state(), saved)
            self.assertEqual(data.cohort_rows(panel)['player_hp0'],
                             [i for i, index in enumerate(panel) if index in data.terminal_indices])

    def test_split_key_retains_source_death_tint_and_sprite_but_ignores_positive_hp(self):
        game = Game(seed=7, endless=True)
        alive_state = game.get_state()
        alive_rgb = game.render(size=32, overlays=False)
        alive_key = render_condition_key(game.state_vector(), game.map)
        alive_state['player']['hp'] = 8.
        game.set_state(alive_state)
        self.assertEqual(alive_key, render_condition_key(game.state_vector(), game.map))
        np.testing.assert_array_equal(alive_rgb, game.render(size=32, overlays=False))
        alive_state['player']['hp'] = 0.
        game.set_state(alive_state)
        self.assertNotEqual(alive_key, render_condition_key(game.state_vector(), game.map))
        expected = ((alive_rgb.astype(np.uint16)+np.array([70, 8, 5]))//2).astype(np.uint8)
        np.testing.assert_array_equal(expected, game.render(size=32, overlays=False))
        # This checks genuine Source label identity, not an inference post-effect.
        alive_state['player']['hp'] = 100.
        alive_state['enemies'][0].update(hp=0., alive=False, death_timer=.1)
        game.set_state(alive_state)
        corpse_key = render_condition_key(game.state_vector(), game.map)
        self.assertNotEqual(alive_key, corpse_key)
        alive_state['enemies'][0]['death_timer'] = .6
        game.set_state(alive_state)
        self.assertNotEqual(corpse_key, render_condition_key(game.state_vector(), game.map))


if __name__ == '__main__':
    unittest.main()
