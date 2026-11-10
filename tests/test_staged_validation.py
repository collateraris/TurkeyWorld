"""Real Source snapshots/episodes, tiny neural queries and strict stage gates."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.engine import BEHAVIOR_VERSION, ENTITY_SLOTS, STATE_DIM, Game
from turkeyworld.staged_validation import evaluate_staged_model
from turkeyworld.structured_state import structured_transition


class TinySampler(nn.Module):
    def __init__(self):
        super().__init__(); self.steps = 12; self.calls = 0
        self.register_buffer('_map_condition', torch.ones(1, 16, 16, dtype=torch.long), persistent=False)
    def sample(self, base, state):
        self.calls += 1
        return base


class TinyWorld(nn.Module):
    def __init__(self, image):
        super().__init__()
        pixels = torch.from_numpy(image).permute(2, 0, 1).float()/255
        self.image_logits = nn.Parameter(torch.logit(pixels.clamp(1e-5, 1-1e-5)))
        self.config = SimpleNamespace(simulation_hz=30., endless=True, game_training_v5=False, image_size=32)
        self.flow_renderer = TinySampler()
        self.visual_encoder = nn.Dropout(.2)
        self.register_buffer('_spawn_walkable', torch.ones(1, 16, 16, dtype=torch.bool), persistent=False)
        self.transitions = self.initializations = self.base_calls = 0
        self.position_error, self.bad_memory, self.on_render = 0., False, None
    def encode_map(self, grid):
        self._spawn_walkable = grid == 0
        self.flow_renderer._map_condition = grid.clone()
        return torch.zeros(len(grid), 2, 16, 16)
    def initialize_from_image(self, images, states, features):
        self.initializations += 1
        assert images.shape[1] == states.shape[1]
        return torch.zeros(len(states), 1, 1)
    def transition(self, state, action, features, memory):
        self.transitions += 1
        target = torch.full((len(state), ENTITY_SLOTS+1), -3.)
        target[:, -1] = 3.
        attacks = torch.full((len(state), ENTITY_SLOTS), -3.)
        result = structured_transition(state, state.clone(), action, target, attacks, endless=True)
        result['state'] = result['state'].clone()
        result['state'][:, 0] += self.position_error
        result['memory'] = torch.zeros(len(state), 2, 1) if self.bad_memory else memory+1
        result['features'] = torch.zeros(len(state), 128)
        return result
    def render_base(self, state, features, maps):
        self.base_calls += 1
        if self.on_render is not None: self.on_render()
        return self.image_logits.sigmoid()[None].expand(len(state), -1, -1, -1)
    def render(self, state, features, maps):
        return self.flow_renderer.sample(self.render_base(state, features, maps), state)


class StagedValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def fixture(root, stage='navigation', *, files=2, frames=8, visible=True):
        static = stage == 'static_enemy'
        directory = root/stage/('snapshots' if static else 'rgb')/'val'
        directory.mkdir(parents=True)
        initial_image = None
        for number in range(files):
            game = Game(73+number, endless=True)
            game.player.update(x=2.5, y=2.5, angle=0.)
            if stage == 'full_game':
                pass  # Genuine normal game initialization, including the live NPC.
            elif static and visible:
                game.enemies[0].update(x=4.5, y=2.5, alive=True, hp=60.)
            else:
                game.enemies[0].update(x=1.5, y=12.5, alive=False, hp=0., death_timer=0.)
            states, images, masks, backgrounds, actions, events, hits, attacks = [], [], [], [], [], [], [], []
            count = frames if static else frames+1
            for index in range(count):
                image = game.render(32, False)
                enemies = game.enemies; game.enemies = []
                background = game.render(32, False); game.enemies = enemies
                states.append(game.state_vector()); images.append(image)
                masks.append(np.any(image != background, axis=-1)); backgrounds.append(background)
                if initial_image is None: initial_image = image
                if not static and index < frames:
                    action = np.zeros(4, np.float32); event = game.step(action)
                    actions.append(action); events.append(Game.event_vector(event)); hits.append(event.get('hit_id', -1))
                    attacks.append([float(bool(event.get('attacker_ids')))])
            payload = dict(states=np.asarray(states, np.float32), frames=np.asarray(images, np.uint8),
                object_masks=np.asarray(masks, bool), background_frames=np.asarray(backgrounds, np.uint8),
                map=game.map, stage=np.asarray(stage), split=np.asarray('val'), behavior_version=np.asarray(BEHAVIOR_VERSION),
                state_dim=np.asarray(STATE_DIM), entity_slots=np.asarray(ENTITY_SLOTS),
                dataset_kind=np.asarray('renderer_snapshots' if static else 'world_transitions'),
                world_transition_compatible=np.asarray(not static), staged_config_fingerprint=np.asarray('fixed-source-fixture'),
                initialization_source=np.asarray('normal_game_start' if stage == 'full_game' else 'staged_valid_initial_state'),
                frame_source=np.asarray('reference_engine'), engine_semantics_modified=np.asarray(False), ai_freeze=np.asarray(False),
                dt=np.asarray(1/30), policy=np.asarray('movement'), seed=np.asarray(73+number))
            if not static:
                payload.update(actions=np.asarray(actions, np.float32), events=np.asarray(events, np.float32),
                    hit_ids=np.asarray(hits, np.int64), attack_flags=np.asarray(attacks, np.float32),
                    respawn_flags=np.zeros((frames, ENTITY_SLOTS), np.float32))
            np.savez_compressed(directory/f'episode_{number:03d}.npz', **payload)
        return directory.parent.parent, initial_image

    def test_navigation_passes_real_recorded_idle_source_and_no_runtime_engine_or_future_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); stage, image = self.fixture(root)
            model = TinyWorld(image).train(); model.visual_encoder.eval()
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (stage/'rgb/val').glob('*.npz')}
            with patch.object(Game, 'step', side_effect=AssertionError('No online engine')),\
                    patch.object(Game, 'render', side_effect=AssertionError('No online Source render')):
                report = evaluate_staged_model(model, stage, 'navigation', context=2, horizon=4, count=2,
                    batch_size=2, output_dir=root/'report')
            self.assertTrue(report['passed'], report['failures'])
            self.assertFalse(report['final_qualified'])
            self.assertFalse(report['future_source_feedback'])
            self.assertEqual(report['metrics']['transitions_evaluated'], 8)
            self.assertAlmostEqual(report['metrics']['player_xy_mean'], 0.)
            self.assertEqual(report['metrics']['wall_windows'], 0)
            self.assertEqual(model.initializations, 2)  # state audit and RGB audit, once per own trajectory
            self.assertEqual(model.transitions, 8)      # four per batched pass; no GT reinitialization at each frame
            self.assertTrue(model.training); self.assertFalse(model.visual_encoder.training)
            self.assertTrue(Path(report['report_path']).is_file())
            self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before})

    def test_static_gate_only_renders_base_and_full_sampler_with_exact_npc_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, image = self.fixture(Path(folder), 'static_enemy', files=1, frames=8)
            model = TinyWorld(image)
            with patch.object(model, 'transition', side_effect=AssertionError('Snapshots are not transitions')),\
                    patch.object(model, 'initialize_from_image', side_effect=AssertionError('No fabricated context')):
                report = evaluate_staged_model(model, stage, 'static_enemy', count=8, batch_size=4)
            self.assertTrue(report['passed'], report['failures'])
            self.assertEqual(report['metrics']['transitions_evaluated'], 0)
            self.assertTrue(report['metrics']['renderer_only'])
            self.assertEqual(report['metrics']['visible_frames'], 8)
            self.assertEqual(report['metrics']['contrast_fraction'], 1.)
            self.assertEqual(model.base_calls, 2); self.assertEqual(model.flow_renderer.calls, 2)
            self.assertIsNone(report['report_path'])

    def test_removed_npc_and_no_visible_cases_cannot_pass_static_gate(self):
        for visible in (True, False):
            with self.subTest(visible=visible), tempfile.TemporaryDirectory() as folder:
                stage, image = self.fixture(Path(folder), 'static_enemy', files=1, frames=8, visible=visible)
                with np.load(next((stage/'snapshots/val').glob('*.npz')), allow_pickle=False) as saved:
                    background = saved['background_frames'][0]
                report = evaluate_staged_model(TinyWorld(background), stage, 'static_enemy', count=8)
                self.assertFalse(report['passed'])
                self.assertFalse(report['final_qualified'])
                if visible:
                    self.assertEqual(report['metrics']['contrast_fraction'], 0.)
                    self.assertTrue(any('contrast_fraction' in value for value in report['failures']))
                else:
                    self.assertEqual(report['metrics']['visible_frames'], 0)
                    self.assertIsNone(report['metrics']['npc_mae'])

    def test_wrong_own_pose_and_nonfinite_memory_fail_without_false_zero_or_mode_leak(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, image = self.fixture(Path(folder), frames=10)
            model = TinyWorld(image).train(); model.position_error = .03
            report = evaluate_staged_model(model, stage, 'navigation', context=2, horizon=8, count=2)
            self.assertFalse(report['passed'])
            self.assertGreater(report['metrics']['player_xy_mean'], .1)
            model.position_error = 0.; model.bad_memory = True
            report = evaluate_staged_model(model, stage, 'navigation', context=2, horizon=4, count=2)
            self.assertFalse(report['passed'])
            self.assertTrue(any('memory' in failure.lower() for failure in report['failures']))
            self.assertTrue(model.training)

    def test_all_rng_modes_and_map_caches_restore_on_success_and_exception(self):
        from turkeyworld.staged_curriculum import SnapshotData
        with tempfile.TemporaryDirectory() as folder:
            stage, image = self.fixture(Path(folder), 'static_enemy', files=1, frames=8)
            data = SnapshotData(stage/'snapshots/val'); data.rng = np.random.default_rng(173)
            model = TinyWorld(image).train(); model.visual_encoder.eval()
            original_map = model.flow_renderer._map_condition.clone()
            original_walkable = model._spawn_walkable.clone()
            saved = (torch.random.get_rng_state().clone(), copy.deepcopy(np.random.get_state()),
                random.getstate(), copy.deepcopy(data.rng.bit_generator.state))
            def mutate():
                torch.rand(3); np.random.random(3); random.random(); data.rng.random(3)
            model.on_render = mutate
            evaluate_staged_model(model, data, 'static_enemy', count=8)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), saved[0]))
            np.testing.assert_array_equal(np.random.get_state()[1], saved[1][1])
            self.assertEqual(random.getstate(), saved[2]); self.assertEqual(data.rng.bit_generator.state, saved[3])
            torch.testing.assert_close(model.flow_renderer._map_condition, original_map)
            torch.testing.assert_close(model._spawn_walkable, original_walkable)
            self.assertTrue(model.training); self.assertFalse(model.visual_encoder.training)
            def fail():
                mutate(); raise RuntimeError('render probe failure')
            model.on_render = fail
            with self.assertRaisesRegex(RuntimeError, 'render probe'):
                evaluate_staged_model(model, data, 'static_enemy', count=8)
            self.assertTrue(model.training); self.assertFalse(model.visual_encoder.training)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), saved[0]))
            self.assertEqual(data.rng.bit_generator.state, saved[3])

    def test_wrong_stage_split_dimensions_and_cross_report_fingerprint_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); stage, image = self.fixture(root)
            model = TinyWorld(image)
            with self.assertRaisesRegex(ValueError, 'wrong stage|wrong|Wrong'):
                evaluate_staged_model(model, stage, 'moving1', horizon=4, count=2)
            report = evaluate_staged_model(model, stage, 'navigation', horizon=4, count=2, output_dir=root/'report')
            self.assertTrue(report['passed'])
            with self.assertRaisesRegex(ValueError, 'different data/stage/thresholds'):
                evaluate_staged_model(model, stage, 'navigation', horizon=4, count=2,
                    thresholds={'rgb_ssim_min': .5}, output_dir=root/'report')
            path = next((stage/'rgb/val').glob('*.npz'))
            with np.load(path, allow_pickle=False) as saved: arrays = {key: saved[key] for key in saved.files}
            arrays['split'] = np.asarray('train'); np.savez(path, **arrays)
            with self.assertRaises(ValueError): evaluate_staged_model(model, stage, 'navigation', horizon=4, count=2)

    def test_full_game_never_qualifies_short_or_skipped_rollouts(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, image = self.fixture(Path(folder), 'full_game', files=1, frames=8)
            model = TinyWorld(image)
            missing = evaluate_staged_model(model, stage, 'full_game', count=1)
            self.assertFalse(missing['passed']); self.assertFalse(missing['final_qualified'])
            self.assertFalse(missing['metrics']['full_rollout_evaluated'])
            self.assertTrue(any('450' in text for text in missing['failures']))
            skipped = evaluate_staged_model(model, stage, 'full_game', count=1, horizon=4, final_rollout=False)
            self.assertFalse(skipped['passed']); self.assertFalse(skipped['final_qualified'])

    def test_full_game_checks_all450_states_and_memory_with_sparse_recorded_rgb_no_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, image = self.fixture(Path(folder), 'full_game', files=1, frames=451)
            model = TinyWorld(image)
            with patch.object(Game, 'step', side_effect=AssertionError('No engine corrections')),\
                    patch.object(Game, 'render', side_effect=AssertionError('No source rendering')):
                report = evaluate_staged_model(model, stage, 'full_game', count=1, batch_size=1,
                    full_probe_count=1, render_every=30)
            self.assertFalse(report['passed'])  # An idle, non-combat oracle is not a playable session.
            self.assertFalse(report['final_qualified'])
            self.assertTrue(report['metrics']['full_rollout_evaluated'])
            self.assertEqual(report['metrics']['full_rollout_steps'], 450)
            self.assertEqual(report['metrics']['transitions_evaluated'], 450)
            self.assertEqual(report['metrics']['full_rollout_windows'], 1)
            self.assertEqual(report['rollout_audits'][0]['checked_steps'], 450)
            self.assertEqual(report['metrics']['engine_corrections'], 0)
            self.assertEqual(report['metrics']['model_resets'], 0)
            self.assertEqual(report['metrics']['rgb_frames'], 16)
            self.assertEqual(model.flow_renderer.calls, 16)
            self.assertTrue(any('coverage' in text for text in report['failures']))

    def test_seeded_valid_long_start_is_allowed_but_repaired_or_non_source_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, image = self.fixture(Path(folder), 'full_game', files=1, frames=8)
            path = next((stage/'rgb/val').glob('*.npz'))
            with np.load(path, allow_pickle=False) as saved: original = {key: saved[key] for key in saved.files}
            for source in ('normal_game_start', 'seeded_valid_continuous_start'):
                arrays = {**original, 'initialization_source': np.asarray(source)}
                np.savez(path, **arrays)
                report = evaluate_staged_model(TinyWorld(image), stage, 'full_game', count=1)
                self.assertFalse(report['passed'])  # Accepted provenance still cannot qualify a short recording.
                self.assertTrue(any('450' in failure for failure in report['failures']))
            for changed in ({'engine_semantics_modified': np.asarray(True)}, {'ai_freeze': np.asarray(True)},
                            {'frame_source': np.asarray('model_generated')},
                            {'initialization_source': np.asarray('health_repaired_during_rollout')}):
                with self.subTest(changed=list(changed)):
                    np.savez(path, **{**original, **changed})
                    with self.assertRaisesRegex(ValueError, 'wrong stage|wrong|Wrong'):
                        evaluate_staged_model(TinyWorld(image), stage, 'full_game', count=1)

    def test_nonfinite_renderer_fails_with_json_safe_absent_roi_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); stage, image = self.fixture(root, 'static_enemy', files=1, frames=8)
            model = TinyWorld(image)
            with patch.object(model, 'render', return_value=torch.full((4, 3, 32, 32), float('nan'))):
                report = evaluate_staged_model(model, stage, 'static_enemy', count=8, batch_size=4,
                                              output_dir=root/'report')
            self.assertFalse(report['passed']); self.assertFalse(report['metrics']['rgb_finite'])
            self.assertIsNone(report['metrics']['npc_mae'])
            self.assertIsNone(report['metrics']['background_mae'])
            self.assertEqual(report['metrics']['contrast_fraction'], 0.)
            json.dumps(report, allow_nan=False)
            self.assertTrue(Path(report['report_path']).is_file())


if __name__ == '__main__':
    unittest.main()
