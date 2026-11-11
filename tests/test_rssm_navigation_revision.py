from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from turkeyworld.rssm_navigation_revision import (consider_navigation_candidate,
    navigation_revision_setting, navigation_validation_rank, new_revision_ledger,
    commit_revision_update, validate_revision_ledger)


PANELS = {'long450': 'two-observation-panel', 'one_frame450': 'one-observation-panel'}


def validation(step=3100, *, walls=4, mean=.4, p95=.8):
    report = {'step': step, 'fingerprint': 'identified-full-report', 'continuation_id': 'continuation-test',
              'source_future_feedback': False, 'future_rgb_encoded': False, 'passed': False}
    for name, context in (('long450', 2), ('one_frame450', 1)):
        report[name] = {'fingerprint': PANELS[name], 'context': context, 'source_held_out': True,
            'source_future_feedback': False, 'horizon': 450, 'windows': 16,
            'all_states_finite': True, 'all_memory_finite': True, 'source_wall_steps': 0,
            'npc_active_steps': 0, 'model_wall_steps': walls, 'model_wall_path_steps': walls,
            'final_player_xy_error_tiles': {'n': 16, 'mean': mean, 'p95': p95}}
    return report


def complete_report(step=3100, mean=.01):
    report = validation(step, walls=0, mean=mean, p95=mean*2)
    report['passed'] = True
    for name, count in (('short', 128), ('contact_generalization', 256)):
        report[name] = {'source_held_out': True,
            'counts': {'samples': count, 'free_axes': 64, 'blocked_axes': 32, 'sliding_samples': 16},
            'cohorts': {key: {'samples': 8} for key in ('positive_turn', 'negative_turn', 'moving_turn', 'idle')},
            'metrics': {'velocity_error_tiles_s': {'mean': .01}, 'false_stop_rate': 0, 'missed_block_rate': 0}}
    report['heldout_idle'] = {'source_held_out': True, 'samples': 32,
        'all_states_memory_finite': True, 'velocity_error_tiles_s_max': 0}
    report['default_spawn'] = {'passed': True}
    return report


def checkpoint(step=3100, weight=1.):
    return {'stage': 'world', 'training_steps': step, 'global_training_step': step,
        'state_schema': {'slots': 1, 'state_dim': 20},
        'config': {'architecture': 'dreamer_rssm', 'navigation_only': True},
        'model': {'core.weight': torch.full((2, 3), weight), 'codec.counter': torch.tensor(11)},
        'rssm_migration': {'frozen_render_assets_bitexact': True, 'source_teacher_step': 1633},
        'source_teacher_checkpoint': 'archived_teacher.pt', 'quality_status': 'Not qualified',
        'optimizer': {'state': {0: {'exp_avg': torch.tensor([.3])}}},
        'torch_rng_state': torch.get_rng_state(), 'run_state': {'do_not_copy': True}}


class RSSMNavigationRevisionTests(unittest.TestCase):
    def test_joint16_cycle_has_three_argmax_one_sample_for_every_pool_context(self):
        settings = [navigation_revision_setting(index) for index in range(16)]
        joint = Counter((row['long_pool'], row['context'], row['latent_mode']) for row in settings)
        self.assertEqual(len(joint), 8)
        for pool in ('original', 'coverage'):
            for context in (1, 2):
                self.assertEqual(joint[pool, context, 'argmax'], 3)
                self.assertEqual(joint[pool, context, 'sample'], 1)
        for row in settings:
            self.assertEqual(row['numeric_context'], row['context'])
            self.assertEqual(row['observed_context'], row['context'])
            self.assertEqual((row['window_start'], row['horizon']), (0, 450))
        self.assertEqual(navigation_revision_setting(16)['cycle_index'], 0)
        for bad in (-1, True, 1.2):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                navigation_revision_setting(bad)

    def test_ledger_resume_and_duplicate_commit_preserve_old_source_clock_and_joint_draws(self):
        ledger = new_revision_ledger(3100, 2900, 'immutable-plan')
        for offset in range(7):
            commit_revision_update(ledger, step=3101+offset, source_completed=2901+offset)
        resumed = json.loads(json.dumps(ledger))
        self.assertFalse(commit_revision_update(resumed, step=3107, source_completed=2907))
        for offset in range(7, 16):
            commit_revision_update(resumed, step=3101+offset, source_completed=2901+offset)
        direct = new_revision_ledger(3100, 2900, 'immutable-plan')
        for offset in range(16):
            commit_revision_update(direct, step=3101+offset, source_completed=2901+offset)
        self.assertEqual(resumed, direct)
        self.assertEqual(direct['auxiliary_draws'], {'corner': 128, 'single_wall': 64, 'legacy': 64})
        self.assertEqual(sum(direct['joint_long_draws'].values()), 16*16)
        self.assertEqual(direct['source_completed'], 2916)
        self.assertEqual(validate_revision_ledger(resumed, step=3116, source_completed=2916), resumed)
        for kwargs in ({'step': 3118, 'source_completed': 2917},
                       {'step': 3117, 'source_completed': 2918},
                       {'step': 3117, 'source_completed': 2917, 'long_batch_size': 8}):
            previous = deepcopy(resumed)
            with self.assertRaises(ValueError):
                commit_revision_update(resumed, **kwargs)
            self.assertEqual(previous, resumed)
        corrupted = deepcopy(resumed)
        corrupted['joint_long_draws'][next(iter(corrupted['joint_long_draws']))] += 1
        with self.assertRaisesRegex(ValueError, 'draws'):
            validate_revision_ledger(corrupted)

    def test_physical_rank_prioritizes_walls_and_requires_both_pinned_finite450_panels(self):
        worse = navigation_validation_rank(validation(walls=2, mean=.01), expected_fingerprints=PANELS)
        better = navigation_validation_rank(validation(walls=1, mean=10), expected_fingerprints=PANELS)
        self.assertLess(better, worse)
        mutations = [('fingerprint', 'wrong'), ('all_memory_finite', False), ('windows', 0),
                     ('horizon', 449), ('source_wall_steps', 1), ('npc_active_steps', 1), ('context', 2)]
        for key, value in mutations:
            report = validation()
            report['one_frame450'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                navigation_validation_rank(report, expected_fingerprints=PANELS)
        report = validation()
        report['long450']['final_player_xy_error_tiles']['mean'] = float('nan')
        with self.assertRaises(ValueError):
            navigation_validation_rank(report, expected_fingerprints=PANELS)
        report = validation()
        del report['one_frame450']
        with self.assertRaises(ValueError):
            navigation_validation_rank(report, expected_fingerprints=PANELS)

    def test_candidate_is_minimal_immutable_and_source_optimizer_rng_migration_are_preserved(self):
        source = checkpoint()
        rng = torch.get_rng_state().clone()
        original_weights = {key: value.clone() for key, value in source['model'].items()}
        original_moment = source['optimizer']['state'][0]['exp_avg'].clone()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            selected = consider_navigation_candidate(root, source, validation(),
                expected_fingerprints=PANELS, assets_metadata={'renderer': 'approvedRF1.pt', 'codec_frozen': True})
            candidate = Path(selected['best_physical']['checkpoint'])
            mtime = candidate.stat().st_mtime_ns
            saved = torch.load(candidate, weights_only=False)
            self.assertNotIn('optimizer', saved)
            self.assertNotIn('run_state', saved)
            self.assertNotIn('torch_rng_state', saved)
            self.assertEqual(saved['rssm_migration'], source['rssm_migration'])
            self.assertEqual(saved['navigation_assets']['renderer'], 'approvedRF1.pt')
            self.assertFalse(saved['navigation_candidate']['qualified'])
            self.assertFalse(selected['automatic_deployment'])
            for key, value in saved['model'].items():
                self.assertTrue(torch.equal(value, original_weights[key]))
            repeated = consider_navigation_candidate(root, source, validation(), expected_fingerprints=PANELS)
            self.assertEqual(repeated, selected)
            self.assertEqual(candidate.stat().st_mtime_ns, mtime)
            self.assertEqual(len(list((root/'candidates').glob('*.pt'))), 1)
        self.assertTrue(torch.equal(source['optimizer']['state'][0]['exp_avg'], original_moment))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for key, value in source['model'].items():
            self.assertTrue(torch.equal(value, original_weights[key]))

    def test_worse_candidate_never_overwrites_better_and_retention_never_deletes_user_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = consider_navigation_candidate(root, checkpoint(), validation(), expected_fingerprints=PANELS)
            previous = Path(first['best_physical']['checkpoint'])
            foreign = root/'candidates'/'my_user_checkpoint.pt'
            foreign.write_bytes(b'preserve-user-file')
            worse = consider_navigation_candidate(root, checkpoint(3101, 2.),
                validation(3101, walls=6, mean=.01), expected_fingerprints=PANELS)
            self.assertEqual(worse, first)
            improved = consider_navigation_candidate(root, checkpoint(3102, 3.),
                validation(3102, walls=1), expected_fingerprints=PANELS)
            self.assertEqual(improved['best_physical']['step'], 3102)
            self.assertFalse(previous.exists())
            self.assertEqual(foreign.read_bytes(), b'preserve-user-file')

    def test_candidate_write_and_selector_failure_do_not_lose_best_then_orphan_retry_recovers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old = consider_navigation_candidate(root, checkpoint(), validation(), expected_fingerprints=PANELS)
            old_pointer = (root/'navigation_best.json').read_bytes()
            new, report = checkpoint(3101, 2.), validation(3101, walls=0)
            with patch('turkeyworld.rssm_navigation_revision.torch.save', side_effect=OSError('disk write failure')):
                with self.assertRaises(OSError):
                    consider_navigation_candidate(root, new, report, expected_fingerprints=PANELS)
            self.assertEqual((root/'navigation_best.json').read_bytes(), old_pointer)
            self.assertTrue(Path(old['best_physical']['checkpoint']).is_file())
            with patch('turkeyworld.rssm_navigation_revision.atomic_json', side_effect=OSError('pointer failure')):
                with self.assertRaises(OSError):
                    consider_navigation_candidate(root, new, report, expected_fingerprints=PANELS)
            self.assertEqual((root/'navigation_best.json').read_bytes(), old_pointer)
            self.assertEqual(len(list((root/'candidates').glob('*.pt'))), 2)
            chosen = consider_navigation_candidate(root, new, report, expected_fingerprints=PANELS)
            self.assertEqual(chosen['best_physical']['step'], 3101)
            self.assertEqual(len(list((root/'candidates').glob('*.pt'))), 1)
            self.assertFalse(list((root/'candidates').glob('*.pending')))

    def test_single_complete_gate_pass_is_not_two_check_qualification_and_both_best_files_can_coexist(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = consider_navigation_candidate(root, checkpoint(), complete_report(), expected_fingerprints=PANELS)
            self.assertIsNone(first['best_qualified'])
            self.assertTrue(first['best_physical']['complete_quality_gate_passed'])
            proof = {'status': 'qualified', 'consecutive_passes': 2, 'required_consecutive_passes': 2,
                     'last_validation_passed': True, 'last_validation_step': 3101,
                     'fingerprint': 'continuation-test'}
            selected = consider_navigation_candidate(root, checkpoint(3101, 2.), complete_report(3101, mean=.02),
                expected_fingerprints=PANELS, qualification=proof)
            self.assertEqual(selected['best_physical']['step'], 3100)
            self.assertEqual(selected['best_qualified']['step'], 3101)
            self.assertTrue(selected['best_qualified']['qualified'])
            self.assertEqual(len(list((root/'candidates').glob('*.pt'))), 2)

    def test_mismatched_step_nonfinite_weights_and_tampered_selected_file_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'identical optimizer step'):
                consider_navigation_candidate(root, checkpoint(3101), validation(), expected_fingerprints=PANELS)
            bad = checkpoint()
            bad['model']['core.weight'][0, 0] = float('inf')
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                consider_navigation_candidate(root, bad, validation(), expected_fingerprints=PANELS)
            selected = consider_navigation_candidate(root, checkpoint(), validation(), expected_fingerprints=PANELS)
            candidate = Path(selected['best_physical']['checkpoint'])
            candidate.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                consider_navigation_candidate(root, checkpoint(3101), validation(3101, walls=0), expected_fingerprints=PANELS)


if __name__ == '__main__':
    unittest.main()
