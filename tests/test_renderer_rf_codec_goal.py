"""Only full pinned RF-Q SSIM can finish the decoder-free RF phase."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from turkeyworld.renderer_rf_codec_goal import (
    PROTOCOL, METRIC, rf_codec_goal_spec, evaluate_rf_codec_goal,
    update_rf_codec_best, validate_saved_rf_codec_goal)


PINS = dict(panel_fingerprint='a'*64, plan_sha256='b'*64, feature_world_sha256='c'*64)


def validation(ssim, step=0):
    return dict(kind='renderer_rf_codec_validation_v1', protocol='renderer_rf_codec_geometry_v1',
        step=step, solver='euler', solver_steps=3, frames=256, native_resolution=32,
        source_held_out=True, renderer_only=True, world_rollout=False, autonomous_rollout=False,
        sr_quality_evaluated=False, restorer_active=False, Source_Q_image_branch=False,
        Source_RGB_into_RF_conditioning=False, future_source_rgb_in_prior=False,
        future_source_pose_in_prior=False, validation_objective_matches_main_training=True,
        **PINS, data_identity_sha256='d'*64,
        models=dict(rf_new=dict(summary=dict(frames=256, ssim_mean=ssim, mae_mean=.9)),
            rf_old=dict(summary=dict(frames=256, ssim_mean=.99)),
            source_codec=dict(summary=dict(frames=256, ssim_mean=1.))),
        source_preservation=dict(passed=False), geometry_preservation=dict(passed=False),
        passed=False, quality_passed=False, val_loss=.12,
        validation_objective=dict(protocol='renderer_rf_codec_geometry_v1',
            input_kind='RF3_terminal_Q_frozen_current_decoder', frames=256,
            source_held_out=True, matches_training_formula=True, main_only=True,
            Source_Q_image_branch=False, loss=.12))


def saved_reports(root, values, *, current_step=None):
    directory = Path(root)/'validation'; directory.mkdir(exist_ok=True)
    history, best = [], None
    for step, ssim in values:
        report = validation(ssim, step)
        payload = json.dumps(report, indent=2).encode('utf-8')
        (directory/f'step_{step:06d}.json').write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        history.append(dict(step=step, ssim=ssim, goal_met=ssim > .85, report_sha256=sha))
        best = update_rf_codec_best(best, report, report_sha256=sha, **PINS)
    return dict(step=values[-1][0] if current_step is None else current_step,
                goal_history=history, best=best, goal_met=history[-1]['goal_met'])


class RendererRFCodecGoalTests(unittest.TestCase):
    def test_strict_threshold_one_full_validation_no_budget_or_secondary_goal(self):
        for value, met in [(-.1, False), (.85, False), (math.nextafter(.85, 1.), True), (.9, True)]:
            result = evaluate_rf_codec_goal(validation(value), **PINS)
            self.assertEqual(result['goal_met'], met)
            self.assertEqual(result['measured_ssim'], value)
            self.assertEqual(result['metric'], METRIC)
            self.assertFalse(result['geometry_preservation_is_goal_requirement'])
            self.assertFalse(result['Source_preservation_is_goal_requirement'])
        spec = rf_codec_goal_spec()
        self.assertIsNone(spec['budget_updates'])
        self.assertEqual(spec['consecutive_successes_required'], 1)
        self.assertFalse(spec['budget_completion_is_success'])
        self.assertFalse(spec['restorer_active'])

    def test_no_Source_Q_old_RF_MAE_loss_or_quality_proxy(self):
        report = validation(.84, 20000)
        report['models']['rf_old']['summary']['ssim_mean'] = 1.
        report['models']['rf_new']['summary']['mae_mean'] = 0.
        report['val_loss'] = report['validation_objective']['loss'] = 0.
        report['quality_passed'] = report['passed'] = True
        self.assertFalse(evaluate_rf_codec_goal(report, **PINS)['goal_met'])
        self.assertTrue(evaluate_rf_codec_goal(validation(.86), **PINS)['goal_met'])

    def test_wrong_scope_partial_stale_or_future_inputs_cannot_complete(self):
        for name, value in [('kind', 'conditional_restorer_validation_v1'), ('protocol', 'old'),
                ('frames', 4), ('frames', 256.), ('solver_steps', 5), ('native_resolution', 128),
                ('source_held_out', False), ('world_rollout', True), ('autonomous_rollout', True),
                ('restorer_active', True), ('Source_Q_image_branch', True),
                ('Source_RGB_into_RF_conditioning', True), ('future_source_pose_in_prior', True),
                ('validation_objective_matches_main_training', False), ('step', True),
                ('panel_fingerprint', 'e'*64), ('plan_sha256', 'e'*64),
                ('feature_world_sha256', 'e'*64), ('data_identity_sha256', 'invalid')]:
            bad = validation(.99); bad[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluate_rf_codec_goal(bad, **PINS)
        for value in (True, '0.9', None, float('nan'), float('inf'), 85., 1.01):
            with self.subTest(ssim=value), self.assertRaises(ValueError):
                evaluate_rf_codec_goal(validation(value), **PINS)
        for change in ('missing_RF', 'summary_frames', 'wrong_objective', 'objective_loss', 'objective_Source_branch'):
            bad = validation(.99)
            if change == 'missing_RF':del bad['models']['rf_new']
            if change == 'summary_frames':bad['models']['rf_new']['summary']['frames'] = 4
            if change == 'wrong_objective':bad['validation_objective']['input_kind'] = 'Source_Q'
            if change == 'objective_loss':bad['validation_objective']['loss'] = .13
            if change == 'objective_Source_branch':bad['validation_objective']['Source_Q_image_branch'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_rf_codec_goal(bad, **PINS)

    def test_best_is_maximum_RF_SSIM_earliest_tie_even_if_loss_and_guards_worsen(self):
        first = update_rf_codec_best(None, validation(.74), **PINS, report_sha256='e'*64)
        report = validation(.75, 100)
        report['val_loss'] = report['validation_objective']['loss'] = 10.
        second = update_rf_codec_best(first, report, **PINS)
        self.assertTrue(second['improved']); self.assertEqual(second['step'], 100)
        lower = update_rf_codec_best(second, validation(.749, 200), **PINS)
        tie = update_rf_codec_best(lower, validation(.75, 300), **PINS)
        self.assertEqual(tie['step'], 100); self.assertFalse(tie['improved'])
        for name, value in [('step', -1), ('goal_met', True), ('frames', True),
                            ('protocol', 'old'), ('data_identity_sha256', 'e'*64)]:
            bad = deepcopy(first); bad[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                update_rf_codec_best(bad, validation(.75, 100), **PINS)

    def test_saved_history_reloads_every_actual_SHA_and_retains_earliest_best(self):
        with tempfile.TemporaryDirectory() as root:
            saved = saved_reports(root, [(0, .74), (100, .75), (200, .75), (300, .749)], current_step=346)
            before = deepcopy(saved)
            best, history, met = validate_saved_rf_codec_goal(saved, run_dir=root, **PINS)
            self.assertEqual(best['step'], 100); self.assertFalse(met)
            self.assertEqual(history, saved['goal_history']); self.assertEqual(saved, before)
            best['ssim'] = 0.; self.assertEqual(saved, before)
            success = saved_reports(root, [(0, .74), (100, .8500001)])
            self.assertTrue(validate_saved_rf_codec_goal(success, run_dir=root, **PINS)[2])

    def test_missing_tampered_or_rehashed_wrong_measurement_reports_reject(self):
        with tempfile.TemporaryDirectory() as root:
            saved = saved_reports(root, [(0, .74), (100, .75)])
            path = Path(root)/'validation'/'step_000100.json'
            payload = path.read_bytes()
            path.write_bytes(payload+b' ')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                validate_saved_rf_codec_goal(saved, run_dir=root, **PINS)
            changed = validation(.9, 100); payload = json.dumps(changed).encode()
            path.write_bytes(payload)
            saved['goal_history'][1]['report_sha256'] = hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                validate_saved_rf_codec_goal(saved, run_dir=root, **PINS)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                validate_saved_rf_codec_goal(saved, run_dir=root, **PINS)

    def test_false_completion_nonmaximum_latest_tie_and_postsuccess_updates_reject(self):
        with tempfile.TemporaryDirectory() as root:
            valid = saved_reports(root, [(0, .74), (100, .75), (200, .75)])
            for mutate in (
                lambda s:s.update(goal_met=True),
                lambda s:s['best'].update(ssim=.99),
                lambda s:s['best'].update(step=200, validation_step=200),
                lambda s:s['goal_history'].reverse(),
                lambda s:s['goal_history'].pop(0),
                lambda s:s['goal_history'][0].update(report_sha256='invalid'),
            ):
                bad = deepcopy(valid); mutate(bad)
                with self.assertRaises(ValueError):
                    validate_saved_rf_codec_goal(bad, run_dir=root, **PINS)
            success = saved_reports(root, [(0, .74), (100, .86)], current_step=101)
            with self.assertRaises(ValueError):
                validate_saved_rf_codec_goal(success, run_dir=root, **PINS)
            after_success = saved_reports(root, [(0, .86), (100, .87)])
            with self.assertRaises(ValueError):
                validate_saved_rf_codec_goal(after_success, run_dir=root, **PINS)

    def test_only_unmeasured_initial_update_zero_can_resume_without_reports(self):
        initial = dict(step=0, goal_history=[], best=None, goal_met=False)
        self.assertEqual(validate_saved_rf_codec_goal(initial, run_dir=None, **PINS), (None, [], False))
        for change in (dict(step=1), dict(goal_met=True), dict(best={})):
            with self.assertRaises(ValueError):
                validate_saved_rf_codec_goal(dict(initial, **change), run_dir=None, **PINS)
        with self.assertRaises(ValueError):
            validate_saved_rf_codec_goal(initial, run_dir=None, **dict(PINS, plan_sha256='invalid'))
        json.dumps(rf_codec_goal_spec(), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
