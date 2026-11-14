"""The user goal is exactly RF-Q held-out SSIM>.85, with immutable pins."""
from copy import deepcopy
import json
import math
import unittest

from turkeyworld.conditional_restorer_goal import (
    PROTOCOL, conditional_restorer_goal_spec, evaluate_restorer_goal,
    update_restorer_best, plateau_report)


PINS = dict(panel_fingerprint='a'*64, plan_sha256='b'*64,
            feature_world_sha256='c'*64, data_identity_sha256='d'*64)


def validation(ssim, step=0):
    return dict(kind='conditional_restorer_validation_v1',
        protocol='conditional_rf_restorer_paired_v1', step=step, solver='euler', solver_steps=3,
        frames=256, native_resolution=32, source_held_out=True, renderer_only=True,
        world_rollout=False, autonomous_rollout=False, sr_quality_evaluated=False,
        Source_RGB_direct_restorer_input=False, Source_RGB_into_RF_conditioning=False,
        future_source_rgb_in_prior=False, future_source_pose_in_prior=False,
        **PINS, models=dict(rf_new=dict(summary=dict(frames=256, ssim_mean=ssim, mae_mean=.9)),
            rf_old=dict(summary=dict(frames=256, ssim_mean=.99)),
            source_new=dict(summary=dict(frames=256, ssim_mean=.999))),
        val_loss=.12, validation_objective=dict(protocol='conditional_rf_restorer_paired_v1',
            input_kind='frozen_RF3_or_Source_quantized_codes_RGB_state_map', frames=256,
            source_held_out=True, loss=.12), source_preservation=dict(passed=False),
        passed=False, quality_passed=False)


class ConditionalRestorerGoalTests(unittest.TestCase):
    def test_strict_threshold_first_full_validation_no_extra_gate_or_consecutive_requirement(self):
        for value, expected in [(-.1, False), (0., False), (.849999999, False),
                                (.85, False), (math.nextafter(.85, 1.), True), (.9, True), (1., True)]:
            with self.subTest(ssim=value):
                result = evaluate_restorer_goal(validation(value), **PINS)
                self.assertEqual(result['goal_met'], expected)
                self.assertEqual(result['measured_ssim'], value)
                self.assertEqual(result['metric'], 'models.rf_new.summary.ssim_mean')
                self.assertTrue(result['eligible'])
                self.assertFalse(result['Source_preservation_is_goal_requirement'])
                self.assertFalse(result['MAE_is_goal_requirement'])
        spec = conditional_restorer_goal_spec()
        self.assertIsNone(spec['budget_updates'])
        self.assertFalse(spec['budget_completion_is_success'])
        self.assertEqual(spec['consecutive_successes_required'], 1)
        self.assertFalse(spec['plateau_is_automatic_stop'])

    def test_no_RF_old_Source_Q_MAE_loss_SR_or_quality_proxy(self):
        report = validation(.84, 2000)
        report['models']['source_new']['summary']['ssim_mean'] = 1.
        report['models']['rf_old']['summary']['ssim_mean'] = 1.
        report['models']['rf_new']['summary']['mae_mean'] = 0.
        report['passed'] = report['quality_passed'] = report['source_preservation']['passed'] = True
        report['val_loss'] = report['validation_objective']['loss'] = 0.
        self.assertFalse(evaluate_restorer_goal(report, **PINS)['goal_met'])
        report = validation(.86, 0)
        self.assertTrue(evaluate_restorer_goal(report, **PINS)['goal_met'])

    def test_nonfinite_bool_percent_and_missing_RF_summary_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf'), True, '0.9', None, 85., 1.01, -1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate_restorer_goal(validation(value), **PINS)
        bad = validation(.99); del bad['models']['rf_new']
        with self.assertRaises(ValueError): evaluate_restorer_goal(bad, **PINS)

    def test_partial_preview_TRAIN_stale_domain_wrong_kind_or_future_input_rejected(self):
        for key, value in [('kind', 'decoder_rf_adaptation_validation_v1'),
                ('protocol', 'other'), ('frames', 4), ('frames', True),
                ('source_held_out', False), ('renderer_only', False), ('world_rollout', True),
                ('autonomous_rollout', True), ('sr_quality_evaluated', True),
                ('native_resolution', 128), ('solver_steps', 5),
                ('Source_RGB_direct_restorer_input', True), ('future_source_rgb_in_prior', True),
                ('panel_fingerprint', 'e'*64), ('plan_sha256', 'e'*64),
                ('feature_world_sha256', 'e'*64), ('data_identity_sha256', 'e'*64),
                ('step', -1), ('step', True)]:
            bad = validation(.99); bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                evaluate_restorer_goal(bad, **PINS)
        for change in ('summary_frames', 'objective_frames', 'objective_protocol', 'objective_loss'):
            bad = validation(.99)
            if change == 'summary_frames': bad['models']['rf_new']['summary']['frames'] = 4
            if change == 'objective_frames': bad['validation_objective']['frames'] = 4
            if change == 'objective_protocol': bad['validation_objective']['protocol'] = 'other'
            if change == 'objective_loss': bad['validation_objective']['loss'] = .2
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_restorer_goal(bad, **PINS)
        for key in PINS:
            bad = dict(PINS); bad[key] = 'not-a-sha'
            with self.subTest(pin=key), self.assertRaises(ValueError):
                evaluate_restorer_goal(validation(.99), **bad)

    def test_best_checkpoint_uses_maximum_RF_SSIM_and_earliest_tie_not_loss_or_mae(self):
        first = update_restorer_best(None, validation(.74, 0), **PINS, report_sha256='e'*64)
        second_report = validation(.75, 100)
        second_report['val_loss'] = second_report['validation_objective']['loss'] = 10.
        second_report['models']['rf_new']['summary']['mae_mean'] = 1.
        second = update_restorer_best(first, second_report, **PINS)
        self.assertEqual(second['ssim'], .75); self.assertEqual(second['step'], 100)
        self.assertTrue(second['improved'])
        lower = update_restorer_best(second, validation(.749, 200), **PINS)
        tied = update_restorer_best(lower, validation(.75, 300), **PINS)
        self.assertEqual(tied['step'], 100); self.assertFalse(tied['improved'])
        success = update_restorer_best(tied, validation(.850001, 400), **PINS)
        self.assertTrue(success['goal_met']); self.assertEqual(success['step'], 400)
        self.assertEqual(first['report_sha256'], 'e'*64)
        json.dumps(success, allow_nan=False)

    def test_foreign_or_tampered_best_cannot_hide_goal_or_cross_a_phase(self):
        best = update_restorer_best(None, validation(.74, 0), **PINS)
        for key, value in [('protocol', 'other'), ('plan_sha256', 'e'*64),
                ('feature_world_sha256', 'e'*64), ('panel_fingerprint', 'e'*64),
                ('data_identity_sha256', 'e'*64), ('ssim', float('nan')),
                ('step', -1), ('goal_met', True), ('target', .8), ('frames', 4),
                ('frames', 256.), ('validation_step', False), ('report_kind', 'other')]:
            bad = deepcopy(best); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                update_restorer_best(bad, validation(.75, 100), **PINS)
        future = update_restorer_best(None, validation(.8, 200), **PINS)
        with self.assertRaises(ValueError):
            update_restorer_best(future, validation(.75, 100), **PINS)

    def test_plateau_is_informational_never_stops_or_reinterprets_budget(self):
        short = plateau_report([validation(.74, 0)], **PINS)
        self.assertIsNone(short['plateau']); self.assertFalse(short['stop_training'])
        flat = plateau_report([validation(.744 + i*.00001, i*100) for i in range(5)], **PINS)
        self.assertTrue(flat['plateau'])
        self.assertFalse(flat['stop_training']); self.assertFalse(flat['automatic_stop'])
        self.assertTrue(flat['diagnostic_only'])
        rising = plateau_report([validation(.74 + i*.01, 2000+i*100) for i in range(5)], **PINS)
        self.assertFalse(rising['plateau']); self.assertFalse(rising['stop_training'])
        self.assertEqual(rising['steps'], [2000, 2100, 2200, 2300, 2400])
        for history in ([validation(.74, 0), validation(.74, 0)],
                        [validation(.74, 0), dict(validation(.8, 100), panel_fingerprint='e'*64)]):
            with self.assertRaises(ValueError): plateau_report(history, **PINS)


if __name__ == '__main__':
    unittest.main()
