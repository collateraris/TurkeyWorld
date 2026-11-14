"""Strict held-out RF-Q + restorer SSIM goal; no update-budget completion.

This module evaluates saved measurements. It never trains, stops a worker,
changes thresholds, or treats Source-code reconstruction as RF performance.
"""
from copy import deepcopy
import math


PROTOCOL = 'conditional_restorer_rf_ssim_goal_v1'
REPORT_KIND = 'conditional_restorer_validation_v1'
REPORT_PROTOCOL = 'conditional_rf_restorer_paired_v1'
INPUT_KIND = 'frozen_RF3_or_Source_quantized_codes_RGB_state_map'
TARGET = .85
FRAMES = 256
METRIC = 'models.rf_new.summary.ssim_mean'


def conditional_restorer_goal_spec():
    return dict(protocol=PROTOCOL, version=1, target=TARGET,
        comparison='strict_greater_than', metric=METRIC, model='rf_new',
        frames=FRAMES, report_kind=REPORT_KIND, report_protocol=REPORT_PROTOCOL,
        native_resolution=32, solver='euler', solver_steps=3, source_held_out=True,
        renderer_only=True, world_rollout=False, sr_quality_evaluated=False,
        first_full_validation_above_target_is_sufficient=True,
        consecutive_successes_required=1, Source_preservation_is_goal_requirement=False,
        MAE_is_goal_requirement=False, Source_code_SSIM_is_goal_measurement=False,
        budget_updates=None, budget_completion_is_success=False,
        stop_policy='goal_success_or_explicit_user_stop_or_error',
        plateau_is_automatic_stop=False, automatic_active_promotion=False)


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(name + ' must be an explicit lowercase SHA256 fingerprint')
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite scalar, not a bool or proxy')
    return float(value)


def _ssim(value):
    value = _number(value, METRIC)
    # Gaussian SSIM may be negative. Values in percent (85) are invalid.
    if not -1 <= value <= 1:
        raise ValueError('RF-Q SSIM must use the native [-1,1] scale, never percent')
    return value


def evaluate_restorer_goal(report, *, panel_fingerprint, plan_sha256,
                          feature_world_sha256, data_identity_sha256=None):
    """Accept ONLY the pinned, complete Source-held-out native32 RF-new mean.

    A valid measurement equal to .85 does not satisfy the user goal. Source-Q
    preservation, MAE and other quality gates remain independent diagnostics.
    """
    pins = dict(panel_fingerprint=_sha(panel_fingerprint, 'panel_fingerprint'),
        plan_sha256=_sha(plan_sha256, 'plan_sha256'),
        feature_world_sha256=_sha(feature_world_sha256, 'feature_world_sha256'))
    if not isinstance(report, dict):
        raise ValueError('RF-Q goal needs an actual full validation report')
    required = dict(kind=REPORT_KIND, protocol=REPORT_PROTOCOL, solver='euler', solver_steps=3,
        frames=FRAMES, native_resolution=32, source_held_out=True, renderer_only=True,
        world_rollout=False, autonomous_rollout=False, sr_quality_evaluated=False,
        Source_RGB_direct_restorer_input=False, Source_RGB_into_RF_conditioning=False,
        future_source_rgb_in_prior=False, future_source_pose_in_prior=False)
    for name, expected in required.items():
        value = report.get(name)
        if value != expected or type(value) is not type(expected):
            raise ValueError('RF-Q goal requires exact validation scope: ' + name)
    for name, expected in pins.items():
        if report.get(name) != expected:
            raise ValueError('RF-Q goal rejects a different pinned ' + name)
    actual_data = _sha(report.get('data_identity_sha256'), 'report.data_identity_sha256')
    if data_identity_sha256 is not None and actual_data != _sha(data_identity_sha256, 'data_identity_sha256'):
        raise ValueError('RF-Q goal rejects a different Source data identity')
    step = report.get('step')
    if type(step) is not int or step < 0:
        raise ValueError('RF-Q goal requires a real nonnegative validation step')
    models = report.get('models')
    rf_new = models.get('rf_new') if isinstance(models, dict) else None
    summary = rf_new.get('summary') if isinstance(rf_new, dict) else None
    if (not isinstance(summary, dict) or type(summary.get('frames')) is not int
            or summary['frames'] != FRAMES):
        raise ValueError('RF-Q goal needs all256 frames in models.rf_new.summary')
    value = _ssim(summary.get('ssim_mean'))
    objective = report.get('validation_objective')
    if (not isinstance(objective, dict) or objective.get('protocol') != REPORT_PROTOCOL
            or objective.get('input_kind') != INPUT_KIND
            or type(objective.get('frames')) is not int or objective['frames'] != FRAMES
            or objective.get('source_held_out') is not True):
        raise ValueError('RF-Q goal needs the matching complete held-out validation objective')
    loss = _number(report.get('val_loss'), 'val_loss')
    if loss < 0 or _number(objective.get('loss'), 'validation_objective.loss') != loss:
        raise ValueError('Validation objective and measured val_loss must agree')
    met = value > TARGET
    return dict(protocol=PROTOCOL, eligible=True, goal_met=met,
        measured_ssim=value, current_ssim=value, target=TARGET,
        comparison='strict_greater_than', metric=METRIC, model='rf_new',
        frames=FRAMES, step=step, validation_step=step,
        report_kind=REPORT_KIND, report_protocol=REPORT_PROTOCOL,
        **pins, data_identity_sha256=actual_data,
        reason='threshold_exceeded' if met else 'threshold_not_exceeded',
        Source_preservation_is_goal_requirement=False, MAE_is_goal_requirement=False,
        first_full_validation_above_target_is_sufficient=True,
        source_held_out=True, world_rollout=False, native_resolution=32)


def update_restorer_best(best, report, *, panel_fingerprint, plan_sha256,
                         feature_world_sha256, data_identity_sha256=None,
                         report_sha256=None):
    """Return the measurement with maximum RF-Q SSIM; retain the earliest tie.

    The caller saves the corresponding checkpoint when ``improved`` is True.
    Loss, MAE, Source-Q scores and preservation gates cannot choose the winner.
    """
    decision = evaluate_restorer_goal(report, panel_fingerprint=panel_fingerprint,
        plan_sha256=plan_sha256, feature_world_sha256=feature_world_sha256,
        data_identity_sha256=data_identity_sha256)
    if report_sha256 is not None:
        report_sha256 = _sha(report_sha256, 'report_sha256')
    if best is not None:
        if not isinstance(best, dict):
            raise ValueError('Best RF-Q measurement must be a pinned goal record')
        for name, expected in dict(protocol=PROTOCOL, metric=METRIC, frames=FRAMES,
                target=TARGET, comparison='strict_greater_than',
                report_kind=REPORT_KIND, report_protocol=REPORT_PROTOCOL,
                panel_fingerprint=decision['panel_fingerprint'], plan_sha256=decision['plan_sha256'],
                feature_world_sha256=decision['feature_world_sha256'],
                data_identity_sha256=decision['data_identity_sha256']).items():
            if best.get(name) != expected or type(best.get(name)) is not type(expected):
                raise ValueError('Best RF-Q record belongs to a different goal/domain: ' + name)
        previous = _ssim(best.get('ssim'))
        if (type(best.get('step')) is not int or best['step'] < 0
                or type(best.get('validation_step')) is not int
                or best.get('validation_step') != best['step']
                or best['step'] > decision['validation_step']
                or type(best.get('goal_met')) is not bool
                or best['goal_met'] != (previous > TARGET)):
            raise ValueError('Best RF-Q record has an inconsistent step or goal result')
        if previous >= decision['measured_ssim']:
            result = deepcopy(best)
            result['improved'] = False
            return result
    return dict(protocol=PROTOCOL, metric=METRIC, target=TARGET,
        comparison='strict_greater_than', frames=FRAMES,
        ssim=decision['measured_ssim'], step=decision['validation_step'],
        validation_step=decision['validation_step'], goal_met=decision['goal_met'],
        panel_fingerprint=decision['panel_fingerprint'], plan_sha256=decision['plan_sha256'],
        feature_world_sha256=decision['feature_world_sha256'],
        data_identity_sha256=decision['data_identity_sha256'],
        report_kind=REPORT_KIND, report_protocol=REPORT_PROTOCOL,
        report_sha256=report_sha256, improved=True)


def plateau_report(history, *, panel_fingerprint, plan_sha256, feature_world_sha256,
                   data_identity_sha256=None, window=5, min_improvement=.001):
    """Describe recent RF-Q progress; a plateau NEVER stops this goal loop."""
    if type(window) is not int or window < 2:
        raise ValueError('Plateau diagnostics require at least two validations')
    minimum = _number(min_improvement, 'min_improvement')
    if minimum < 0:
        raise ValueError('Plateau minimum improvement must be nonnegative')
    measurements = [evaluate_restorer_goal(report, panel_fingerprint=panel_fingerprint,
        plan_sha256=plan_sha256, feature_world_sha256=feature_world_sha256,
        data_identity_sha256=data_identity_sha256) for report in history]
    measurements.sort(key=lambda item: item['validation_step'])
    steps = [item['validation_step'] for item in measurements]
    if len(set(steps)) != len(steps):
        raise ValueError('Plateau diagnostics require distinct full validation steps')
    result = dict(protocol=PROTOCOL, metric=METRIC, diagnostic_only=True,
        automatic_stop=False, stop_training=False, target=TARGET,
        window=window, measured_validations=len(measurements),
        min_improvement=minimum, plateau=None, reason='insufficient_validations')
    if len(measurements) < window:
        return result
    selected = measurements[-window:]
    values = [item['measured_ssim'] for item in selected]
    gain = max(values) - values[0]
    result.update(plateau=gain <= minimum, reason='measured_RF_Q_progress_only',
        steps=[item['validation_step'] for item in selected],
        first_ssim=values[0], last_ssim=values[-1], best_ssim=max(values),
        net_improvement=values[-1]-values[0], best_gain_from_window_start=gain,
        observed_range=max(values)-min(values))
    return result
