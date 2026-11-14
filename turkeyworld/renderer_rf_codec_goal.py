"""Pinned RF3/Q/frozen-D SSIM goal and auditable resume measurements."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


PROTOCOL = 'renderer_rf_codec_ssim_goal_v1'
REPORT_KIND = 'renderer_rf_codec_validation_v1'
REPORT_PROTOCOL = 'renderer_rf_codec_geometry_v1'
INPUT_KIND = 'RF3_terminal_Q_frozen_current_decoder'
TARGET = .85
FRAMES = 256
METRIC = 'models.rf_new.summary.ssim_mean'


def rf_codec_goal_spec():
    return dict(protocol=PROTOCOL, version=1, target=TARGET,
        comparison='strict_greater_than', metric=METRIC, model='rf_new',
        frames=FRAMES, report_kind=REPORT_KIND, report_protocol=REPORT_PROTOCOL,
        native_resolution=32, solver='euler', solver_steps=3,
        source_held_out=True, renderer_only=True, world_rollout=False,
        restorer_active=False, Source_Q_image_branch=False,
        first_full_validation_above_target_is_sufficient=True,
        consecutive_successes_required=1, Source_preservation_is_goal_requirement=False,
        geometry_preservation_is_goal_requirement=False, MAE_is_goal_requirement=False,
        Source_code_SSIM_is_goal_measurement=False, budget_updates=None,
        budget_completion_is_success=False, plateau_is_automatic_stop=False,
        stop_policy='goal_success_or_explicit_user_stop_or_error', automatic_active_promotion=False)


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(name + ' must be a lowercase SHA256 fingerprint')
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ' must be finite and cannot be a bool or proxy')
    return float(value)


def _ssim(value):
    value = _number(value, METRIC)
    if not -1 <= value <= 1:
        raise ValueError('RF-Q SSIM must use the native [-1,1] scale')
    return value


def _exact(mapping, expected, context):
    if not isinstance(mapping, dict):
        raise ValueError(context + ' must be a dictionary')
    for name, value in expected.items():
        if mapping.get(name) != value or type(mapping.get(name)) is not type(value):
            raise ValueError(context + ' has a missing or different ' + name)


def evaluate_rf_codec_goal(report, *, panel_fingerprint, plan_sha256, feature_world_sha256):
    """Accept only a full pinned RF-Q native32 measurement, strictly above .85."""
    pins = {name: _sha(value, name) for name, value in dict(
        panel_fingerprint=panel_fingerprint, plan_sha256=plan_sha256,
        feature_world_sha256=feature_world_sha256).items()}
    _exact(report, dict(kind=REPORT_KIND, protocol=REPORT_PROTOCOL, solver='euler', solver_steps=3,
        frames=FRAMES, native_resolution=32, source_held_out=True, renderer_only=True,
        world_rollout=False, autonomous_rollout=False, sr_quality_evaluated=False,
        restorer_active=False, Source_Q_image_branch=False,
        Source_RGB_into_RF_conditioning=False, future_source_rgb_in_prior=False,
        future_source_pose_in_prior=False, validation_objective_matches_main_training=True,
        **pins), 'RF-codec validation report')
    step = report.get('step')
    if type(step) is not int or step < 0:
        raise ValueError('RF-codec measurement requires an actual nonnegative validation step')
    data_sha = _sha(report.get('data_identity_sha256'), 'data_identity_sha256')
    models = report.get('models')
    head = models.get('rf_new') if isinstance(models, dict) else None
    summary = head.get('summary') if isinstance(head, dict) else None
    _exact(summary, dict(frames=FRAMES), 'RF-Q summary')
    value = _ssim(summary.get('ssim_mean'))
    objective = report.get('validation_objective')
    _exact(objective, dict(protocol=REPORT_PROTOCOL, input_kind=INPUT_KIND,
        frames=FRAMES, source_held_out=True, matches_training_formula=True,
        main_only=True, Source_Q_image_branch=False), 'RF-codec held-out MAIN objective')
    loss = _number(report.get('val_loss'), 'val_loss')
    if loss < 0 or _number(objective.get('loss'), 'validation_objective.loss') != loss:
        raise ValueError('RF-codec held-out MAIN objective and val_loss must agree')
    met = value > TARGET
    return dict(protocol=PROTOCOL, eligible=True, measured_ssim=value, current_ssim=value,
        goal_met=met, target=TARGET, comparison='strict_greater_than', metric=METRIC,
        model='rf_new', frames=FRAMES, step=step, validation_step=step,
        report_kind=REPORT_KIND, report_protocol=REPORT_PROTOCOL, **pins,
        data_identity_sha256=data_sha, source_held_out=True, world_rollout=False,
        native_resolution=32, restorer_active=False, Source_Q_image_branch=False,
        reason='threshold_exceeded' if met else 'threshold_not_exceeded',
        first_full_validation_above_target_is_sufficient=True,
        Source_preservation_is_goal_requirement=False, geometry_preservation_is_goal_requirement=False,
        MAE_is_goal_requirement=False)


def update_rf_codec_best(best, report, *, panel_fingerprint, plan_sha256,
                         feature_world_sha256, report_sha256=None):
    """Keep maximum actual RF SSIM, retaining the earliest exact tie."""
    decision = evaluate_rf_codec_goal(report, panel_fingerprint=panel_fingerprint,
        plan_sha256=plan_sha256, feature_world_sha256=feature_world_sha256)
    if report_sha256 is not None:
        report_sha256 = _sha(report_sha256, 'report_sha256')
    identity = dict(protocol=PROTOCOL, metric=METRIC, target=TARGET,
        comparison='strict_greater_than', frames=FRAMES, report_kind=REPORT_KIND,
        report_protocol=REPORT_PROTOCOL, panel_fingerprint=decision['panel_fingerprint'],
        plan_sha256=decision['plan_sha256'], feature_world_sha256=decision['feature_world_sha256'],
        data_identity_sha256=decision['data_identity_sha256'])
    if best is not None:
        _exact(best, identity, 'Best RF-codec measurement')
        previous = _ssim(best.get('ssim'))
        if (type(best.get('step')) is not int or not 0 <= best['step'] <= decision['step']
                or type(best.get('validation_step')) is not int or best['validation_step'] != best['step']
                or type(best.get('goal_met')) is not bool or best['goal_met'] != (previous > TARGET)
                or type(best.get('improved')) is not bool):
            raise ValueError('Best RF-codec measurement has inconsistent clocks or goal metadata')
        if best.get('report_sha256') is not None:
            _sha(best['report_sha256'], 'best.report_sha256')
        if previous >= decision['measured_ssim']:
            result = deepcopy(best)
            result['improved'] = False
            return result
    return dict(**identity, ssim=decision['measured_ssim'], step=decision['step'],
        validation_step=decision['step'], goal_met=decision['goal_met'],
        report_sha256=report_sha256, improved=True)


def validate_saved_rf_codec_goal(saved, *, panel_fingerprint, plan_sha256,
                                feature_world_sha256, run_dir):
    """Rebuild completion and best selection from each exact saved report.

    Reading one closed byte string supplies both the SHA and the JSON, avoiding
    a separate hash/read race. A saved boolean alone cannot complete training.
    """
    if not isinstance(saved, dict):
        raise ValueError('Saved RF-codec goal state must be a dictionary')
    step, history, best, met = (saved.get(key) for key in ('step', 'goal_history', 'best', 'goal_met'))
    if type(step) is not int or step < 0 or type(met) is not bool or not isinstance(history, list):
        raise ValueError('Saved RF-codec goal needs exact clocks, history and completion boolean')
    pins = dict(panel_fingerprint=panel_fingerprint, plan_sha256=plan_sha256,
                feature_world_sha256=feature_world_sha256)
    for name, value in pins.items():
        _sha(value, name)
    if not history:
        if step != 0 or best is not None or met:
            raise ValueError('An unmeasured RF-codec goal cannot resume beyond initial update0')
        return None, [], False
    if run_dir is None:
        raise ValueError('Saved RF-codec goal requires its preserved validation reports')
    expected_best, previous_step = None, -1
    for item in history:
        if (not isinstance(item, dict) or type(item.get('step')) is not int
                or not previous_step < item['step'] <= step
                or type(item.get('goal_met')) is not bool):
            raise ValueError('Saved RF-codec history has invalid or unordered validation clocks')
        value = _ssim(item.get('ssim'))
        if item['goal_met'] != (value > TARGET):
            raise ValueError('Saved RF-codec history has a false goal result')
        report_sha = _sha(item.get('report_sha256'), 'history.report_sha256')
        path = Path(run_dir) / 'validation' / f"step_{item['step']:06d}.json"
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError('Saved RF-codec validation report is missing') from error
        if hashlib.sha256(payload).hexdigest() != report_sha:
            raise ValueError('Saved RF-codec validation report SHA changed')
        try:
            report = json.loads(payload.decode('utf-8-sig'))
        except (UnicodeError, ValueError) as error:
            raise ValueError('Saved RF-codec validation report is invalid JSON') from error
        decision = evaluate_rf_codec_goal(report, **pins)
        if (decision['step'] != item['step'] or decision['measured_ssim'] != value
                or decision['goal_met'] != item['goal_met']):
            raise ValueError('Saved RF-codec history disagrees with its actual measurement')
        expected_best = update_rf_codec_best(expected_best, report, report_sha256=report_sha, **pins)
        previous_step = item['step']
    if (history[0]['step'] != 0 or any(item['goal_met'] for item in history[:-1])
            or met != history[-1]['goal_met'] or (met and step != history[-1]['step'])):
        raise ValueError('RF-codec completion must follow its first successful full validation')
    if not isinstance(best, dict) or type(best.get('improved')) is not bool:
        raise ValueError('Saved RF-codec best measurement is missing')
    if set(best) != set(expected_best) or any(
            best[key] != value or type(best[key]) is not type(value)
            for key, value in expected_best.items() if key != 'improved'):
        raise ValueError('Saved RF-codec best must be the exact earliest maximum held-out measurement')
    return deepcopy(best), deepcopy(history), met
