"""RF-chain best selection from full reports; fixed3000 completion is separate."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


KIND = 'renderer_rf_codec_validation_v1'
PROTOCOL = 'renderer_rf_codec_geometry_v1'
METRIC = 'models.rf_new.summary.ssim_mean'


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(name + ' must be a lowercase SHA256 fingerprint')
    return value


def _exact(mapping, expected, context):
    if not isinstance(mapping, dict) or any(
            mapping.get(key) != value or type(mapping.get(key)) is not type(value)
            for key, value in expected.items()):
        raise ValueError(context + ' has missing or changed scope, pins or measurements')


def _ssim(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError('Raw Base SSIM must be a finite [-1,1] scalar, not a bool or proxy')
    return float(value)


def rf_chain_measurement(report, *, panel_fingerprint, plan_sha256, feature_world_sha256):
    from .renderer_rf_codec_goal import evaluate_rf_codec_goal
    measured=evaluate_rf_codec_goal(report,panel_fingerprint=panel_fingerprint,
        plan_sha256=plan_sha256,feature_world_sha256=feature_world_sha256)
    identity={key:_sha(report.get(key),key)for key in
        ('data_identity_sha256','codec_sha256','base_student_identity','reference_RF_sha256')}
    return dict(step=measured['step'],ssim=measured['measured_ssim'],metric=METRIC,
        panel_fingerprint=panel_fingerprint,plan_sha256=plan_sha256,
        feature_world_sha256=feature_world_sha256,**identity)

def validate_saved_rf_chain_best(saved, *, panel_fingerprint, plan_sha256,
                             feature_world_sha256, run_dir):
    """Rebuild the earliest maximum from all saved, SHA-pinned full reports."""
    pins = dict(panel_fingerprint=panel_fingerprint, plan_sha256=plan_sha256,
                feature_world_sha256=feature_world_sha256)
    for key, value in pins.items():_sha(value, key)
    if not isinstance(saved, dict) or type(saved.get('step')) is not int or saved['step'] < 0:
        raise ValueError('Saved Base selection needs an actual nonnegative optimizer clock')
    history, best = saved.get('validation_history'), saved.get('best')
    if not isinstance(history, list):
        raise ValueError('Saved Base selection must retain its validation history')
    if not history:
        if saved['step'] != 0 or best is not None:
            raise ValueError('An unmeasured Base selection may only resume initial update0')
        return None, []
    if run_dir is None:
        raise ValueError('Saved Base selection requires its preserved validation reports')
    previous, maximum, domain = -1, None, None
    for item in history:
        if (not isinstance(item, dict) or type(item.get('step')) is not int
                or not previous < item['step'] <= saved['step']):
            raise ValueError('Saved Base validation clocks must be ascending and not in the future')
        score = _ssim(item.get('ssim'))
        sha = _sha(item.get('report_sha256'), 'history.report_sha256')
        path = Path(run_dir)/'validation'/f"step_{item['step']:06d}.json"
        try:payload = path.read_bytes()
        except OSError as error:raise ValueError('Saved Base validation report is missing') from error
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError('Saved Base validation report SHA changed')
        try:report = json.loads(payload.decode('utf-8-sig'))
        except (UnicodeError, ValueError) as error:raise ValueError('Saved Base report is invalid JSON') from error
        measured = rf_chain_measurement(report, **pins)
        if measured['step'] != item['step'] or measured['ssim'] != score:
            raise ValueError('Saved Base history disagrees with its raw held-out measurement')
        current_domain = tuple(measured[name] for name in ('data_identity_sha256', 'codec_sha256', 'base_student_identity','reference_RF_sha256'))
        if domain is not None and domain != current_domain:
            raise ValueError('Saved Base history changed Source data or frozen RF/codec identity')
        domain = current_domain
        if maximum is None or score > maximum['ssim']:
            maximum = dict(step=item['step'], ssim=score, metric=METRIC, **pins, report_sha256=sha)
        previous = item['step']
    if history[0]['step'] != 0:
        raise ValueError('Saved Base history must retain its original validation0')
    if (not isinstance(best, dict) or set(best) != set(maximum)
            or any(best[key] != value or type(best[key]) is not type(value) for key, value in maximum.items())):
        raise ValueError('Saved Base best must be the exact earliest maximum raw Base SSIM')
    return deepcopy(best), deepcopy(history)


