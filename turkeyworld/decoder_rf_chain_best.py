"""Report-backed best RF-Q decoder selection for a fixed 3000-update stage."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


PROTOCOL='decoder_rf_source_paired_adaptation_v1'
METRIC='models.rf_new.summary.ssim_mean'


def decoder_chain_measurement(report,*,panel_fingerprint,plan_sha256,feature_world_sha256):
    pins=dict(panel_fingerprint=panel_fingerprint,plan_sha256=plan_sha256,feature_world_sha256=feature_world_sha256)
    if any(not isinstance(v,str)or len(v)!=64 or any(c not in '0123456789abcdef'for c in v)for v in pins.values()):
        raise ValueError('Decoder chain selection requires exact SHA256 pins')
    required=dict(kind='decoder_rf_adaptation_validation_v1',protocol=PROTOCOL,frames=256,
        native_resolution=32,solver_steps=3,source_held_out=True,renderer_only=True,world_rollout=False,
        Source_RGB_into_RF_conditioning=False,future_source_rgb_in_prior=False,
        future_source_pose_in_prior=False,reference_codec_tensors_unchanged=True,
        encoder_quantizer_tensors_unchanged=True,**pins)
    if not isinstance(report,dict)or any(report.get(k)!=v or type(report.get(k))is not type(v)for k,v in required.items()):
        raise ValueError('Decoder chain selection changed its true held-out RF-Q scope or pins')
    step=report.get('step');summary=report.get('models',{}).get('rf_new',{}).get('summary',{})
    score=summary.get('ssim_mean')
    if (type(step)is not int or step<0 or summary.get('frames')!=256 or type(summary.get('frames'))is not int
            or type(score)not in(int,float)or not math.isfinite(score)or not -1<=score<=1):
        raise ValueError('Decoder best requires an actual finite full256 RF-Q SSIM measurement')
    identity={k:report.get(k)for k in('data_identity_sha256','reference_codec_sha256')}
    if any(not isinstance(v,str)or len(v)!=64 or any(c not in '0123456789abcdef'for c in v)for v in identity.values()):
        raise ValueError('Decoder selection requires unchanged Source and reference-codec identities')
    return dict(step=step,ssim=float(score),metric=METRIC,**pins,**identity)


def validate_saved_decoder_chain_best(saved,*,panel_fingerprint,plan_sha256,feature_world_sha256,run_dir):
    history=saved.get('validation_history');best=saved.get('best');clock=saved.get('step')
    if type(clock)is not int or clock<0 or not isinstance(history,list):raise ValueError('Missing decoder selection history')
    if not history:
        if clock!=0 or best is not None:raise ValueError('Only the unmeasured initial decoder may have no best')
        return None,[]
    previous=-1;maximum=None;domain=None
    pins=dict(panel_fingerprint=panel_fingerprint,plan_sha256=plan_sha256,feature_world_sha256=feature_world_sha256)
    for item in history:
        if not isinstance(item,dict)or type(item.get('step'))is not int or not previous<item['step']<=clock:
            raise ValueError('Decoder validation history must be ordered and not future-dated')
        path=Path(run_dir)/'validation'/f"step_{item['step']:06d}.json"
        try:raw=path.read_bytes();report=json.loads(raw.decode('utf-8-sig'))
        except (OSError,ValueError,UnicodeError)as error:raise ValueError('Missing decoder held-out report')from error
        sha=hashlib.sha256(raw).hexdigest()
        measured=decoder_chain_measurement(report,**pins)
        if item!={k:measured[k]for k in('step','ssim')}|dict(report_sha256=sha):
            raise ValueError('Decoder best history differs from its SHA-pinned report')
        current=tuple(measured[k]for k in('data_identity_sha256','reference_codec_sha256'))
        if domain is not None and current!=domain:raise ValueError('Decoder history changed its Source or frozen reference')
        domain=current
        if maximum is None or measured['ssim']>maximum['ssim']:
            maximum={k:v for k,v in measured.items()if k not in('data_identity_sha256','reference_codec_sha256')}
            maximum['report_sha256']=sha
        previous=item['step']
    if history[0]['step']!=0 or best!=maximum or any(type(best[k])is not type(v)for k,v in maximum.items()):
        raise ValueError('Saved decoder best must be the earliest maximum actual RF-Q SSIM')
    return deepcopy(best),deepcopy(history)
