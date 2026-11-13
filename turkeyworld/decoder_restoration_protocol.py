"""Pinned raw-RGBA warmup and a measured, persistent gradual SR blend."""
from copy import deepcopy
import math

PROTOCOL='decoder_restoration_sr_v3'
SETTINGS=dict(version=1,protocol=PROTOCOL,warmup_minimum_updates=200,
    blend_ramp_updates=400,validation_every=100,prediction_every=15,
    quality_gate='raw_foreground_and_edge_strict_improvement_background_empty_not_worse_than_initial',
    raw_reference='fixed_initial_held_out_NPR_panel_Source_background_labels_only',
    source_targets_are_inference_inputs=False,automatic_deployment=False)
METRICS=('raw_foreground_mae','raw_foreground_edge_mae','raw_empty_mae','raw_background_mae')


def initial_restoration_schedule():
    return dict(settings=deepcopy(SETTINGS),updates=0,last_optimizer_step=0,
        phase='standalone',blend_started_at=None,blend_scale=0.,
        baseline_report_sha256=None,baseline_panel_fingerprint=None,
        baseline_data_identity_sha256=None,gate_history=[])


def _scale(updates,start):
    return 0. if start is None else min(1.,(updates-start)/SETTINGS['blend_ramp_updates'])


def validate_restoration_schedule(state, *, step=None, config_scale=None):
    updates=state.get('updates');start=state.get('blend_started_at')
    if (state.get('settings')!=SETTINGS or type(updates)is not int or updates<0
            or state.get('last_optimizer_step')!=updates or step is not None and updates!=step
            or start is not None and (type(start)is not int or not 200<=start<=updates)):
        raise ValueError('Restoration phase clock or immutable schedule changed')
    expected=_scale(updates,start)
    phase='standalone' if start is None else 'restoration' if expected==1. else 'blending'
    history=state.get('gate_history')
    if (state.get('phase')!=phase or type(state.get('blend_scale'))not in(float,int)
            or state['blend_scale']!=expected or config_scale is not None and config_scale!=expected
            or not isinstance(history,list)
            or any(type(row.get('step'))is not int or not 0<row['step']<=updates for row in history)
            or any(a['step']>=b['step']for a,b in zip(history,history[1:]))
            or start is not None and not any(row.get('passed')is True and row['step']==start for row in history)):
        raise ValueError('Restoration saved scale, gate history or phase changed')
    return state


def next_restoration_blend_scale(state):
    validate_restoration_schedule(state)
    return _scale(state['updates']+1,state['blend_started_at'])


def commit_restoration_update(state, *, step):
    validate_restoration_schedule(state)
    if type(step)is not int or step!=state['updates']+1:
        raise ValueError('Restoration schedule commits once per complete optimizer update')
    state.update(updates=step,last_optimizer_step=step)
    scale=_scale(step,state['blend_started_at'])
    state.update(blend_scale=scale,phase='standalone'if state['blend_started_at']is None
                 else 'restoration'if scale==1. else 'blending')
    return validate_restoration_schedule(state)


def bind_restoration_baseline(state,report,*,sha256):
    validate_restoration_schedule(state)
    identity=(report.get('panel_fingerprint'),report.get('data_identity_sha256'),sha256)
    if (report.get('step')!=0 or report.get('source_held_out')is not True
            or any(not isinstance(value,str)or not value for value in identity)
            or 'restoration_raw'not in report):
        raise ValueError('Restoration needs its fixed initial held-out raw-RGBA report')
    previous=(state.get('baseline_panel_fingerprint'),state.get('baseline_data_identity_sha256'),
              state.get('baseline_report_sha256'))
    if previous!=(None,None,None)and previous!=identity:
        raise ValueError('Restoration initial raw-quality reference changed')
    state.update(baseline_panel_fingerprint=identity[0],baseline_data_identity_sha256=identity[1],
                 baseline_report_sha256=identity[2])


def consider_restoration_gate(state,report,baseline,*,step):
    validate_restoration_schedule(state,step=step)
    if (report.get('step')!=step or step<=0 or step%SETTINGS['validation_every']
            or report.get('source_held_out')is not True
            or report.get('panel_fingerprint')!=state['baseline_panel_fingerprint']
            or report.get('data_identity_sha256')!=state['baseline_data_identity_sha256']
            or baseline.get('panel_fingerprint')!=state['baseline_panel_fingerprint']
            or baseline.get('data_identity_sha256')!=state['baseline_data_identity_sha256']):
        raise ValueError('Restoration quality gate requires its exact complete held-out validation')
    if state['blend_started_at']is not None:
        return None
    if state['gate_history']and state['gate_history'][-1]['step']==step:
        raise ValueError('Restoration gate cannot consume one validation twice')
    current,initial=report['restoration_raw'],baseline['restoration_raw']
    finite=lambda value:type(value)in(float,int)and math.isfinite(value)and value>=0
    if any(row.get(name)is not None and not finite(row[name])for row in(current,initial)for name in METRICS):
        raise ValueError('Restoration raw-quality decision cannot consume nonfinite metrics')
    available=all(finite(row.get(name))for row in(current,initial)for name in METRICS)
    passed=(available and step>=SETTINGS['warmup_minimum_updates']
        and all(current[name]<initial[name]for name in METRICS[:2])
        and all(current[name]<=initial[name]for name in METRICS[2:]))
    record=dict(step=step,passed=bool(passed),minimum_warmup_completed=step>=200,
        metrics_available=available,current={name:current.get(name)for name in METRICS},
        initial={name:initial.get(name)for name in METRICS})
    state['gate_history'].append(record)
    if passed:
        state.update(blend_started_at=step,phase='blending',blend_scale=0.)
    validate_restoration_schedule(state,step=step)
    return record
