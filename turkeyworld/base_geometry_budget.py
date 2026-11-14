"""Explicit same-phase Base2000->TOTAL3000 revision; no counter reset."""
from copy import deepcopy
import re

KIND='base_geometry_budget_revision_v1'
PROTOCOL='base_expanded_geometry_v1'

def _sha(value):
    return isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None

def base_budget_revision_plan(saved,*,source_checkpoint,source_checkpoint_sha256,total_updates=3000):
    phase=saved.get('phase',{})
    if (phase.get('protocol',{}).get('name')!=PROTOCOL
            or phase.get('configuration',{}).get('steps')!=2000
            or type(saved.get('step'))is not int or not 0<=saved['step']<=2000
            or saved.get('base_training_steps')!=29209+saved['step']
            or not _sha(saved.get('plan_sha256'))or not _sha(source_checkpoint_sha256)
            or type(total_updates)is not int or total_updates!=3000
            or saved.get('base_budget_revision')is not None or not source_checkpoint):
        raise ValueError('Base revision needs its complete original2000 phase and source boundary')
    return dict(kind=KIND,version=1,protocol=PROTOCOL,legacy_total_updates=2000,total_updates=3000,
        source_step=saved['step'],source_checkpoint=str(source_checkpoint),
        source_checkpoint_sha256=source_checkpoint_sha256,phase_plan_sha256=saved['plan_sha256'],
        previous_updates_retained=True,optimizer_reset=False,sampler_reset=False,
        best_history_retained=True,budget_completion_is_quality=False)

def bind_base_budget_revision(saved,plan,*,path,sha256,loaded_checkpoint_sha256):
    if (not isinstance(plan,dict)or plan.get('kind')!=KIND or plan.get('version')!=1
            or plan.get('protocol')!=PROTOCOL or plan.get('legacy_total_updates')!=2000
            or type(plan.get('total_updates'))is not int or plan['total_updates']!=3000
            or type(plan.get('source_step'))is not int or not 0<=plan['source_step']<=2000
            or not _sha(plan.get('source_checkpoint_sha256'))or not _sha(sha256)
            or saved.get('plan_sha256')!=plan.get('phase_plan_sha256')
            or type(saved.get('step'))is not int or not plan['source_step']<=saved['step']<=3000
            or saved.get('phase',{}).get('configuration',{}).get('steps')!=2000
            or any(plan.get(k)is not value for k,value in dict(previous_updates_retained=True,
                optimizer_reset=False,sampler_reset=False,best_history_retained=True,
                budget_completion_is_quality=False).items())):
        raise ValueError('Base revision changed its immutable phase, source or preservation contract')
    record=dict(kind=KIND,version=1,plan_path=str(path),plan_sha256=sha256,
        source_step=plan['source_step'],source_checkpoint_sha256=plan['source_checkpoint_sha256'],
        phase_plan_sha256=plan['phase_plan_sha256'],legacy_total_updates=2000,total_updates=3000)
    existing=saved.get('base_budget_revision')
    if existing is None:
        if saved['step']!=plan['source_step']or loaded_checkpoint_sha256!=plan['source_checkpoint_sha256']:
            raise ValueError('First Base budget activation requires its exact stopped source')
    elif existing!=record:raise ValueError('Saved Base budget revision pin changed on resume')
    return deepcopy(record)
