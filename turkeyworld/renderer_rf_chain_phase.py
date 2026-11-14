"""Train RF3 after selected expanded Base; preserve all seven Adam groups."""
from copy import deepcopy
from types import SimpleNamespace

import torch

from .base_expanded_experts import base_expanded_expert_metadata
from .base_geometry_phase import _OldBaseParameters
from .renderer_native_phase import _same
from .renderer_rf5_experts import rf5_expert_metadata
from .renderer_rf_codec_phase import (configure_rf_codec_training,rf_codec_frozen_fingerprint,
    rf_codec_parameter_counts,rf_codec_trainable_names,_expected_groups)


PROTOCOL='renderer_rf3_after_expanded_base_v1'


def rf_chain_protocol():
    return dict(name=PROTOCOL,version=1,local_updates=3000,solver='euler',solver_steps=3,
        parent_protocol='base_expanded_geometry_v1',parent_bundle_format=6,output_bundle_format=7,
        Base_frozen=True,codec_frozen=True,private_opacity_frozen=True,world_frozen=True,
        restorer_active=False,optimizer_groups_preserved=7,new_optimizer_groups=0,
        source_samplers_and_global_RNG_preserved=True,selected_parent='complete_best_Base_checkpoint',
        budget_completion_is_quality=False,stop_on_ssim=False,automatic_promotion=False)


def configure_rf_chain_training(model):
    base_expanded_expert_metadata(model.base_student)
    return configure_rf_codec_training(model)


def rf_chain_frozen_fingerprint(model):return rf_codec_frozen_fingerprint(model)
def rf_chain_parameter_counts(model):return rf_codec_parameter_counts(model)


def rf_chain_canonical_names(model):
    base=model.base_student;meta=base_expanded_expert_metadata(base)
    view=SimpleNamespace(base_student=_OldBaseParameters(base,meta['legacy_parameter_name_map']),flow_renderer=model.flow_renderer)
    old=_expected_groups(view,rf5_expert_metadata(model.flow_renderer.output_moe))
    remap={'base_student.'+a:'base_student.'+b for a,b in meta['legacy_parameter_name_map'].items()}
    return [[remap.get(n,n)for n in group]for group in old]+[
        ['base_student.'+n for n in meta['added_parameter_names']]]


def adopt_rf_chain_optimizer(model,saved):
    names=rf_chain_canonical_names(model);flat=[n for g in names for n in g]
    parameters=dict(model.named_parameters());step=saved.get('step')
    if (saved.get('phase',{}).get('protocol',{}).get('name')!='base_expanded_geometry_v1'
            or type(step)is not int or not 0<=step<=3000
            or saved.get('renderer_training_steps')!=30717
            or saved.get('base_training_steps')!=29209+step
            or saved.get('optimizer_parameter_names')!=names
            or {n for n,p in parameters.items()if p.requires_grad}!=set(rf_codec_trainable_names(model))):
        raise ValueError('RF chain requires the selected whole expanded-Base checkpoint and canonical seven-group optimizer')
    for key,module in (('base',model.base_student),('flow',model.flow_renderer)):
        expected=module.state_dict();old=saved.get(key,{})
        if (set(old)!=set(expected)or any(not torch.is_tensor(v)or not bool(torch.isfinite(v).all())
                or v.shape!=expected[n].shape or v.dtype!=expected[n].dtype or not _same(v,expected[n])for n,v in old.items())):
            raise ValueError('RF chain must start from exact selected Base/RF/currentD/EQ/private tensors')
    optimizer=saved.get('optimizer',{});groups=optimizer.get('param_groups',[]);states=optimizer.get('state',{})
    ids=[i for group in groups for i in group.get('params',[])]
    fresh_base=set(groups[-1].get('params',[]))if len(groups)==7 else set()
    missing=set(ids)-set(states)
    if (len(groups)!=7 or ids!=list(range(len(flat)))
            or any(len(group.get('params',[]))!=len(n)for group,n in zip(groups,names))
            or set(states)!=(set(ids)-fresh_base if step==0 else set(ids))
            or any(parameters[flat[i]].requires_grad for i in missing)):
        raise ValueError('RF chain cannot omit an initialized Adam state; only Basebest0 fresh80 may remain absent and frozen')
    for i,moment in states.items():
        parameter=parameters[flat[i]];age=moment.get('step')
        if (not torch.is_tensor(age)or age.numel()!=1 or not bool(torch.isfinite(age))
                or float(age)<1 or float(age)!=int(float(age))
                or i in fresh_base and float(age)!=step
                or any(not torch.is_tensor(moment.get(k))or moment[k].shape!=parameter.shape
                    or moment[k].dtype!=parameter.dtype or not bool(torch.isfinite(moment[k]).all())
                    for k in ('exp_avg','exp_avg_sq'))or bool((moment['exp_avg_sq']<0).any())):
            raise ValueError('RF chain parent Adam age/moment is incomplete, nonfinite or shape-mismatched')
    configured=[dict(**deepcopy({k:v for k,v in group.items()if k!='params'}),
        params=[parameters[n]for n in named])for group,named in zip(groups,names)]
    result=torch.optim.AdamW(configured);result.load_state_dict(deepcopy(optimizer))
    if not _same(result.state_dict(),optimizer):raise AssertionError('RF chain reset an old Adam state or option')
    proof=dict(kind=PROTOCOL,version=1,source_step=step,parent_Base_training_steps=saved['base_training_steps'],
        parent_RF_training_steps=30717,optimizer_parameter_names=deepcopy(names),old_Adam_states=len(states),
        optimizer_parameter_tensors=len(flat),old_optimizer_groups=7,new_optimizer_groups=0,
        all_old_Adam_states_options_bitexact=True,missing_frozen_Base_state_ids=sorted(missing),
        initial_missing_Base_moments_invented=False,old_model_tensors_bitexact=True,
        trainable_parameter_names=rf_codec_trainable_names(model),parameters=rf_chain_parameter_counts(model),
        frozen_fingerprint=rf_chain_frozen_fingerprint(model),samplers_and_global_RNG_unchanged=True,
        local_counter_starts_zero=True,Base_frozen=True,codec_frozen=True,restorer_active=False)
    return result,proof
