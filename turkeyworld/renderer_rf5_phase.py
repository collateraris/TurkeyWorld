"""Named full-Adam continuation for RF5; no checkpoint or sampler is written here."""
from copy import deepcopy
import math

import torch

from .renderer_native_phase import _same
from .renderer_refinement import frozen_fingerprint
from .renderer_rf5_experts import SETTINGS, rf5_expert_metadata


PROTOCOL = 'renderer_rf_expanded_experts_v1'
PREFIX = 'flow_renderer.output_moe.'


def renderer_rf5_protocol(solver_steps=3):
    if type(solver_steps) is not int or solver_steps not in (3,5):
        raise ValueError('Expanded RF training supports explicit three or five Euler steps')
    return dict(name=PROTOCOL,version=1,solver='euler',solver_steps=solver_steps,
        parent_protocol='one_enemy_renderer_token_moe_five_v1',
        parent_solver_steps=1,expert_architecture=deepcopy(SETTINGS),
        Base_frozen=True,VQGAN_frozen=True,private_opacity_frozen=True,world_frozen=True,
        old_RF_trunk_experts_and_router_trainable=True,
        old_Adam_groups_preserved=5,fresh_Adam_group=6,new_optimizer_lr=1e-4,
        old_Base_training_clock_frozen=True,local_counter_starts_zero=True,
        Source_targets_are_labels_only=True,Source_future_feedback=False,
        sampler_and_global_RNG_restoration='caller_restores_full_parent_exactly',
        automatic_active_promotion=False,gameplay_qualified=False)


def rf5_frozen_fingerprint(model):
    """Hash EVERY buffer and parameter outside the current trainable RF set."""
    return frozen_fingerprint(model)


def rf5_parameter_counts(model):
    flow=model.flow_renderer
    return dict(base=sum(p.numel()for p in model.base_student.parameters()),
        rf_total_without_codec=sum(p.numel()for n,p in flow.named_parameters()if not n.startswith('codec.')),
        rf_trainable=sum(p.numel()for p in flow.parameters()if p.requires_grad),
        codec_frozen=sum(p.numel()for p in flow.codec.parameters()),
        private_opacity_frozen=sum(p.numel()for p in flow.geometry_opacity.parameters()),
        experts=5,parameters_per_expert=SETTINGS['parameters_per_expert'],
        added_parameters=SETTINGS['added_parameters'],added_parameter_tensors=80,
        total_model=sum(p.numel()for p in model.parameters()),
        trainable_model=sum(p.numel()for p in model.parameters()if p.requires_grad))


def configure_rf5_training(model):
    """Keep all pretrained assets frozen while training RF and its five expansions."""
    flow=getattr(model,'flow_renderer',None)
    if (not isinstance(getattr(model,'base_student',None),torch.nn.Module)
            or flow is None or getattr(flow,'kind',None)!='student'
            or getattr(flow,'mode',None)!='rectified_flow' or type(flow.steps)is not int or flow.steps not in (3,5)
            or not isinstance(getattr(flow,'codec',None),torch.nn.Module)
            or not isinstance(getattr(flow,'geometry_opacity',None),torch.nn.Module)):
        raise ValueError('RF5 training requires the complete compact Base, five-step student, codec and private opacity')
    metadata=rf5_expert_metadata(flow.output_moe)
    model.eval().requires_grad_(False)
    flow.train()
    for name,parameter in flow.named_parameters():
        parameter.requires_grad_(not name.startswith(('codec.','geometry_opacity.')))
    flow.codec.eval().requires_grad_(False)
    flow.geometry_opacity.eval().requires_grad_(False)
    model.base_student.eval().requires_grad_(False)
    return dict(protocol=PROTOCOL,solver='euler',solver_steps=flow.steps,
        Base_frozen=True,VQGAN_frozen=True,private_opacity_frozen=True,world_frozen=True,
        expert_architecture=metadata,parameters=rf5_parameter_counts(model),
        frozen_fingerprint=rf5_frozen_fingerprint(model))


def _canonical_names(model,metadata):
    """Rebuild the exact original five-group order, excluding only new capacity."""
    renamed={PREFIX+old:PREFIX+new for old,new in metadata['legacy_parameter_name_map'].items()}
    inverse={new:old for old,new in renamed.items()}
    added=[PREFIX+name for name in metadata['added_parameter_names']]
    if len(added)!=80 or len(set(added))!=80 or len(inverse)!=len(renamed):
        raise ValueError('RF5 parameter mapping is not a complete canonical bijection')
    added_set=set(added);groups=[[]for _ in range(5)]
    for head,module in(('base_student',model.base_student),('flow_renderer',model.flow_renderer)):
        for local,parameter in module.named_parameters():
            full=head+'.'+local
            if full in added_set or head=='flow_renderer'and local.startswith('codec.'):
                continue
            old=inverse.get(full,full)
            name=old[len(head)+1:]
            if name.startswith(('output_moe.experts.3.','output_moe.experts.4.','output_moe.split_router.')):index=4
            elif name.startswith('output_moe.'):index=3
            elif head=='flow_renderer'and name.startswith('geometry_opacity.'):index=2
            else:index=0 if head=='base_student'else 1
            groups[index].append(old)
    if any(not group for group in groups)or len(groups[2])!=4:
        raise ValueError('RF5 needs all five complete original Adam groups, including four private-opacity tensors')
    return groups,renamed,added


def build_rf5_optimizer(model,saved):
    """Adopt every old named moment/options entry, then append one fresh group.

    The caller has already loaded the complete parent and appended zero-output
    expert residuals. No sampler, global RNG, model tensor or curriculum changes.
    """
    flow=model.flow_renderer
    metadata=rf5_expert_metadata(flow.output_moe)
    required={'base','flow','optimizer','optimizer_parameter_names','step',
              'renderer_training_steps','base_training_steps','identity'}
    if (not isinstance(saved,dict)or not required<=saved.keys()
            or type(saved['step'])is not int or saved['step']<1
            or any(type(saved[k])is not int or saved[k]<1 for k in('renderer_training_steps','base_training_steps'))
            or saved['identity'].get('renderer_moe_protocol',{}).get('name')!='one_enemy_renderer_token_moe_five_v1'):
        raise ValueError('RF5 adoption requires a complete trained five-role parent with explicit cumulative clocks')
    expected,renamed,added=_canonical_names(model,metadata)
    names=saved['optimizer_parameter_names']
    if names!=expected:
        raise ValueError('RF5 parent Adam names/order differ from the complete canonical five groups')
    parameters=dict(model.named_parameters())
    mapped=[[renamed.get(name,name)for name in group]for group in names]
    flat=[name for group in mapped for name in group]
    if len(set(flat))!=len(flat)or set(flat)&set(added)or any(name not in parameters for name in flat+added):
        raise ValueError('RF5 old/new parameter names are not a one-to-one complete mapping')
    # Verify source weights and buffers before any optimizer state is adopted.
    current={head+'.'+name:value for head,module in(('base_student',model.base_student),('flow_renderer',flow))
             for name,value in module.state_dict().items()}
    old={head+'.'+name:value for head,key in(('base_student','base'),('flow_renderer','flow'))
         for name,value in saved[key].items()}
    remapped={}
    for name,value in old.items():
        target=renamed.get(name,name)
        if target in remapped:raise ValueError('RF5 parent model name mapping is not a bijection')
        remapped[target]=value
    if (set(current)!=set(remapped)|set(added)
            or any(not _same(value,current[name])for name,value in remapped.items())
            or any(not bool(torch.isfinite(current[name]).all())for name in added)):
        raise ValueError('RF5 adoption changed an old Base/RF/codec/opacity tensor or added invalid capacity')
    if any(bool(current[name].count_nonzero())for name in added if '.private_output.'in name):
        raise ValueError('RF5 private residual outputs must be exactly zero before adoption')
    trainable={name for name,p in model.named_parameters()if p.requires_grad}
    wanted={name for name in flat+added if name.startswith('flow_renderer.')
            and not name.startswith(('flow_renderer.codec.','flow_renderer.geometry_opacity.'))}
    if trainable!=wanted:
        raise ValueError('Configure RF5 first: only RF trunk/experts/router may be trainable')
    old_optimizer=saved['optimizer'];old_groups=old_optimizer.get('param_groups',[])
    old_states=old_optimizer.get('state',{})
    if len(old_groups)!=5 or any(len(group.get('params',[]))!=len(group_names)for group,group_names in zip(old_groups,names)):
        raise ValueError('RF5 adoption lost a complete ordered parent Adam group')
    old_ids=[index for group in old_groups for index in group['params']]
    if old_ids!=list(range(len(flat)))or set(old_states)!=set(old_ids):
        raise ValueError('RF5 parent Adam IDs/moments must cover every canonical old parameter exactly once')
    for index,name in zip(old_ids,flat):
        moment=old_states[index];parameter=parameters[name]
        age=moment.get('step')
        if (not {'step','exp_avg','exp_avg_sq'}<=moment.keys()
                or not torch.is_tensor(age)or age.numel()!=1
                or not math.isfinite(float(age))or float(age)<1 or float(age)!=int(float(age))
                or any(not torch.is_tensor(moment[k])or moment[k].shape!=parameter.shape
                       or moment[k].dtype!=parameter.dtype or not bool(torch.isfinite(moment[k]).all())
                       for k in('exp_avg','exp_avg_sq'))):
            raise ValueError('RF5 parent Adam has a missing, nonfinite, wrong-shape or reset moment/age')
        if any(torch.is_tensor(v)and not bool(torch.isfinite(v).all())for v in moment.values()):
            raise ValueError('RF5 parent Adam contains a nonfinite state tensor')
    groups=[{**deepcopy({k:v for k,v in group.items()if k!='params'}),
             'params':[parameters[name]for name in group_names]}
            for group,group_names in zip(old_groups,mapped)]
    fresh={**deepcopy({k:v for k,v in old_groups[1].items()if k!='params'}),
           'params':[parameters[name]for name in added],'lr':renderer_rf5_protocol()['new_optimizer_lr']}
    optimizer=torch.optim.AdamW([*groups,fresh])
    constructed=optimizer.state_dict()
    optimizer.load_state_dict(dict(state=deepcopy(old_states),param_groups=deepcopy(constructed['param_groups'])))
    adopted=optimizer.state_dict()
    if (not _same(adopted['state'],old_states)
            or not _same(adopted['param_groups'][:5],old_groups)
            or set(adopted['param_groups'][5]['params'])&set(adopted['state'])):
        raise AssertionError('RF5 optimizer failed exact old-Adam preservation or fresh-group isolation')
    all_names=[*mapped,added]
    proof=dict(kind=PROTOCOL,version=1,source_step=saved['step'],start_step=0,
        parent_renderer_training_steps=saved['renderer_training_steps'],
        frozen_base_training_steps=saved['base_training_steps'],solver='euler',solver_steps=flow.steps,
        parent_solver_steps=1,expert_architecture=metadata,
        old_model_tensors_bitexact=True,zero_residual_initialization=True,
        old_Adam_states=len(old_states),old_optimizer_groups=5,
        all_old_Adam_states_options_bitexact=True,
        old_parameter_name_map={name:renamed.get(name,name)for group in names for name in group},
        renamed_legacy_parameter_names=renamed,optimizer_parameter_names=all_names,
        added_parameter_names=added,added_parameter_tensors=len(added),
        added_parameters=sum(parameters[name].numel()for name in added),new_Adam_states_at_start=0,
        new_optimizer_lr=fresh['lr'],old_Base_world_codec_opacity_frozen=True,
        frozen_fingerprint=rf5_frozen_fingerprint(model),parameters=rf5_parameter_counts(model),
        sampler_and_global_RNG_unchanged_by_optimizer_builder=True,
        automatic_active_promotion=False,gameplay_qualified=False)
    return optimizer,proof
