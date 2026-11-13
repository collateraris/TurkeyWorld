"""Preserve full v7 training while adding a private renderer-opacity MLP."""
from copy import deepcopy
import re

import torch

from .renderer_native_phase import _same,restore_renderer_native_rng


PRIVATE_PREFIX='geometry_opacity.'
PROTOCOL=dict(name='one_enemy_private_geometry_renderer_v8',version=1,
    parent_protocol='one_enemy_all_depth_wall_geometry_renderer_v7',
    private_parameters=6241,private_optimizer_lr=.00005,opacity_depth_weight=1.,
    depth_weight_warmup_updates=500,
    original_shared_geometry_frozen=True,codec_world_map_SR_frozen=True,
    original_Base_RF_weights_Adam_samplers_RNG_preserved=True,
    depth_source='training_only_Source_camera_forward_DDA_32_columns',
    runtime_geometry='learned_private_opacity_on_frozen_map_samples_no_Source_queries',
    original_geometry_auxiliary_batch=8,geometry_auxiliary_weight=1.,
    source_RGB_is_inference_condition=False,new_trainable_tensor_count=4,
    automatic_active_promotion=False,gameplay_qualified=False)


def opacity_geometry_protocol():return deepcopy(PROTOCOL)


def restore_renderer_opacity_parent(saved,model,optimizer,data,far_data,geometry_data,
        identity,configuration,frozen,*,parent_checkpoint_sha256,protocol):
    if protocol!=opacity_geometry_protocol()or not re.fullmatch('[0-9a-f]{64}',parent_checkpoint_sha256):
        raise ValueError('Private-geometry phase needs the exact protocol and full source SHA')
    required=('base','flow','optimizer','step','identity','configuration','frozen_fingerprint',
              'sampler','far_wall_aux_sampler','geometry_wall_aux_sampler','rng_torch','rng_numpy','rng_python','rng_cuda')
    if any(key not in saved for key in required):raise ValueError('Private-geometry phase needs a full v7 boundary')
    old=saved['configuration'];step=saved['step']
    if (type(step)is not int or step<1 or old.get('steps')!=step or configuration.get('steps',0)<=step
            or old.get('geometry_wall_supervision')is not True or old.get('private_geometry_supervision',False)
            or configuration.get('private_geometry_supervision')is not True
            or not configuration.get('private_geometry_parent')):
        raise ValueError('Private-geometry activation requires a completed all-depth v7 parent')
    changes={'run_dir','steps','private_geometry_supervision','private_geometry_parent'}
    if not _same({k:v for k,v in old.items()if k not in changes},
                 {k:v for k,v in configuration.items()if k not in changes}):
        raise ValueError('Private-geometry continuation changed an inherited training setting')
    if (identity.get('private_geometry_protocol')!=protocol
            or not _same(saved['identity'],{k:v for k,v in identity.items()if k!='private_geometry_protocol'})
            or saved['frozen_fingerprint']!=frozen):
        raise ValueError('Private geometry changed Source data, frozen features, codec or old validation panels')
    private=getattr(model.flow_renderer,'geometry_opacity',None)
    if private is None or sum(p.numel()for p in private.parameters())!=6241:
        raise ValueError('Expected exactly the6241-parameter private opacity network')
    shared=model.vjepa_backend.renderer.geometry_reader.opacity_network
    if not _same(private.state_dict(),shared.state_dict()):
        raise ValueError('New private opacity must exactly copy the frozen shared geometry network')
    old_flow=model.flow_renderer.state_dict()
    added={k:v for k,v in old_flow.items()if k.startswith(PRIVATE_PREFIX)}
    if len(added)!=4 or set(old_flow)!=set(saved['flow'])|set(added):
        raise ValueError('Private opacity must add only four new tensors')
    for name,state in (('base',model.base_student.state_dict()),('flow',{k:v for k,v in old_flow.items()if k not in added})):
        if (saved[name].keys()!=state.keys()or any(not torch.is_tensor(v)or v.shape!=state[k].shape
                or v.dtype!=state[k].dtype or not bool(torch.isfinite(v).all())for k,v in saved[name].items())):
            raise ValueError('Private geometry changed inherited Base/RF tensor architecture')
    params=[list(model.base_student.parameters()),[p for n,p in model.flow_renderer.named_parameters()
            if p.requires_grad and not n.startswith(('codec.',PRIVATE_PREFIX))],list(private.parameters())]
    current=optimizer.state_dict();groups=saved['optimizer']['param_groups']
    if (len(groups)!=2 or len(current['param_groups'])!=3 or len(params[2])!=4
            or any([id(p)for p in group['params']]!=[id(p)for p in wanted]
                   for group,wanted in zip(optimizer.param_groups,params))
            or current['param_groups'][2]['lr']!=PROTOCOL['private_optimizer_lr']):
        raise ValueError('Private optimizer must preserve two old groups plus four fresh opacity tensors')
    for old_group,new_group,wanted in zip(groups,current['param_groups'],params):
        if len(old_group['params'])!=len(wanted)or not _same(
                {k:v for k,v in old_group.items()if k!='params'},
                {k:v for k,v in new_group.items()if k!='params'}):
            raise ValueError('Inherited Base/RF optimizer groups or options changed')
    old_ids=[i for group in groups for i in group['params']]
    if len(old_ids)!=len(set(old_ids))or set(old_ids)!=set(saved['optimizer']['state']):
        raise ValueError('Every inherited Adam moment must be present exactly once')
    for group,wanted in zip(groups,params):
        for index,param in zip(group['params'],wanted):
            state=saved['optimizer']['state'][index]
            if (float(state['step'])!=step or any(not torch.is_tensor(state[key])or state[key].shape!=param.shape
                    or not bool(torch.isfinite(state[key]).all())for key in ('exp_avg','exp_avg_sq'))):
                raise ValueError('Inherited Adam moments or completed optimizer clock changed')
    # New group uses canonical optimizer IDs after all inherited parameters.
    new_group=deepcopy(current['param_groups'][2])
    if set(new_group['params'])&set(old_ids):raise ValueError('Private Adam IDs overlap inherited states')
    migrated={'state':deepcopy(saved['optimizer']['state']),
              'param_groups':deepcopy(groups)+[new_group]}
    flow={**old_flow,**saved['flow']}
    model.base_student.load_state_dict(saved['base'],strict=True)
    model.flow_renderer.load_state_dict(flow,strict=True)
    optimizer.load_state_dict(migrated)
    for source,key in ((data,'sampler'),(far_data,'far_wall_aux_sampler'),(geometry_data,'geometry_wall_aux_sampler')):
        source.restore_sampler(deepcopy(saved[key]))
        if not _same(source.sampler_state(),saved[key]):raise AssertionError('Existing geometry Source sampler changed')
    if (not _same(saved['base'],model.base_student.state_dict())or
            not _same(saved['flow'],{k:v for k,v in model.flow_renderer.state_dict().items()if k not in added})or
            not _same(added,{k:v for k,v in model.flow_renderer.state_dict().items()if k in added})or
            not _same(saved['optimizer']['state'],optimizer.state_dict()['state'])):
        raise AssertionError('Private initialization changed old weights, moments or initial opacity copy')
    rng=restore_renderer_native_rng(saved,device=next(model.parameters()).device)
    return dict(kind=PROTOCOL['name'],version=1,start_step=step,parent_optimizer_step=step,
        parent_checkpoint_sha256=parent_checkpoint_sha256,identity=deepcopy(identity),
        old_Base_RF_weights_bitexact=True,old_Adam_options_moments_bitexact=True,
        parent_sampler_and_private_rng_bitexact=True,private_parameters=6241,fresh_parameter_tensors=4,
        inherited_Adam_states=len(old_ids),new_Adam_states_at_start=0,
        parent_mix_counts=deepcopy(saved['sampler']['parent']['mix_counts']),
        sampler_draws_at_adoption=saved['sampler']['phase_draws'],
        original_shared_geometry_frozen=True,private_initialization='exact_copy_of_original_shared_opacity',
        **rng,automatic_active_promotion=False,gameplay_qualified=False)
