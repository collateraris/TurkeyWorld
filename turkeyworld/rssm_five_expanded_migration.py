"""One-million dynamics expert continuation, preserving full named Adam/history."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch

from .models import NeuralWorldModel,WorldModelConfig,STATE_SCHEMA,parameter_counts
from .renderer_native_phase import _same
from .rssm_five_migration import file_sha,report_sha
from .rssm_five_expanded import PREFIX,PROTOCOL,SETTINGS,expand_five_experts,expanded_five_metadata
from .rssm_migration import configure_rssm_training
from .training_pipeline import atomic_json,replace_file


def enable_expanded_five_checkpoint(saved,*,source_checkpoint_sha256=None):
    from .renderer_student import renderer_condition_identity,_hash_tensors
    from .rssm_moe_protocol import moe_training_horizon
    if (not isinstance(saved,dict)or saved.get('stage')!='world'or saved.get('state_schema')!=STATE_SCHEMA
            or not isinstance(saved.get('rssm_five_migration'),dict)):
        raise ValueError('Expert expansion requires a complete migrated five-role world checkpoint')
    old_config=WorldModelConfig(**saved['config']);step=saved.get('training_steps');run=saved.get('run_state')
    if (not old_config.rssm_five_role_enabled or old_config.rssm_five_expanded_enabled
            or old_config.rssm_fusion_moe_width!=256 or old_config.rssm_fusion_moe_heads!=4
            or old_config.rssm_fusion_moe_expert_hidden!=512
            or type(step)is not int or step<1 or saved.get('global_training_step')!=step
            or not isinstance(run,dict)or run.get('global_training_step')!=step):
        raise ValueError('Expert expansion needs the intact unexpanded d256 five-role optimizer boundary')
    moe_training_horizon(run)
    state=saved['model']
    if (not isinstance(state,dict)or not state or any(not torch.is_tensor(v)or v.device.type!='cpu'
            or v.is_floating_point()and not bool(torch.isfinite(v).all())for v in state.values())):
        raise ValueError('Expert expansion requires finite complete CPU checkpoint tensors')
    config={**asdict(old_config),'rssm_five_expanded_enabled':True}
    with torch.random.fork_rng(devices=[]),torch.device('meta'):
        old=NeuralWorldModel(old_config);target=NeuralWorldModel(WorldModelConfig(**config))
    if (set(old.state_dict())!=set(state)or any(v.shape!=old.state_dict()[n].shape
            or v.dtype!=old.state_dict()[n].dtype for n,v in state.items())):
        raise ValueError('Source dynamics tensor namespace differs from its complete configuration')
    old.load_state_dict(dict(state),strict=True,assign=True)
    fusion=deepcopy(old.vjepa_backend.five_role_fusion)
    meta=expand_five_experts(fusion);target.vjepa_backend.five_role_fusion=fusion
    rename={PREFIX+a:PREFIX+b for a,b in meta['legacy_parameter_name_map'].items()}
    added=[PREFIX+n for n in meta['added_parameter_names']]
    remapped={rename.get(n,n):v for n,v in state.items()}
    missing,unexpected=target.load_state_dict(remapped,strict=False,assign=True)
    if unexpected or set(missing)!=set(added)or len(remapped)!=len(state):
        raise ValueError('Expanded dynamics namespace is not an exact old-name bijection plus70 tensors')
    new_state=target.state_dict()
    if any(not _same(value,new_state[rename.get(name,name)])for name,value in state.items()):
        raise AssertionError('Expert expansion changed a retained old tensor')
    configure_rssm_training(old);configure_rssm_training(target)
    canonical_old=[n for n,p in old.named_parameters()if p.requires_grad]
    canonical_new=[n for n,p in target.named_parameters()if p.requires_grad]
    names=saved.get('optimizer_parameter_names');optimizer=saved.get('optimizer',{})
    if names!=canonical_old or [n for n in canonical_new if n not in set(added)]!=[rename.get(n,n)for n in names]:
        raise ValueError('Expert expansion lost the canonical old optimizer name order')
    groups=optimizer.get('param_groups',[]);ids=[i for g in groups for i in g.get('params',[])]
    if len(groups)!=5 or ids!=list(range(len(names)))or set(optimizer.get('state',{}))!=set(ids):
        raise ValueError('Expert expansion requires all five named old Adam groups and every existing state')
    for i,name in enumerate(names):
        moment=optimizer['state'][i]
        if (not {'step','exp_avg','exp_avg_sq'}<=moment.keys()or float(moment['step'])<1
                or any(torch.is_tensor(v)and(v.device.type!='cpu'or not bool(torch.isfinite(v).all())
                    or k!='step'and v.shape!=state[name].shape)for k,v in moment.items())):
            raise ValueError('Invalid old Adam moment/age at '+name)
    fresh=deepcopy({k:v for k,v in groups[0].items()if k not in('params','param_names')})
    fresh['params']=list(range(len(names),len(names)+len(added)))
    mapped_names=[rename.get(n,n)for n in names]+added
    mapped_groups=deepcopy(groups)
    for group in mapped_groups:
        if 'param_names'in group:group['param_names']=[mapped_names[i]for i in group['params']]
    # The old trained heads are nonzero; compare ALL old fusion outputs instead
    # of incorrectly asserting that the complete task residual is zero.
    old_fusion=deepcopy(old.vjepa_backend.five_role_fusion).eval();fusion.eval()
    with torch.no_grad():
        context=torch.arange(2*3*256).reshape(2,3,256).float()/1000
        inputs=(context,torch.zeros(2,20),torch.zeros(2,4));rng=torch.get_rng_state().clone()
        a,b=old_fusion(*inputs),fusion(*inputs)
        if not _same(a,b)or not torch.equal(rng,torch.get_rng_state()):
            raise AssertionError('Expanded expert initialization changed the trained fusion function or RNG')
    identity=renderer_condition_identity(old)
    if renderer_condition_identity(target)!=identity or target.vjepa_backend.memory_size!=old.vjepa_backend.memory_size:
        raise AssertionError('Expert expansion changed recurrent memory or frozen rendering assets')
    counts=parameter_counts(target)
    report=dict(format_version=1,kind='expand_five_private_dynamics_experts',source_step=step,
        source_checkpoint_sha256=source_checkpoint_sha256,source_config=asdict(old_config),target_config=config,
        allowed_config_changes=['rssm_five_expanded_enabled'],old_model_tensors_bitexact=True,
        legacy_parameter_name_map=rename,old_model_tensor_name_map={n:rename.get(n,n)for n in state},
        source_model_sha256=_hash_tensors(state),added_parameter_names=added,
        added_parameter_count=meta['added_parameters'],added_tensor_count=len(added),
        old_tensor_count=len(state),memory_size=old.vjepa_backend.memory_size,
        frozen_render_assets_bitexact=True,zero_residual_initialization=True,
        source_renderer_conditioning_identity=identity,architecture=meta,
        source_optimizer_parameter_names=list(names),optimizer_parameter_names=mapped_names,
        old_Adam_states=len(optimizer['state']),old_optimizer_groups=5,new_optimizer_groups=1,
        new_optimizer_state_entries=0,all_old_Adam_states_options_bitexact=True,
        all_old_RNG_sampler_gate_history_bitexact=True,no_objective_or_horizon_restart=True,
        model_parameter_counts=counts,automatic_active_promotion=False,gameplay_qualified=False)
    report['revision_sha256']=report_sha(report)
    newrun=deepcopy(run)
    newrun.update(rssm_five_expanded_migration=report,rssm_five_expanded_training=dict(
        protocol=PROTOCOL,enabled=True,migration_sha256=report['revision_sha256'],source_step=step,
        current_step=step,parameters_per_expert=1005088,added_parameters=2389920,architecture=meta),
        model_parameters={**run.get('model_parameters',{}),**counts})
    newrun.setdefault('training_protocol',{})['rssm_five_expansion']=dict(
        protocol=PROTOCOL,migration_sha256=report['revision_sha256'],source_step=step)
    payload={**saved,'config':config,'model':{n:v.detach().clone()if n in added else v for n,v in new_state.items()},
        'optimizer_parameter_names':mapped_names,'optimizer':{**optimizer,'state':deepcopy(optimizer['state']),
            'param_groups':mapped_groups+[fresh]},'run_state':newrun,'rssm_five_expanded_migration':report}
    return payload,report


def prepare_expanded_five_continuation(source,destination):
    source,destination=Path(source).resolve(strict=True),Path(destination).resolve()
    if source==destination or destination.exists():raise FileExistsError('Expert expansion needs a separate immutable destination')
    source_sha=file_sha(source)
    saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    payload,report=enable_expanded_five_checkpoint(saved,source_checkpoint_sha256=source_sha)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    try:
        with temporary.open('wb')as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        if file_sha(source)!=source_sha:raise ValueError('Source changed during expert expansion')
        replace_file(temporary,destination)
    finally:temporary.unlink(missing_ok=True)
    receipt={**report,'source_checkpoint':str(source),'checkpoint':str(destination),'checkpoint_sha256':file_sha(destination)}
    atomic_json(destination.with_suffix(destination.suffix+'.expanded-five.json'),receipt)
    return receipt


def bind_expanded_five_plan(state,plan,*,path,sha256,report):
    if (plan.get('kind')!='rssm_five_expansion_revision_v1'or plan.get('version')!=1
            or plan.get('migration_sha256')!=report['revision_sha256']
            or plan.get('source_checkpoint_sha256')!=report['source_checkpoint_sha256']
            or plan.get('source_step')!=report['source_step']or plan.get('architecture')!=report['architecture']):
        raise ValueError('Expanded dynamics revision plan differs from the immutable migration')
    record=dict(protocol=PROTOCOL,plan_path=str(path),plan_sha256=sha256,
        migration_sha256=report['revision_sha256'],source_step=report['source_step'])
    existing=state.get('training_protocol',{}).get('rssm_five_expansion',{})
    if existing and any(existing.get(k)!=v for k,v in record.items()if k in existing):
        raise ValueError('Saved expanded dynamics revision plan pin changed')
    state.setdefault('training_protocol',{})['rssm_five_expansion']=record
    state['rssm_five_expanded_training'].update(plan_sha256=sha256,plan_path=str(path))
    return record
