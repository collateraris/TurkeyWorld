"""Reuse exact D Adam while retaining the selected Base/RF optimizer ledger."""
from copy import deepcopy
import math

import torch

from .decoder_rf_adaptation_objectives import configure_decoder_adaptation, TRAINABLE_PREFIXES
from .renderer_native_phase import _same
from .renderer_rf_chain_phase import rf_chain_canonical_names
from .train_decoder_rf_adaptation import decoder_state


PROTOCOL='decoder_rf_source_paired_adaptation_v1'
PHASE_KIND='decoder_after_selected_Base_RF3_v1'


def configure_decoder_rf_chain(model):
    if (getattr(model.flow_renderer,'conditional_restorer',None)is not None
            or not getattr(model.base_student.config,'output_moe_expanded_enabled',False)
            or model.flow_renderer.steps!=3):
        raise ValueError('Decoder chain requires expanded Base, RF3 and no restorer')
    return configure_decoder_adaptation(model)


def _moment(state, parameter, label, *, expected_age=None):
    age=state.get('step')
    if (not torch.is_tensor(age)or age.numel()!=1 or not bool(torch.isfinite(age))
            or float(age)<1 or float(age)!=int(float(age))
            or expected_age is not None and float(age)!=expected_age
            or any(not torch.is_tensor(state.get(key))or state[key].shape!=parameter.shape
                or state[key].dtype!=parameter.dtype or not bool(torch.isfinite(state[key]).all())
                for key in ('exp_avg','exp_avg_sq'))or bool((state['exp_avg_sq']<0).any())):
        raise ValueError(label+' has missing, stale or invalid Adam moments')


def adopt_decoder_rf_chain_optimizer(model, rf_parent, decoder_parent, *, lr=1e-4):
    """Keep seven RF/Base groups, add the exact previous D group with remapped IDs."""
    configure_decoder_rf_chain(model)
    parameters=dict(model.named_parameters())
    old_names=rf_parent.get('optimizer_parameter_names')
    opt=rf_parent.get('optimizer',{})
    groups,states=opt.get('param_groups',[]),opt.get('state',{})
    if (not isinstance(old_names,list)or len(old_names)!=7 or len(groups)!=7
            or old_names!=rf_chain_canonical_names(model)
            or any(not isinstance(group,list) for group in old_names)):
        raise ValueError('Decoder chain needs all seven named selected-RF optimizer groups')
    flat=[name for group in old_names for name in group]
    ids=[index for group in groups for index in group.get('params',[])]
    if (len(flat)!=617 or len(set(flat))!=len(flat)or ids!=list(range(len(flat)))
            or any(len(group['params'])!=len(names) for group,names in zip(groups,old_names))
            or any(name not in parameters or parameters[name].requires_grad for name in flat)):
        raise ValueError('Selected RF/Base names, frozen policy or ordered Adam IDs changed')
    missing=set(ids)-set(states)
    if missing and not (missing==set(groups[6]['params']) and len(missing)==80
            and all(flat[index].startswith('base_student.output_moe.experts.')
                    and '.legacy.' not in flat[index] for index in missing)):
        raise ValueError('Only the untouched fresh80 frozen Base states may be absent at selected Base0')
    if set(states)-set(ids):raise ValueError('RF parent has unnamed optimizer states')
    for index,state in states.items():_moment(state,parameters[flat[index]],flat[index])
    for prefix,key in (('base_student.','base'),('flow_renderer.','flow')):
        current=model.base_student.state_dict() if key=='base' else model.flow_renderer.state_dict()
        if not _same(current,rf_parent.get(key)):
            raise ValueError('Selected full RF boundary differs from its loaded '+key)
    names=[name for name,parameter in model.named_parameters()if parameter.requires_grad]
    dopt=decoder_parent.get('optimizer',{})
    dgroups,dstates=dopt.get('param_groups',[]),dopt.get('state',{})
    age=decoder_parent.get('step')
    if (type(age)is not int or age<1 or decoder_parent.get('optimizer_parameter_names')!=names
            or len(names)!=62 or len(dgroups)!=1 or dgroups[0].get('params')!=list(range(62))
            or set(dstates)!=set(range(62))or not _same(decoder_parent.get('decoder'),decoder_state(model))
            or type(lr)not in(int,float)or not math.isfinite(lr)or lr<=0 or dgroups[0].get('lr')!=lr):
        raise ValueError('Decoder Adam parent must exactly match current D weights,62 names,age and LR')
    for index,name in enumerate(names):_moment(dstates[index],parameters[name],name,expected_age=age)
    if any(not name.startswith(TRAINABLE_PREFIXES) for name in names)or set(flat)&set(names):
        raise ValueError('Only decoder/post_quant_conv may receive the appended optimizer group')
    configured=[dict(**deepcopy({key:value for key,value in group.items()if key!='params'}),
        params=[parameters[name] for name in names_])for group,names_ in zip(groups,old_names)]
    configured.append(dict(**deepcopy({key:value for key,value in dgroups[0].items()if key!='params'}),
                           params=[parameters[name] for name in names]))
    optimizer=torch.optim.AdamW(configured)
    dgroup=deepcopy(dgroups[0]);dgroup['params']=list(range(len(flat),len(flat)+62))
    merged={**deepcopy(opt),'state':{**deepcopy(states),**{len(flat)+index:deepcopy(state)
            for index,state in dstates.items()}},'param_groups':deepcopy(groups)+[dgroup]}
    optimizer.load_state_dict(merged)
    if not _same(optimizer.state_dict(),merged):raise AssertionError('Decoder adoption changed an Adam state or option')
    return optimizer,dict(kind=PHASE_KIND,version=1,old_RF_parameter_names=deepcopy(old_names),
        optimizer_parameter_names=deepcopy(old_names)+[names],old_RF_Adam_states=len(states),
        old_RF_optimizer_groups=7,decoder_Adam_states=62,decoder_parent_updates=age,
        decoder_Adam_and_options_bitexact=True,old_RF_Adam_and_options_bitexact=True,
        missing_frozen_Base_state_ids=sorted(missing),fresh_Adam_states=0,total_Adam_states=len(merged['state']),
        optimizer_groups=8,trainable_parameter_names=names,RF_Base_world_frozen=True)


def validate_decoder_rf_chain_optimizer(model,saved,rf_parent,decoder_parent,*,local_updates):
    """Reject reset/tampered frozen moments and nonfinite or wrong-age D resume states."""
    if type(local_updates)is not int or local_updates<0:raise ValueError('Invalid decoder local optimizer clock')
    rf=rf_parent['optimizer'];d=decoder_parent['optimizer'];groups=saved.get('param_groups',[]);states=saved.get('state',{})
    expected_d=deepcopy(d['param_groups'][0]);expected_d['params']=list(range(617,679))
    if (len(groups)!=8 or not _same(groups[:7],rf['param_groups'])or not _same(groups[7],expected_d)
            or set(states)!=set(rf['state'])|set(range(617,679))
            or any(not _same(states[key],value)for key,value in rf['state'].items())):
        raise ValueError('Decoder resume changed frozen RF/Base Adam states, group options or D group mapping')
    parameters=dict(model.named_parameters());names=decoder_parent['optimizer_parameter_names']
    for i,name in enumerate(names):
        _moment(states[617+i],parameters[name],name,expected_age=decoder_parent['step']+local_updates)
