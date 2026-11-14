"""Small learned output experts over existing renderer tokens, never game rules."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F

KIND='renderer_token_output_moe_v1'
ROLES=('scene_walls','enemy','combat')
THREE_SCHEMA='scene_enemy_combat_v1'
FIVE_SCHEMA='near_far_enemy_motion_enemy_combat_player_combat_v2'
FIVE_KIND='renderer_token_output_moe_five_v2'
FIVE_ROLES=('near_geometry_player','far_geometry','enemy_motion','enemy_combat','player_combat')
FIVE_ROLE_EXPERT_INDICES=(0,3,1,4,2)


def validate_role_schema(schema,experts,enabled=True):
    if (type(experts)is not int or (schema,experts)not in((THREE_SCHEMA,3),(FIVE_SCHEMA,5))
            or not enabled and schema!=THREE_SCHEMA):
        raise ValueError('Renderer output MoE needs an explicit matching three/five-role schema and expert count')


def base_parameter_budget(config):
    if getattr(config,'output_moe_expanded_enabled',False):return 2_500_000
    return 1_005_000 if config.output_moe_role_schema==FIVE_SCHEMA else 1_000_000


def runtime_role_priors(state,grid):
    """Weak current-state image-space hints; no wall/visibility/Source queries."""
    from .camera import project_sprite_bounds
    if state.ndim!=2 or state.shape[1]!=20 or not state.is_floating_point():
        raise ValueError('Renderer MoE needs current state[B,20]')
    pose=state.detach().float()
    boxes=project_sprite_bounds(pose,image_size=32)['boxes']
    y,x=torch.meshgrid(torch.arange(grid,device=state.device),torch.arange(grid,device=state.device),indexing='ij')
    patch=torch.stack([x,y,x+1,y+1],-1).reshape(1,grid*grid,1,4).float()/grid
    overlap=(torch.minimum(patch[...,2:],boxes[:,None,:,2:])-
             torch.maximum(patch[...,:2],boxes[:,None,:,:2])).clamp_min(0).prod(-1)
    enemy=overlap.sum(-1).mul(grid*grid).clamp(0,1)
    gun=pose.new_tensor([.165,.33,.835,1.])[None,None,None]
    gun_overlap=(torch.minimum(patch[...,2:],gun[...,2:])-
                 torch.maximum(patch[...,:2],gun[...,:2])).clamp_min(0).prod(-1)
    gun_overlap=gun_overlap[:,:,0].mul(grid*grid)
    flash=pose[:,7].clamp(0,1)[:,None]
    hurt=pose[:,15].clamp(0,1)[:,None]
    combat=(gun_overlap*flash+enemy*hurt).clamp(0,1)
    return torch.stack([(1-enemy).clamp_min(0),enemy,combat],-1)


class RendererTokenMoE(nn.Module):
    """Three private screen-token Transformers behind a shared small bottleneck."""
    def __init__(self,width,output_dim,expert_width,grid,role_bias=2.,*,role_schema=THREE_SCHEMA,experts=3):
        super().__init__()
        if (any(type(v)is not int or v<1 for v in(width,output_dim,expert_width,grid))
                or expert_width not in (8,32)
                or type(role_bias)not in(int,float)or not math.isfinite(role_bias)or role_bias<0):
            raise ValueError('Invalid compact renderer MoE dimensions/prior bias')
        self.width,self.output_dim,self.expert_width,self.grid=width,output_dim,expert_width,grid
        self.expert_heads=2 if expert_width==8 else 4
        self.role_bias=float(role_bias)
        validate_role_schema(role_schema,experts)
        self.role_schema,self.expert_count=THREE_SCHEMA,3
        self.input=nn.Linear(width,expert_width)
        self.experts=nn.ModuleList(nn.TransformerEncoderLayer(expert_width,self.expert_heads,
            dim_feedforward=expert_width*2,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
            for _ in ROLES)
        self.output=nn.Linear(expert_width,output_dim)
        self.router=nn.Linear(expert_width,3)
        nn.init.zeros_(self.router.weight);nn.init.zeros_(self.router.bias)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.last_aux=None
        if role_schema==FIVE_SCHEMA:
            _append_five_experts(self)

    def forward(self,tokens,state,*,role_priors=None):
        if self.role_schema==FIVE_SCHEMA:
            return self._forward_five(tokens,state,role_priors=role_priors)
        if tokens.ndim!=3 or tokens.shape[1:]!=(self.grid*self.grid,self.width)or len(tokens)!=len(state):
            raise ValueError('Renderer MoE token/state batch or screen grid differs')
        priors=runtime_role_priors(state,self.grid)if role_priors is None else role_priors
        if priors.shape!=(*tokens.shape[:2],3)or priors.device!=tokens.device:
            raise ValueError('Renderer MoE requires own-state role priors[B,N,3]')
        projected=self.input(tokens.to(self.input.weight.dtype))
        logits=self.router(projected).float()+self.role_bias*priors.detach().float()
        gates=logits.softmax(-1)
        expert_tokens=torch.stack([expert(projected)for expert in self.experts],-2)
        outputs=self.output(expert_tokens)
        residual=(gates[...,None].to(outputs.dtype)*outputs).sum(-2).to(tokens.dtype)
        self.last_aux={'router_logits':logits,'gates':gates,'runtime_role_priors':priors.detach(),
            'expert_outputs':outputs,'expert_tokens':expert_tokens,'role_names':ROLES,'grid':self.grid}
        return residual

    def _forward_five(self,tokens,state,*,role_priors=None):
        if tokens.ndim!=3 or tokens.shape[1:]!=(self.grid*self.grid,self.width)or len(tokens)!=len(state):
            raise ValueError('Renderer MoE token/state batch or screen grid differs')
        priors=runtime_role_priors(state,self.grid)if role_priors is None else role_priors
        if priors.shape!=(*tokens.shape[:2],3)or priors.device!=tokens.device:
            raise ValueError('Five-role renderer retains own-state parent role priors[B,N,3]')
        projected=self.input(tokens.to(self.input.weight.dtype))
        parent_logits=self.router(projected).float()+self.role_bias*priors.detach().float()
        parent_gates=parent_logits.softmax(-1)
        split_logits=self.split_router(projected).float()
        splits=split_logits.sigmoid()
        # Keep the old three-expert call/output shape and reduction order. Clones
        # contribute through zero differences, rather than summing five rounded
        # terms or changing the pretrained shared output GEMM's shape.
        tokens_by_expert=([expert(projected,query_context=tokens)for expert in self.experts]
            if getattr(self,'base_expanded_enabled',False)else[expert(projected)for expert in self.experts])
        primary_tokens=torch.stack(tokens_by_expert[:3],-2)
        parent_outputs=self.output(primary_tokens)
        differences=torch.stack([tokens_by_expert[3]-tokens_by_expert[0],
                                 tokens_by_expert[4]-tokens_by_expert[1]],-2)
        delta=F.linear(differences,self.output.weight,None)
        outputs=torch.stack([parent_outputs[...,0,:]+splits[...,0,None].to(delta.dtype)*delta[...,0,:],
                             parent_outputs[...,1,:]+splits[...,1,None].to(delta.dtype)*delta[...,1,:],
                             parent_outputs[...,2,:]],-2)
        residual=(parent_gates[...,None].to(outputs.dtype)*outputs).sum(-2).to(tokens.dtype)
        gates=torch.stack([parent_gates[...,0]*(1-splits[...,0]),parent_gates[...,0]*splits[...,0],
            parent_gates[...,1]*(1-splits[...,1]),parent_gates[...,1]*splits[...,1],parent_gates[...,2]],-1)
        parent_log=parent_logits.log_softmax(-1)
        logits=torch.stack([parent_log[...,0]+F.logsigmoid(-split_logits[...,0]),
            parent_log[...,0]+F.logsigmoid(split_logits[...,0]),
            parent_log[...,1]+F.logsigmoid(-split_logits[...,1]),
            parent_log[...,1]+F.logsigmoid(split_logits[...,1]),parent_log[...,2]],-1)
        logical_outputs=torch.stack([parent_outputs[...,0,:],parent_outputs[...,0,:]+delta[...,0,:],
            parent_outputs[...,1,:],parent_outputs[...,1,:]+delta[...,1,:],parent_outputs[...,2,:]],-2)
        logical_priors=torch.stack([priors[...,0]*(1-splits[...,0].detach()),
            priors[...,0]*splits[...,0].detach(),priors[...,1]*(1-splits[...,1].detach()),
            priors[...,1]*splits[...,1].detach(),priors[...,2]],-1)
        self.last_aux={'router_logits':logits,'gates':gates,'runtime_role_priors':logical_priors.detach(),
            'expert_outputs':logical_outputs,
            'expert_tokens':torch.stack([tokens_by_expert[i]for i in FIVE_ROLE_EXPERT_INDICES],-2),
            'role_names':FIVE_ROLES,'grid':self.grid,'role_schema':FIVE_SCHEMA,
            'parent_router_logits':parent_logits,'parent_gates':parent_gates,
            'parent_runtime_role_priors':priors.detach(),'split_logits':split_logits,'split_gates':splits}
        return residual

    def pop_diagnostics(self):
        result=self.last_aux;self.last_aux=None;return result


def _append_five_experts(module):
    if module.role_schema==FIVE_SCHEMA:
        if len(module.experts)!=5 or not hasattr(module,'split_router'):
            raise ValueError('Incomplete five-role renderer expert module')
        return
    if module.role_schema!=THREE_SCHEMA or len(module.experts)!=3 or hasattr(module,'split_router'):
        raise ValueError('Five-role migration requires the intact trained three-expert module')
    parameter=module.input.weight
    devices=[parameter.device.index]if parameter.is_cuda else[]
    with torch.random.fork_rng(devices=devices):
        clones=[deepcopy(module.experts[0]),deepcopy(module.experts[1])]
        split=nn.Linear(module.expert_width,2).to(device=parameter.device,dtype=parameter.dtype)
        nn.init.zeros_(split.weight);nn.init.zeros_(split.bias)
    module.experts.extend(clones)
    module.split_router=split.train(module.training)
    module.role_schema,module.expert_count=FIVE_SCHEMA,5
    module.last_aux=None


def output_moe_metadata(head):
    from .renderer_student import _hash_tensors
    module=getattr(head,'output_moe',None)
    if module is None:return None
    if not isinstance(module,RendererTokenMoE):raise ValueError('Unknown renderer output expert module')
    result=dict(kind=KIND,version=1,roles=list(ROLES),width=module.width,expert_width=module.expert_width,
        expert_heads=module.expert_heads,expert_depth=1,expert_ff_width=module.expert_width*2,
        experts_are_private_transformers=True,dropout=0.,norm_first=True,
        output_dim=module.output_dim,grid=module.grid,role_bias=module.role_bias,
        parameters=sum(p.numel()for p in module.parameters()),state_sha256=_hash_tensors(module.state_dict()),
        runtime_inputs='existing_tokens_and_current_numeric_state_only',Source_future_inputs=False)
    if module.role_schema==FIVE_SCHEMA:
        result.update(kind=FIVE_KIND,version=2,roles=list(FIVE_ROLES),role_schema=FIVE_SCHEMA,experts=5,
            role_expert_indices=list(FIVE_ROLE_EXPERT_INDICES),routing='hierarchical_parent3_scene_enemy_sigmoid_splits',
            parent_roles=list(ROLES),parent_group_mass_preserved=True,
            runtime_depth_oracle=False,shared_output_delta_preserves_parent_reduction=True)
    expanded=bool(getattr(getattr(head,'config',None),'output_moe_expanded_enabled',False))
    if expanded or getattr(module,'base_expanded_enabled',False):
        from .base_expanded_experts import KIND as BASE_KIND,base_expanded_expert_metadata
        architecture=base_expanded_expert_metadata(head)
        result.update(kind=BASE_KIND,version=1,base_expert_expansion=architecture,
            private_query_context_width=144,private_expert_width=192,private_expert_heads=6,
            private_expert_ff_width=314,expert_depth=2,parameters_per_expert=300058,
            legacy_kind=FIVE_KIND,legacy_version=2)
    return result


def _matching_moe_roles(base_meta,flow_meta):
    if (base_meta.get('kind'),base_meta.get('version'))==(flow_meta.get('kind'),flow_meta.get('version')):
        return True
    from .base_expanded_experts import KIND as BASE_KIND,validate_base_expanded_expert_metadata
    if ((base_meta.get('kind'),base_meta.get('version'))!=(BASE_KIND,1)
            or(flow_meta.get('kind'),flow_meta.get('version'))!=(FIVE_KIND,2)
            or base_meta.get('role_schema')!=FIVE_SCHEMA or flow_meta.get('role_schema')!=FIVE_SCHEMA
            or base_meta.get('roles')!=list(FIVE_ROLES)or flow_meta.get('roles')!=list(FIVE_ROLES)):
        return False
    validate_base_expanded_expert_metadata(base_meta.get('base_expert_expansion'))
    return True


def renderer_moe_metadata(flow,base):
    base_meta=output_moe_metadata(base)if base is not None else None
    flow_meta=output_moe_metadata(flow)
    if base_meta is None and flow_meta is None:return None
    if base_meta is None or flow_meta is None:
        raise ValueError('Renderer output MoE requires both its compact Base and RF experts')
    if not _matching_moe_roles(base_meta,flow_meta):
        raise ValueError('Renderer Base and RF must share their explicit role schema')
    return dict(kind=flow_meta['kind'],version=flow_meta['version'],base=base_meta,flow=flow_meta,
        shared_transformers_unchanged=True,new_RSSM_memory=False,Source_future_inputs=False)


def validate_renderer_moe_bundle(flow,base_bundle,header):
    actual=output_moe_metadata(flow)
    base=(base_bundle or{}).get('output_moe')
    enabled=bool((base_bundle or{}).get('config',{}).get('output_moe_enabled',False))
    expanded=(base_bundle or{}).get('config',{}).get('output_moe_expanded_enabled',False)
    if actual is None:
        if header is not None or base is not None or enabled:
            raise ValueError('Renderer output MoE metadata requires its actual RF module')
        return
    expected=dict(kind=actual['kind'],version=actual['version'],base=base,flow=actual,
        shared_transformers_unchanged=True,new_RSSM_memory=False,Source_future_inputs=False)
    from .base_expanded_experts import KIND as BASE_KIND
    if (not enabled or not isinstance(base,dict)or header!=expected
            or type(expanded)is not bool or expanded!=(base.get('kind')==BASE_KIND)
            or not _matching_moe_roles(base,actual)):
        raise ValueError('Renderer output MoE and compact Base binding/hash differ')


def _new_module(owner,*,width,output_dim,expert_width,grid,role_schema=THREE_SCHEMA,experts=3):
    parameter=next(owner.parameters())
    devices=[parameter.device.index]if parameter.is_cuda else[]
    with torch.random.fork_rng(devices=devices):
        module=RendererTokenMoE(width,output_dim,expert_width,grid,role_schema=role_schema,experts=experts)
    return module.to(device=parameter.device,dtype=parameter.dtype).train(owner.training)


def enable_renderer_moe(model):
    """Append only new expert tensors, preserving old weights/buffers/modes/RNG."""
    from .base_student import CompactBaseHead
    base=getattr(model,'base_student',None);flow=getattr(model,'flow_renderer',None)
    if (not isinstance(base,CompactBaseHead)or flow is None or flow.kind!='student'or flow.steps!=1
            or base.config.pixel_residual):
        raise ValueError('Output MoE requires the compact Base/no-pixel-residual and RF1')
    if hasattr(base,'output_moe')or hasattr(flow,'output_moe'):
        if not(hasattr(base,'output_moe')and hasattr(flow,'output_moe')):
            raise ValueError('Cannot migrate a partially attached renderer output MoE')
        if (not base.config.output_moe_enabled or not model.config.flow_output_moe_enabled
                or not flow._renderer_student_config.get('flow_output_moe_enabled')):
            raise ValueError('Renderer output MoE modules require their explicit configuration flags')
        return renderer_moe_metadata(flow,base)
    upgraded_config=replace(model.config,flow_output_moe_enabled=True)
    b=_new_module(base,width=base.config.width,output_dim=12,expert_width=8,grid=16)
    f=_new_module(flow,width=flow.width,output_dim=flow.latent_channels*flow.patch*flow.patch,expert_width=32,grid=flow.grid)
    if sum(p.numel()for p in base.parameters())+sum(p.numel()for p in b.parameters())>1_000_000:
        raise ValueError('Renderer output MoE exceeds the one-million Base budget')
    base.output_moe=b;flow.output_moe=f
    base.config=replace(base.config,output_moe_enabled=True);model.config=upgraded_config
    flow._renderer_student_config={**flow._renderer_student_config,'flow_output_moe_enabled':True}
    return renderer_moe_metadata(flow,base)


def configure_renderer_moe(model):
    """Train Base/RF experts/shared readouts; keep all world/codec/opacity frozen."""
    enable_renderer_moe(model)
    model.eval().requires_grad_(False)
    model.base_student.train().requires_grad_(True)
    model.flow_renderer.train().requires_grad_(True)
    model.flow_renderer.codec.eval().requires_grad_(False)
    if hasattr(model.flow_renderer,'geometry_opacity'):
        model.flow_renderer.geometry_opacity.eval().requires_grad_(False)
    return {**renderer_moe_metadata(model.flow_renderer,model.base_student),
        'trainable_parameters':sum(p.numel()for p in model.parameters()if p.requires_grad),
        'private_geometry_frozen':True,'codec_frozen':True,'dynamics_frozen':True}


def migrate_renderer_moe_five(model):
    """Clone trained scene/enemy experts; retain every old parameter name/value."""
    from .base_student import CompactBaseHead
    base=getattr(model,'base_student',None);flow=getattr(model,'flow_renderer',None)
    if (not isinstance(base,CompactBaseHead)or flow is None
            or not all(isinstance(getattr(head,'output_moe',None),RendererTokenMoE)for head in(base,flow))):
        raise ValueError('Five-role migration requires the paired trained Base/RF output MoE')
    schemas=[head.output_moe.role_schema for head in(base,flow)]
    if schemas==[FIVE_SCHEMA,FIVE_SCHEMA]:
        return renderer_moe_metadata(flow,base)
    if schemas!=[THREE_SCHEMA,THREE_SCHEMA]:
        raise ValueError('Cannot migrate mismatched Base/RF role schemas')
    new_base=replace(base.config,output_moe_role_schema=FIVE_SCHEMA,output_moe_experts=5)
    new_world=replace(model.config,flow_output_moe_role_schema=FIVE_SCHEMA,flow_output_moe_experts=5)
    added_base=2*sum(p.numel()for p in base.output_moe.experts[0].parameters())+2*(base.output_moe.expert_width+1)
    if sum(p.numel()for p in base.parameters())+added_base>base_parameter_budget(new_base):
        raise ValueError('Five-role Base exceeds its explicit1.005M budget')
    for head in(base,flow):_append_five_experts(head.output_moe)
    base.config,model.config=new_base,new_world
    flow._renderer_student_config={**flow._renderer_student_config,
        'flow_output_moe_role_schema':FIVE_SCHEMA,'flow_output_moe_experts':5}
    return renderer_moe_metadata(flow,base)
