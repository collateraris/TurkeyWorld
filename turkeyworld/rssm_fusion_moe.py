"""Causal feedforward fusion of two continuously updated RSSM memories.

Inputs are previous own h/z, current own numerical state/map and known action.
There is no observation, Source label, future-state or recurrent routing input.
Three role queries share a Transformer, then use a top-two sparse FFN mixture.
Zero task readouts make all six additions exactly zero at activation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


ROLES = ('player_navigation','npc_navigation','interaction')
TOKEN_NAMES = ('previous_player_hz','previous_enemy_hz','map','state','action',
               'spatial_player','spatial_enemy',*ROLES)
COMBAT_SLICES = {'target':slice(0,2),'attack':slice(2,3),
                 'spawn_tile':slice(3,259),'spawn_anchor':slice(259,268)}


@dataclass
class RSSMFusionMoEConfig:
    width:int=256
    depth:int=2
    heads:int=4
    expert_hidden:int=512
    experts:int=3
    top_k:int=2
    role_bias:float=2.

    def __post_init__(self):
        import math
        if (any(type(v)is not int or v<1 for v in
                (self.width,self.depth,self.heads,self.expert_hidden,self.experts,self.top_k))
                or self.width%self.heads or self.experts!=3 or self.top_k!=2
                or not isinstance(self.role_bias,(int,float))or isinstance(self.role_bias,bool)
                or not math.isfinite(self.role_bias)or self.role_bias<0):
            raise ValueError('Fusion MoE requires valid widths, three semantic experts and top-two routing')


class RSSMFusionMoE(nn.Module):
    """Return additive head residuals; never replace, reset or route RSSM h/z.

    Both previous memories remain differentiable. Each role query prefers its
    own expert with bias ``role_bias``; the next cyclic expert receives 0.1.
    The latter breaks secondary-route ties without stochastic routing. Role CE
    targets the fixed three output roles, not future game-mode/event labels.
    """
    def __init__(self,config=None,*,player_memory_dim,enemy_memory_dim,map_width,spatial_width=None):
        super().__init__()
        config=RSSMFusionMoEConfig()if config is None else config
        if isinstance(config,dict):config=RSSMFusionMoEConfig(**config)
        if not isinstance(config,RSSMFusionMoEConfig):
            raise TypeError('Expected explicit RSSMFusionMoEConfig')
        spatial_width=map_width if spatial_width is None else spatial_width
        if any(type(v)is not int or v<1 for v in
               (player_memory_dim,enemy_memory_dim,map_width,spatial_width)):
            raise ValueError('Fusion input dimensions must be positive integers')
        self.config=config
        self.player_memory_dim,self.enemy_memory_dim=player_memory_dim,enemy_memory_dim
        self.map_width,self.spatial_width=map_width,spatial_width
        width=config.width
        self.player_projection=nn.Linear(player_memory_dim,width)
        self.enemy_projection=nn.Linear(enemy_memory_dim,width)
        self.map_projection=nn.Linear(map_width,width)
        self.state_projection=nn.Linear(20,width)
        self.action_projection=nn.Linear(4,width)
        self.spatial_projection=nn.Linear(spatial_width,width)
        self.role_queries=nn.Parameter(torch.empty(3,width))
        self.token_types=nn.Parameter(torch.empty(len(TOKEN_NAMES),width))
        nn.init.normal_(self.role_queries,std=.02);nn.init.normal_(self.token_types,std=.02)
        self.input_norm=nn.LayerNorm(width)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(width,config.heads,
            dim_feedforward=config.expert_hidden,dropout=0.,activation='gelu',
            batch_first=True,norm_first=True)for _ in range(config.depth)])
        self.shared_norm=nn.LayerNorm(width)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(width,config.expert_hidden),nn.GELU(),
            nn.Linear(config.expert_hidden,width))for _ in range(config.experts)])
        self.router=nn.Linear(width,config.experts)
        nn.init.normal_(self.router.weight,std=.01);nn.init.zeros_(self.router.bias)
        bias=torch.eye(3)*config.role_bias
        bias[torch.arange(3),(torch.arange(3)+1)%3]=.1
        self.register_buffer('role_router_bias',bias)
        self.role_norm=nn.LayerNorm(width)
        self.player_head=nn.Linear(width,2)
        self.enemy_head=nn.Linear(width,2)
        self.combat_head=nn.Linear(width,268)
        for head in (self.player_head,self.enemy_head,self.combat_head):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def _inputs(self,player,enemy,state,action,map_context,spatial):
        if not torch.is_tensor(player)or player.ndim!=2 or len(player)<1:
            raise ValueError('Fusion requires nonempty previous player memory[B,P]')
        batch=len(player);parameter=self.player_projection.weight
        shapes=((batch,self.player_memory_dim),(batch,self.enemy_memory_dim),
                (batch,20),(batch,4),(batch,self.map_width),(batch,2,self.spatial_width))
        values=(player,enemy,state,action,map_context,spatial)
        if any(not torch.is_tensor(value)or tuple(value.shape)!=shape
               or not value.is_floating_point()or value.device!=parameter.device
               for value,shape in zip(values,shapes)):
            raise ValueError('Fusion previous-memory/state/action/map/two-spatial-token shapes/devices differ')
        if not all(bool(torch.isfinite(value).all())for value in values):
            raise FloatingPointError('Nonfinite own fusion inputs; no reset or correction')
        return tuple(value.to(dtype=parameter.dtype)for value in values)

    def forward(self,previous_player_memory,previous_enemy_memory,state,action,map_context,spatial_tokens,
                *,combat_context=None):
        player,enemy,state,action,maps,spatial=self._inputs(previous_player_memory,
            previous_enemy_memory,state,action,map_context,spatial_tokens)
        batch=len(player)
        controls=torch.cat([action[:,:3].clamp(-1,1),action[:,3:4].clamp(0,1)],-1)
        context=torch.stack([self.player_projection(player),self.enemy_projection(enemy),
            self.map_projection(maps),self.state_projection(state),self.action_projection(controls)],1)
        queries=self.role_queries[None].expand(batch,-1,-1)
        if combat_context is not None:
            if (not torch.is_tensor(combat_context)
                    or tuple(combat_context.shape)!=(batch,self.config.width)
                    or not combat_context.is_floating_point()or combat_context.device!=player.device):
                raise ValueError('Previous Combat fusion context must match [B,fusion_width]')
            if not bool(torch.isfinite(combat_context).all()):
                raise FloatingPointError('Nonfinite previous Combat fusion context')
            # Reuse the existing interaction token. A new attention token,
            # even zero-valued, would change the pretrained softmax function.
            queries=torch.cat([queries[:,:2],
                queries[:,2:3]+combat_context[:,None].to(queries.dtype)],1)
        tokens=torch.cat([context,self.spatial_projection(spatial),queries],1)
        tokens=self.input_norm(tokens+self.token_types[None])
        for block in self.blocks:tokens=block(tokens)
        shared=self.shared_norm(tokens[:,-3:])
        logits=self.router(shared).float()+self.role_router_bias.float()[None]
        probabilities=logits.softmax(-1)
        top_probabilities,top_indices=probabilities.topk(self.config.top_k,dim=-1)
        top_weights=top_probabilities/top_probabilities.sum(-1,keepdim=True)
        flat=shared.reshape(-1,self.config.width)
        indices=top_indices.reshape(-1,self.config.top_k)
        weights=top_weights.reshape(-1,self.config.top_k)
        mixed=torch.zeros_like(flat)
        # Evaluate only selected token/expert pairs. Both recurrent cores still
        # execute outside this module on every simulation tick.
        for index,expert in enumerate(self.experts):
            selected=torch.nonzero(indices==index,as_tuple=False)
            if not selected.numel():continue
            rows,slots=selected.unbind(-1)
            value=expert(flat.index_select(0,rows))
            value=value*weights[rows,slots,None].to(value.dtype)
            mixed=mixed.index_add(0,rows,value)
        roles=self.role_norm(shared+mixed.reshape_as(shared))
        targets=torch.arange(3,device=logits.device)[None].expand(batch,-1)
        role_loss=F.cross_entropy(logits.reshape(-1,3),targets.reshape(-1))
        importance=probabilities.mean((0,1))
        load=F.one_hot(top_indices,3).float().mean((0,1,2))
        balance=3*(importance*load.detach()).sum()
        player=self.player_head(roles[:,0]).float()
        enemy=self.enemy_head(roles[:,1]).float()
        combat=self.combat_head(roles[:,2]).float()
        return {'player_pass_residual':player,'enemy_motion_residual':enemy,
            'interaction_target_residual':combat[:,COMBAT_SLICES['target']],
            'interaction_attack_residual':combat[:,COMBAT_SLICES['attack']],
            'interaction_spawn_tile_residual':combat[:,COMBAT_SLICES['spawn_tile']][:,None],
            'interaction_spawn_anchor_residual':combat[:,COMBAT_SLICES['spawn_anchor']][:,None],
            'role_tokens':roles.float(),'router_logits':logits,'router_probabilities':probabilities,
            'router_top_indices':top_indices,'router_top_weights':top_weights,
            'router_role_loss':role_loss,'router_balance_loss':balance,
            'router_importance':importance,'router_load':load}
