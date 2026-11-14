"""Separate learned reward/continuation heads with recorded Source targets.

No engine, renderer, map oracle or future observation is used by prediction.
Reward shaping is explicit; a corpus/time-limit cutoff is never player death.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from .dreamer_actor_critic_math import make_value_bins,twohot_mean,twohot_loss,twohot_target_clip_stats

PROTOCOL='dreamer_source_reward_continue_v1'
REWARD_SETTINGS=dict(version=1,kill=1.,hit=.1,hp_damage=1.,new_death=1.,
    continuation='next_player_HP_positive',terminal_mode='endless_player_death_only',
    corpus_cutoff_is_terminal=False,post_death_padding_is_valid=False,
    source_events=['shot','hit','kill','player_hit'],reward_scale=1.)


def reward_settings(config=None):
    value=deepcopy(REWARD_SETTINGS)
    if config is not None:
        if not isinstance(config,dict)or set(config)!=set(value):raise ValueError('Reward settings require the complete explicit schema')
        value=deepcopy(config)
    for key in('kill','hit','hp_damage','new_death','reward_scale'):
        if(type(value[key])not in(int,float)or not math.isfinite(value[key])or value[key]<0):
            raise ValueError('Reward scales must be finite nonnegative declared values')
    if(any(value[k]!=REWARD_SETTINGS[k]for k in set(value)-{'kill','hit','hp_damage','new_death','reward_scale'})):
        raise ValueError('Reward continuation/time-limit semantics changed')
    return value


def source_reward_continue(previous_state,next_state,events,*,config=None,truncated=None):
    """Targets for recorded [...,20] -> [...,20] transitions, never imagined ones."""
    settings=reward_settings(config)
    if(not torch.is_tensor(previous_state)or not torch.is_tensor(next_state)
            or previous_state.shape!=next_state.shape or previous_state.ndim<2 or previous_state.shape[-1]!=20
            or not previous_state.numel()or not torch.is_tensor(events)or events.shape!=previous_state.shape[:-1]+(4,)):
        raise ValueError('Source reward needs aligned nonempty state20 transitions and recorded event4')
    previous=previous_state.detach().float();target=next_state.detach().to(previous);event=events.detach().to(previous)
    if(not bool(torch.isfinite(previous).all()and torch.isfinite(target).all()and torch.isfinite(event).all())
            or bool(((event<0)|(event>1)|(event!=event.round())).any())
            or bool(((previous[...,4]<0)|(previous[...,4]>1)|(target[...,4]<0)|(target[...,4]>1)).any())):
        raise ValueError('Source states/events must be finite with normalized player HP and binary recorded events')
    alive=previous[...,4]>0;terminated=target[...,4]<=0
    kills=target[...,18]-previous[...,18]
    if(bool((kills<-.0001).any())or bool(((kills-event[...,2]).abs()>.0001).logical_and(alive).any())
            or bool(((target[...,4]-previous[...,4])>1e-5).any())):
        raise ValueError('Source reward window crossed a reset or disagrees with recorded kill/HP bookkeeping')
    if truncated is None:cutoff=torch.zeros_like(alive)
    else:
        cutoff=torch.as_tensor(truncated,device=previous.device)
        if cutoff.shape!=alive.shape or cutoff.dtype!=torch.bool:raise ValueError('Truncation labels must be aligned bool values')
        cutoff=cutoff.detach()
    components=dict(kills=kills.clamp_min(0),hits=event[...,1],
        hp_damage=(previous[...,4]-target[...,4]).clamp_min(0),new_death=(alive&terminated).float(),
        shots=event[...,0],player_hits=event[...,3])
    reward=settings['reward_scale']*(settings['kill']*components['kills']+settings['hit']*components['hits']
        -settings['hp_damage']*components['hp_damage']-settings['new_death']*components['new_death'])
    reward=reward*alive
    return dict(reward=reward,continue_target=(~terminated).float(),valid=alive,terminated=terminated,
        truncated=cutoff,components=components,settings=settings,Source_labels_only=True)


@dataclass
class RewardContinueConfig:
    feature_dim:int
    hidden:int=256
    depth:int=2
    bins:int=255
    low:float=-20.
    high:float=20.
    encoding:str='symexp_twohot'

    def __post_init__(self):
        if(any(type(v)is not int or v<1 for v in(self.feature_dim,self.hidden,self.depth,self.bins))
                or self.bins<3 or self.encoding not in('symexp_twohot','symlog_twohot')):
            raise ValueError('Invalid separate reward/continue head configuration')


class DreamerRewardContinue(nn.Module):
    def __init__(self,feature_dim=None,hidden=256,depth=2,config=None):
        super().__init__()
        self.config=(config if isinstance(config,RewardContinueConfig)else RewardContinueConfig(**config)
            if config is not None else RewardContinueConfig(feature_dim,hidden,depth))
        c=self.config;layers=[];incoming=c.feature_dim
        for _ in range(c.depth):
            layers.extend([nn.Linear(incoming,c.hidden),nn.LayerNorm(c.hidden),nn.SiLU()]);incoming=c.hidden
        self.trunk=nn.Sequential(*layers);self.reward_head=nn.Linear(c.hidden,c.bins);self.continue_head=nn.Linear(c.hidden,1)
        nn.init.zeros_(self.reward_head.weight);nn.init.zeros_(self.reward_head.bias)
        nn.init.zeros_(self.continue_head.weight);nn.init.zeros_(self.continue_head.bias)
        self.register_buffer('reward_bins',make_value_bins(c.bins,c.low,c.high,encoding=c.encoding))

    def forward(self,features):
        if(not torch.is_tensor(features)or features.ndim<2 or features.shape[-1]!=self.config.feature_dim
                or not features.is_floating_point()or not bool(torch.isfinite(features).all())):
            raise ValueError('Reward heads require finite full world features')
        shape=features.shape[:-1];p=next(self.parameters());value=features.detach().to(p).reshape(-1,self.config.feature_dim)
        h=self.trunk(value)
        return dict(reward_logits=self.reward_head(h).reshape(*shape,self.config.bins),
            continue_logit=self.continue_head(h).reshape(*shape),reward_bins=self.reward_bins,
            encoding=self.config.encoding)

    def predict(self,features):
        out=self(features)
        return dict(reward=twohot_mean(out['reward_logits'],out['reward_bins'],encoding=self.config.encoding),
            continue_prob=out['continue_logit'].sigmoid())

    def metadata(self):
        return dict(protocol=PROTOCOL,config=asdict(self.config),parameters=sum(p.numel()for p in self.parameters()),
            reward_settings=reward_settings(),world_frozen=True,Source_labels_training_only=True)


def reward_continue_loss(outputs,labels,weights=None):
    logits=outputs['reward_logits'];continuation=outputs['continue_logit']
    reward=labels['reward'].detach().to(logits);target=labels['continue_target'].detach().to(logits)
    valid=labels['valid'].detach().to(device=logits.device,dtype=torch.bool)
    if(logits.shape[:-1]!=reward.shape or continuation.shape!=reward.shape or target.shape!=reward.shape or valid.shape!=reward.shape
            or not bool(valid.any())or not bool(torch.isfinite(logits).all()and torch.isfinite(continuation).all())):
        raise ValueError('Reward/continuation loss needs aligned finite predictions and at least one real pre-terminal transition')
    weight=torch.ones_like(reward)if weights is None else torch.as_tensor(weights,device=logits.device,dtype=logits.dtype).detach()
    if(weight.shape!=reward.shape or not bool(torch.isfinite(weight).all())or bool((weight<0).any())
            or float((weight*valid).sum())<=0):raise ValueError('Importance weights must be aligned finite nonnegative values with positive valid mass')
    weight=weight*valid;denominator=weight.sum()
    reduce=lambda value:(value*weight).sum()/denominator
    reward_terms=twohot_loss(logits,reward,outputs['reward_bins'],encoding=outputs.get('encoding','symexp_twohot'))
    continue_terms=F.binary_cross_entropy_with_logits(continuation,target,reduction='none')
    reward_loss,continue_loss=reduce(reward_terms),reduce(continue_terms);loss=reward_loss+continue_loss
    mean=twohot_mean(logits,outputs['reward_bins'],encoding=outputs.get('encoding','symexp_twohot'));prob=continuation.sigmoid()
    death=valid&(target==0);alive=valid&(target==1);predicted_death=prob<.5
    components=labels.get('components',{})
    metrics=dict(loss=float(loss.detach()),reward_loss=float(reward_loss.detach()),continue_loss=float(continue_loss.detach()),
        reward_MAE=float(reduce((mean-reward).abs()).detach()),continue_Brier=float(reduce((prob-target).square()).detach()),
        unweighted_reward_MAE=float((mean-reward).abs()[valid].mean().detach()),
        unweighted_continue_BCE=float(continue_terms[valid].mean().detach()),
        unweighted_continue_Brier=float((prob-target).square()[valid].mean().detach()),
        valid_transitions=int(valid.sum()),death_transitions=int(death.sum()),alive_transitions=int(alive.sum()),
        death_recall=float(predicted_death[death].float().mean())if bool(death.any())else None,
        false_death_rate=float(predicted_death[alive].float().mean())if bool(alive.any())else None,
        importance_weight_sum=float(denominator),effective_weighted_samples=float(denominator.square()/weight.square().sum().clamp_min(1e-12)),
        implicit_class_balancing=False,world_gradients=False,protocol=PROTOCOL)
    for name,value in components.items():metrics['Source_'+name+'_support']=float(value.detach().to(logits)[valid].sum())
    metrics.update({'reward_target_'+key:value for key,value in twohot_target_clip_stats(
        reward[valid],outputs['reward_bins'],encoding=outputs.get('encoding','symexp_twohot')).items()})
    return loss,metrics
