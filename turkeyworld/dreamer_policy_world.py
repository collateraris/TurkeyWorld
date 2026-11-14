"""Frozen learned three-RSSM environment for a separate actor/critic sidecar.

Observation methods are replay-training/initialization only. Imagination uses
only the model's priors, complete packed h/z, static map features and actions.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch

PROTOCOL='dreamer_three_rssm_imagination_adapter_v1'


@dataclass(frozen=True)
class WorldState:
    state:torch.Tensor
    memory:torch.Tensor
    map_features:torch.Tensor
    grid:torch.Tensor


class RSSMImaginationAdapter:
    def __init__(self,world,heads=None,*,sample_latents=True):
        backend=getattr(world,'vjepa_backend',None)
        if(getattr(getattr(world,'config',None),'architecture',None)!='dreamer_rssm'
                or backend is None or not hasattr(backend,'enemy_rssm')or not hasattr(backend,'combat_rssm')
                or type(sample_latents)is not bool):
            raise ValueError('Policy imagination requires the complete player/enemy/combat RSSM world')
        self.world=world;self.heads=heads;self.sample_latents=sample_latents
        self.memory_size=int(backend.memory_size);self.feature_dim=20+self.memory_size
        if self.memory_size<1:raise ValueError('Missing persistent three-RSSM memory')
        for parameter in self.world.parameters():parameter.requires_grad_(False)
        self.world.eval()
        if heads is not None and heads.config.feature_dim!=self.feature_dim:
            raise ValueError('Reward heads and world feature dimensions differ')

    @contextmanager
    def _prior_mode(self):
        modes={module:module.training for module in self.world.modules()}
        try:
            self.world.eval()
            # Player core controls categorical sampling in all three branches;
            # the surrounding structured readout stays in hard inference mode.
            self.world.vjepa_backend.core.train(self.sample_latents)
            yield
        finally:
            for module,training in modes.items():module.training=training

    def _validate_state(self,current):
        if(not isinstance(current,WorldState)or current.state.ndim!=2 or current.state.shape[-1]!=20
                or not len(current.state)or current.memory.shape!=(len(current.state),self.memory_size)
                or current.map_features.shape[0]!=len(current.state)
                or current.grid.shape!=(len(current.state),16,16)
                or not bool(torch.isfinite(current.state).all()and torch.isfinite(current.memory).all())):
            raise ValueError('Imagination state must preserve complete state20 and all three packed h/z memories')

    def features(self,current):
        self._validate_state(current)
        return torch.cat((current.state.float(),current.memory.float()),-1).detach()

    def _bind_map_context(self,current):
        """Restore this stream's static walkability cache without re-encoding/RGB."""
        renderer=self.world.vjepa_backend.renderer
        if hasattr(renderer,'_spawn_walkable'):
            renderer._spawn_walkable=(current.grid==0).to(device=current.map_features.device)

    @torch.no_grad()
    def initialize(self,state,grid,rgb):
        if(state.ndim not in(2,3)or state.shape[-1]!=20 or not len(state)
                or grid.shape!=(len(state),16,16)or not bool(torch.isfinite(state).all())):
            raise ValueError('Observed initialization requires aligned state20 and grid16')
        if rgb.ndim==4:rgb=rgb[:,None]
        if state.ndim==2:state=state[:,None]
        if(rgb.ndim!=5 or rgb.shape[:2]!=state.shape[:2]or rgb.shape[2]!=3
                or rgb.shape[-2:]!=(self.world.config.image_size,self.world.config.image_size)
                or not bool(torch.isfinite(rgb).all())):
            raise ValueError('Every observed context image needs its matching current Source pose')
        try:p=next(self.world.parameters());device=p.device
        except StopIteration:device=state.device
        state=state.detach().to(device=device,dtype=torch.float32);grid=grid.detach().to(device=device)
        rgb=rgb.detach().to(device=device)
        with self._prior_mode():
            features=self.world.encode_map(grid)
            memory=self.world.initialize_from_image(rgb,state,features)
        current=WorldState(state[:,-1].clone(),memory.detach().float(),features.detach(),grid.detach().clone())
        self._validate_state(current)
        return current

    def _action(self,current,action):
        self._validate_state(current)
        if(not torch.is_tensor(action)or action.shape!=(len(current.state),4)
                or not bool(torch.isfinite(action).all())):
            raise ValueError('Imagination needs finite action4 for every actor')
        action=action.detach().to(current.state)
        if(bool((action[:,:3].abs()>1).any())or bool(((action[:,3]<0)|(action[:,3]>1)).any())):
            raise ValueError('Policy actions exceed forward/strafe/turn[-1,1] or fire[0,1]')
        return action

    @torch.no_grad()
    def transition_prior(self,current,action):
        action=self._action(current,action)
        with self._prior_mode():
            self._bind_map_context(current)
            result=self.world.transition(current.state,action,current.map_features,current.memory)
        following=WorldState(result['state'].detach().float(),result['memory'].detach().float(),current.map_features,current.grid)
        self._validate_state(following)
        return following

    @torch.no_grad()
    def step(self,current,action):
        if self.heads is None:raise ValueError('Imagination reward/continue heads must be supplied before actor training')
        following=self.transition_prior(current,action);features=self.features(following)
        prediction=self.heads.predict(features)
        return dict(world_state=following,next_features=features,reward=prediction['reward'].detach(),
            continue_prob=prediction['continue_prob'].detach(),state=following.state,memory=following.memory)

    @torch.no_grad()
    def observe(self,current,action,next_state,next_rgb):
        """Replay-only posterior update; Source next observations are training data."""
        action=self._action(current,action)
        if(next_state.shape!=current.state.shape or next_rgb.shape!=(len(next_state),3,self.world.config.image_size,self.world.config.image_size)
                or not bool(torch.isfinite(next_state).all()and torch.isfinite(next_rgb).all())):
            raise ValueError('Source posterior update needs aligned true next-state/RGB labels')
        with self._prior_mode():
            self._bind_map_context(current)
            result=self.world.vjepa_backend.observe_step(current.state,action,current.map_features,
                current.memory,next_rgb.detach().to(current.state),sample=self.sample_latents)
        following=WorldState(next_state.detach().to(current.state),result['memory'].detach().float(),current.map_features,current.grid)
        self._validate_state(following)
        return following

    @torch.no_grad()
    def observe_batch(self,batch,*,return_states=False):
        """Return [B,T+1,D], including the initial observed seed then T posteriors."""
        states,frames,targets,actions=(batch[key]for key in('states','context_frames','target_states','actions'))
        next_frames=batch['target_frames'];context=frames.shape[1]
        if(actions.ndim!=3 or actions.shape[-1]!=4 or targets.shape!=actions.shape[:2]+(20,)
                or next_frames.shape[:2]!=actions.shape[:2]or states.shape[1]<context):
            raise ValueError('Source observation batch has inconsistent context/transition alignment')
        current=self.initialize(states[:,:context],batch['map'],frames);observed=[current]
        for tick in range(actions.shape[1]):
            current=self.observe(current,actions[:,tick],targets[:,tick],next_frames[:,tick]);observed.append(current)
        features=torch.stack([self.features(value)for value in observed],1)
        return (features,observed)if return_states else features

    @torch.no_grad()
    def select_observed(self,states_list,indices):
        if not states_list:raise ValueError('No real observation seeds were supplied')
        batch=len(states_list[0].state);choice=torch.as_tensor(indices)
        if(choice.shape!=(batch,)or choice.dtype not in(torch.int32,torch.int64)
                or bool(((choice<0)|(choice>=len(states_list))).any())):
            raise ValueError('Select one valid real observation index per actor')
        for value in states_list:self._validate_state(value)
        indices=choice.detach().cpu().tolist()
        return WorldState(*(torch.stack([getattr(states_list[tick],field)[actor]for actor,tick in enumerate(indices)])
            for field in('state','memory','map_features','grid')))

    @torch.no_grad()
    def imagine(self,initial,actor,*,horizon=15,deterministic=False,generator=None):
        if type(horizon)is not int or horizon<1:raise ValueError('Imagination horizon must be positive')
        current=initial;features=[self.features(current)];actions=[];rewards=[];continues=[]
        for _ in range(horizon):
            action=(actor.act(features[-1],deterministic=deterministic,generator=generator)
                if hasattr(actor,'act')else actor(features[-1]))
            step=self.step(current,action);current=step['world_state']
            actions.append(action.detach());rewards.append(step['reward']);continues.append(step['continue_prob']);features.append(step['next_features'])
        return dict(features=torch.stack(features,1),actions=torch.stack(actions,1),rewards=torch.stack(rewards,1),
            continues=torch.stack(continues,1),initial_alive=(initial.state[:,4]>0).detach(),final_world_state=current)
