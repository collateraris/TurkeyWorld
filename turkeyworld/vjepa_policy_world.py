"""Frozen V-JEPA action-conditioned prior and causal real-observation adapter."""
from contextlib import contextmanager
import torch
from .dreamer_policy_world import RSSMImaginationAdapter,WorldState

class VJEPAImaginationAdapter(RSSMImaginationAdapter):
    def __init__(self,world,heads=None,*,sample_latents=False):
        if world.config.architecture!='vjepa2_ac':raise ValueError('Expected V-JEPA 2-AC dynamics')
        self.world=world;self.heads=heads;self.sample_latents=False
        core=world.vjepa_backend.core
        self.memory_size=core.memory_tokens*core.width
        self.feature_dim=148
        world.eval().requires_grad_(False)
        self.feedback_settings=getattr(world,'vjepa_sr_feedback',None)
        self.feedback_renderer=None
        if self.feedback_settings:
            from .vjepa_sr_feedback import FrozenBaseSR
            self.feedback_renderer=FrozenBaseSR(self.feedback_settings,next(world.parameters()).device)
        if heads is not None and heads.config.feature_dim!=148:raise ValueError('V-JEPA policy feature dimensions changed')
    @contextmanager
    def _prior_mode(self):
        modes={m:m.training for m in self.world.modules()}
        try:self.world.eval();yield
        finally:
            for m,mode in modes.items():m.training=mode
    def _validate_state(self,current):
        if (not isinstance(current,WorldState) or current.state.shape!=(len(current.state),20)
                or current.memory.shape!=(len(current.state),self.memory_size)
                or current.grid.shape!=(len(current.state),16,16)
                or not bool(torch.isfinite(current.state).all() and torch.isfinite(current.memory).all())):
            raise ValueError('Invalid complete V-JEPA action/state/visual-token memory')
    @torch.no_grad()
    def initialize(self,state,grid,rgb):
        if state.ndim==2:state=state[:,None]
        if rgb.ndim==4:rgb=rgb[:,None]
        device=next(self.world.parameters()).device
        state=state.to(device).float();rgb=rgb.to(device).float();grid=grid.to(device)
        if rgb.shape[:2]!=state.shape[:2] or rgb.shape[2:]!=(3,32,32):raise ValueError('Unaligned V-JEPA observed context')
        if self.feedback_renderer is not None:
            b,t=state.shape[:2]
            rgb=self.feedback_renderer.observation(state.reshape(b*t,20),grid.repeat_interleave(t,0),navigation_only=self.world.config.navigation_only).reshape(b,t,3,32,32)
        fmap=self.world.encode_map(grid);memory=self.world.initialize_from_image(rgb,state,fmap)
        current=WorldState(state[:,-1].detach(),memory.detach().flatten(1),fmap.detach(),grid.detach())
        self._validate_state(current);return current
    @torch.no_grad()
    def features(self,current):
        self._validate_state(current)
        return torch.cat((current.state,self.world.vjepa_backend.latent_features(current.memory)), -1).detach().float()
    @torch.no_grad()
    def transition_prior(self,current,action):
        following, _ = self.transition_with_info(current, action)
        return following
    @torch.no_grad()
    def transition_with_info(self,current,action):
        """Advance the same autonomous prior and expose its accepted events to evaluation."""
        action=self._action(current,action)
        with self._prior_mode():
            self._bind_map_context(current)
            result=self.world.transition(current.state,action,current.map_features,current.memory)
        memory=result['memory']
        if self.feedback_renderer is not None:
            from .vjepa_sr_feedback import feedback_active,blend_memory
            active=feedback_active(result['state'],self.feedback_settings)
            if bool(active.any()):
                rgb=self.feedback_renderer.observation(result['state'],current.grid,navigation_only=self.world.config.navigation_only)
                memory=blend_memory(self.world,memory,rgb,self.feedback_settings,active)
        following=WorldState(result['state'].detach().float(),memory.detach().flatten(1).float(),current.map_features,current.grid)
        self._validate_state(following);return following,result
    @torch.no_grad()
    def observe(self,current,action,next_state,next_rgb):
        action=self._action(current,action);backend=self.world.vjepa_backend;core=backend.core
        if next_state.shape!=current.state.shape or next_rgb.shape!=(len(next_state),3,32,32):raise ValueError('Unaligned Source observation')
        self._bind_map_context(current)
        if self.feedback_renderer is not None:next_rgb=self.feedback_renderer.observation(next_state,current.grid,navigation_only=self.world.config.navigation_only)
        latents=core.encode_video(next_rgb.to(current.state)[:,None])
        # Retain all prior real observations; only the newest frame enters here.
        block=core.pack_context(latents,torch.zeros_like(action)[:,None],next_state[:,None])
        context=backend._map_context(current.map_features,next_state)
        block=torch.cat((block[:,:1],block[:,1:2]+context[:,None],block[:,2:]),1)
        previous=current.memory.reshape(len(next_state),core.context_frames,core.block_tokens,core.width)
        action_token=core.predictor.action_encoder(action.to(core.predictor.action_encoder.weight.dtype))[:,None]
        # The observed state/map token and visual tokens are already aligned;
        # preserve their exact bytes while recording this transition's action.
        last=torch.cat((action_token.to(previous),previous[:,-1,1:]),1)
        history=torch.cat((previous[:,:-1].flatten(1,2),last),1)
        memory=torch.cat((history[:,core.block_tokens:],block),1)
        following=WorldState(next_state.detach().float(),memory.detach().flatten(1).float(),current.map_features,current.grid)
        self._validate_state(following);return following


def imagination_adapter(world,heads=None,*,sample_latents=True):
    cls=VJEPAImaginationAdapter if world.config.architecture=='vjepa2_ac' else RSSMImaginationAdapter
    return cls(world,heads,sample_latents=sample_latents)
