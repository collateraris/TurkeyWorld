"""Frozen Base/SR observations and soft visual-token feedback; no Source queries."""
import torch
from torch.nn import functional as F
from .models import load_world_checkpoint,load_sr_checkpoint
from .renderer_student import load_renderer_student
from .base_only_renderer import enable_base_only
from .rssm_combat_protocol import environment_profile
from .workspace_paths import resolve_path
from .small_enemy_data import digest
from .sprite_scene import base_scene_state, set_sprite_scene

class FrozenBaseRenderer:
    """A frozen renderer world with its bound compact Base head and private opacity."""
    def __init__(self, world, renderer, device):
        self.model, saved = load_world_checkpoint(resolve_path(world), 'cpu')
        del saved
        load_renderer_student(self.model, resolve_path(renderer), allow_unqualified_codec=True)
        enable_base_only(self.model)
        self.model.to(device).eval().requires_grad_(False)
        self._static_map = None

    @torch.no_grad()
    def cache_game_map(self, grid):
        """Retain frozen renderer features for one interactive game's map."""
        self._static_map = grid.clone(), self.model.encode_map(grid)

    @torch.no_grad()
    def base(self, state, grid, *, navigation_only=False):
        with environment_profile(self.model, navigation_only=navigation_only):
            if self._static_map is None:
                fmap = self.model.encode_map(grid)
            else:
                cached_grid, fmap = self._static_map
                if cached_grid.device != grid.device or not torch.equal(cached_grid, grid):
                    self.cache_game_map(grid)
                    _, fmap = self._static_map
            return self.model.render(state, torch.zeros(len(state), 128, device=state.device), fmap)


class FrozenBaseSR(FrozenBaseRenderer):
    def __init__(self,settings,device='cuda'):
        self.settings=settings;self.device=torch.device(device)
        for key in ('world','renderer','sr'):
            if digest(resolve_path(settings[key]))!=settings[key+'_sha256']:raise ValueError('Frozen feedback asset changed: '+key)
        super().__init__(settings['world'], settings['renderer'], device)
        self.sr,saved=load_sr_checkpoint(resolve_path(settings['sr']),'cpu');del saved
        self.sr.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def image(self,state,grid,*,navigation_only=False):
        background = base_scene_state(state, self.sr, navigation_only=navigation_only)
        low = self.base(background, grid, navigation_only=navigation_only)
        set_sprite_scene(self.sr, state, grid, navigation_only=navigation_only)
        return self.sr(low)
    @torch.no_grad()
    def observation(self,state,grid,*,navigation_only=False):
        return F.interpolate(self.image(state,grid,navigation_only=navigation_only),size=(32,32),mode='area')


def feedback_active(state,settings):
    return torch.round(state[:,-1]*9000).long().remainder(settings['every']).eq(0)


def blend_memory(world,memory,observation,settings,active=None):
    core=world.vjepa_backend.core
    encoded=core.encode_video(observation[:,None])[:,-1]
    blocks=memory.reshape(len(memory),core.context_frames,core.block_tokens,core.width)
    visual=blocks[:,-1,2:]
    amount=visual.new_full((len(memory),1,1),settings['blend'])
    if active is not None:amount=amount*active[:,None,None]
    newest=torch.cat((blocks[:,-1,:2],visual*(1-amount)+encoded*amount),1)
    return torch.cat((blocks[:,:-1],newest[:,None]),1).reshape_as(memory)
