"""Learned own-clock/candidate spawn conditioning, never a placement rule."""
from copy import deepcopy
from dataclasses import replace
import math

import torch
from torch import nn

from .spawn import SPAWN_ANCHORS, SPAWN_TICK_HZ, SPAWN_PHASE_TICKS


PROTOCOL = 'rssm_current_state_spawn_conditioning_v1'
KIND = 'rssm_spawn_candidate_conditioning_v1'
PREFIX = 'vjepa_backend.spawn_conditioning.'
SEED = 173


class RSSMSpawnConditioning(nn.Module):
    """Nine learned additive logits from current actors, time and candidates."""
    def __init__(self, width=512, hidden=64, seed=SEED):
        super().__init__()
        if type(width) is not int or width < 1 or hidden != 64 or type(hidden) is not int or seed != SEED or type(seed) is not int:
            raise ValueError('Spawn conditioning requires positive actor width, hidden64 and fixed seed173')
        self.width, self.hidden = width, hidden
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.query = nn.Sequential(nn.Linear(3*width+20+10, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.candidate_id = nn.Embedding(9, 16)
            self.candidate = nn.Sequential(nn.Linear(25, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            # A shared scalar output bias would cancel from every nine-way CE
            # and argmax, so it is intentionally absent rather than dead capacity.
            self.score = nn.Sequential(nn.Linear(2*hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1, bias=False))
            nn.init.zeros_(self.score[-1].weight)
        axis = (torch.arange(16, dtype=torch.float32)+.5)/16
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('_centers', torch.stack((xx, yy), -1).reshape(256, 2), persistent=False)
        indices = torch.tensor([16*y+x for x,y in SPAWN_ANCHORS], dtype=torch.long)
        self.register_buffer('_anchor_indices', indices, persistent=False)

    def _inputs(self, state, pooled, spatial, walkable, dt):
        count = len(state) if torch.is_tensor(state) and state.ndim else 0
        if (count < 1 or not torch.is_tensor(state) or state.shape != (count, 20)
                or not torch.is_tensor(pooled) or pooled.shape != (count, self.width)
                or not torch.is_tensor(spatial) or spatial.shape != (count, 2, self.width)
                or any(not value.is_floating_point() for value in (state, pooled, spatial))
                or any(value.device != self.query[0].weight.device for value in (state, pooled, spatial))
                or any(not bool(torch.isfinite(value).all()) for value in (state, pooled, spatial))
                or type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0):
            raise ValueError('Spawn conditioning requires finite own state20, pooledW, two spatialW tokens and positive dt')
        if not torch.is_tensor(walkable) or walkable.dtype != torch.bool:
            raise ValueError('Spawn conditioning requires the known bool walkable initialization map')
        if walkable.ndim == 2:walkable = walkable[None]
        if walkable.shape == (1, 16, 16):walkable = walkable.expand(count, -1, -1)
        if walkable.shape != (count, 16, 16) or walkable.device != state.device:
            raise ValueError('Spawn conditioning needs a matching known walkable map[B,16,16]')
        return state.detach(), walkable

    def candidate_geometry(self, state, walkable):
        """Match existing floor/player/NPC safety masks, without choosing a tile."""
        dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
        own = state.detach().to(dtype)
        centers = self._centers.to(dtype).to(state.device)
        floor = walkable.flatten(1)
        player_delta = (centers[None]-own[:, None, :2])*16
        distance = player_delta.square().sum(-1)
        old = own[:, 8:18].reshape(len(own), 1, 10)
        alive = (old[..., 5] >= .5) & (old[..., 4] > 0)
        npc_distance = ((centers[None, :, None]-old[:, None, :, :2])*16).square().sum(-1)
        legal = floor & (distance >= 16.) & ~((npc_distance < 1.) & alive[:, None]).any(-1)
        anchor_xy = centers[self._anchor_indices][None].expand(len(own), -1, -1)
        is_anchor = torch.zeros(256, dtype=torch.bool, device=state.device)
        is_anchor[self._anchor_indices] = True
        fallback = legal & ~is_anchor[None]
        mass = fallback.sum(-1)
        fallback_xy = (centers[None]*fallback[..., None]).sum(1)/mass.clamp_min(1)[:, None]
        xy = torch.cat((anchor_xy, fallback_xy[:, None]), 1)
        delta = xy-own[:, None, :2]
        anchor_floor, anchor_legal = floor[:, self._anchor_indices], legal[:, self._anchor_indices]
        candidate_legal = torch.cat((anchor_legal, (mass > 0)[:, None]), 1)
        candidate_floor = torch.cat((anchor_floor, (mass > 0)[:, None]), 1)
        fraction = torch.cat((anchor_legal.to(dtype), (mass.to(dtype)/248)[:, None]), 1)
        fallback_flag = torch.zeros(len(own), 9, 1, dtype=dtype, device=state.device)
        fallback_flag[:, 8] = 1
        geometry = torch.cat((xy, delta, delta.norm(dim=-1, keepdim=True),
            candidate_floor[..., None].to(dtype), candidate_legal[..., None].to(dtype),
            fraction[..., None], fallback_flag), -1)
        return geometry, xy, candidate_legal

    def forward(self, state, pooled, spatial, walkable, *, dt=1/30):
        own, walkable = self._inputs(state, pooled, spatial, walkable, dt)
        ticks = torch.round((own[:, 19].double()*300+dt)*SPAWN_TICK_HZ).long()
        phase_index = torch.div(ticks, SPAWN_PHASE_TICKS, rounding_mode='floor').remainder(8)
        within = ticks.remainder(SPAWN_PHASE_TICKS).double()/SPAWN_PHASE_TICKS
        onehot = torch.nn.functional.one_hot(phase_index, 8)
        dtype = self.query[0].weight.dtype
        clock = torch.cat(((own[:, 19].double()+dt/300)[:, None], onehot.double(), within[:, None]), -1)
        query = self.query(torch.cat((pooled.to(dtype), spatial.flatten(1).to(dtype), own.to(dtype), clock.to(dtype)), -1))
        geometry, xy, legal = self.candidate_geometry(own, walkable)
        identities = self.candidate_id(torch.arange(9, device=own.device))[None].expand(len(own), -1, -1)
        candidates = self.candidate(torch.cat((identities, geometry.to(dtype)), -1))
        scores = self.score(torch.cat((query[:, None].expand(-1, 9, -1), candidates), -1)).squeeze(-1)
        residual = scores[:, None].float()
        if not bool(torch.isfinite(residual).all()):raise FloatingPointError('Nonfinite learned spawn conditioning')
        return dict(anchor_residual=residual, phase_index=phase_index, phase_one_hot=onehot,
            within_phase=within.float(), candidate_xy=xy, candidate_legal=legal,
            candidate_features=geometry, next_tick=ticks)


def spawn_conditioning_metadata(module):
    if not isinstance(module, RSSMSpawnConditioning):raise ValueError('Expected the actual learned spawn conditioning module')
    return dict(kind=KIND, version=1, protocol=PROTOCOL, width=module.width, hidden=64,
        parameters=sum(value.numel() for value in module.parameters()),
        parameter_tensors=len(list(module.parameters())),
        parameter_names=[name for name,_ in module.named_parameters()],
        classes=9, anchors=8, fallback_class=8, candidate_id_width=16,
        candidate_geometry_features=['absolute_x','absolute_y','player_relative_x','player_relative_y',
            'player_distance_normalized','floor','legal','legal_fraction','is_fallback'],
        clock='round_own_next_seconds_times30_then_onehot_floor_tick_over90_mod8_and_fraction',
        fallback='mean_known_legal_nonanchor_centers_and_legal_fraction',
        output='additive_nine_class_anchor_residual_before_existing_hierarchy',
        seed=SEED, zero_output_initialization=True, dropout=0., recurrent_memory_added=0,
        Source_future_inputs=False, Source_spawn_rank_used=False, external_engine_queries=False)


def enable_spawn_conditioning(model, *, seed=SEED):
    """Append the new module without altering any old parameter or RNG state."""
    config, backend = model.config, model.vjepa_backend
    if getattr(config, 'rssm_spawn_conditioning_enabled', False):
        if not hasattr(backend, 'spawn_conditioning'):raise ValueError('Spawn flag requires its actual module')
        return spawn_conditioning_metadata(backend.spawn_conditioning)
    if hasattr(backend, 'spawn_conditioning'):raise ValueError('Unconfigured spawn conditioning module')
    updated = replace(config, rssm_spawn_conditioning_enabled=True)
    module = RSSMSpawnConditioning(width=config.vjepa_width or 512, seed=seed)
    parameter = next(backend.parameters())
    module.to(device=parameter.device, dtype=parameter.dtype).train(backend.training)
    module.requires_grad_(any(value.requires_grad for value in backend.parameters()))
    backend.spawn_conditioning = module
    model.config = backend.config = updated
    return spawn_conditioning_metadata(module)
