"""DreamerV3-style RSSM dynamics with the existing frozen game renderer.

The numerical game state and map are explicit conditioning. The backend name
in world checkpoints remains ``vjepa_backend`` solely to preserve the exact
renderer/appearance tensor identities used by existing distilled bundles.
There is no JEPA predictor, EMA target encoder, actor, or critic in this class.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .structured_state import GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS, STATE_DIM
from .vjepa2_ac import VJEPA2ACWorldBackend


class ObservationEncoder(nn.Module):
    def __init__(self, image_size, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(256*(image_size//16)**2, embed_dim),
            nn.LayerNorm(embed_dim), nn.SiLU())

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError('RSSM observation requires native BCHW RGB')
        first = self.net[0].weight
        if rgb.dtype == torch.uint8:
            rgb = rgb.float()/255
        return self.net(rgb.to(device=first.device, dtype=first.dtype))


class RSSMWorldBackend(VJEPA2ACWorldBackend):
    """Prior-only autonomous dynamics; observations enter only via observe."""
    def __init__(self, config):
        # Reuse only the established spatial readout methods. In particular,
        # never construct VJEPA2ACWorldBackend or its Transformer/EMA machinery.
        nn.Module.__init__(self)
        from .models import initialize_event_heads
        from .rssm import DreamerRSSM, RSSMConfig
        from .transformer_model import CameraTransformerWorldModel
        from .spatial_entities import SpatialEntityEncoder
        from .combat_objectives import COMBAT_COUNT_FEATURE_DIM

        self.config = config
        width = config.vjepa_width or 512
        map_channels = config.map_channels
        rank = min(128, width)
        if config.rssm_action_dim != 4+config.rssm_embed_dim:
            raise ValueError('RSSM action conditioning must contain four controls plus the map/state embedding')
        self.renderer = CameraTransformerWorldModel(config, render_only=True)
        self.core = DreamerRSSM(RSSMConfig(
            deter_dim=config.rssm_deter_dim, hidden_dim=config.rssm_hidden_dim,
            stoch=config.rssm_stoch, classes=config.rssm_classes,
            embed_dim=config.rssm_embed_dim, action_dim=config.rssm_action_dim,
            blocks=config.rssm_blocks, unimix=config.rssm_unimix))
        self.obs_encoder = ObservationEncoder(config.image_size, config.rssm_embed_dim)
        feature_size = config.rssm_deter_dim+config.rssm_stoch*config.rssm_classes
        self.rssm_readout = nn.Sequential(nn.Linear(feature_size, config.rssm_hidden_dim),
            nn.LayerNorm(config.rssm_hidden_dim), nn.SiLU(), nn.Linear(config.rssm_hidden_dim, width))
        self.map_projection = nn.Sequential(nn.Linear(map_channels*6, rank), nn.GELU(),
            nn.Linear(rank, width), nn.LayerNorm(width))
        self.condition = nn.Sequential(nn.Linear(STATE_DIM+3*width, config.rssm_embed_dim),
            nn.LayerNorm(config.rssm_embed_dim), nn.SiLU(),
            nn.Linear(config.rssm_embed_dim, config.rssm_embed_dim), nn.Tanh())
        neck_width = width*config.vjepa_readout_mlp_ratio
        self.readout_neck = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, neck_width),
            nn.GELU(), nn.Linear(neck_width, width))
        self.delta_head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 2*(ENTITY_SLOTS+1)))
        nn.init.normal_(self.delta_head[-1].weight, std=.001)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.hit_target_head = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, ENTITY_SLOTS+1))
        self.attack_head = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, ENTITY_SLOTS))
        initialize_event_heads(self.hit_target_head, self.attack_head)
        self.feature_head = nn.Linear(width, 128)
        nn.init.zeros_(self.feature_head.weight)
        nn.init.zeros_(self.feature_head.bias)
        if config.endless:
            self.spawn_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, ENTITY_SLOTS*256))
        self.spatial_encoder = SpatialEntityEncoder(width, map_channels)
        self.spatial_motion_context = nn.Linear(width, rank)
        self.spatial_motion_head = nn.Sequential(nn.Linear(width+rank, rank), nn.GELU(), nn.Linear(rank, 2))
        nn.init.zeros_(self.spatial_motion_head[-1].weight)
        nn.init.zeros_(self.spatial_motion_head[-1].bias)
        self.spatial_combat_head = nn.Sequential(
            nn.Linear(3*width+2*COMBAT_COUNT_FEATURE_DIM, rank), nn.GELU(), nn.Linear(rank, 2))
        nn.init.zeros_(self.spatial_combat_head[-1].weight)
        nn.init.zeros_(self.spatial_combat_head[-1].bias)
        if config.endless:
            spawn_rank = min(64, width)
            self.spatial_spawn_query = nn.Sequential(nn.Linear(3*width+3, spawn_rank), nn.GELU())
            self.spatial_spawn_entity_key = nn.Linear(width, spawn_rank, bias=False)
            self.spatial_spawn_entity_value = nn.Linear(width, spawn_rank, bias=False)
            self.spatial_spawn_output = nn.Linear(spawn_rank, spawn_rank)
            self.spatial_spawn_map_key = nn.Linear(map_channels, spawn_rank, bias=False)
            nn.init.zeros_(self.spatial_spawn_output.weight)
            nn.init.zeros_(self.spatial_spawn_output.bias)
            if config.game_training_v5:
                self.spatial_spawn_anchor_head = nn.Linear(spawn_rank, 9)
                nn.init.zeros_(self.spatial_spawn_anchor_head.weight)
                nn.init.zeros_(self.spatial_spawn_anchor_head.bias)
        self.player_pass_head = nn.Sequential(nn.Linear(2*width+6, 128), nn.GELU(), nn.Linear(128, 2))
        nn.init.zeros_(self.player_pass_head[-1].weight)
        nn.init.zeros_(self.player_pass_head[-1].bias)
        # A training-only observation decoder gives the stochastic posterior
        # an image reconstruction objective independently of exact GT pose.
        self.observation_decoder = nn.Sequential(nn.Linear(width, 512), nn.SiLU(),
            nn.Linear(512, 3*config.image_size**2), nn.Sigmoid())

    def initial_memory(self, batch, device=None, dtype=None):
        return self.core.pack(self.core.initial(batch, device=device)).float()

    def _conditioned_action(self, state, action, map_features):
        spatial = self.spatial_tokens(state, map_features)
        map_context = self._map_context(map_features, state)
        value = torch.cat([state.float(), map_context.float(), spatial.flatten(1).float()], -1)
        conditioning = self.condition(value.to(self.condition[0].weight.dtype)).float()
        controls = torch.cat([action[:, :3].float().clamp(-1, 1), action[:, 3:4].float().clamp(0, 1)], -1)
        return torch.cat([controls, conditioning], -1), spatial

    def initialize_from_image(self, rgb, state, map_features=None):
        if map_features is None:
            raise ValueError('RSSM initialization requires the persistent map features')
        if rgb.ndim == 4:
            rgb = rgb[:, None]
        if state.ndim == 2:
            state = state[:, None].expand(-1, rgb.shape[1], -1)
        if rgb.shape[:2] != state.shape[:2]:
            raise ValueError('Observed RSSM RGB and state contexts must align')
        memory = self.core.initial(rgb.shape[0], device=rgb.device)
        action = state.new_zeros(state.shape[0], 4)
        for index in range(rgb.shape[1]):
            controls, _ = self._conditioned_action(state[:, index], action, map_features)
            memory, _ = self.core.observe_step(memory, controls, self.obs_encoder(rgb[:, index]))
        return self.core.pack(memory).float()

    def _pooled(self, latent):
        features = self.core.features(latent)
        pooled = self.rssm_readout(features.to(self.rssm_readout[0].weight.dtype))
        return pooled+self.readout_neck(pooled.to(self.readout_neck[1].weight.dtype))

    def latent_features(self, memory):
        pooled = self._pooled(self.core.unpack(memory))
        return self.feature_head(pooled.to(self.feature_head.weight.dtype))

    def _readout(self, state, action, map_features, latent, spatial, projection, *, reconstruct=False):
        from .spatial_entities import entity_motion_basis, world_to_local, local_to_world
        from .structured_state import structured_transition, player_command_delta
        pooled = self._pooled(latent)
        delta = self.delta_head(pooled.to(self.delta_head[0].weight.dtype)).float()
        local_context = self.spatial_motion_context(pooled.to(self.spatial_motion_context.weight.dtype))
        local_input = torch.cat([spatial, local_context[:, None].expand(-1, ENTITY_SLOTS+1, -1)], -1)
        residual = self.spatial_motion_head(local_input.to(self.spatial_motion_head[0].weight.dtype)).float()
        basis = entity_motion_basis(state, action, dt=1/self.config.simulation_hz)
        local_prior = world_to_local(delta.reshape(-1, ENTITY_SLOTS+1, 2), basis)
        world_residual = local_to_world(residual, basis)
        motion = delta+world_residual.flatten(1)
        command = player_command_delta(state, action, dt=1/self.config.simulation_hz)
        controls = torch.cat([action[:, :3].float().clamp(-1, 1), action[:, 3:4].float().clamp(0, 1)], -1)
        inputs = torch.cat([pooled.float(), spatial[:, 0].float(), controls, command], -1)
        logits = self.player_pass_head(inputs.to(self.player_pass_head[0].weight.dtype)).float()
        fraction = logits.sigmoid()
        player_delta = command*fraction
        entities = state[:, GLOBAL_DIM:-2].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)
        candidate_entities = torch.cat([entities[:, :, :2]+.1*motion[:, 2:].reshape(-1, ENTITY_SLOTS, 2),
            entities[:, :, 2:]], -1)
        candidate = torch.cat([state[:, :2]+player_delta, state[:, 2:8],
            candidate_entities.flatten(1), state[:, -2:]], -1)
        candidate = projection(candidate, previous=state)
        targets, attacks = self.combat_logits(state, pooled, spatial)
        spawn = {}
        if self.config.endless:
            from .spawn import decode_spawn_positions
            heads = self.spawn_outputs(state, pooled, spatial, map_features)
            positions, valid = decode_spawn_positions(heads['spawn_logits'], state,
                self.renderer._spawn_walkable, 1/self.config.simulation_hz)
            spawn = {**heads, 'spawn_positions': positions, 'spawn_valid': valid}
        result = structured_transition(state, candidate, action.float(), targets, attacks,
            dt=1/self.config.simulation_hz, training=self.training, endless=self.config.endless,
            respawn_enabled=not self.config.navigation_only,
            spawn_positions=spawn.get('spawn_positions'), spawn_valid=spawn.get('spawn_valid'))
        result.update(spawn)
        result.update(memory=self.core.pack(latent).float(),
            features=self.feature_head(pooled.to(self.feature_head.weight.dtype)),
            rssm_readout=pooled.float(), rssm_h=latent['deter'], rssm_z=latent['stoch'],
            rssm_prior_logits=latent['logits'],
            target_logits=targets, attack_logits=attacks,
            player_pass_logits=logits, player_pass_fraction=fraction,
            player_command_delta=command, raw_player_delta=player_delta,
            spatial_tokens=spatial, local_motion_prediction=local_prior+residual,
            motion_basis_forward=basis['forward'], motion_basis_right=basis['right'],
            proposed_npc_displacement=candidate[:, GLOBAL_DIM:-2].reshape_as(entities)[:, :, :2]-entities[:, :, :2])
        if reconstruct:
            result['rssm_reconstruction'] = self.observation_decoder(
                pooled.to(self.observation_decoder[0].weight.dtype)).reshape(-1, 3,
                    self.config.image_size, self.config.image_size)
        return result

    def transition(self, state, action, map_features, memory, projection):
        state = state.float()
        if memory is None:
            memory = self.initial_memory(len(state), device=state.device)
        conditioned_action, spatial = self._conditioned_action(state, action, map_features)
        prior = self.core.imagine_step(self.core.unpack(memory), conditioned_action)
        return self._readout(state, action, map_features, prior, spatial, projection)

    def observe_step(self, state, action, map_features, memory, rgb, *, sample=None):
        """Auxiliary observed stream; prior_result never depends on next RGB."""
        from .models import project_state
        state = state.float()
        conditioned_action, spatial = self._conditioned_action(state, action, map_features)
        prior = self.core.imagine_step(self.core.unpack(memory), conditioned_action, sample=sample)
        prior_result = self._readout(state, action, map_features, prior, spatial, project_state)
        posterior = self.core.posterior(prior, self.obs_encoder(rgb), sample=sample)
        result = self._readout(state, action, map_features, posterior, spatial, project_state, reconstruct=True)
        result.update(prior_result=prior_result, rssm_prior_logits=prior['logits'],
            rssm_posterior_logits=posterior['logits'])
        return result
