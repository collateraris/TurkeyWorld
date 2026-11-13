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
        # Append the optional branch after every legacy module: disabled
        # checkpoints retain the exact namespace, initialization and RNG use.
        if getattr(config, 'enemy_rssm_enabled', False):
            from .enemy_rssm import EnemyRSSM, EnemyRSSMConfig
            self.enemy_rssm = EnemyRSSM(EnemyRSSMConfig(
                deter_dim=config.enemy_rssm_deter_dim, hidden_dim=config.enemy_rssm_hidden_dim,
                stoch=config.enemy_rssm_stoch, classes=config.enemy_rssm_classes,
                embed_dim=config.enemy_rssm_embed_dim, action_dim=config.enemy_rssm_action_dim,
                blocks=config.enemy_rssm_blocks, unimix=config.enemy_rssm_unimix),
                width=width, observation_dim=config.rssm_embed_dim)
        if getattr(config, 'rssm_fusion_moe_enabled', False):
            from .rssm_fusion_moe import RSSMFusionMoE, RSSMFusionMoEConfig
            self.fusion_moe = RSSMFusionMoE(RSSMFusionMoEConfig(
                width=config.rssm_fusion_moe_width, depth=config.rssm_fusion_moe_depth,
                heads=config.rssm_fusion_moe_heads, expert_hidden=config.rssm_fusion_moe_expert_hidden,
                experts=config.rssm_fusion_moe_experts, top_k=config.rssm_fusion_moe_top_k,
                role_bias=config.rssm_fusion_moe_role_bias),
                player_memory_dim=self.core.memory_size, enemy_memory_dim=self.enemy_rssm.memory_size,
                map_width=width, spatial_width=width)
        # Registration is append-only, after the pretrained P/E and MoE.
        if getattr(config, 'combat_rssm_enabled', False):
            from .combat_rssm import CombatRSSM, CombatRSSMConfig
            self.combat_rssm = CombatRSSM(CombatRSSMConfig(
                deter_dim=config.combat_rssm_deter_dim, hidden_dim=config.combat_rssm_hidden_dim,
                stoch=config.combat_rssm_stoch, classes=config.combat_rssm_classes,
                embed_dim=config.combat_rssm_embed_dim, action_dim=config.combat_rssm_action_dim,
                blocks=config.combat_rssm_blocks, unimix=config.combat_rssm_unimix),
                width=width, observation_dim=config.rssm_embed_dim,
                fusion_width=config.rssm_fusion_moe_width)
        if getattr(config, 'rssm_five_role_enabled', False):
            from .rssm_five_role_fusion import RSSMFiveRoleFusion
            # Appended after every pretrained branch; the new zero task heads
            # neither consume nor reset any recurrent memory or RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.five_role_fusion = RSSMFiveRoleFusion(
                    width=config.rssm_fusion_moe_width, heads=config.rssm_fusion_moe_heads,
                    expert_hidden=config.rssm_fusion_moe_expert_hidden,
                    role_bias=config.rssm_fusion_moe_role_bias)

    @property
    def memory_size(self):
        return (self.core.memory_size
            +(self.enemy_rssm.memory_size if hasattr(self, 'enemy_rssm') else 0)
            +(self.combat_rssm.memory_size if hasattr(self, 'combat_rssm') else 0))

    def split_memory(self, memory):
        """Legacy two-way view; appended Combat memory never enters Enemy h/z."""
        player, enemy, _ = self.split_all_memory(memory)
        return player, enemy

    def split_all_memory(self, memory):
        if not hasattr(self, 'enemy_rssm'):
            return memory, None, None
        if memory.ndim != 2 or memory.shape[-1] != self.memory_size:
            raise ValueError('RSSM memory must concatenate the complete player/enemy/optional Combat states')
        player_end = self.core.memory_size
        enemy_end = player_end+self.enemy_rssm.memory_size
        return (memory[:, :player_end].float(), memory[:, player_end:enemy_end].float(),
            memory[:, enemy_end:].float() if hasattr(self, 'combat_rssm') else None)

    def _pack_memory(self, player, enemy, combat=None):
        parts = [self.core.pack(player).float()]
        if enemy is not None:
            parts.append(enemy['memory'].float())
        if hasattr(self, 'combat_rssm'):
            if combat is None:
                raise ValueError('Persistent Combat state cannot be omitted from packed RSSM memory')
            parts.append(combat['memory'].float())
        return torch.cat(parts, -1) if len(parts)>1 else parts[0]

    def initial_memory(self, batch, device=None, dtype=None):
        parts = [self.core.pack(self.core.initial(batch, device=device)).float()]
        if hasattr(self, 'enemy_rssm'):
            parts.append(self.enemy_rssm.initial(batch, device=device))
        if hasattr(self, 'combat_rssm'):
            parts.append(self.combat_rssm.initial(batch, device=device))
        return torch.cat(parts, -1) if len(parts)>1 else parts[0]

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
        enemy = None
        combat = None
        for index in range(rgb.shape[1]):
            controls, spatial = self._conditioned_action(state[:, index], action, map_features)
            embed = self.obs_encoder(rgb[:, index])
            memory, _ = self.core.observe_step(memory, controls, embed)
            if hasattr(self, 'enemy_rssm'):
                context = self._map_context(map_features, state[:, index])
                if enemy is None:
                    enemy = self.enemy_rssm.observe_initial(embed.detach(), state[:, index],
                        context, spatial, sample=self.core.training)
                else:
                    prior = self.enemy_rssm.prior(enemy['memory'], state[:, index], action,
                        state[:, index, :2], context, spatial, sample=self.core.training)
                    enemy = self.enemy_rssm.posterior(prior, embed.detach(), sample=self.core.training)
            if hasattr(self, 'combat_rssm'):
                contexts = {'player_context': self._pooled(memory), 'enemy_context': enemy['pooled'],
                    'sample': self.core.training}
                if combat is None:
                    combat = self.combat_rssm.observe_initial(embed.detach(), state[:, index],
                        context, spatial, **contexts)
                else:
                    prior = self.combat_rssm.prior(combat['memory'], state[:, index], action,
                        state[:, index, :2], context, spatial, **contexts)
                    combat = self.combat_rssm.posterior(prior, embed.detach(), sample=self.core.training)
        return self._pack_memory(memory, enemy, combat)

    def _pooled(self, latent):
        features = self.core.features(latent)
        pooled = self.rssm_readout(features.to(self.rssm_readout[0].weight.dtype))
        return pooled+self.readout_neck(pooled.to(self.readout_neck[1].weight.dtype))

    def latent_features(self, memory):
        if hasattr(self, 'enemy_rssm'):
            memory, _ = self.split_memory(memory)
        pooled = self._pooled(self.core.unpack(memory))
        return self.feature_head(pooled.to(self.feature_head.weight.dtype))

    def _readout(self, state, action, map_features, latent, spatial, projection, *, reconstruct=False,
                 enemy_memory=None, enemy_result=None, enemy_conditioned_player_xy=None, sample=None,
                 fusion=None, combat_memory=None, combat_result=None):
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
        if fusion is not None:
            logits = logits+fusion['player_pass_residual'].float()
        fraction = logits.sigmoid()
        player_delta = command*fraction
        if hasattr(self, 'enemy_rssm'):
            if enemy_conditioned_player_xy is None:
                enemy_conditioned_player_xy = (state[:, :2]+player_delta).detach()
            if enemy_result is None:
                if enemy_memory is None:
                    raise ValueError('Enemy RSSM requires its persistent previous memory')
                enemy_result = self.enemy_rssm.prior(enemy_memory, state, action,
                    enemy_conditioned_player_xy, self._map_context(map_features, state), spatial,
                    sample=self.core.training if sample is None else sample)
            enemy_motion = enemy_result['motion'].float()
            if fusion is not None:
                enemy_motion = enemy_motion+fusion['enemy_motion_residual'].float()
            motion = torch.cat([motion[:, :2], enemy_motion], -1)
        if hasattr(self, 'combat_rssm') and combat_result is None:
            if combat_memory is None:
                raise ValueError('Combat RSSM requires its persistent previous memory')
            combat_result = self.combat_rssm.prior(combat_memory, state, action,
                enemy_conditioned_player_xy, self._map_context(map_features, state), spatial,
                player_context=pooled, enemy_context=enemy_result['pooled'],
                sample=self.core.training if sample is None else sample)
        entities = state[:, GLOBAL_DIM:-2].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)
        candidate_entities = torch.cat([entities[:, :, :2]+.1*motion[:, 2:].reshape(-1, ENTITY_SLOTS, 2),
            entities[:, :, 2:]], -1)
        candidate = torch.cat([state[:, :2]+player_delta, state[:, 2:8],
            candidate_entities.flatten(1), state[:, -2:]], -1)
        candidate = projection(candidate, previous=state)
        targets, attacks = self.combat_logits(state, pooled, spatial)
        if enemy_result is not None:
            attacks = enemy_result['attack_logits'].float()
        if fusion is not None:
            targets = targets+fusion['interaction_target_residual'].float()
            attacks = attacks+fusion['interaction_attack_residual'].float()
        if combat_result is not None:
            targets = targets+combat_result['target_residual'].float()
            attacks = attacks+combat_result['attack_residual'].float()
        spawn = {}
        if self.config.endless:
            from .spawn import decode_spawn_positions
            heads = self.spawn_outputs(state, pooled, spatial, map_features)
            if fusion is not None or combat_result is not None:
                tile_residual = (fusion['interaction_spawn_tile_residual'].float()
                    if fusion is not None else torch.zeros_like(heads['spawn_logits']))
                if combat_result is not None:
                    tile_residual = tile_residual+combat_result['spawn_tile_residual'].float()
                if 'spawn_fallback_logits' in heads:
                    from .anchor_spawn import hierarchical_spawn_logits
                    anchor_residual = (fusion['interaction_spawn_anchor_residual'].float()
                        if fusion is not None else torch.zeros_like(heads['spawn_anchor_residual']))
                    if combat_result is not None:
                        anchor_residual = anchor_residual+combat_result['spawn_anchor_residual'].float()
                    heads = hierarchical_spawn_logits(heads['spawn_fallback_logits']+tile_residual,
                        heads['spawn_anchor_residual']+anchor_residual,
                        state, self.renderer._spawn_walkable, 1/self.config.simulation_hz)
                else:
                    heads = {**heads, 'spawn_logits': heads['spawn_logits']+tile_residual}
            positions, valid = decode_spawn_positions(heads['spawn_logits'], state,
                self.renderer._spawn_walkable, 1/self.config.simulation_hz)
            spawn = {**heads, 'spawn_positions': positions, 'spawn_valid': valid}
        result = structured_transition(state, candidate, action.float(), targets, attacks,
            dt=1/self.config.simulation_hz, training=self.training, endless=self.config.endless,
            respawn_enabled=not self.config.navigation_only,
            spawn_positions=spawn.get('spawn_positions'), spawn_valid=spawn.get('spawn_valid'))
        result.update(spawn)
        result.update(memory=self._pack_memory(latent, enemy_result, combat_result),
            features=self.feature_head(pooled.to(self.feature_head.weight.dtype)),
            rssm_readout=pooled.float(), rssm_h=latent['deter'], rssm_z=latent['stoch'],
            rssm_prior_logits=latent['logits'],
            target_logits=targets, attack_logits=attacks,
            player_pass_logits=logits, player_pass_fraction=fraction,
            player_command_delta=command, raw_player_delta=player_delta,
            spatial_tokens=spatial, local_motion_prediction=local_prior+residual,
            motion_basis_forward=basis['forward'], motion_basis_right=basis['right'],
            proposed_npc_displacement=candidate[:, GLOBAL_DIM:-2].reshape_as(entities)[:, :, :2]-entities[:, :, :2])
        if enemy_result is not None:
            enemy_local = world_to_local(motion.reshape(-1, ENTITY_SLOTS+1, 2), basis)[:, 1:]
            result.update(local_motion_prediction=torch.cat([(local_prior+residual)[:, :1], enemy_local], 1),
                enemy_rssm_latent=enemy_result, enemy_rssm_memory=enemy_result['memory'].float(),
                enemy_rssm_readout=enemy_result['pooled'].float(),
                enemy_rssm_h=enemy_result['rssm_state']['deter'],
                enemy_rssm_z=enemy_result['rssm_state']['stoch'],
                enemy_rssm_prior_logits=enemy_result['prior_logits'],
                enemy_rssm_motion=enemy_motion, enemy_rssm_attack_logits=attacks,
                enemy_rssm_conditioned_player_xy=enemy_conditioned_player_xy.detach())
            if 'posterior_logits' in enemy_result:
                result['enemy_rssm_posterior_logits'] = enemy_result['posterior_logits']
        if combat_result is not None:
            result.update(combat_rssm_latent=combat_result,
                combat_rssm_memory=combat_result['memory'].float(),
                combat_rssm_readout=combat_result['pooled'].float(),
                combat_rssm_h=combat_result['rssm_state']['deter'],
                combat_rssm_z=combat_result['rssm_state']['stoch'],
                combat_rssm_prior_logits=combat_result['prior_logits'])
            if 'posterior_logits' in combat_result:
                result['combat_rssm_posterior_logits'] = combat_result['posterior_logits']
        if reconstruct:
            result['rssm_reconstruction'] = self.observation_decoder(
                pooled.to(self.observation_decoder[0].weight.dtype)).reshape(-1, 3,
                    self.config.image_size, self.config.image_size)
        if fusion is not None:
            result.update(rssm_fusion_moe=fusion,
                rssm_fusion_moe_router_role_loss=fusion['router_role_loss'],
                rssm_fusion_moe_router_balance_loss=fusion['router_balance_loss'])
            addition = fusion.get('five_role_fusion')
            if addition is not None:
                result.update(five_role_fusion=addition,
                    five_role_router_role_loss=addition['router_role_loss'],
                    five_role_router_balance_loss=addition['router_balance_loss'])
        return result

    def _previous_memory_fusion(self, player_memory, enemy_memory, state, action, map_features, spatial,
                                combat_memory=None):
        if not hasattr(self, 'fusion_moe'):
            return None
        if enemy_memory is None:
            raise ValueError('Shared RSSM fusion requires both persistent previous memories')
        extra = ({'combat_context': self.combat_rssm.fusion_context(combat_memory)}
            if hasattr(self, 'combat_rssm') else {})
        result = self.fusion_moe(player_memory, enemy_memory, state, action,
            self._map_context(map_features, state), spatial, **extra)
        if hasattr(self, 'five_role_fusion'):
            from .rssm_five_role_fusion import RESIDUAL_KEYS
            addition = self.five_role_fusion(result['role_tokens'], state, action,
                navigation_only=self.config.navigation_only)
            result = {**result, **{key: result[key]+addition[key] for key in RESIDUAL_KEYS},
                'router_role_loss': result['router_role_loss']+addition['router_role_loss'],
                'router_balance_loss': result['router_balance_loss']+addition['router_balance_loss'],
                'five_role_fusion': addition}
        return result

    def transition(self, state, action, map_features, memory, projection):
        state = state.float()
        if memory is None:
            memory = self.initial_memory(len(state), device=state.device)
        memory, enemy_memory, combat_memory = self.split_all_memory(memory)
        conditioned_action, spatial = self._conditioned_action(state, action, map_features)
        fusion = self._previous_memory_fusion(memory, enemy_memory, state, action, map_features, spatial,
            combat_memory)
        prior = self.core.imagine_step(self.core.unpack(memory), conditioned_action)
        return self._readout(state, action, map_features, prior, spatial, projection,
            enemy_memory=enemy_memory, fusion=fusion, combat_memory=combat_memory)

    def observe_step(self, state, action, map_features, memory, rgb, *, sample=None):
        """Auxiliary observed stream; prior_result never depends on next RGB."""
        from .models import project_state
        state = state.float()
        memory, enemy_memory, combat_memory = self.split_all_memory(memory)
        conditioned_action, spatial = self._conditioned_action(state, action, map_features)
        fusion = self._previous_memory_fusion(memory, enemy_memory, state, action, map_features, spatial,
            combat_memory)
        prior = self.core.imagine_step(self.core.unpack(memory), conditioned_action, sample=sample)
        prior_result = self._readout(state, action, map_features, prior, spatial, project_state,
            enemy_memory=enemy_memory, sample=sample, fusion=fusion, combat_memory=combat_memory)
        embed = self.obs_encoder(rgb)
        posterior = self.core.posterior(prior, embed, sample=sample)
        enemy_posterior = (self.enemy_rssm.posterior(prior_result['enemy_rssm_latent'], embed.detach(),
            sample=self.core.training if sample is None else sample) if hasattr(self, 'enemy_rssm') else None)
        combat_posterior = (self.combat_rssm.posterior(prior_result['combat_rssm_latent'], embed.detach(),
            sample=self.core.training if sample is None else sample) if hasattr(self, 'combat_rssm') else None)
        result = self._readout(state, action, map_features, posterior, spatial, project_state, reconstruct=True,
            enemy_result=enemy_posterior,
            enemy_conditioned_player_xy=(prior_result['enemy_rssm_conditioned_player_xy']
                if enemy_posterior is not None else None), fusion=fusion, combat_result=combat_posterior)
        result.update(prior_result=prior_result, rssm_prior_logits=prior['logits'],
            rssm_posterior_logits=posterior['logits'])
        return result
