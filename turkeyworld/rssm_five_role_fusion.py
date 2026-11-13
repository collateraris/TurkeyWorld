"""Five private Transformer experts refining the pretrained RSSM fusion trunk.

Only previous-memory trunk tokens, current own state and known action enter.
The module has no recurrent state, observation encoder or simulator dependency.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


FIVE_ROLE_PROTOCOL = 'rssm_five_private_transformer_roles_v1'
FIVE_ROLES = ('player_without_enemy', 'player_with_enemy_quiet',
              'player_with_enemy_combat', 'enemy_motion_spawn', 'combat_health_hit_death')
RESIDUAL_KEYS = ('player_pass_residual', 'enemy_motion_residual',
                 'interaction_target_residual', 'interaction_attack_residual',
                 'interaction_spawn_tile_residual', 'interaction_spawn_anchor_residual')


def own_player_mode(state, action, *, navigation_only=False):
    """A causal, soft-routing hint; absent NPC differs from a respawning NPC."""
    absent = ((state[:, 8:10].abs().sum(-1) == 0) & (state[:, 12] <= 0)
              & (state[:, 13] < .5) & (state[:, 16] <= 0))
    if navigation_only:
        absent = torch.ones_like(absent)
    combat = ((action[:, 3] > 0) | (state[:, 7] > 0) | (state[:, 14] > 0)
              | (state[:, 15] > 0) | (state[:, 16] > 0) | (state[:, 4] <= 0))
    return torch.where(absent, torch.zeros_like(absent, dtype=torch.long),
                       torch.where(combat, torch.full_like(absent, 2, dtype=torch.long),
                                   torch.ones_like(absent, dtype=torch.long)))


class RSSMFiveRoleFusion(nn.Module):
    """Soft five-expert routing and five task heads; exact zero new residuals.

The first three heads propose player-axis pass-logit corrections. Their own-
state mode probabilities mix those corrections. Head4 controls NPC displacement
and spawn selection, head5 controls shot target/attack logits. HP, hit counts
and deaths continue to follow those events through structured bookkeeping.
"""
    def __init__(self, width=256, heads=4, expert_hidden=512, *, role_bias=2.):
        super().__init__()
        if (any(type(value) is not int or value < 1 for value in (width, heads, expert_hidden))
                or width % heads or type(role_bias) not in (int, float)
                or not math.isfinite(role_bias) or role_bias < 0):
            raise ValueError('Five RSSM experts need valid Transformer widths and causal role bias')
        self.width, self.heads, self.expert_hidden = width, heads, expert_hidden
        self.role_bias = float(role_bias)
        self.state_projection = nn.Linear(20, width)
        self.action_projection = nn.Linear(4, width)
        self.role_queries = nn.Parameter(torch.empty(5, width))
        nn.init.normal_(self.role_queries, std=.02)
        self.input_norm = nn.LayerNorm(width)
        self.experts = nn.ModuleList(nn.TransformerEncoderLayer(width, heads,
            dim_feedforward=expert_hidden, dropout=0., activation='gelu',
            batch_first=True, norm_first=True) for _ in FIVE_ROLES)
        self.router = nn.Linear(width, 5)
        nn.init.zeros_(self.router.weight); nn.init.zeros_(self.router.bias)
        self.role_norm = nn.LayerNorm(width)
        self.player_mode_router = nn.Linear(width, 3)
        nn.init.zeros_(self.player_mode_router.weight); nn.init.zeros_(self.player_mode_router.bias)
        self.player_heads = nn.ModuleList(nn.Linear(width, 2) for _ in range(3))
        self.enemy_head = nn.Linear(width, 267)
        self.combat_head = nn.Linear(width, 3)
        for head in (*self.player_heads, self.enemy_head, self.combat_head):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, context, state, action, *, navigation_only=False):
        weight = self.state_projection.weight
        batch = len(state)
        if (type(navigation_only) is not bool or batch < 1
                or context.shape != (batch, 3, self.width) or state.shape != (batch, 20)
                or action.shape != (batch, 4)
                or any(not value.is_floating_point() or value.device != weight.device
                       for value in (context, state, action))):
            raise ValueError('Five-role fusion requires own trunk[B,3,W], state[B,20], action[B,4]')
        if not all(bool(torch.isfinite(value).all()) for value in (context, state, action)):
            raise FloatingPointError('Nonfinite own five-role inputs; no state reset or correction')
        controls = torch.cat([action[:, :3].clamp(-1, 1), action[:, 3:4].clamp(0, 1)], -1)
        shared = torch.cat([context.to(weight.dtype), self.state_projection(state.to(weight.dtype))[:, None],
                            self.action_projection(controls.to(weight.dtype))[:, None]], 1)
        private = []
        for index, expert in enumerate(self.experts):
            query = self.role_queries[index][None, None].expand(batch, 1, -1)
            tokens = self.input_norm(torch.cat([shared, query], 1))
            private.append(expert(tokens)[:, -1])
        private = torch.stack(private, 1)
        logits = self.router(private).float() + self.role_bias * torch.eye(5, device=state.device)[None]
        probabilities = logits.softmax(-1)
        mixed = torch.einsum('bqe,bew->bqw', probabilities.to(private.dtype), private)
        roles = self.role_norm(private + mixed)
        targets = torch.arange(5, device=state.device)[None].expand(batch, -1)
        role_loss = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1))
        importance = probabilities.mean((0, 1))
        balance = (5 * importance.square().sum() - 1).clamp_min(0)
        mode_targets = own_player_mode(state.detach(), controls.detach(), navigation_only=navigation_only)
        mode_logits = self.player_mode_router(roles[:, :3].mean(1)).float()
        mode_logits = mode_logits + self.role_bias * F.one_hot(mode_targets, 3).float()
        mode_weights = mode_logits.softmax(-1)
        mode_loss = F.cross_entropy(mode_logits, mode_targets)
        player_candidates = torch.stack([head(roles[:, index]) for index, head in enumerate(self.player_heads)], 1)
        player = (player_candidates.float() * mode_weights[:, :, None]).sum(1)
        enemy, combat = self.enemy_head(roles[:, 3]).float(), self.combat_head(roles[:, 4]).float()
        return dict(player_pass_residual=player, enemy_motion_residual=enemy[:, :2],
            interaction_target_residual=combat[:, :2], interaction_attack_residual=combat[:, 2:3],
            interaction_spawn_tile_residual=enemy[:, 2:258, None].transpose(1, 2),
            interaction_spawn_anchor_residual=enemy[:, 258:267, None].transpose(1, 2),
            role_tokens=roles.float(), expert_tokens=private.float(), router_logits=logits,
            router_probabilities=probabilities, router_role_loss=role_loss + .25 * mode_loss,
            router_balance_loss=balance, router_importance=importance,
            player_mode_logits=mode_logits, player_mode_weights=mode_weights,
            player_mode_targets=mode_targets, player_mode_loss=mode_loss,
            role_names=FIVE_ROLES, routing='soft_five_private_transformers')


def five_role_metadata(module):
    if not isinstance(module, RSSMFiveRoleFusion):
        raise TypeError('Expected the explicit five-role RSSM adapter')
    result = dict(protocol=FIVE_ROLE_PROTOCOL, roles=list(FIVE_ROLES), experts=5,
        private_Transformer_blocks_per_expert=1, width=module.width, heads=module.heads,
        expert_hidden=module.expert_hidden, role_bias=module.role_bias,
        parameters=sum(p.numel() for p in module.parameters()),
        inputs='previous-memory shared trunk tokens/current own state/known action',
        new_recurrent_memory=False, Source_future_inputs=False, routing='soft_five_private_transformers',
        zero_task_readouts_at_activation=True,
        enemy_head_outputs='NPC XY residual +256 spawn tiles +9 spawn anchors',
        combat_head_outputs='shot target and enemy attack logits; structured events change HP/hits/deaths')
    if hasattr(module.experts[0],'legacy'):
        from .rssm_five_expanded import expanded_five_metadata
        result.update(expanded_experts=True,expanded_architecture=expanded_five_metadata(module),
            private_Transformer_blocks_per_expert=2,parameters_per_expert=1005088)
    return result
