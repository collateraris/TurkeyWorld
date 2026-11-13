"""Five loss-only role masks; frozen causal conditions and old RGB losses stay intact."""
import torch
from torch.nn import functional as F

from .renderer_moe_objectives import ProfiledFrames, moe_conditions
from .renderer_moe_five_phase import ROLES, renderer_moe_five_protocol


class FiveProfiledFrames(ProfiledFrames):
    def batch(self, size, *, indices=None):
        batch = super().batch(size, indices=indices)
        rows = [self.profile_rows[int(i)] for i in batch['row_index'].detach().cpu().tolist()]
        batch['source_enemy_combat_event'] = torch.tensor([
            bool(set(row.get('combat_cases', ())) & {'hit', 'death', 'respawn', 'player_hurt'})
            for row in rows], dtype=torch.bool)
        batch['source_player_hurt_event'] = torch.tensor([
            'player_hurt' in row.get('combat_cases', ()) for row in rows], dtype=torch.bool)
        batch['source_player_combat_event'] = torch.tensor([
            bool(set(row.get('combat_cases', ())) & {'hit', 'miss', 'player_hurt'})
            or bool(row.get('source_events', [0])[0] > .5) for row in rows], dtype=torch.bool)
        return batch


def source_five_role_labels(batch, *, device):
    """Scene depth, visible NPC phase and exact gun/tint coverage are labels only."""
    from .renderer_wall_depth import source_wall_depth_labels
    from .renderer_wall_supervision import _gun_masks
    from .engine import Game
    import numpy as np

    state = batch['state'].detach().float().to(device)
    enemy = batch['object_mask'].detach().float().to(device)
    grid = batch['map'].detach().to(device)
    if (state.shape != (len(state), 20) or enemy.shape != (len(state), 1, 32, 32)
            or not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(enemy).all())
            or bool(((enemy < 0) | (enemy > 1)).any())):
        raise ValueError('Five-role labels need finite native Source states/maps/masks')
    # No target RGB enters DDA, role selection or the ordinary renderer forward.
    depth = source_wall_depth_labels(grid, state, enemy)['depth'].detach().to(device)
    flash = state[:, 7].clamp(0, 1) * Game.FLASH_DURATION
    frame = torch.where(flash > .045, 1, torch.where(flash > 0, 2, 0)).long()
    gun = torch.as_tensor(np.stack(_gun_masks()), device=device, dtype=state.dtype)[frame][:, None]
    previous = batch.get('previous_state')
    dynamic = batch.get('dynamic', torch.zeros(len(state), dtype=torch.bool, device=device)).detach().to(device)
    hp_drop = torch.zeros(len(state), dtype=torch.bool, device=device)
    npc_attack_reset = torch.zeros_like(hp_drop)
    if previous is not None:
        previous = previous.detach().to(state)
        if previous.shape != state.shape: raise ValueError('Wrong previous Source state shape')
        hp_drop = dynamic & (state[:, 4] < previous[:, 4] - 1e-6)
        npc_attack_reset = dynamic & (state[:, 14] > previous[:, 14] + 1e-6)
    def event(name):
        value = batch.get(name)
        if value is None: return torch.zeros(len(state), dtype=torch.bool, device=device)
        if value.shape != (len(state),) or value.dtype != torch.bool:
            raise ValueError('Recorded combat-event role labels must be bool[B]')
        return value.detach().to(device)
    # state[18] is the kill counter, not a visual overlay. Source has no full-frame
    # hurt tint in these overlays=False queries; only player HP0 darkens the frame.
    player_dead = state[:, 4] <= 0
    player_combat = ((flash > 0) | hp_drop | event('source_player_hurt_event')
                     | event('source_player_combat_event'))
    player = gun * player_combat[:, None, None, None]
    player = torch.maximum(player, player_dead[:, None, None, None].to(gun))
    idle_gun = gun * (~player_combat & ~player_dead)[:, None, None, None]
    visible_enemy = enemy * (1 - gun) * (~player_dead)[:, None, None, None]
    enemy_combat = ((state[:, 15] > 0) | (state[:, 13] < .5) | (state[:, 12] <= 0)
                    | (state[:, 14] > 1 / 1.2) | npc_attack_reset | event('source_enemy_combat_event'))
    combat = visible_enemy * enemy_combat[:, None, None, None]
    motion = visible_enemy - combat
    scene = (1 - player - visible_enemy).clamp_min(0)
    # An idle near-camera weapon remains the player/near role even when its wall
    # column hits distant geometry. Combat weapon pixels use the fifth role.
    near = scene * (depth < 5.).to(scene) + idle_gun * (depth >= 5.).to(scene)
    far = scene - near
    labels = torch.cat((near, far, motion, combat, player), 1).detach()
    if not bool(torch.allclose(labels.sum(1), torch.ones_like(labels[:, 0]), atol=1e-6, rtol=0)):
        raise AssertionError('Five Source role labels do not partition every pixel')
    return labels


def five_token_role_loss(router_logits, gates, labels):
    if (router_logits.ndim != 3 or router_logits.shape[-1] != 5 or gates.shape != router_logits.shape
            or labels.ndim != 4 or labels.shape[1] != 5):
        raise ValueError('Five-role logits/gates and native Source labels differ')
    side = int(router_logits.shape[1] ** .5)
    if side * side != router_logits.shape[1] or len(labels) != len(router_logits):
        raise ValueError('Five-role labels need the actual square token grid')
    target = F.adaptive_avg_pool2d(labels.detach().to(router_logits), (side, side)).flatten(2).transpose(1, 2)
    mass = target.sum((0, 1)); present = mass > 0
    per_role = -(router_logits.float().log_softmax(-1) * target).sum((0, 1)) / mass.clamp_min(1e-12)
    role = per_role[present].mean() if bool(present.any()) else router_logits.sum() * 0
    means = gates.float().mean((0, 1))
    balance = (5 * means.square().sum() - 1).clamp_min(0)
    return role, balance, dict(role_ce=float(role.detach()), load_balance=float(balance.detach()),
        gates_mean=means.detach().cpu().tolist(), label_mass=mass.detach().cpu().tolist(),
        present_roles=int(present.sum()), tokens=int(router_logits.shape[1]), role_names=list(ROLES))


def moe_five_refinement_loss(model, batch, *, kind='main', inherited=None):
    from .renderer_enemy_objectives import enemy_refinement_loss, far_wall_refinement_loss
    from .renderer_geometry_objectives import geometry_refinement_loss
    protocol = renderer_moe_five_protocol(); inherited = inherited or {}
    modules = (model.base_student.output_moe, model.flow_renderer.output_moe)
    for module in modules: module.pop_diagnostics()
    detached = not inherited.get('couple_base_flow', False)
    if kind == 'main':
        rgb, metrics = enemy_refinement_loss(model, batch,
            ssim_weight=inherited.get('ssim_weight', .2), edge_weight=inherited.get('edge_weight', .1),
            contrast_weight=inherited.get('contrast_weight', .1),
            latent_weight=inherited.get('source_latent_weight', .1), detach_base=detached,
            base_regression_weight=inherited.get('base_regression_weight', 1.),
            object_crop_weight=inherited.get('object_crop_weight', 0.),
            small_enemy_rebalance=True, far_wall_supervision=True, conditions_fn=moe_conditions)
    elif kind == 'far':
        rgb, metrics = far_wall_refinement_loss(model, batch, detach_base=detached, conditions_fn=moe_conditions)
    elif kind == 'geometry':
        rgb, metrics = geometry_refinement_loss(model, batch, detach_base=detached,
            opacity_depth_weight=0., conditions_fn=moe_conditions)
    else: raise ValueError('Unknown five-role renderer RGB objective')
    labels = source_five_role_labels(batch, device=next(model.parameters()).device)
    role, balance, heads = rgb * 0, rgb * 0, {}
    for name, module in zip(('base', 'rf1'), modules):
        aux = module.pop_diagnostics()
        if not aux or tuple(aux['role_names']) != tuple(ROLES):
            raise ValueError('Missing actual five-role forward diagnostics')
        value, regularizer, report = five_token_role_loss(aux['router_logits'], aux['gates'], labels)
        for key in ('parent_gates', 'split_gates'):
            if key in aux: report[key + '_mean'] = aux[key].detach().float().mean((0, 1)).cpu().tolist()
        heads[name] = report; role = role + value; balance = balance + regularizer
    loss = rgb + protocol['router_role_weight'] * role + protocol['router_balance_weight'] * balance
    metrics.update(train_loss=float(loss.detach()), renderer_rgb_loss=float(rgb.detach()),
        renderer_moe_role_loss=float(role.detach()), renderer_moe_balance_loss=float(balance.detach()),
        renderer_moe_role_weight=protocol['router_role_weight'],
        renderer_moe_balance_weight=protocol['router_balance_weight'], renderer_moe_heads=heads,
        source_future_feedback=False, Source_role_masks_are_loss_only=True,
        source_environment_profile_from_recorded_metadata=True, static_appearance_features_zero=True,
        private_opacity_training_active=False, near_far_boundary_tiles=5.)
    return loss, metrics
