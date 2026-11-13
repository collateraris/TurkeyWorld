"""Causal environment-matched features and loss-only token-role labels."""
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from .renderer_moe_phase import renderer_moe_protocol
from .rssm_combat_protocol import environment_profile


class ProfiledFrames:
    """Add recorded environment metadata; keep every original sampler intact."""
    def __init__(self, parent, rows=None):
        self.parent = parent
        self.profile_rows = parent.rows if rows is None else rows

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def batch(self, size, *, indices=None):
        batch = self.parent.batch(size, indices=indices)
        result = dict(batch)
        values = []
        for index in batch['row_index'].detach().cpu().tolist():
            row = self.profile_rows[int(index)]
            values.append(row.get('source_kind') == 'empty_navigation'
                          or row.get('category') == 'empty_navigation')
        result['navigation_only'] = torch.tensor(values, dtype=torch.bool)
        return result


def moe_conditions(model, batch):
    """Only recorded past enters the prior; the current pose is a render query."""
    device = next(model.parameters()).device
    state = batch['state'].detach().to(device=device, dtype=torch.float32)
    grid = batch['map'].detach().to(device=device, dtype=torch.long)
    dynamic = batch['dynamic'].detach().to(device=device)
    nav = batch.get('navigation_only')
    if (nav is None or nav.shape != (len(state),) or nav.dtype != torch.bool
            or state.shape != (len(state), 20) or grid.shape != (len(state), 16, 16)
            or dynamic.shape != (len(state),) or dynamic.dtype != torch.bool):
        raise ValueError('MoE conditions need explicit recorded navigation flags and native Source queries')
    nav = nav.detach().to(device)
    features = state.new_zeros(len(state), 128)
    with torch.no_grad():
        for profile in (False, True):
            ids = (dynamic & (nav == profile)).nonzero(as_tuple=False).flatten()
            if not len(ids):
                continue
            previous = batch['previous_state'].detach().to(device=device, dtype=torch.float32)[ids]
            image = batch['previous_rgb'].detach().to(device=device, dtype=torch.float32)[ids]
            action = batch['action'].detach().to(device=device, dtype=torch.float32)[ids]
            if (previous.shape != (len(ids), 20) or image.shape != (len(ids), 3, 32, 32)
                    or action.shape != (len(ids), 4)
                    or any(not bool(torch.isfinite(value).all()) for value in (previous, image, action))
                    or bool(((image < 0) | (image > 1)).any())):
                raise ValueError('MoE dynamic features need genuine finite previous observations')
            with environment_profile(model, navigation_only=profile):
                fmap = model.encode_map(grid[ids])
                memory = model.initialize_from_image(image, previous, fmap)
                result = model.transition(previous, action, fmap, memory)
                if result['features'].shape != (len(ids), 128):
                    raise ValueError('MoE renderer requires the actual128-dimensional appearance readout')
                features[ids] = result['features'].detach().float()
        fmap = model.encode_map(grid)
        geometry = {key: value.detach() for key, value in model.render_geometry(state, fmap).items()
                    if torch.is_tensor(value)}
    target, background, mask = (batch[key].detach().to(device=device, dtype=torch.float32)
                                 for key in ('target', 'background', 'object_mask'))
    if (target.shape != (len(state), 3, 32, 32) or background.shape != target.shape
            or mask.shape != (len(state), 1, 32, 32)
            or any(not bool(torch.isfinite(value).all()) for value in (state, features, target, background, mask))
            or any(bool(((value < 0) | (value > 1)).any()) for value in (target, background, mask))):
        raise ValueError('MoE render labels or causal features are invalid')
    return dict(state=state, map=grid, features=features.detach(), geometry=geometry,
                target=target, background=background, object_mask=mask,
                dynamic=dynamic, dynamic_indices=dynamic.nonzero(as_tuple=False).flatten(),
                navigation_only=nav)


def source_role_labels(batch, *, device):
    """Exact native Source gun/visible NPC masks, used only as routing labels."""
    from .renderer_wall_supervision import _gun_masks
    from .engine import Game
    state = batch['state'].detach().float().cpu()
    enemy = batch['object_mask'].detach().float().to(device).clamp(0, 1)
    flash = state[:, 7].clamp(0, 1) * Game.FLASH_DURATION
    frame = torch.where(flash > .045, 1, torch.where(flash > 0, 2, 0)).long()
    masks = np.stack(_gun_masks())
    combat = torch.from_numpy(masks[frame.numpy()]).to(device=device, dtype=torch.float32)[:, None]
    # Weapon pixels cover NPC pixels in the Source rasterization order.
    enemy = enemy * (1 - combat)
    scene = (1 - enemy - combat).clamp(0, 1)
    return torch.cat((scene, enemy, combat), 1).detach()


def token_role_loss(router_logits, gates, labels):
    if (router_logits.ndim != 3 or router_logits.shape[-1] != 3
            or gates.shape != router_logits.shape or labels.ndim != 4 or labels.shape[1] != 3):
        raise ValueError('Renderer routing requires matching token logits/gates and three Source role labels')
    side = int(router_logits.shape[1] ** .5)
    if side * side != router_logits.shape[1] or len(labels) != len(router_logits):
        raise ValueError('Renderer role labels must match an actual square token grid')
    target = F.adaptive_avg_pool2d(labels.detach().to(router_logits), (side, side)).flatten(2).transpose(1, 2)
    mass = target.sum((0, 1))
    present = mass > 0
    per_role = -(router_logits.float().log_softmax(-1) * target).sum((0, 1)) / mass.clamp_min(1e-12)
    role = per_role[present].mean() if bool(present.any()) else router_logits.sum() * 0
    means = gates.float().mean((0, 1))
    balance = (3 * means.square().sum() - 1).clamp_min(0)
    return role, balance, dict(role_ce=float(role.detach()), load_balance=float(balance.detach()),
        gates_mean=means.detach().cpu().tolist(), label_mass=mass.detach().cpu().tolist(),
        present_roles=int(present.sum()), tokens=int(router_logits.shape[1]))


def moe_refinement_loss(model, batch, *, kind='main', inherited=None):
    """Retain the parent RGB losses; add actual per-head role diagnostics."""
    from .renderer_enemy_objectives import enemy_refinement_loss, far_wall_refinement_loss
    from .renderer_geometry_objectives import geometry_refinement_loss
    protocol = renderer_moe_protocol()
    for module in (model.base_student.output_moe, model.flow_renderer.output_moe):
        module.pop_diagnostics()
    inherited = inherited or {}
    detached = not inherited.get('couple_base_flow', False)
    if kind == 'main':
        loss, metrics = enemy_refinement_loss(model, batch,
            ssim_weight=inherited.get('ssim_weight', .2), edge_weight=inherited.get('edge_weight', .1),
            contrast_weight=inherited.get('contrast_weight', .1),
            latent_weight=inherited.get('source_latent_weight', .1), detach_base=detached,
            base_regression_weight=inherited.get('base_regression_weight', 1.),
            object_crop_weight=inherited.get('object_crop_weight', 0.),
            small_enemy_rebalance=True, far_wall_supervision=True, conditions_fn=moe_conditions)
    elif kind == 'far':
        loss, metrics = far_wall_refinement_loss(model, batch, detach_base=detached, conditions_fn=moe_conditions)
    elif kind == 'geometry':
        loss, metrics = geometry_refinement_loss(model, batch, detach_base=detached,
            opacity_depth_weight=0., conditions_fn=moe_conditions)
    else:
        raise ValueError('Unknown renderer MoE source objective')
    rgb = loss
    labels = source_role_labels(batch, device=next(model.parameters()).device)
    role = rgb * 0; balance = rgb * 0; heads = {}
    for name, module in (('base', model.base_student.output_moe), ('rf1', model.flow_renderer.output_moe)):
        aux = module.pop_diagnostics()
        if not aux:
            raise ValueError('Missing actual renderer MoE forward diagnostics: ' + name)
        value, regularizer, report = token_role_loss(aux['router_logits'], aux['gates'], labels)
        role = role + value; balance = balance + regularizer; heads[name] = report
    loss = rgb + protocol['router_role_weight'] * role + protocol['router_balance_weight'] * balance
    metrics.update(train_loss=float(loss.detach()), renderer_rgb_loss=float(rgb.detach()),
        renderer_moe_role_loss=float(role.detach()), renderer_moe_balance_loss=float(balance.detach()),
        renderer_moe_role_weight=protocol['router_role_weight'],
        renderer_moe_balance_weight=protocol['router_balance_weight'], renderer_moe_heads=heads,
        source_future_feedback=False, Source_role_masks_are_loss_only=True,
        source_environment_profile_from_recorded_metadata=True,
        static_appearance_features_zero=True, private_opacity_training_active=False)
    return loss, metrics
