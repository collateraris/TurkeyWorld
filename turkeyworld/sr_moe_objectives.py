"""Atlas recognition losses and detached Source-region supervision for five SR experts."""
import numpy as np
import torch
from torch.nn import functional as F

from .atlas_enemy_objectives import atlas_enemy_loss, LOSS_WEIGHTS as ATLAS_WEIGHTS, _proposal_confirmed
from .enemy_decoder_objectives import balanced_binary_loss
from .sr_moe_phase import PROTOCOL, ROLES


LOSS_WEIGHTS = {**ATLAS_WEIGHTS, 'near_wall_l1': .25, 'far_wall_l1': .25,
    'near_wall_edge': .1, 'far_wall_edge': .1, 'near_wall_ssim': .1, 'far_wall_ssim': .1,
    'router_role': .01, 'router_balance': .001, 'guided_confirmation_bce': .1}


def source_sr_moe_labels(batch, *, device):
    from .renderer_wall_depth import source_wall_depth_labels
    from .renderer_wall_supervision import _gun_masks
    from .engine import Game
    state = batch['state'].detach().to(device=device, dtype=torch.float32)
    hr_mask = batch['mask_hr'].detach().to(device=device, dtype=torch.float32)
    grid = batch['map'].detach().to(device=device, dtype=torch.long)
    if (state.shape != (len(state), 20) or hr_mask.shape != (len(state), 1, 128, 128)
            or not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(hr_mask).all())
            or bool(((hr_mask < 0) | (hr_mask > 1)).any())):
        raise ValueError('SR expert labels require detached true Source128 visible masks/state20/map16')
    # Area pooling retains an enemy visible in even one true HR pixel.
    enemy = F.adaptive_avg_pool2d(hr_mask, 32)
    geometry = source_wall_depth_labels(grid, state, enemy)
    depth = geometry['depth'].detach().to(device)
    flash = state[:, 7].clamp(0, 1) * Game.FLASH_DURATION
    frame = torch.where(flash > .045, 1, torch.where(flash > 0, 2, 0)).long()
    gun = torch.as_tensor(np.stack(_gun_masks()), device=device, dtype=state.dtype)[frame][:, None]
    # No kills-as-hurt heuristic: state18 is kills. Only HP0 is a global Source tint.
    player = torch.maximum(gun, (state[:, 4] <= 0)[:, None, None, None].to(gun))
    enemy = enemy * (1 - player)
    scene = (1 - player - enemy).clamp_min(0)
    near, far = scene * (depth < 5), scene * (depth >= 5)
    roles = torch.cat((.5 * enemy, .5 * enemy, far, near, player), 1).detach()
    if not bool(torch.allclose(roles.sum(1), torch.ones_like(roles[:, 0]), atol=1e-6, rtol=0)):
        raise AssertionError('Five SR role targets do not sum to one per native pixel')
    def wall(bands):
        native = sum(geometry['bands'][name]['wall'].detach().to(device) for name in bands)
        return F.interpolate(native, (128, 128), mode='nearest') * (1 - hr_mask) * (1 - F.interpolate(player, (128, 128), mode='nearest'))
    return dict(roles=roles, near_wall=wall(('near_0_2', 'mid_2_5')),
                far_wall=wall(('far_5_8', 'distant_8_12', 'thin_12_plus')))


def _per_image_region(error, mask):
    mask = mask.detach().to(error)
    mass = mask.flatten(1).sum(-1) * error.shape[1]
    values = (error * mask).flatten(1).sum(-1) / mass.clamp_min(1e-12)
    return values, mass > 0


def _present_mean(values, present):
    return values[present].mean() if bool(present.any()) else values.sum() * 0


def _ssim_error(image, target):
    axis = torch.arange(11, device=image.device, dtype=torch.float32) - 5
    kernel = (-axis.square() / (2 * 1.5 ** 2)).exp(); kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None]).expand(3, 1, 11, 11).contiguous()
    a, b = image.float(), target.detach().float()
    mu_a, mu_b = F.conv2d(a, window, groups=3), F.conv2d(b, window, groups=3)
    va = (F.conv2d(a.square(), window, groups=3) - mu_a.square()).clamp_min(0)
    vb = (F.conv2d(b.square(), window, groups=3) - mu_b.square()).clamp_min(0)
    covariance = F.conv2d(a * b, window, groups=3) - mu_a * mu_b
    return 1 - ((2 * mu_a * mu_b + .01 ** 2) * (2 * covariance + .03 ** 2) /
                ((mu_a.square() + mu_b.square() + .01 ** 2) * (va + vb + .03 ** 2))).clamp(-1, 1)


def region_image_terms(image, target, labels):
    target = target.detach().to(image); ssim = _ssim_error(image, target)
    result, support = {}, {}
    for name in ('near_wall', 'far_wall'):
        mask = labels[name].detach().to(image)
        values, present = _per_image_region((image - target).abs(), mask)
        result[name + '_l1'] = _present_mean(values, present)
        support[name + '_frames'] = int(present.sum()); support[name + '_pixels'] = float(mask.sum())
        edges, edge_masks = [], []
        for axis in (-1, -2):
            a, b = [slice(None)] * 4, [slice(None)] * 4
            a[axis], b[axis] = slice(1, None), slice(None, -1)
            edge_mask = torch.maximum(mask[tuple(a)], mask[tuple(b)])
            error = (image.diff(dim=axis) - target.diff(dim=axis)).abs()
            value, valid = _per_image_region(error, edge_mask); edges.append(value); edge_masks.append(valid)
        valid = torch.stack(edge_masks)
        per_image = (torch.stack(edges) * valid).sum(0) / valid.sum(0).clamp_min(1)
        result[name + '_edge'] = _present_mean(per_image, valid.any(0))
        value, valid = _per_image_region(ssim, F.avg_pool2d(mask, 11, stride=1))
        result[name + '_ssim'] = _present_mean(value, valid)
    return result, support


def sr_moe_loss(output, batch, *, label_renderer):
    old, metrics = atlas_enemy_loss(output, batch, label_renderer=label_renderer)
    labels = source_sr_moe_labels(batch, device=output['image'].device)
    terms, supports = region_image_terms(output['image'], batch['hr_source'], labels)
    guided=output['guided_confirmation_logits'].float()
    applicable=output['guided_applicable'].detach().bool().reshape(-1)
    original=output['guided_hypothesis_full_bbox'].detach()
    confirmed=_proposal_confirmed(original,batch['mask_hr'].detach().to(guided.device))
    terms['guided_confirmation_bce']=(balanced_binary_loss(guided[applicable],confirmed[applicable])
        if bool(applicable.any()) else guided.sum()*0)
    supports.update(guided_confirmation_frames=int(applicable.sum()),
        guided_confirmation_positive_frames=int((applicable&confirmed).sum()),
        guided_confirmation_negative_frames=int((applicable&~confirmed).sum()))
    aux = output.get('sr_moe')
    if not isinstance(aux, dict) or tuple(aux.get('role_names', ())) != ROLES:
        raise ValueError('SR expert objective requires the actual five-role forward diagnostics')
    logits, gates = aux['router_logits'], aux['gates']
    if logits.ndim != 3 or logits.shape[1:] != (64, 5) or gates.shape != logits.shape:
        raise ValueError('SR role routing requires actual8x8 token logits and five probabilities')
    target = F.adaptive_avg_pool2d(labels['roles'], (8, 8)).flatten(2).transpose(1, 2).to(logits)
    mass = target.sum((0, 1)); present = mass > 0
    ce = -(logits.float().log_softmax(-1) * target).sum((0, 1)) / mass.clamp_min(1e-12)
    terms['router_role'] = ce[present].mean() if bool(present.any()) else logits.sum() * 0
    means = gates.float().mean((0, 1)); terms['router_balance'] = (5 * means.square().sum() - 1).clamp_min(0)
    total = old + sum(LOSS_WEIGHTS[name] * value for name, value in terms.items())
    if not bool(torch.isfinite(total)): raise FloatingPointError('Nonfinite five-expert Atlas SR objective')
    metrics.update({name:float(value.detach()) for name,value in terms.items()})
    metrics.update({name + '_weighted': LOSS_WEIGHTS[name] * metrics[name] for name in terms})
    metrics.update(**supports, train_loss=float(total.detach()), loss_protocol=PROTOCOL, objective_protocol=PROTOCOL,
        sr_moe_gates_mean=means.detach().cpu().tolist(), sr_moe_label_mass=mass.detach().cpu().tolist(),
        sr_moe_present_roles=int(present.sum()), source_role_masks_loss_only=True,
        source_wall_masks_scope='Source32_DDA_wall_regions_nearest_to128_minus_true_HR_enemy_and_weapon',
        source_future_inputs=False, old_SR_trainable=False, GAN_active=False)
    return total, metrics
