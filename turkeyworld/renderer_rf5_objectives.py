"""Native Source supervision for RF3/RF5, with a frozen Base and VQ codec."""
import math

import torch

from .object_losses import decoded_ssim_loss, balanced_edge_l1, foreground_contrastive_loss
from .renderer_enemy_objectives import balanced_base_regression_loss, source_object_crop_ssim
from .renderer_far_wall_protocol import far_wall_loss
from .renderer_geometry_objectives import geometry_wall_loss
from .renderer_moe_five_objectives import source_five_role_labels, five_token_role_loss
from .renderer_moe_five_phase import ROLES
from .renderer_moe_objectives import moe_conditions
from .renderer_small_enemy_protocol import separate_region_loss, native_crop_loss
from .renderer_wall_depth import source_wall_depth_labels
from .renderer_wall_supervision import source_wall_masks, wall_loss_terms


ROUTER_ROLE_WEIGHT = .01
ROUTER_BALANCE_WEIGHT = .001


def _validate_frozen(model, checkpoint_velocity):
    flow, base = model.flow_renderer, model.base_student
    codec = flow.codec
    private = getattr(flow, 'geometry_opacity', None)
    if (type(checkpoint_velocity) is not bool or flow.kind != 'student'
            or type(flow.steps) is not int or flow.steps not in (3, 5)
            or base.training or any(p.requires_grad for p in base.parameters())
            or codec.training or codec.quantize.training
            or any(p.requires_grad for p in codec.parameters())
            or private is not None and (private.training or any(p.requires_grad for p in private.parameters()))):
        raise ValueError('RF refinement requires three/five Euler steps and frozen eval Base/codec/private opacity')
    if not callable(getattr(getattr(flow, 'output_moe', None), 'pop_diagnostics', None)):
        raise ValueError('RF5 requires the actual five-role renderer output expert diagnostics')


def rf5_forward(model, conditions, *, checkpoint_velocity=True):
    """Return Base, decoded RF, endpoint and its three/five router graphs.

    The hook only retains existing tensor references. It is removed before any
    backward can trigger checkpoint recomputation, so recomputation cannot
    append extra routing samples or change its tensor-operation sequence.
    Source target images and masks in ``conditions`` are never read here.
    """
    _validate_frozen(model, checkpoint_velocity)
    flow = model.flow_renderer
    steps = flow.steps
    with torch.no_grad():
        base = model.base_student(conditions['state'], conditions['features'], conditions['geometry']).detach()
    module = flow.output_moe
    module.pop_diagnostics()
    routers = []
    def capture(head, _inputs, _output):
        # Deliberately no Tensor operation: non-reentrant checkpoint must see
        # exactly the same velocity computation with/without this hook.
        routers.append(head.last_aux)
    handle = module.register_forward_hook(capture)
    try:
        endpoint = flow.latent_sample(base, conditions['state'], noise=flow.fixed_noise,
            steps=steps, checkpoint_velocity=checkpoint_velocity, map_condition=conditions['map'])
    finally:
        handle.remove()
        module.pop_diagnostics()
    if (len(routers) != steps or any(not isinstance(aux, dict)
            or tuple(aux.get('role_names', ())) != tuple(ROLES) for aux in routers)):
        raise ValueError('RF needs one real five-role router graph from each actual Euler velocity call')
    # Frozen parameters do not imply no_grad: reconstruction gradients must
    # cross terminal quantization and decoding back to all RF velocity calls.
    quantized = flow.codec.quantize(flow.denormalize_latents(endpoint))['quantized']
    rgb = flow.codec.decode(quantized)
    if (base.shape != rgb.shape or rgb.shape[1:] != (3, 32, 32)
            or not all(bool(torch.isfinite(value).all()) for value in (base, rgb, endpoint))):
        raise FloatingPointError('Invalid native32 RF5 prediction or latent endpoint')
    return base, rgb, endpoint, routers


def _weights(inherited):
    result = dict(ssim_weight=.2, edge_weight=.1, contrast_weight=.1,
        source_latent_weight=.1, base_regression_weight=1., object_crop_weight=0.)
    for key in result:
        result[key] = inherited.get(key, result[key])
        if (type(result[key]) not in (int, float) or not math.isfinite(result[key]) or result[key] < 0):
            raise ValueError('RF5 ' + key + ' must be finite and nonnegative')
    return result


def rf5_refinement_loss(model, batch, *, kind='main', inherited=None, checkpoint_velocity=True):
    """Supervise RF only; Source state/geometry/images remain query/labels.

    Main keeps independently normalized NPC/background, native crops, wall
    geometry and quantized Source-latent targets. Auxiliary streams compute
    only their named far/all-depth geometry objective. No Base or GAN loss is
    included. Router regularization is a mean over all actual Euler calls.
    """
    if kind not in ('main', 'far', 'geometry'):
        raise ValueError('Unknown RF5 Source objective kind')
    weights = _weights(inherited or {})
    _validate_frozen(model, checkpoint_velocity)
    conditions = moe_conditions(model, batch)
    base, rgb, endpoint, routers = rf5_forward(model, conditions, checkpoint_velocity=checkpoint_velocity)
    steps = model.flow_renderer.steps
    head = 'rf' + str(steps)
    target, mask, background = (conditions[name].detach() for name in ('target', 'object_mask', 'background'))
    depth_labels = source_wall_depth_labels(conditions['map'], conditions['state'], mask)
    metrics, terms = {}, {}
    if kind == 'main':
        regions = separate_region_loss(rgb, target, mask)
        terms[head+'_source_l1'] = regions['loss']
        terms[head+'_source_ssim_loss'] = weights['ssim_weight'] * decoded_ssim_loss(rgb, target, mask)
        terms[head+'_source_edge'] = weights['edge_weight'] * balanced_edge_l1(rgb, target, mask)
        terms[head+'_foreground_contrast'] = weights['contrast_weight'] * foreground_contrastive_loss(rgb, target, background, mask)
        crop, crop_metrics = native_crop_loss(rgb, target, mask)
        terms[head+'_native_crop'] = crop
        wall = wall_loss_terms(rgb, target, source_wall_masks(conditions['map'], conditions['state'], mask))
        terms[head+'_wall'] = wall['loss']
        far, far_metrics = far_wall_loss(rgb, target, depth_labels)
        terms[head+'_far_wall'] = far
        if weights['source_latent_weight']:
            with torch.no_grad():
                target_z = model.flow_renderer.normalize_latents(model.flow_renderer.codec.encode(target)['quantized']).detach()
            latent = separate_region_loss(endpoint, target_z, mask, squared=True)['loss']
            terms[head+'_source_latent'] = weights['source_latent_weight'] * latent
            metrics['source_latent_loss'] = float(latent.detach())
        if weights['base_regression_weight']:
            regression = balanced_base_regression_loss(rgb, target, base, mask)
            terms[head+'_base_regression'] = weights['base_regression_weight'] * regression
            metrics['base_regression_loss'] = float(regression.detach())
        if weights['object_crop_weight']:
            terms[head+'_object_crop_ssim'] = weights['object_crop_weight'] * source_object_crop_ssim(rgb, target, mask)
        metrics.update({head+'_foreground_l1':float(regions['foreground'].detach()),
            head+'_background_l1':float(regions['background'].detach()),
            head+'_foreground_images':regions['foreground_images'],head+'_background_images':regions['background_images']})
        metrics.update(
            native_crop={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in crop_metrics.items()},
            wall={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in wall.items()}, far_wall=far_metrics)
    elif kind == 'far':
        far, metrics['far_wall'] = far_wall_loss(rgb, target, depth_labels)
        terms[head+'_far_wall'] = far
    else:
        geometry, metrics['geometry_wall'] = geometry_wall_loss(rgb, target, depth_labels)
        terms[head+'_geometry_wall'] = geometry
    rgb_loss = sum(terms.values())
    labels = source_five_role_labels(batch, device=rgb.device)
    roles, balances, reports = [], [], []
    for index, aux in enumerate(routers):
        role, balance, report = five_token_role_loss(aux['router_logits'], aux['gates'], labels)
        reports.append(dict(report, euler_index=index, euler_time=index/steps))
        roles.append(role); balances.append(balance)
    role = torch.stack(roles).mean()
    balance = torch.stack(balances).mean()
    total = rgb_loss + ROUTER_ROLE_WEIGHT*role + ROUTER_BALANCE_WEIGHT*balance
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite RF5 Source-only objective')
    metrics.update({key:float(value.detach()) for key,value in terms.items()})
    metrics.update(train_loss=float(total.detach()), renderer_rgb_loss=float(rgb_loss.detach()),
        renderer_moe_role_loss=float(role.detach()), renderer_moe_balance_loss=float(balance.detach()),
        renderer_moe_role_weight=ROUTER_ROLE_WEIGHT, renderer_moe_balance_weight=ROUTER_BALANCE_WEIGHT,
        renderer_moe_heads={head:dict(role_names=list(ROLES), tokens=reports[0]['tokens'],
            gates_mean=[sum(row['gates_mean'][i] for row in reports)/steps for i in range(5)],
            role_ce=sum(row['role_ce'] for row in reports)/steps,
            load_balance=sum(row['load_balance'] for row in reports)/steps)},
        rf_router_steps=reports, router_velocity_calls=steps, router_reduction='mean_over_actual_Euler_velocity_calls',
        solver='euler', solver_steps=steps, checkpoint_velocity=checkpoint_velocity,
        objective_kind=kind, examples=len(rgb), base_loss_active=False, base_router_loss_active=False,
        base_frozen=True, codec_frozen=True, private_opacity_frozen=True, GAN_active=False,
        source_latent_active=kind=='main' and weights['source_latent_weight']>0,
        source_future_feedback=False, Source_role_masks_are_loss_only=True,
        future_source_rgb_in_prior=False, future_source_state_in_prior=False,
        source_pose_aligned=True, flow_base_detached=True)
    metrics[head+'_router_steps'] = reports
    return total, metrics
