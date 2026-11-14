"""RF3 -> terminal Q -> frozen decoder, supervised by native Source labels.

The restorer is deliberately absent.  Only RF learns; Source RGB supplies
image and detached latent targets, never an RF conditioning input.
"""
from copy import deepcopy

import torch

from .object_losses import balanced_edge_l1, decoded_ssim_loss
from .renderer_far_wall_protocol import far_wall_loss, far_wall_protocol
from .renderer_geometry_objectives import geometry_wall_loss, geometry_wall_protocol
from .renderer_moe_five_objectives import source_five_role_labels, five_token_role_loss
from .renderer_moe_five_phase import ROLES
from .renderer_moe_objectives import moe_conditions
from .renderer_rf5_objectives import rf5_forward, _validate_frozen
from .renderer_small_enemy_protocol import separate_region_loss, native_crop_loss
from .renderer_wall_depth import source_wall_depth_labels


PROTOCOL = 'renderer_rf_codec_geometry_v1'
SETTINGS = dict(
    name=PROTOCOL, version=1, solver='euler', solver_steps=3,
    inference_path='RF3_then_terminal_Q_then_frozen_current_decoder',
    weights=dict(global_ssim=1., region_l1=.5, region_edge=.1,
                 native_crop_l1=.25, native_crop_edge=.1, geometry_wall=1.,
                 source_latent=.1, router_role=.01, router_balance=.001),
    main_loss='global_Gaussian_SSIM_plus_region_RGB_edges_crops_geometry_and_Source_Q_latent',
    source_latent_reduction='foreground_background_balanced_squared_error',
    router_reduction='mean_over_three_actual_Euler_velocity_calls',
    outer_stream_weights=dict(main=1., far=.25, geometry=1.),
    Base_frozen=True, encoder_frozen=True, codebook_frozen=True, decoder_frozen=True,
    private_opacity_frozen=True, world_frozen=True, restorer_active=False,
    Source_Q_image_branch=False, Base_regression_loss=False, GAN_active=False,
    Source_RGB_into_RF_conditioning=False, Source_labels_only=True,
    Source_Q_is_detached_latent_target_only=True, source_future_feedback=False,
)


def renderer_rf_codec_protocol():
    result = deepcopy(SETTINGS)
    result['geometry_wall_protocol'] = geometry_wall_protocol()
    result['far_wall_protocol'] = far_wall_protocol()
    return result


def _validate(model, checkpoint_velocity):
    flow = model.flow_renderer
    if (type(flow.steps) is not int or flow.steps != 3
            or getattr(flow, 'conditional_restorer', None) is not None):
        raise ValueError('RF-codec objective requires exactly RF3 and no conditional restorer')
    _validate_frozen(model, checkpoint_velocity)


def rf_codec_forward(model, conditions, *, checkpoint_velocity=True):
    """One differentiable RF3/Q/D forward; never reads target RGB or masks."""
    _validate(model, checkpoint_velocity)
    return rf5_forward(model, conditions, checkpoint_velocity=checkpoint_velocity)


def rf_codec_refinement_loss(model, batch, *, kind='main', checkpoint_velocity=True):
    """Return the actual named-stream loss and JSON-ready measurements.

    The caller applies the pinned outer stream weights (1, .25, 1).  Main
    optimizes unmasked whole-image SSIM; NPC/corner/thin-wall objectives retain
    their independent area normalization.  Far/geometry streams have no main
    RGB or Source-code loss.  All use the actual three router graphs.
    """
    if kind not in ('main', 'far', 'geometry'):
        raise ValueError('Unknown RF-codec Source objective kind')
    _validate(model, checkpoint_velocity)
    conditions = moe_conditions(model, batch)
    base, rgb, endpoint, routers = rf_codec_forward(
        model, conditions, checkpoint_velocity=checkpoint_velocity)
    target, mask = (conditions[key].detach() for key in ('target', 'object_mask'))
    depth_labels = source_wall_depth_labels(conditions['map'], conditions['state'], mask)
    weights = SETTINGS['weights']
    terms, raw, metrics = {}, {}, {}
    if kind == 'main':
        regions = separate_region_loss(rgb, target, mask)
        raw['global_ssim_loss'] = decoded_ssim_loss(rgb, target, mask=None)
        raw['region_l1'] = regions['loss']
        raw['region_edge_l1'] = balanced_edge_l1(rgb, target, mask)
        crop, crop_metrics = native_crop_loss(rgb, target, mask)
        geometry, geometry_metrics = geometry_wall_loss(rgb, target, depth_labels)
        with torch.no_grad():
            target_z = model.flow_renderer.normalize_latents(
                model.flow_renderer.codec.encode(target)['quantized']).detach()
        raw['source_latent_mse'] = separate_region_loss(
            endpoint, target_z, mask, squared=True)['loss']
        terms.update(global_ssim=weights['global_ssim'] * raw['global_ssim_loss'],
            region_l1=weights['region_l1'] * raw['region_l1'],
            region_edge=weights['region_edge'] * raw['region_edge_l1'],
            native_crop=crop, geometry_wall=weights['geometry_wall'] * geometry,
            source_latent=weights['source_latent'] * raw['source_latent_mse'])
        metrics.update(rf3_foreground_l1=float(regions['foreground'].detach()),
            rf3_background_l1=float(regions['background'].detach()),
            rf3_foreground_images=regions['foreground_images'],
            rf3_background_images=regions['background_images'],
            native_crop={key: float(value.detach()) if torch.is_tensor(value) else value
                         for key, value in crop_metrics.items()},
            geometry_wall=geometry_metrics)
    elif kind == 'far':
        terms['far_wall'], metrics['far_wall'] = far_wall_loss(rgb, target, depth_labels)
    else:
        terms['geometry_wall'], metrics['geometry_wall'] = geometry_wall_loss(rgb, target, depth_labels)

    image_loss = sum(terms.values())
    labels = source_five_role_labels(batch, device=rgb.device)
    roles, balances, reports = [], [], []
    for index, aux in enumerate(routers):
        role, balance, report = five_token_role_loss(aux['router_logits'], aux['gates'], labels)
        roles.append(role); balances.append(balance)
        reports.append(dict(report, euler_index=index, euler_time=index / 3))
    role, balance = torch.stack(roles).mean(), torch.stack(balances).mean()
    total = image_loss + weights['router_role'] * role + weights['router_balance'] * balance
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite RF-codec Source objective')
    metrics.update({key: float(value.detach()) for key, value in raw.items()})
    metrics.update(protocol=PROTOCOL, objective_protocol=PROTOCOL,
        train_loss=float(total.detach()), renderer_rgb_loss=float(image_loss.detach()),
        weighted_terms={key: float(value.detach()) for key, value in terms.items()},
        renderer_moe_role_loss=float(role.detach()), renderer_moe_balance_loss=float(balance.detach()),
        renderer_moe_role_weight=weights['router_role'], renderer_moe_balance_weight=weights['router_balance'],
        renderer_moe_heads=dict(rf3=dict(role_names=list(ROLES), tokens=reports[0]['tokens'],
            gates_mean=[sum(row['gates_mean'][i] for row in reports) / 3 for i in range(5)],
            role_ce=sum(row['role_ce'] for row in reports) / 3,
            load_balance=sum(row['load_balance'] for row in reports) / 3)),
        rf_router_steps=reports, rf3_router_steps=reports, router_velocity_calls=3,
        router_reduction=SETTINGS['router_reduction'], solver='euler', solver_steps=3,
        objective_kind=kind, examples=len(base), global_ssim_active=kind == 'main',
        global_ssim_region_masked=False, source_latent_active=kind == 'main',
        source_latent_weight=weights['source_latent'] if kind == 'main' else 0.,
        checkpoint_velocity=checkpoint_velocity, base_loss_active=False,
        base_regression_loss_active=False, base_router_loss_active=False,
        base_frozen=True, codec_frozen=True, private_opacity_frozen=True,
        restorer_active=False, Source_Q_image_branch=False, GAN_active=False,
        source_future_feedback=False, Source_role_masks_are_loss_only=True,
        future_source_rgb_in_prior=False, future_source_state_in_prior=False,
        source_pose_aligned=True, flow_base_detached=True)
    return total, metrics
