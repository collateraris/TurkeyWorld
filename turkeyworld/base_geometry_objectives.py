"""Raw compact Base geometry training; no RF forward or restorer path."""
from copy import deepcopy

import torch

from .base_expanded_experts import base_expanded_expert_metadata
from .object_losses import balanced_edge_l1, decoded_ssim_loss
from .renderer_far_wall_protocol import far_wall_loss, far_wall_protocol
from .renderer_geometry_objectives import geometry_wall_loss, geometry_wall_protocol
from .renderer_moe_five_objectives import source_five_role_labels, five_token_role_loss
from .renderer_moe_five_phase import ROLES
from .renderer_moe_objectives import moe_conditions
from .renderer_small_enemy_protocol import separate_region_loss, native_crop_loss
from .renderer_wall_depth import source_wall_depth_labels


PROTOCOL = 'base_expanded_geometry_v1'
SETTINGS = dict(name=PROTOCOL, version=1, inference_path='compact_Base_raw_RGB32',
    weights=dict(global_ssim=1., region_l1=.5, region_edge=.1, native_crop_l1=.25,
        native_crop_edge=.1, geometry_wall=1., source_latent=.1, router_role=.01, router_balance=.001),
    outer_stream_weights=dict(main=1., far=.25, geometry=1.),
    RF_forward_active=False, Base_only_trainable=True, full_codec_frozen=True,
    private_opacity_frozen=True, world_frozen=True, restorer_active=False,
    frozen_encoder_input_gradient_allowed=True,
    latent_alignment='Q(E(Base_raw_RGB)) versus detached Q(E(Source32)); unchanged normalization',
    Source_Q_image_branch=False, Source_RGB_into_Base_conditioning=False,
    Source_labels_only=True, Source_future_feedback=False, GAN_active=False)


def base_geometry_protocol():
    result = deepcopy(SETTINGS)
    result.update(geometry_wall_protocol=geometry_wall_protocol(), far_wall_protocol=far_wall_protocol())
    return result


def _validate(model, *, expanded=False):
    flow, base = model.flow_renderer, model.base_student
    codec = flow.codec
    if (getattr(flow, 'conditional_restorer', None) is not None
            or flow.training or codec.training or codec.quantize.training
            or any(p.requires_grad for name, p in model.named_parameters() if not name.startswith('base_student.'))
            or any(p.requires_grad for p in flow.parameters())
            or not callable(getattr(getattr(base, 'output_moe', None), 'pop_diagnostics', None))):
        raise ValueError('Base geometry training requires frozen eval RF/codec/world/private and no restorer')
    if expanded:
        base_expanded_expert_metadata(base)


def base_geometry_forward(model, conditions):
    """Return raw RGB and its actual live router graph, without reading labels.

    The same forward helper can measure the frozen old five-expert Base; only
    the training objective below requires the opt-in expanded architecture.
    """
    _validate(model)
    module = model.base_student.output_moe
    module.pop_diagnostics()
    rgb = model.base_student(conditions['state'], conditions['features'], conditions['geometry'])
    auxiliary = module.pop_diagnostics()
    if (rgb.shape != (len(conditions['state']), 3, 32, 32)
            or not bool(torch.isfinite(rgb).all()) or not isinstance(auxiliary, dict)
            or tuple(auxiliary.get('role_names', ())) != tuple(ROLES)):
        raise ValueError('Base geometry needs native32 RGB and the actual five-role router graph')
    return rgb, auxiliary


def base_geometry_refinement_loss(model, batch, *, kind='main'):
    if kind not in ('main', 'far', 'geometry'):
        raise ValueError('Unknown Base geometry Source objective kind')
    _validate(model, expanded=True)
    conditions = moe_conditions(model, batch)
    rgb, auxiliary = base_geometry_forward(model, conditions)
    target, mask = conditions['target'].detach(), conditions['object_mask'].detach()
    labels = source_wall_depth_labels(conditions['map'], conditions['state'], mask)
    weights = SETTINGS['weights']
    terms, raw, metrics = {}, {}, {}
    if kind == 'main':
        regions = separate_region_loss(rgb, target, mask)
        raw['global_ssim_loss'] = decoded_ssim_loss(rgb, target, mask=None)
        raw['region_l1'] = regions['loss']
        raw['region_edge_l1'] = balanced_edge_l1(rgb, target, mask)
        crop, crop_metrics = native_crop_loss(rgb, target, mask)
        geometry, geometry_metrics = geometry_wall_loss(rgb, target, labels)
        codec, flow = model.flow_renderer.codec, model.flow_renderer
        # Frozen E/Q weights still propagate the INPUT gradient into Base.
        prediction_z = flow.normalize_latents(codec.encode(rgb)['quantized'])
        with torch.no_grad():
            target_z = flow.normalize_latents(codec.encode(target)['quantized']).detach()
        raw['source_latent_mse'] = separate_region_loss(prediction_z, target_z, mask, squared=True)['loss']
        terms.update(global_ssim=raw['global_ssim_loss'], region_l1=.5*raw['region_l1'],
            region_edge=.1*raw['region_edge_l1'], native_crop=crop, geometry_wall=geometry,
            source_latent=.1*raw['source_latent_mse'])
        metrics.update(base_foreground_l1=float(regions['foreground'].detach()),
            base_background_l1=float(regions['background'].detach()),
            base_foreground_images=regions['foreground_images'], base_background_images=regions['background_images'],
            native_crop={key: float(value.detach()) if torch.is_tensor(value) else value
                         for key, value in crop_metrics.items()}, geometry_wall=geometry_metrics)
    elif kind == 'far':
        terms['far_wall'], metrics['far_wall'] = far_wall_loss(rgb, target, labels)
    else:
        terms['geometry_wall'], metrics['geometry_wall'] = geometry_wall_loss(rgb, target, labels)
    image_loss = sum(terms.values())
    role_labels = source_five_role_labels(batch, device=rgb.device)
    role, balance, report = five_token_role_loss(auxiliary['router_logits'], auxiliary['gates'], role_labels)
    total = image_loss + weights['router_role']*role + weights['router_balance']*balance
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite Base-only Source geometry objective')
    metrics.update({key: float(value.detach()) for key, value in raw.items()})
    metrics.update(protocol=PROTOCOL, objective_protocol=PROTOCOL, objective_kind=kind,
        train_loss=float(total.detach()), renderer_rgb_loss=float(image_loss.detach()),
        weighted_terms={key: float(value.detach()) for key, value in terms.items()},
        renderer_moe_role_loss=float(role.detach()), renderer_moe_balance_loss=float(balance.detach()),
        renderer_moe_role_weight=weights['router_role'], renderer_moe_balance_weight=weights['router_balance'],
        renderer_moe_heads=dict(base=dict(report, role_names=list(ROLES))),
        examples=len(rgb), global_ssim_active=kind=='main', global_ssim_region_masked=False,
        source_latent_active=kind=='main', source_latent_weight=.1 if kind=='main' else 0.,
        Base_only_trainable=True, RF_forward_active=False, RF_frozen=True, codec_frozen=True,
        private_opacity_frozen=True, world_frozen=True, restorer_active=False,
        frozen_encoder_input_gradient_allowed=True, Source_Q_image_branch=False,
        source_future_feedback=False, Source_role_masks_are_loss_only=True,
        future_source_rgb_in_prior=False, future_source_state_in_prior=False, GAN_active=False)
    return total, metrics
