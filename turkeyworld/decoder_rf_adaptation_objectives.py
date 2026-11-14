"""Decoder-only paired RF3 / Source-code adaptation at native32.

Generate both detached quantized code sets BEFORE enabling D gradients.  The
RF sampler's codec guard freezes its entire codec, so it must never be called
between enabling D and backward.  Source RGB is an encoder/reconstruction
label branch; it never conditions the RF endpoint.
"""
from copy import deepcopy
from contextlib import contextmanager
import random

import numpy as np
import torch

from .object_losses import balanced_edge_l1, decoded_ssim_loss
from .renderer_moe_objectives import moe_conditions
from .renderer_small_enemy_protocol import separate_region_loss, native_crop_loss
from .renderer_wall_supervision import source_wall_masks, wall_loss_terms


PROTOCOL = 'decoder_rf_source_paired_adaptation_v1'
TRAINABLE_PREFIXES = ('flow_renderer.codec.decoder.', 'flow_renderer.codec.post_quant_conv.')
SETTINGS = dict(name=PROTOCOL, version=1, native_size=32, solver='euler', solver_steps=3,
    branch_weights=dict(rf_quantized=.5, source_quantized=.5),
    weights=dict(region_l1=1., ssim=.2, region_edge=.1, native_crop_l1=.25,
                 native_crop_edge=.1, wall_boundary_l1=.2),
    foreground_background_weights=dict(foreground=.5, background=.5),
    empty_foreground_excluded_from_foreground_denominator=True,
    decoder_examples_per_Source_example=2,
    trainable_prefixes=list(TRAINABLE_PREFIXES),
    RF_Base_world_private_geometry_encoder_quant_conv_codebook_frozen=True,
    quantizer_eval=True, RF_codes='Q(denormalize(RF3_endpoint))_detached',
    Source_codes='frozen_Q(frozen_E(Source32))_detached',
    Source_RGB_into_RF_conditioning=False, Source_geometry_and_masks_are_loss_labels_only=True,
    future_Source_feedback=False, GAN_active=False,
    code_generation_before_decoder_enable=True, code_generation_RNG_preserved=True)


def decoder_rf_adaptation_protocol():
    return deepcopy(SETTINGS)


def _codec(model):
    flow = getattr(model, 'flow_renderer', None)
    codec = getattr(flow, 'codec', None)
    if (flow is None or getattr(flow, 'kind', None) != 'student'
            or getattr(flow, 'mode', None) != 'rectified_flow'
            or type(getattr(flow, 'steps', None)) is not int or flow.steps != 3
            or codec is None or codec.config.image_size != 32
            or not bool(flow.codec_loaded) or not bool(codec.quantize.initialized)
            or not isinstance(getattr(model, 'base_student', None), torch.nn.Module)
            or not isinstance(getattr(codec, 'decoder', None), torch.nn.Module)
            or not isinstance(getattr(codec, 'post_quant_conv', None), torch.nn.Module)):
        raise ValueError('Decoder adaptation requires an initialized compact codec and native32 RF3 student')
    return codec


@contextmanager
def _preserved_rng(model):
    """Code generation must not change the caller's optimizer/sampler draw clock."""
    numpy, python = np.random.get_state(), random.getstate()
    device = next(model.parameters()).device
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == 'cuda' else [])
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        np.random.set_state(numpy)
        random.setstate(python)


def configure_decoder_adaptation(model):
    """Freeze every existing asset, then enable ONLY D and post-quantization conv.

    No optimizer, existing gradient, parameter, buffer or random state changes.
    Call again after each code-generation phase, never via flow.train().
    """
    codec = _codec(model)
    model.eval().requires_grad_(False)
    codec.decoder.train().requires_grad_(True)
    codec.post_quant_conv.train().requires_grad_(True)
    wanted = {name for name, _ in model.named_parameters() if name.startswith(TRAINABLE_PREFIXES)}
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not wanted or actual != wanted:
        raise ValueError('Decoder adaptation enabled parameters outside D/post_quant_conv')
    return dict(protocol=PROTOCOL, trainable_parameter_names=sorted(actual),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_parameter_tensors=len(actual), encoder_frozen=True, codebook_frozen=True,
        RF_frozen=True, Base_frozen=True, private_geometry_frozen=True)


def generate_decoder_training_codes(model, batch):
    """Return (RF quantized codes, Source quantized codes, detached conditions).

    Both code sets correspond to exactly the same B examples.  This function
    leaves the complete model frozen/eval, including D; the caller may enable
    D only once all RF/E/Q forwards are complete.  It performs no D forward.
    """
    codec = _codec(model)
    model.eval().requires_grad_(False)
    flow = model.flow_renderer
    with _preserved_rng(model), torch.no_grad():
        conditions = moe_conditions(model, batch)
        base = model.base_student(conditions['state'], conditions['features'], conditions['geometry'])
        endpoint = flow.latent_sample(base, conditions['state'], noise=flow.fixed_noise,
            steps=flow.steps, checkpoint_velocity=False, map_condition=conditions['map'])
        codes_rf = codec.quantize(flow.denormalize_latents(endpoint))['quantized'].detach()
        codes_source = codec.encode(conditions['target'])['quantized'].detach()
    _validate_codes(codec, codes_rf, codes_source, conditions)
    return codes_rf, codes_source, conditions


def _validate_codes(codec, codes_rf, codes_source, conditions):
    if not isinstance(conditions, dict) or not {'target', 'object_mask', 'map', 'state'} <= conditions.keys():
        raise ValueError('Prepared decoder codes need detached aligned native Source labels')
    target, mask = conditions['target'], conditions['object_mask']
    if (not torch.is_tensor(target) or target.ndim != 4 or target.shape[1:] != (3, 32, 32)
            or len(target) < 1 or not torch.is_tensor(mask) or mask.shape != (len(target), 1, 32, 32)
            or not all(bool(torch.isfinite(v).all()) for v in (target, mask))
            or any(bool(((v < 0) | (v > 1)).any()) for v in (target, mask))):
        raise ValueError('Prepared decoder Source RGB/mask must be finite native32 values in [0,1]')
    shape = (len(target), codec.config.embedding_dim, codec.config.latent_size, codec.config.latent_size)
    device = codec.post_quant_conv.weight.device
    if any(not torch.is_tensor(value) or tuple(value.shape) != shape or not value.is_floating_point()
           or value.device != device or value.requires_grad or value.grad_fn is not None
           or not bool(torch.isfinite(value).all()) for value in (codes_rf, codes_source)):
        raise ValueError('Prepared RF/Source codes must be finite detached quantized embeddings with identical batch shape')
    if target.device != device or mask.device != device:
        raise ValueError('Prepared decoder labels and codec must share a device')


def _branch_loss(prediction, target, mask, wall_labels):
    regions = separate_region_loss(prediction, target, mask)
    ssim = decoded_ssim_loss(prediction, target, mask)
    edge = balanced_edge_l1(prediction, target, mask)
    crop, crop_metrics = native_crop_loss(prediction, target, mask)
    wall = wall_loss_terms(prediction, target, wall_labels)
    from .visual_quality import image_quality_metrics
    with torch.no_grad():
        image_metrics = image_quality_metrics(prediction.detach(), target.detach())
    terms = dict(region_l1=regions['loss'], ssim=.2 * ssim, region_edge=.1 * edge,
                 native_crop=crop, wall_boundary=.2 * wall['boundary_l1'])
    total = sum(terms.values())
    metrics = dict(loss=float(total.detach()), foreground_l1=float(regions['foreground'].detach()),
        background_l1=float(regions['background'].detach()),
        foreground_images=regions['foreground_images'], background_images=regions['background_images'],
        region_l1=float(regions['loss'].detach()), ssim_loss=float(ssim.detach()), edge_l1=float(edge.detach()),
        **{k: float(v.detach()) if torch.is_tensor(v) else v for k, v in crop_metrics.items()},
        **{k: float(v.detach()) if torch.is_tensor(v) else v for k, v in wall.items() if k != 'loss'},
        wall_boundary_error=float(wall['boundary_l1'].detach()),
        image_mae=float(image_metrics['mae'].mean()), image_ssim=float(image_metrics['ssim'].mean()),
        weighted_terms={name: float(value.detach()) for name, value in terms.items()})
    return total, metrics


def decoder_rf_adaptation_loss(model, batch=None, *, prepared=None):
    """Exact paired 50:50 RF-code and Source-code decoder reconstruction loss.

    Pass the tuple from generate_decoder_training_codes as ``prepared`` to
    avoid every RF/E/Q forward in the decoder training phase.  An ordinary
    batch call performs generation first, then enables D.  RGB supervision is
    Source-only, with independently normalized visible NPC/background regions.
    """
    codec = _codec(model)
    if (prepared is None) == (batch is None):
        raise ValueError('Pass exactly one Source batch or prepared decoder-code tuple')
    if prepared is None:
        prepared = generate_decoder_training_codes(model, batch)
    if not isinstance(prepared, (tuple, list)) or len(prepared) != 3:
        raise ValueError('Prepared decoder training requires RF codes, Source codes and conditions')
    codes_rf, codes_source, conditions = prepared
    _validate_codes(codec, codes_rf, codes_source, conditions)
    configure_decoder_adaptation(model)
    target, mask = conditions['target'].detach(), conditions['object_mask'].detach()
    wall_labels = source_wall_masks(conditions['map'], conditions['state'], mask)
    # Do not call flow, _check_codec, E or Q after this point: its guard would
    # silently disable the decoder gradients which are intentionally enabled.
    prediction_rf = codec.decode(codes_rf)
    prediction_source = codec.decode(codes_source)
    if any(value.shape != target.shape or not bool(torch.isfinite(value).all())
           for value in (prediction_rf, prediction_source)):
        raise FloatingPointError('Nonfinite or wrongly shaped decoder RF/Source reconstruction')
    loss_rf, report_rf = _branch_loss(prediction_rf, target, mask, wall_labels)
    loss_source, report_source = _branch_loss(prediction_source, target, mask, wall_labels)
    total = .5 * loss_rf + .5 * loss_source
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite paired RF/Source decoder-only objective')
    metrics = dict(protocol=PROTOCOL, objective_protocol=PROTOCOL, train_loss=float(total.detach()),
        decoder_rf_loss=float(loss_rf.detach()), decoder_source_loss=float(loss_source.detach()),
        branches=dict(rf_quantized=report_rf, source_quantized=report_source),
        branch_weights=deepcopy(SETTINGS['branch_weights']), solver='euler', solver_steps=3,
        Source_examples=len(target), decoder_examples=2 * len(target),
        RF_examples=len(target), Source_code_examples=len(target), paired_examples=True,
        decoder_only=True, encoder_frozen=True, quantizer_frozen=True, RF_frozen=True, Base_frozen=True,
        private_geometry_frozen=True, GAN_active=False, source_future_feedback=False,
        Source_RGB_into_RF_conditioning=False, Source_masks_and_geometry_are_loss_labels_only=True)
    return total, metrics
