"""Paired RF/Source-code supervision of an independent RGB restorer only.

The pretrained RF3, Base, complete codec (including D/post-quantization conv),
geometry and world are frozen. Source RGB/masks are reconstruction labels;
raw Source RGB never enters the RF or the conditional restorer's arguments.
"""
from copy import deepcopy

import torch

from .decoder_rf_adaptation_objectives import (
    _codec, _validate_codes, _branch_loss, generate_decoder_training_codes,
    decoder_rf_adaptation_protocol)
from .renderer_wall_supervision import source_wall_masks


PROTOCOL = 'conditional_rf_restorer_paired_v1'
TRAINABLE_PREFIXES = ('flow_renderer.conditional_restorer.',)


def conditional_restorer_protocol():
    previous = decoder_rf_adaptation_protocol()
    return dict(name=PROTOCOL, version=1, native_size=32, solver='euler', solver_steps=3,
        branch_weights=deepcopy(previous['branch_weights']), weights=deepcopy(previous['weights']),
        foreground_background_weights=deepcopy(previous['foreground_background_weights']),
        empty_foreground_excluded_from_foreground_denominator=True,
        trainable_prefixes=list(TRAINABLE_PREFIXES), only_conditional_restorer_trainable=True,
        RF_Base_world_private_geometry_full_codec_frozen=True, encoder_frozen=True,
        decoder_post_quant_conv_frozen=True, quantizer_eval=True,
        RF_codes='Q(denormalize(RF3_endpoint))_detached',
        Source_codes='frozen_Q(frozen_E(Source32))_detached',
        restorer_inputs=['quantized_codes', 'frozen_decoded_RGB32', 'state20', 'raw_map16x16'],
        Source_RGB_direct_restorer_input=False, Source_RGB_into_RF_conditioning=False,
        Source_code_branch_is_frozen_E_Q_reconstruction_supervision=True,
        Source_geometry_and_masks_are_loss_labels_only=True, future_Source_feedback=False,
        decoder_examples_per_Source_example=2, GAN_active=False,
        code_generation_before_restorer_enable=True)


def _restorer(model):
    codec = _codec(model)
    module = getattr(model.flow_renderer, 'conditional_restorer', None)
    if not isinstance(module, torch.nn.Module) or not list(module.parameters()):
        raise ValueError('Conditional restoration requires its explicit learned flow_renderer.conditional_restorer')
    return codec, module


def configure_conditional_restorer(model):
    """Enable the independent restorer, preserving every pretrained tensor."""
    codec, restorer = _restorer(model)
    model.eval().requires_grad_(False)
    restorer.train().requires_grad_(True)
    expected = {name for name, _ in model.named_parameters() if name.startswith(TRAINABLE_PREFIXES)}
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not expected or expected != actual or any(p.requires_grad for p in codec.parameters()):
        raise ValueError('Conditional restoration enabled parameters outside the new restorer')
    return dict(protocol=PROTOCOL, trainable_parameter_names=sorted(actual),
        trainable_parameters=sum(p.numel() for p in restorer.parameters()),
        trainable_parameter_tensors=len(actual), only_conditional_restorer_trainable=True,
        encoder_frozen=True, quantizer_frozen=True, decoder_frozen=True,
        post_quant_conv_frozen=True, RF_frozen=True, Base_frozen=True,
        private_geometry_frozen=True, world_frozen=True)


def generate_conditional_restorer_inputs(model, batch):
    """The same detached, paired codes used by the decoder adaptation phase."""
    _restorer(model)
    return generate_decoder_training_codes(model, batch)


def conditional_restorer_predictions(model, prepared):
    """Return old/new RF/Source images; no target/mask enters the restorer.

    This prepared path performs no RF, encoder or quantizer forward. Frozen D
    decodes before enabling the restorer; gradients cross only its two calls.
    """
    codec, restorer = _restorer(model)
    if not isinstance(prepared, (tuple, list)) or len(prepared) != 3:
        raise ValueError('Prepared restoration needs RF codes, Source codes and aligned conditions')
    codes_rf, codes_source, conditions = prepared
    _validate_codes(codec, codes_rf, codes_source, conditions)
    configure_conditional_restorer(model)
    state = conditions['state'].detach()
    grid = conditions['map'].detach()
    if state.shape != (len(codes_rf), 20) or grid.shape != (len(codes_rf), 16, 16):
        raise ValueError('Conditional restoration needs aligned state20 and raw map16x16')
    with torch.no_grad():
        rgb_rf = codec.decode(codes_rf).detach()
        rgb_source = codec.decode(codes_source).detach()
    result = dict(rf_old=rgb_rf, source_old=rgb_source,
        rf_new=restorer(codes_rf.detach(), rgb_rf, state, grid),
        source_new=restorer(codes_source.detach(), rgb_source, state, grid))
    shape = conditions['target'].shape
    if any(value.shape != shape or not bool(torch.isfinite(value).all()) for value in result.values()):
        raise FloatingPointError('Nonfinite or incorrectly shaped native32 conditional restoration')
    return result


def conditional_restorer_loss(model, batch=None, *, prepared=None):
    """Exact 50:50 RF-Q / Source-EQ image objective, updating the restorer only."""
    if (prepared is None) == (batch is None):
        raise ValueError('Pass exactly one Source batch or prepared conditional-restorer tuple')
    if prepared is None:
        prepared = generate_conditional_restorer_inputs(model, batch)
    predictions = conditional_restorer_predictions(model, prepared)
    conditions = prepared[2]
    target, mask = conditions['target'].detach(), conditions['object_mask'].detach()
    labels = source_wall_masks(conditions['map'], conditions['state'], mask)
    rf_loss, rf_metrics = _branch_loss(predictions['rf_new'], target, mask, labels)
    source_loss, source_metrics = _branch_loss(predictions['source_new'], target, mask, labels)
    total = .5 * rf_loss + .5 * source_loss
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite conditional-restorer objective')
    return total, dict(protocol=PROTOCOL, objective_protocol=PROTOCOL, train_loss=float(total.detach()),
        restorer_rf_loss=float(rf_loss.detach()), restorer_source_loss=float(source_loss.detach()),
        branches=dict(rf_quantized=rf_metrics, source_quantized=source_metrics),
        branch_weights=dict(rf_quantized=.5, source_quantized=.5),
        solver='euler', solver_steps=3, Source_examples=len(target),
        restorer_examples=2 * len(target), decoder_examples=2 * len(target), paired_examples=True,
        only_conditional_restorer_trainable=True, full_codec_frozen=True, encoder_frozen=True,
        quantizer_frozen=True, decoder_frozen=True, post_quant_conv_frozen=True,
        RF_frozen=True, Base_frozen=True, private_geometry_frozen=True, world_frozen=True,
        Source_RGB_direct_restorer_input=False, Source_RGB_into_RF_conditioning=False,
        Source_masks_and_geometry_are_loss_labels_only=True, source_future_feedback=False, GAN_active=False)
