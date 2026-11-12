"""Exact zero-residual checks in an isolated, matching inference context."""
from __future__ import annotations

import torch

from .renderer_enemy_objectives import enemy_conditions
from .renderer_refinement import refinement_forward
from .staged_validation import _isolated_validation


def verify_zero_pixel_residual(model, original_base, batch):
    """Require bitexact Base/final RGB, restoring modes, dtype and every RNG.

    Attention can choose different CUDA kernels for parameter gradient flags,
    even under no-grad/inference mode. Both heads use matching eval modes and
    frozen parameter flags; all original modes/flags resume afterwards.
    This check cannot consume the training prior's stochastic draw clock.
    """
    parameter=next(model.base_student.parameters())
    original_parameter=next(original_base.parameters())
    old_device,old_dtype=original_parameter.device,original_parameter.dtype
    modes=[(module,module.training)for module in original_base.modules()]
    gradient_flags=[(parameter,parameter.requires_grad)
                    for head in (original_base,model.base_student)for parameter in head.parameters()]
    missing=object()
    caches=[(module,name,getattr(module,name,missing))for module in model.modules()
            for name in ('_spawn_walkable','_map_condition')]
    try:
        original_base.to(device=parameter.device,dtype=parameter.dtype).eval()
        original_base.requires_grad_(False)
        model.base_student.requires_grad_(False)
        # Includes no-grad, matching eval modes/precision, all RNG restoration,
        # and restoration of the working model's previous child-module modes.
        with _isolated_validation(model,None,173):
            conditions=enemy_conditions(model,batch)
            old_base=original_base(conditions['state'],conditions['features'],conditions['geometry'])
            new_base,residual=model.base_student.forward_with_residual(
                conditions['state'],conditions['features'],conditions['geometry'])
            flow=model.flow_renderer
            endpoint=flow.latent_sample(old_base,conditions['state'],noise=flow.fixed_noise,map_condition=conditions['map'])
            old_rgb=flow.codec.decode(flow.codec.quantize(flow.denormalize_latents(endpoint))['quantized'])
            _,new_rgb,_=refinement_forward(model,conditions)
            base_equal=torch.equal(old_base,new_base)
            rgb_equal=torch.equal(old_rgb,new_rgb)
            residual_zero=residual is not None and not bool(residual.any())
            if not(base_equal and rgb_equal and residual_zero):
                raise AssertionError('Zero pixel residual changed the parent renderer function: '
                    f'base_equal={base_equal}, final_rgb_equal={rgb_equal}, residual_zero={residual_zero}')
            return {'base_rgb_bitexact_at_initialization':True,'final_rgb_bitexact_at_initialization':True,
                'comparison_mode':'eval','comparison_dtype':str(parameter.dtype),
                'comparison_parameter_gradients_enabled':False,
                'verification_rng_isolated':True,'training_module_modes_restored':True,
                'parameter_gradient_flags_restored':True}
    finally:
        original_base.to(device=old_device,dtype=old_dtype)
        for module,mode in modes:module.training=mode
        for parameter,enabled in gradient_flags:parameter.requires_grad_(enabled)
        for module,name,value in caches:
            if value is missing:
                if hasattr(module,name):delattr(module,name)
            else:setattr(module,name,value)
