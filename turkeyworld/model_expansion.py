"""Function-preserving teacher growth by a zero-initialized velocity residual.

The trained flow backbone and world/renderer weights retain their exact names,
shapes and values. The additional conditional Transformer learns a residual
velocity. Its zero output projection initially leaves every generated frame
unchanged, including the complete Heun trajectory and cached runtime noise.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from .models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint


def expand_teacher_model(model, *, adapter_width=512, adapter_depth=12, adapter_heads=8,
                         adapter_mlp_ratio=4):
    """Return a grown model and a verified weight-preservation report.

    No optimizer is reused by numeric parameter position: adding modules makes
    old unnamed optimizer IDs ambiguous. Learned weights are preserved exactly;
    the caller starts a fresh optimizer or explicitly migrates moments by name.
    """
    config = asdict(model.config)
    if config['kind'] != 'teacher' or config['renderer_mode'] != 'flow_matching':
        raise ValueError("Only a Flow Matching teacher can use this expansion")
    if config.get('flow_adapter_width', 0):
        raise ValueError("Teacher already has a residual adapter; reuse its expanded checkpoint")
    config.update(flow_adapter_width=adapter_width, flow_adapter_depth=adapter_depth,
                  flow_adapter_heads=adapter_heads, flow_adapter_mlp_ratio=adapter_mlp_ratio)
    first = next(model.parameters())
    expanded = NeuralWorldModel(WorldModelConfig(**config)).to(device=first.device, dtype=first.dtype)
    original = model.state_dict()
    copied = expanded.load_state_dict(original, strict=False)
    prefix = 'flow_renderer.residual_adapter.'
    if copied.unexpected_keys or any(not key.startswith(prefix) for key in copied.missing_keys):
        raise ValueError(f"Unexpected checkpoint migration mismatch: {copied}")
    if not copied.missing_keys:
        raise ValueError("Expansion must add a real conditional residual network")
    new_state = expanded.state_dict()
    if any(not torch.equal(value, new_state[key]) for key, value in original.items()):
        raise RuntimeError("An existing trained tensor changed during expansion")
    adapter = expanded.flow_renderer.residual_adapter
    if bool(torch.count_nonzero(adapter.output.weight)) or bool(torch.count_nonzero(adapter.output.bias)):
        raise RuntimeError("Residual output must be exactly zero at expansion")
    old_parameters = dict(model.named_parameters())
    for name, parameter in expanded.named_parameters():
        if name in old_parameters:
            parameter.requires_grad_(old_parameters[name].requires_grad)
    expanded.train(model.training)
    old_modules = dict(model.named_modules())
    for name, module in expanded.named_modules():
        if name in old_modules:
            module.training = old_modules[name].training
    adapter.train(model.flow_renderer.training)
    total = sum(parameter.numel() for parameter in expanded.parameters())
    added = sum(parameter.numel() for parameter in adapter.parameters())
    report = {'method': 'zero_initialized_conditional_velocity_residual',
              'generator_parameters_before': sum(parameter.numel() for parameter in model.parameters()),
              'generator_parameters_after': total, 'added_parameters': added,
              'discriminator_parameters_included': False,
              'adapter': {'width': adapter_width, 'depth': adapter_depth,
                          'heads': adapter_heads, 'mlp_ratio': adapter_mlp_ratio},
              'copied_tensor_count': len(original), 'new_tensor_count': len(copied.missing_keys),
              'all_original_tensors_exact': True, 'residual_output_initially_zero': True,
              'initial_function_preserved': True, 'optimizer_reset_required': True}
    return expanded, report


def _file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def expand_teacher_checkpoint(source, output, *, adapter_width=512, adapter_depth=12,
                              adapter_heads=8, adapter_mlp_ratio=4):
    """Write a separate CPU-portable warm-start checkpoint, preserving the source.

    This artifact contains expanded model weights and the original training
    progress. It deliberately omits unnamed optimizer state and previous visual
    pass streaks. New optimization/gating starts in a distinct experiment phase.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Expansion output must differ from the trained source checkpoint")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an expansion artifact: {output}")
    model, checkpoint = load_world_checkpoint(source, 'cpu')
    expanded, report = expand_teacher_model(model, adapter_width=adapter_width,
        adapter_depth=adapter_depth, adapter_heads=adapter_heads, adapter_mlp_ratio=adapter_mlp_ratio)
    report.update(source_checkpoint=str(source), source_checkpoint_sha256=_file_hash(source),
                  source_training_steps=int(checkpoint.get('training_steps', 0)),
                  output_checkpoint=str(output), created_at=datetime.now(timezone.utc).isoformat())
    excluded = {'model', 'optimizer', 'teacher_quality_gate'}
    payload = {key: value for key, value in checkpoint.items() if key not in excluded}
    payload.update(config=asdict(expanded.config), model=expanded.state_dict(), stage='world',
                   pipeline_phase='teacher_expanded_warm_start', expansion=report,
                   optimizer_reset_required=True,
                   quality_status='Trained function preserved at expansion; new visual/adversarial and gameplay validation pending.')
    if 'teacher_quality_gate' in checkpoint:
        payload['source_teacher_quality_gate'] = checkpoint['teacher_quality_gate']
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+f'.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    for attempt in range(40):
        try:
            os.replace(temporary, output)
            return report
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter-width', type=int, default=512)
    parser.add_argument('--adapter-depth', type=int, default=12)
    parser.add_argument('--adapter-heads', type=int, default=8)
    args = parser.parse_args(argv)
    print(json.dumps(expand_teacher_checkpoint(args.source, args.output,
        adapter_width=args.adapter_width, adapter_depth=args.adapter_depth,
        adapter_heads=args.adapter_heads), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
