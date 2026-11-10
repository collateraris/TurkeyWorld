"""Explicit eight-to-one actor warm start; the old20D loader is never faked.

All compatible generator tensors are copied exactly. Changed interfaces retain
the selected NPC and the old none class. Removing actors changes the function;
this is an honest warm start, not a claim of exact whole-world equivalence.
Optimizer/critic/EMA state is reset for the new data/schema and training run.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from .models import NeuralWorldModel, WorldModelConfig, STATE_SCHEMA
from .structured_state import STATE_DIM, STATE_SCHEMA_VERSION


def legacy_state_indices(selected_enemy=0):
    if not isinstance(selected_enemy, int) or isinstance(selected_enemy, bool) or not 0 <= selected_enemy < 8:
        raise ValueError('The retained legacy NPC must be0..7')
    return list(range(8))+list(range(8+10*selected_enemy, 18+10*selected_enemy))+[88, 89]


def _tensor_for_single(name, value, expected, selected_enemy):
    if value.shape == expected.shape:
        return value, 'unchanged'
    columns = legacy_state_indices(selected_enemy)
    if value.ndim == 2 and value.shape[1] == 90 and expected.shape == (value.shape[0], STATE_DIM):
        result = value[:, columns].clone()
        result[:, -2] /= 8  # old kills/8 input vs the new kills/1 input.
        return result, 'state_columns_and_kill_normalization'
    if name.endswith('render_condition.0.weight') and value.ndim == 2 and value.shape[1] == 90+128:
        result = value[:, columns+list(range(90, 218))].clone()
        result[:, STATE_DIM-2] /= 8
        return result, 'state_columns_kill_normalization_and_appearance_features'
    if name.endswith(('hit_target_head.2.weight', 'hit_target_head.2.bias')) and value.shape[0] == 9:
        return value[[selected_enemy, 8]], 'retained_target_and_none'
    if name.endswith(('attack_head.2.weight', 'attack_head.2.bias')) and value.shape[0] == 8:
        return value[selected_enemy:selected_enemy+1], 'retained_attack'
    if name.endswith(('spawn_head.2.weight', 'spawn_head.2.bias')) and value.shape[0] == 8*256:
        return value[selected_enemy*256:(selected_enemy+1)*256], 'retained_spawn_slot'
    if name.endswith(('delta_head.2.weight', 'delta_head.2.bias')) and value.shape[0] == 18:
        return value[[0, 1, 2+2*selected_enemy, 3+2*selected_enemy]], 'player_and_retained_npc_xy'
    if name.endswith(('slot_embedding', 'slot_embedding.weight', 'identity.weight')) and value.ndim == 2:
        if value.shape[0] == 9 and expected.shape[0] == 2:
            return value[[0, selected_enemy+1]], 'player_and_retained_npc_identity'
        if value.shape[0] == 8 and expected.shape[0] == 1:
            return value[selected_enemy:selected_enemy+1], 'retained_npc_identity'
    raise ValueError(f'No explicit safe single-enemy migration rule for {name}: {tuple(value.shape)}->{tuple(expected.shape)}')


def migrate_single_enemy_checkpoint(source, output, *, selected_enemy=0):
    """Write a new counter0 inference-generator artifact and return its report."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise ValueError('Single-enemy migration requires a new output checkpoint')
    columns = legacy_state_indices(selected_enemy)
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    old_config = dict(saved.get('config', {}))
    if saved.get('stage') != 'world' or old_config.get('architecture') != 'vjepa2_ac':
        raise ValueError('Single-enemy migration requires a trusted V-JEPA world checkpoint')
    old = saved['model']
    if not any(name.endswith('state_encoder.weight') and value.shape[1] == 90 for name, value in old.items()):
        raise ValueError('Expected a legacy90D eight-enemy checkpoint, not an already migrated model')
    config = dict(old_config, entity_slots=1, state_schema_version=STATE_SCHEMA_VERSION)
    with torch.random.fork_rng(devices=[]):
        model = NeuralWorldModel(WorldModelConfig(**config)).eval()
    mapped, retained, changed = {}, [], {}
    target = model.state_dict()
    if set(target) != set(old):
        raise ValueError(f'Source generator keys differ: missing={sorted(set(target)-set(old))}, extra={sorted(set(old)-set(target))}')
    for name, expected in target.items():
        value, rule = _tensor_for_single(name, old[name], expected, selected_enemy)
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f'Unexpected mapped shape/dtype for {name}')
        mapped[name] = value
        if rule == 'unchanged':
            retained.append(name)
        else:
            changed[name] = {'rule': rule, 'source_shape': list(old[name].shape), 'shape': list(value.shape)}
    model.load_state_dict(mapped, strict=True)
    copied = model.state_dict()
    if any(not torch.equal(copied[name], old[name]) for name in retained):
        raise RuntimeError('Compatible tensor changed during single-enemy migration')
    codec_keys = [name for name in copied if name.startswith('flow_renderer.codec.')]
    if any(not torch.equal(copied[name], old[name]) for name in codec_keys):
        raise RuntimeError('Frozen codec changed during actor-schema migration')
    source_step = int(saved.get('global_training_step', saved.get('training_steps', 0)))
    report = {'schema_version': STATE_SCHEMA_VERSION, 'entity_slots': 1, 'state_dim': STATE_DIM,
        'selected_legacy_enemy': selected_enemy, 'source_checkpoint': str(source), 'source_optimizer_step': source_step,
        'source_state_columns': columns, 'source_config_fingerprint': hashlib.sha256(
            json.dumps(old_config, sort_keys=True).encode()).hexdigest(),
        'generator_parameters': sum(parameter.numel() for parameter in model.parameters()),
        'compatible_tensor_count': len(retained), 'transformed_tensors': changed,
        'compatible_tensors_bit_exact': True, 'codec_tensors_bit_exact': True,
        'optimizer_reset': True, 'discriminator_reset': True, 'ema_reset': True,
        'whole_function_preserved': False, 'training_steps': 0, 'global_training_step': 0,
        'created_at': datetime.now(timezone.utc).isoformat(), 'initial_checkpoint': str(output)}
    payload = {'format_version': 2, 'stage': 'world', 'config': asdict(model.config), 'state_schema': STATE_SCHEMA,
        'model': copied, 'training_steps': 0, 'global_training_step': 0, 'pipeline_phase': 'staged_single_initial',
        'source_optimizer_step': source_step, 'single_enemy_migration': report,
        'initialization': {'mode': 'single_enemy_warm_start', 'pretrained_world_weights_loaded': True,
            'pretrained_renderer_weights_loaded': True, 'trained_codec_reused': True,
            'source_optimizer_step': source_step, 'source_checkpoint': str(source)},
        'quality_status': 'Unqualified new single-enemy model; no old visual or rollout gate transfers.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(output)
    return report
