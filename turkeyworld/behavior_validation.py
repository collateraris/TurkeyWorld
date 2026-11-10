"""Periodic held-out behavior measurements, isolated from training state/RNG."""
from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import torch

from .behavior_heads import evaluate_heads
from .training_pipeline import atomic_json
from .visual_quality import strict_render_precision


def behavior_validation_due(step, previous, protocol):
    """Catch a100-update boundary skipped by a checkpointed stop/resume."""
    return (previous.get('loss_protocol') != protocol
            or int(step)//100 > int(previous.get('step', -1))//100)


def behavior_summary(report, *, step, protocol):
    """Keep explicit denominators and distinguish event-enriched cohorts."""
    cohorts = report['cohorts']
    uniform = cohorts['uniform']
    attack = cohorts['attack']['binary']['attack_accepted']
    spawn = cohorts['spawn']['metrics']
    hits = cohorts['hit']['metrics']['hit_target_accuracy']
    trajectories = report['heldout_trajectory']
    last = trajectories['key_offsets']['16']
    return {'step': int(step), 'every': 100, 'loss_protocol': protocol,
        'source_held_out': True, 'future_source_feedback': False,
        'uniform_windows': uniform['windows'],
        'trajectory_windows': trajectories['count'], 'trajectory_horizon': 16,
        'player_xy_error_tiles_h16': last['player_xy_error_tiles'],
        'npc_xy_error_tiles_h16': last['npc_xy_error_tiles'],
        'windows_with_wall_collision': trajectories['windows_with_wall_collision'],
        'attack_positive_cases': attack['source_positives'],
        'attack_recall': attack['recall'], 'attack_precision': attack['precision'],
        'attack_f1': attack['f1'], 'hit_target_accuracy': hits,
        'spawn_tile_accuracy': spawn['spawn_decoded_top1_tile_accuracy'],
        'spawn_xy_error_tiles': spawn['spawn_decoded_xy_error_tiles'],
        'all_states_finite': trajectories['all_states_finite'],
        'all_memory_finite': trajectories['all_memory_finite_and_shape_valid'],
        'scope': 'Fixed short held-out trajectories and event-enriched one-step cases; not full requested-duration qualification'}


def evaluate_behavior(model, data, output_dir, *, step, protocol, context=2, seed=173):
    """Run the same fixed panels every100 updates without changing train state."""
    root = Path(output_dir)
    modes = [(module, module.training) for module in model.modules()]
    caches = [(module, name, getattr(module, name).detach().clone())
        for module in model.modules() for name in ('_spawn_walkable', '_map_condition')
        if torch.is_tensor(getattr(module, name, None))]
    numpy_state, python_state = np.random.get_state(), random.getstate()
    generator = getattr(data, 'rng', None)
    data_state = copy.deepcopy(generator.bit_generator.state) if isinstance(generator, np.random.Generator) else None
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), strict_render_precision():
            report = evaluate_heads(model, data, root, context=context, batch_size=4, seed=seed)
        summary = behavior_summary(report, step=step, protocol=protocol)
        summary['report'] = str((root/'heldout_head_validation.json').resolve())
        atomic_json(root/'summary.json', summary)
        return summary
    finally:
        for module, name, value in caches:
            setattr(module, name, value)
        for module, training in modes:
            module.training = training
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        if data_state is not None:
            generator.bit_generator.state = data_state
