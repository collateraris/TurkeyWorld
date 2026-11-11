"""Fixed Source-pose action tests and closed-loop450 navigation measurements.

Only initial observations and future controls enter the model. Source future
states are labels; they never repair a prediction or latent memory. Default
spawn regressions are reported separately from held-out pose generalization.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch

from .behavior_trajectory import evaluate_state_trajectories
from .navigation_recovery_data import digest, fingerprint, source_axis_labels
from .runtime_checks import circle_is_clear, segment_is_clear, PLAYER_RADIUS
from .staged_validation import _isolated_validation
from .training_pipeline import atomic_json


THRESHOLDS = {'short_velocity_error_mean_max': .35, 'false_stop_rate_max': .05,
              'missed_block_rate_max': .05, 'long_final_xy_mean_max': .10,
              'long_final_xy_p95_max': .25, 'long_wall_steps_max': 0}


def _stats(values):
    values = np.asarray(values, float).reshape(-1)
    return {'n': int(values.size), 'mean': float(values.mean()) if values.size else None,
            'p95': float(np.percentile(values, 95)) if values.size else None,
            'max': float(values.max()) if values.size else None}


def _data(value, model):
    if hasattr(value, 'batch'):
        return value
    from .staged_curriculum import StagedSequenceData
    return StagedSequenceData(Path(value), str(next(model.parameters()).device), mode='stream', cache_bytes=32*1024**2)


def _short(model, data, *, context, count, batch_size, seed, default_only=False):
    if data.split != ('train' if default_only else 'val'):
        raise ValueError('Action tests require held-out val; default regression explicitly uses train')
    eligible = []
    for index, episode in enumerate(data.episodes):
        path = Path(episode['path'])
        with np.load(path, allow_pickle=False) as source:
            if default_only and str(source['initialization_source']) != 'default_spawn':
                continue
        eligible.append(index)
    if not eligible:
        raise ValueError('No eligible action-response episodes')
    if default_only:
        selected = eligible
    else:
        if len(eligible) < count:
            raise ValueError('Not enough distinct held-out action-response episodes')
        selected = sorted(np.random.default_rng(seed).choice(eligible, count, replace=False).tolist())
    panel = [{'episode': Path(data.episodes[index]['path']).name, 'source_sha256': digest(data.episodes[index]['path']),
              'episode_index': index, 'start': 0} for index in selected]
    device = next(model.parameters()).device
    predicted, source, previous, controls, pass_predictions = [], [], [], [], []
    head_present = True
    with _isolated_validation(model, data, seed):
        for first in range(0, len(selected), batch_size):
            indices = np.asarray([[index, 0] for index in selected[first:first+batch_size]], np.int64)
            batch = data.batch(len(indices), context=context, horizon=1, indices=indices)
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            fmap = model.encode_map(batch['map'])
            state = batch['states'][:, context-1].float()
            memory = model.initialize_from_image(batch['context_frames'], batch['states'][:, :context], fmap)
            result = model.transition(state, batch['actions'][:, 0], fmap, memory)
            if not bool(torch.isfinite(result['state']).all()) or not bool(torch.isfinite(result['memory']).all()):
                raise FloatingPointError('Nonfinite action response or latent memory')
            # Targets are inspected only after the independent prediction.
            predicted.append(result['state'].float().cpu().numpy())
            source.append(batch['target_states'][:, 0].float().cpu().numpy())
            previous.append(state.cpu().numpy())
            controls.append(batch['actions'][:, 0].float().cpu().numpy())
            if 'player_pass_logits' in result:
                pass_predictions.append(result['player_pass_logits'].float().sigmoid().cpu().numpy())
            else:
                head_present = False
    pred, target, before, actions = map(np.concatenate, (predicted, source, previous, controls))
    command, fraction, active = [], [], []
    for old, new, action in zip(before, target, actions):
        cmd, frac, mask, _ = source_axis_labels(np.stack([old, new]), action[None], 1/model.config.simulation_hz)
        command.append(cmd[0]); fraction.append(frac[0]); active.append(mask[0])
    command, fraction, active = np.stack(command), np.stack(fraction), np.stack(active)
    actual, estimated = (target[:, :2]-before[:, :2])*16, (pred[:, :2]-before[:, :2])*16
    free, blocked = active & (fraction >= .95), active & (np.abs(fraction) <= .05)
    partial = active & ~free & ~blocked
    sliding = free.any(-1) & blocked.any(-1)
    false_stop = free & (np.abs(estimated) < .2*np.abs(command))
    missed_block = blocked & (np.abs(estimated) > .2*np.abs(command))
    dt = 1/model.config.simulation_hz
    counts = {'samples': len(pred), 'active_axes': int(active.sum()), 'free_axes': int(free.sum()),
              'blocked_axes': int(blocked.sum()), 'partial_axes': int(partial.sum()),
              'sliding_samples': int(sliding.sum()),
              'false_stop_axes': int(false_stop.sum()), 'missed_block_axes': int(missed_block.sum())}
    metrics = {'velocity_error_tiles_s': _stats(np.linalg.norm(estimated-actual, axis=-1)/dt),
               'free_axis_velocity_error_tiles_s': _stats(np.abs(estimated-actual)[free]/dt),
               'blocked_axis_velocity_error_tiles_s': _stats(np.abs(estimated-actual)[blocked]/dt),
               'partial_axis_velocity_error_tiles_s': _stats(np.abs(estimated-actual)[partial]/dt),
               'sliding_velocity_error_tiles_s': _stats(np.linalg.norm(estimated-actual, axis=-1)[sliding]/dt),
               'false_stop_rate': float(false_stop.sum()/free.sum()) if free.any() else None,
               'missed_block_rate': float(missed_block.sum()/blocked.sum()) if blocked.any() else None}
    if head_present:
        pass_pred = np.concatenate(pass_predictions)
        metrics['pass_fraction_absolute_error'] = _stats(np.abs(pass_pred-fraction)[active])
    else:
        pass_pred = np.full_like(fraction, np.nan)
        metrics['pass_fraction_absolute_error'] = None
    groups = {'positive_turn': actions[:, 2] > 0, 'negative_turn': actions[:, 2] < 0,
              'moving_turn': (np.linalg.norm(actions[:, :2], axis=-1) > 0) & (np.abs(actions[:, 2]) > 0),
              'idle': np.abs(actions[:, :3]).sum(-1) == 0}
    cohorts = {name: {'samples': int(mask.sum()), 'velocity_error_tiles_s': _stats(
                    np.linalg.norm(estimated-actual, axis=-1)[mask]/dt)} for name, mask in groups.items()}
    # Source checks whole X then Y steps: a slide usually has one free and one
    # blocked axis, not a fractional progress target inside an individual axis.
    coverage = all(counts[key] > 0 for key in ('free_axes', 'blocked_axes', 'sliding_samples')) and all(
        cohorts[key]['samples'] > 0 for key in ('positive_turn', 'negative_turn', 'moving_turn', 'idle'))
    rates_ok = ((metrics['false_stop_rate'] is None or metrics['false_stop_rate'] <= THRESHOLDS['false_stop_rate_max'])
                and (metrics['missed_block_rate'] is None or metrics['missed_block_rate'] <= THRESHOLDS['missed_block_rate_max']))
    passed = ((coverage or default_only)
              and metrics['velocity_error_tiles_s']['mean'] <= THRESHOLDS['short_velocity_error_mean_max'] and rates_ok)
    report = {'passed': bool(passed), 'source_held_out': not default_only, 'default_spawn_regression': default_only,
        'context': context, 'initial_window_start': 0, 'horizon': 1, 'counts': counts, 'metrics': metrics,
        'cohorts': cohorts, 'coverage_complete': coverage, 'axis_fraction_head_present': head_present,
        'source_future_feedback': False, 'future_rgb_encoded': False,
        'panel': panel, 'fingerprint': fingerprint({'panel': panel, 'context': context, 'seed': seed})}
    arrays = {'predicted': pred, 'reference': target, 'previous': before, 'actions': actions,
              'axis_command_tiles': command, 'axis_pass_target': fraction, 'axis_active': active,
              'axis_pass_prediction': pass_pred}
    return report, arrays


def _long_summary(predicted, reference, maps):
    errors = np.linalg.norm((predicted[:, 1:, :2]-reference[:, 1:, :2])*16, axis=-1)
    walls, paths, reference_walls = [], [], []
    for row, grid in enumerate(maps):
        for offset in range(1, predicted.shape[1]):
            walls.append(not circle_is_clear(grid, predicted[row, offset, :2]*16, PLAYER_RADIUS))
            paths.append(not segment_is_clear(grid, predicted[row, offset-1, :2]*16,
                                               predicted[row, offset, :2]*16, PLAYER_RADIUS))
            reference_walls.append(not circle_is_clear(grid, reference[row, offset, :2]*16, PLAYER_RADIUS))
    return {'final_player_xy_error_tiles': _stats(errors[:, -1]), 'all_steps_player_xy_error_tiles': _stats(errors),
            'model_wall_steps': int(sum(walls)), 'model_wall_path_steps': int(sum(paths)),
            'source_wall_steps': int(sum(reference_walls)), 'windows': len(predicted), 'horizon': predicted.shape[1]-1,
            'all_states_finite': bool(np.isfinite(predicted).all()),
            'npc_active_steps': int(((predicted[:, 1:, 12] > 0) | (predicted[:, 1:, 13] >= .5)).sum()),
            'source_future_feedback': False, 'scope': 'Player-only no-enemy450-step closed loop'}


def evaluate_prior_navigation_tape(model, path, output_dir, *, seed=173):
    """Re-run exactly the prior control tape, encoding only its initial image."""
    from .engine import Game
    from types import SimpleNamespace
    with np.load(path, allow_pickle=False) as saved:
        reference, actions = saved['reference'].copy(), saved['actions'].copy()
        grid = saved['map'].copy() if 'map' in saved.files else saved['maps'][0].copy()
    if reference.ndim == 3:
        reference = reference[0]
    game = Game(seed=0, endless=True, enemies_enabled=False)
    if not np.array_equal(game.map, grid):
        raise ValueError('Prior navigation tape map differs from the current Source map')
    game.set_state_vector(reference[0])
    initial_rgb = game.render(size=model.config.image_size, overlays=False)
    del game
    device = next(model.parameters()).device
    with _isolated_validation(model, SimpleNamespace(), seed):
        fmap = model.encode_map(torch.tensor(grid, device=device, dtype=torch.long)[None])
        state = torch.tensor(reference[0], device=device, dtype=torch.float32)[None]
        image = torch.tensor(initial_rgb, device=device).permute(2, 0, 1)[None].float()/255
        memory = model.initialize_from_image(image, state, fmap)
        predicted = [state.cpu().numpy()[0].copy()]
        for control in actions:
            result = model.transition(state, state.new_tensor(control)[None], fmap, memory)
            state, memory = result['state'], result['memory']
            if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(memory).all()):
                raise FloatingPointError('Nonfinite autonomous prior-tape state or memory')
            predicted.append(state.cpu().numpy()[0].copy())
    predicted = np.stack(predicted)
    summary = _long_summary(predicted[None], reference[None], grid[None])
    summary.update(initial_context=1, initial_source_renders=1, future_source_renders=0,
                   controls_fingerprint=fingerprint(actions.tolist()), source_path=str(Path(path).resolve()))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'prior_tape_states.npz', predicted=predicted, reference=reference, actions=actions, map=grid)
    atomic_json(output/'prior_tape_validation.json', summary)
    return summary


def evaluate_motion_recovery(model, short_data, long_data, output_dir, *, context=1, count=128,
                             long_count=16, batch_size=16, seed=173,
                             default_regression_data=None, prior_tape_path=None):
    """Require expanded short action response AND new450-step held-out quality."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    short_data, long_data = _data(short_data, model), _data(long_data, model)
    if short_data.split != 'val' or long_data.split != 'val' or context not in (1, 2):
        raise ValueError('Recovery validation requires explicit val splits and one/two observed frames')
    short, arrays = _short(model, short_data, context=context, count=count, batch_size=batch_size, seed=seed)
    np.savez_compressed(output/'short_motion_states.npz', **arrays)
    atomic_json(output/'short_motion_validation.json', short)
    long = evaluate_state_trajectories(model, long_data, output/'long450', context=2,
        horizon=450, count=long_count, batch_size=min(batch_size, 4), seed=seed)
    with np.load(output/'long450'/'heldout_trajectory_states.npz', allow_pickle=False) as saved:
        summary = _long_summary(saved['predicted'], saved['reference'], saved['maps'])
    summary.update(fingerprint=long['fingerprint'], all_memory_finite=long['all_memory_finite_and_shape_valid'])
    long_passed = (summary['all_states_finite'] and summary['all_memory_finite']
        and summary['final_player_xy_error_tiles']['mean'] <= THRESHOLDS['long_final_xy_mean_max']
        and summary['final_player_xy_error_tiles']['p95'] <= THRESHOLDS['long_final_xy_p95_max']
        and summary['model_wall_steps'] == 0 and summary['model_wall_path_steps'] == 0 and summary['npc_active_steps'] == 0)
    summary['passed'] = bool(long_passed)
    default = None
    if default_regression_data is not None:
        default, default_arrays = _short(model, _data(default_regression_data, model), context=1,
            count=16, batch_size=batch_size, seed=seed, default_only=True)
        np.savez_compressed(output/'default_spawn_states.npz', **default_arrays)
    prior = evaluate_prior_navigation_tape(model, prior_tape_path, output/'prior_tape', seed=seed) if prior_tape_path else None
    result = {'passed': bool(short['passed'] and long_passed and (default is None or default['passed'])), 'short': short, 'long450': summary,
        'default_spawn': default, 'prior_tape450': prior, 'thresholds': THRESHOLDS,
        'source_future_feedback': False, 'future_rgb_encoded': False, 'rendering_evaluated': False,
        'scope': 'Expanded held-out navigation action response plus own450-step state/memory; default regression is separate',
        'fingerprint': fingerprint({'short': short['fingerprint'], 'long': long['fingerprint'], 'thresholds': THRESHOLDS})}
    atomic_json(output/'motion_recovery_validation.json', result)
    return result


def main(argv=None):
    import argparse
    import gc
    from .models import load_world_checkpoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prior-tape', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--context', choices=(1, 2), type=int, default=1)
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--long-count', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.count, args.long_count, args.batch_size, args.threads) < 1:
        parser.error('Counts, threads and batch size must be positive')
    torch.set_num_threads(args.threads)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA requested but unavailable')
        torch.cuda.init()
    model, metadata = load_world_checkpoint(args.world, 'cpu')
    step = metadata.get('training_steps')
    del metadata
    gc.collect()
    model.float().to(args.device).eval()
    result = evaluate_motion_recovery(model, args.data_root/'short/rgb/val', args.data_root/'long/rgb/val',
        args.output_dir, context=args.context, count=args.count, long_count=args.long_count,
        batch_size=args.batch_size, seed=args.seed, default_regression_data=args.data_root/'short/rgb/train',
        prior_tape_path=args.prior_tape)
    result.update(checkpoint=str(args.world.resolve()), training_steps=step, device=args.device,
                  precision='fp32_tf32_disabled')
    atomic_json(args.output_dir/'motion_recovery_validation.json', result)
    print(json.dumps({'passed': result['passed'], 'training_steps': step,
                      'report': str(args.output_dir/'motion_recovery_validation.json'),
                      'short_metrics': result['short']['metrics'], 'long450': result['long450']}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    main()
