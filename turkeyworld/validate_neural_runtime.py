"""Validate the actual game runtime, including FP16 renderer and learned SR.

Each independent case observes one recorded initial image and numerical pose.
All450 subsequent states, memories and displayed images are produced by the
unchanged NeuralRuntime.step path. Source futures are measurement labels only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from .engine import Game, STATE_DIM
from .neural_play import NeuralRuntime, _arguments as runtime_arguments
from .record_neural_video import _codec_hash, scripted_control
from .runtime_checks import PLAYER_RADIUS, circle_is_clear, segment_is_clear
from .visual_quality import image_quality_metrics


THRESHOLDS = {'model_wall_steps_max': 0, 'model_wall_path_steps_max': 0,
              'final_xy_error_mean_max': .1, 'final_xy_error_p95_max': .25}


def _json(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


def _stats(values):
    values = np.asarray(values, np.float64)
    return {'n': int(values.size), 'mean': float(values.mean()),
            'p95': float(np.percentile(values, 95)), 'max': float(values.max())}


def load_heldout_cases(directory, steps=450):
    """Choose genuine recorded slide and out/back tapes, without model selection."""
    candidates = []
    for path in sorted(Path(directory).glob('*.npz')):
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['split']) != 'val':
                raise ValueError('Runtime held-out tapes require explicit val provenance')
            if str(saved['environment_profile']) != 'navigation_no_enemies_v1':
                raise ValueError('Expected genuine no-enemy navigation environment')
            if len(saved['actions']) < steps or saved['states'].shape[1:] != (STATE_DIM,):
                continue
            action = saved['actions'][:steps]
            reverse = np.any((action[:-1, 0] > .9) & (action[1:, 0] < -.9)
                             & (np.abs(action[:-1, 1:3]).sum(-1) < 1e-5)
                             & (np.abs(action[1:, 1:3]).sum(-1) < 1e-5))
            candidates.append((path, str(saved['movement_case']), bool(reverse)))
    selected = []
    for name, predicate in (('heldout_slide', lambda item: item[1] == 'slide'),
                            ('heldout_return', lambda item: item[2])):
        eligible = [item for item in candidates if predicate(item) and item[0] not in [p for _, p in selected]]
        if not eligible:
            raise ValueError(f'No eligible recorded450-step tape for {name}')
        selected.append((name, eligible[0][0]))
    cases = []
    for name, path in selected:
        with np.load(path, allow_pickle=False) as saved:
            frames = saved['frames'][:steps+1].copy()
            states, actions = saved['states'][:steps+1].copy(), saved['actions'][:steps].copy()
            grid = saved['map'].copy()
            dt = float(np.asarray(saved['dt']).reshape(-1)[0])
            label = str(saved['movement_case'])
        if frames.shape != (steps+1, 32, 32, 3) or np.any(states[:, 13] > .5):
            raise ValueError('Expected native32 no-enemy recorded frames and state20')
        cases.append({'name': name, 'initial_rgb': frames[0], 'states': states, 'actions': actions,
                      'frames': frames, 'map': grid, 'dt': dt, 'source_held_out': True,
                      'source_path': str(path.resolve()), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'movement_case': label, 'initial_context': 1,
                      'selection': 'first sorted slide tape' if name.endswith('slide') else
                                   'first distinct sorted tape with genuine recorded forward-to-back reversal'})
    return cases


def default_case(steps=450, seed=0):
    """Generate an ordinary offline reference; never keep a live Source actor."""
    game = Game(seed=seed, endless=True, enemies_enabled=False)
    grid = game.map.copy()
    states, frames = [game.state_vector().copy()], [game.render(size=32, overlays=False)]
    actions = []
    for index in range(steps):
        _, controls = scripted_control(index, steps)
        game.step(controls, dt=1/30)
        actions.append(controls)
        states.append(game.state_vector().copy())
        frames.append(game.render(size=32, overlays=False))
    del game
    return {'name': 'default_spawn_controls', 'initial_rgb': frames[0], 'states': np.asarray(states),
            'actions': np.asarray(actions, np.float32), 'frames': np.asarray(frames), 'map': grid,
            'dt': 1/30, 'source_held_out': False, 'source_path': None, 'seed': seed,
            'selection': 'fixed10-segment movement/turn controls; default Source initialization',
            'reference_precomputed_offline': True, 'initial_context': 1}


class _RecordedInitializer:
    def __init__(self, case):
        self.map = case['map'].copy()
        self.vector = case['states'][0].copy()
        self.rgb = case['initial_rgb'].copy()
        self.observations = 0

    def state_vector(self):
        return self.vector.copy()

    def render(self, *, size, overlays):
        if size != self.rgb.shape[0] or overlays or self.observations:
            raise AssertionError('Exactly one native initial RGB observation is allowed')
        self.observations += 1
        return self.rgb.copy()


def initialize_runtime(args, case, shared=None):
    """Run the public game initializer using only a recorded initial observation.

    Reuse already validated immutable weights between independent cases. The
    model's own memory/state is initialized anew only at each case boundary.
    """
    initializer = _RecordedInitializer(case)
    with ExitStack() as stack:
        stack.enter_context(patch('turkeyworld.neural_play.Game', return_value=initializer))
        if shared is not None:
            stack.enter_context(patch('turkeyworld.neural_play.load_world_checkpoint',
                                      return_value=(shared.model, shared.world_metadata)))
            stack.enter_context(patch('turkeyworld.neural_play.load_sr_checkpoint',
                                      return_value=(shared.sr, shared.sr_metadata)))
            stack.enter_context(patch('turkeyworld.renderer_student.load_renderer_student',
                                      return_value=shared.renderer_metadata))
        runtime = NeuralRuntime(args)
    if initializer.observations != 1 or runtime.engine_renders_for_initialization != 1:
        raise AssertionError('Runtime must encode exactly one initial recorded observation')
    if not np.allclose(runtime.initial_state[0].float().cpu().numpy(), case['states'][0], atol=1e-6):
        raise AssertionError('Runtime initial pose differs from the recorded initial pose')
    return runtime


def _measure(case, predicted):
    reference, grid, dt = case['states'], case['map'], case['dt']
    error = np.linalg.norm((predicted[1:, :2]-reference[1:, :2])*16, axis=-1)
    point, swept, source_point, source_swept = [], [], [], []
    for first, second, true_first, true_second in zip(predicted[:-1], predicted[1:], reference[:-1], reference[1:]):
        point.append(not circle_is_clear(grid, second[:2]*16, PLAYER_RADIUS))
        swept.append(not segment_is_clear(grid, first[:2]*16, second[:2]*16, PLAYER_RADIUS))
        source_point.append(not circle_is_clear(grid, true_second[:2]*16, PLAYER_RADIUS))
        source_swept.append(not segment_is_clear(grid, true_first[:2]*16, true_second[:2]*16, PLAYER_RADIUS))
    first_wall = next((index+1 for index, (a, b) in enumerate(zip(point, swept)) if a or b), None)
    velocity = np.linalg.norm((np.diff(predicted[:, :2], axis=0)-np.diff(reference[:, :2], axis=0))*16/dt, axis=-1)
    return {'player_xy_error_tiles': _stats(error), 'final_player_xy_error_tiles': float(error[-1]),
            'paired_trajectory_velocity_error_tiles_s': _stats(velocity),
            'velocity_scope': 'Paired own-trajectory deltas; includes pose drift and differs from GT-pose one-step response',
            'model_wall_steps': sum(point), 'model_wall_path_steps': sum(swept),
            'source_wall_steps': sum(source_point), 'source_wall_path_steps': sum(source_swept),
            'first_model_wall_step': first_wall, 'first_model_wall_simulated_seconds': first_wall*dt if first_wall else None,
            'geometry_scope': 'Player circle endpoints and swept separated X-then-Y legs; measurement only',
            'all_states_finite': bool(np.isfinite(predicted).all()),
            'npc_active_steps': int((predicted[1:, 13] >= .5).sum())}


def evaluate_runtime_case(runtime, case, output_dir, *, progress=None):
    """Execute the identical display-producing game step with no Source access."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    count = len(case['actions'])
    if runtime.steps != 0 or abs(runtime.simulation_dt-case['dt']) > 1e-7:
        raise ValueError('Each case needs a fresh matching30Hz runtime')
    codec_before = _codec_hash(runtime)
    offsets = {value for value in (1, 90, 180, 270, 360, count) if value <= count}
    native = []
    displayed = []
    picked = []
    predicted = [runtime.state[0].detach().float().cpu().numpy().copy()]
    flow = runtime.model.flow_renderer
    if flow.kind != 'student' or flow.steps != 1:
        raise ValueError('Actual game validation requires the active one-Euler RF1 renderer')
    def capture_native(module, inputs, result):
        if runtime.steps+1 in offsets:
            rgb = result['rgb'].detach().float()
            if not bool(torch.isfinite(rgb).all()):
                raise FloatingPointError('Nonfinite native renderer output')
            native.append(rgb[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
    # model.step is a public method, so capture its output without changing any inputs.
    actual_step = runtime.model.step
    def step_with_capture(*args, **kwargs):
        result = actual_step(*args, **kwargs)
        capture_native(None, None, result)
        return result
    started = time.perf_counter()
    forbidden = AssertionError('Reference engine access is forbidden during autonomous runtime validation')
    with patch.object(runtime.model, 'step', side_effect=step_with_capture), \
            patch.object(flow, 'velocity', wraps=flow.velocity) as velocity, \
            patch.object(Game, 'step', side_effect=forbidden), \
            patch.object(Game, 'render', side_effect=forbidden), \
            patch.object(Game, 'set_state_vector', side_effect=forbidden):
        for index, action in enumerate(case['actions']):
            frame = runtime.step(action)
            if not bool(torch.isfinite(runtime.memory).all()):
                raise FloatingPointError('Nonfinite runtime memory')
            predicted.append(runtime.state[0].detach().float().cpu().numpy().copy())
            if index+1 in offsets:
                displayed.append(frame.copy())
                picked.append(index+1)
            if progress and (index == 0 or (index+1) % 50 == 0 or index+1 == count):
                progress(case['name'], index+1, count, time.perf_counter()-started)
        calls = velocity.call_count
    wall = time.perf_counter()-started
    if calls != count or runtime.steps != count or _codec_hash(runtime) != codec_before:
        raise AssertionError('Runtime solver/update count or frozen-codec state changed')
    predicted = np.asarray(predicted, np.float32)
    metrics = _measure(case, predicted)
    arrays = {'predicted_states': predicted, 'reference_states': case['states'], 'actions': case['actions'],
              'map': case['map'], 'source_frame_offsets': np.asarray(picked),
              'source_rgb': case['frames'][picked], 'neural_native_rgb': np.asarray(native),
              'neural_screen_rgb': np.asarray(displayed)}
    np.savez_compressed(output_dir/'trajectory.npz', **arrays)
    source = torch.as_tensor(arrays['source_rgb']).permute(0, 3, 1, 2).float()/255
    images = torch.as_tensor(arrays['neural_native_rgb']).permute(0, 3, 1, 2).float()/255
    visual = {key: _stats(value.numpy()) for key, value in image_quality_metrics(images, source).items()}
    report = runtime.report(wall)
    report.update({'name': case['name'], 'scope': 'Actual neural-game runtime navigation; full render/SR path',
                   'source_held_out': case['source_held_out'], 'source_path': case.get('source_path'),
                   'source_sha256': case.get('source_sha256'), 'selection': case['selection'],
                   'metrics': metrics, 'actual_renderer_velocity_evaluations': calls,
                   'one_euler_per_step_verified': calls == count,
                   'initial_observed_rgb_frames': 1, 'numerical_resets_during_rollout': 0,
                   'all_memory_finite': True, 'codec_all_state_dict_tensors_bitexact': True,
                   'codec_sha256': codec_before, 'source_future_feedback': False,
                   'source_future_rgb_encoded': False, 'future_reference_used_only_after_prediction': True,
                   'rgb_quality_scope': 'Combined own-pose dynamics and renderer at sparse offsets; not an isolated renderer score',
                   'sparse_rgb_quality': visual, 'sparse_rgb_offsets': picked})
    _json(output_dir/'report.json', report)
    _save_grid(output_dir, arrays)
    return report


def _save_grid(output, arrays):
    offsets = arrays['source_frame_offsets']
    size, header = 160, 26
    labels = ['SOURCE reference pose32', 'MODEL own pose32', 'Actual game screen1024']
    sheet = Image.new('RGB', (len(offsets)*size, 3*(size+header)), (14, 18, 24))
    draw = ImageDraw.Draw(sheet)
    for row, key in enumerate(('source_rgb', 'neural_native_rgb', 'neural_screen_rgb')):
        for column, (offset, frame) in enumerate(zip(offsets, arrays[key])):
            x, y = column*size, row*(size+header)
            draw.text((x+3, y+3), f'{labels[row]} | {offset}', fill='white')
            sheet.paste(Image.fromarray(frame).resize((size, size), Image.Resampling.NEAREST), (x, y+header))
    sheet.save(output/'source_model_comparison.png')
    grid = arrays['map']
    scale = 32
    path = Image.fromarray(np.where(grid[..., None] > 0, np.array([48, 52, 60]), np.array([224, 228, 222])).astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST)
    paint = ImageDraw.Draw(path)
    for key, color in (('reference_states', (30, 135, 75)), ('predicted_states', (220, 55, 55))):
        xy = arrays[key][:, :2]*16*scale
        paint.line([tuple(value) for value in xy], fill=color, width=2)
        x, y = xy[-1]
        paint.rectangle((x-3, y-3, x+3, y+3), fill=color)
    paint.rectangle((0, 0, 512, 20), fill=(20, 23, 28))
    paint.text((4, 4), 'Source GREEN | Neural RED | squares=end | no corrections', fill='white')
    path.save(output/'player_paths.png')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--concurrent-training', action='store_true')
    args, remaining = parser.parse_known_args(argv)
    runtime_args = runtime_arguments(remaining)
    if (runtime_args.display_size != 1024 or not runtime_args.smooth or runtime_args.sr is None
            or runtime_args.two_stage_upscale or runtime_args.steps not in (0, 450)):
        parser.error('Require450 steps, --sr --smooth --display-size1024; no fixed two-stage upscale')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _json(args.output_dir/'status.json', {'status': 'preparing_offline_reference'})
    cases = [default_case(), *load_heldout_cases(args.data_dir)]
    if runtime_args.device == 'cuda':
        torch.cuda.init()
        torch.empty(1, device='cuda')
    shared = None
    reports = []
    def progress(name, step, total, wall):
        value = {'status': 'validating', 'case': name, 'completed': step, 'total': total, 'wall_seconds': wall}
        _json(args.output_dir/'status.json', value)
        print(json.dumps(value), flush=True)
    for case in cases:
        runtime = initialize_runtime(runtime_args, case, shared)
        if not runtime.model.config.navigation_only or runtime.model.config.image_size != 32:
            raise ValueError('Require the native32 navigation-only game architecture')
        # No warmup transitions: the single observed pose/memory starts the case.
        report = evaluate_runtime_case(runtime, case, args.output_dir/case['name'], progress=progress)
        reports.append(report)
        shared = runtime
    final = np.asarray([row['metrics']['final_player_xy_error_tiles'] for row in reports])
    walls = sum(row['metrics']['model_wall_steps'] for row in reports)
    paths = sum(row['metrics']['model_wall_path_steps'] for row in reports)
    passed = (walls == 0 and paths == 0 and final.mean() <= .1 and np.percentile(final, 95) <= .25
              and all(row['metrics']['npc_active_steps'] == 0 for row in reports))
    result = {'status': 'complete', 'passed': bool(passed), 'cases': reports,
              'final_player_xy_error_tiles': _stats(final), 'model_wall_steps': walls,
              'model_wall_path_steps': paths, 'thresholds': THRESHOLDS,
              'heldout_cases': 2, 'default_regression_cases': 1, 'horizon': 450,
              'simulated_seconds_per_case': 15, 'precision': 'fp16' if runtime_args.fp16 else 'fp32',
              'source_future_feedback': False, 'source_engine_corrections': 0,
              'runtime_path': 'NeuralRuntime.step including renderer, learned SR and device readback',
              'gpu_timing_contended_by_training': args.concurrent_training,
              'timing_scope': 'Shared GPU runtime measurement, not isolated game FPS' if args.concurrent_training
                              else 'Diagnostic runtime timings; no interactive display',
              'generated_at': datetime.now(timezone.utc).isoformat()}
    _json(args.output_dir/'report.json', result)
    _json(args.output_dir/'status.json', {'status': 'complete', 'passed': bool(passed),
                                        'report': str((args.output_dir/'report.json').resolve())})
    print(json.dumps({'status': 'complete', 'passed': bool(passed), 'report': str(args.output_dir/'report.json')}), flush=True)


if __name__ == '__main__':
    main()
