"""Record actual autonomous neural playback, preserving its measured wall-clock speed.

The reference engine is used only by NeuralRuntime's single initialization. The
MP4 repeats each produced image until the next image was actually ready; it does
not play a slow inference tape at an artificial 30 simulated updates per second.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .neural_play import NeuralRuntime, _arguments as runtime_arguments


CONTROL_SEGMENTS = (
    ("W", (1, 0, 0, 0)), ("S", (-1, 0, 0, 0)),
    ("A", (0, -1, 0, 0)), ("D", (0, 1, 0, 0)),
    ("Q", (0, 0, -1, 0)), ("E", (0, 0, 1, 0)),
    ("W+Q", (1, 0, -0.6, 0)), ("W+E", (1, 0, 0.6, 0)),
    ("S+D", (-1, 1, 0, 0)), ("W+A", (1, -1, 0, 0)),
)


def scripted_control(index, steps):
    """Ten equal, open-loop control segments; never inspect a Source state."""
    if steps < 1 or not 0 <= index < steps:
        raise ValueError('Control index must be within the positive recording length')
    name, action = CONTROL_SEGMENTS[min(len(CONTROL_SEGMENTS)-1, index*len(CONTROL_SEGMENTS)//steps)]
    return name, list(action)


def frame_durations(ready_seconds, final_hold):
    values = np.asarray(ready_seconds, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0) or not math.isfinite(final_hold) or final_hold <= 0):
        raise ValueError('Frame-ready timestamps must be finite and strictly increasing')
    return np.r_[np.diff(values), float(final_hold)].tolist()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _codec_hash(runtime):
    codec = getattr(getattr(runtime.model, 'flow_renderer', None), 'codec', None)
    if codec is None:
        return None
    digest = hashlib.sha256()
    for name, tensor in sorted(codec.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode('utf-8'))
        digest.update(str((array.dtype, array.shape)).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _overlay(frame, controls, step, simulated, elapsed, training_step):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    size = max(12, image.width//48)
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', size) if Path('C:/Windows/Fonts/consola.ttf').is_file() else ImageFont.load_default()
    text = (f'EXPERIMENTAL NEURAL GAME | RSSM g{training_step} | NAVIGATION\n'
            f'Keys: {controls:<5} | update {step} | sim {simulated:.2f}s | wall {elapsed:.2f}s\n'
            '32px neural -> learned SR 128px -> bilinear 1024px | no engine corrections')
    box = draw.multiline_textbbox((8, 6), text, font=font, spacing=4)
    draw.rectangle((0, 0, image.width, box[3]+8), fill=(8, 8, 12))
    draw.multiline_text((8, 6), text, font=font, fill=(245, 245, 245), spacing=4)
    return image


def capture_frames(runtime, directory, steps=450, *, overlay=True, clock=time.perf_counter,
                   sleep=time.sleep, progress=None):
    """Save actual model outputs; only initial state/RGB exist outside the model."""
    directory.mkdir(parents=True, exist_ok=False)
    runtime.synchronize()
    started = clock()
    rows = []
    source_forbidden = AssertionError('Reference engine calls are forbidden after neural initialization')
    with patch('turkeyworld.neural_play.Game.step', side_effect=source_forbidden), \
            patch('turkeyworld.neural_play.Game.render', side_effect=source_forbidden), \
            patch('turkeyworld.neural_play.Game.set_state_vector', side_effect=source_forbidden):
        for index in range(steps):
            controls, action = scripted_control(index, steps)
            before = clock()-started
            frame = runtime.step(action)
            after = clock()-started
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError('Neural runtime must return a finite RGB uint8 image')
            if not bool(torch.isfinite(runtime.state).all()) or not bool(torch.isfinite(runtime.memory).all()):
                raise FloatingPointError('Nonfinite state or memory during neural video capture')
            # A fast system keeps the game's maximum simulation rate of30Hz.
            delay = (index+1)*runtime.simulation_dt-(clock()-started)
            if delay > 0:
                sleep(delay)
            visible = clock()-started
            if rows and visible <= rows[-1]['ready_wall_seconds']:
                raise ValueError('Monotonic clock did not advance between model images')
            output = directory/f'frame_{index:06d}.png'
            image = (_overlay(frame, controls, index+1, (index+1)*runtime.simulation_dt,
                              visible, runtime.world_metadata.get('training_steps', 0))
                     if overlay else Image.fromarray(frame))
            image.save(output, compress_level=1)
            rows.append({'step': index+1, 'action': action, 'keys': controls,
                         'simulated_seconds': (index+1)*runtime.simulation_dt,
                         'inference_start_wall_seconds': before, 'inference_end_wall_seconds': after,
                         'ready_wall_seconds': visible, 'frame': output.name})
            if progress is not None and (index == 0 or (index+1) % 30 == 0 or index+1 == steps):
                progress(index+1, clock()-started)
    runtime.synchronize()
    wall = clock()-started
    final_hold = max(runtime.simulation_dt, wall-rows[-1]['ready_wall_seconds'])
    durations = frame_durations([row['ready_wall_seconds'] for row in rows], final_hold)
    return rows, durations, wall


def encode_video(ffmpeg, directory, rows, durations, output, fps=30):
    # Relative generated filenames avoid platform-specific concat path quoting.
    concat = directory/'frames.ffconcat'
    lines = ['ffconcat version 1.0']
    for row, duration in zip(rows, durations):
        lines.extend([f"file '{row['frame']}'", f'duration {duration:.9f}'])
    lines.append(f"file '{rows[-1]['frame']}'")
    concat.write_text('\n'.join(lines)+'\n', encoding='ascii')
    duration = sum(durations)
    command = [str(ffmpeg), '-hide_banner', '-loglevel', 'warning', '-y', '-safe', '1',
               '-i', str(concat.resolve()), '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
               '-pix_fmt', 'yuv420p', '-r', str(fps), '-fps_mode', 'cfr', '-t', f'{duration:.9f}',
               '-movflags', '+faststart', str(output.resolve())]
    with (output.parent/'ffmpeg.stderr.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=log, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path)
    parser.add_argument('--video-fps', type=int, default=30)
    parser.add_argument('--no-overlay', action='store_true')
    parser.add_argument('--keep-frames', action='store_true')
    args, remaining = parser.parse_known_args(argv)
    runtime_args = runtime_arguments(remaining)
    if runtime_args.two_stage_upscale or runtime_args.display_size != 1024 or not runtime_args.smooth or runtime_args.sr is None:
        parser.error('This recording requires --sr --display-size1024 --smooth and no --two-stage-upscale')
    if args.video_fps < 1 or args.video_fps > 120:
        parser.error('--video-fps must be between1 and120')
    ffmpeg = args.ffmpeg or shutil.which('ffmpeg')
    if not ffmpeg or not Path(ffmpeg).is_file():
        parser.error('FFmpeg with libx264 is required; provide --ffmpeg PATH')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir/'status.json'
    _write_json(status, {'status': 'initializing', 'started_at': datetime.now(timezone.utc).isoformat()})
    # Establish CUDA before loading large CPU checkpoint tensors on Windows.
    if runtime_args.device == 'cuda':
        torch.cuda.init()
        torch.empty(1, device='cuda')
    runtime = NeuralRuntime(runtime_args)
    if not getattr(runtime.model.config, 'navigation_only', False):
        raise ValueError('This diagnostic video is explicitly navigation-only')
    if runtime.model.config.image_size != 32 or runtime.engine_renders_for_initialization != 1:
        raise ValueError('Expected native32 and exactly one initial Source image')
    runtime.warmup()
    flow = getattr(runtime.model, 'flow_renderer', None)
    if flow is None or flow.kind != 'student' or flow.steps != 1:
        raise ValueError('This recording requires the active one-Euler student renderer')
    codec_before = _codec_hash(runtime)
    steps = runtime_args.steps or 450
    def progress(completed, wall):
        payload = {'status': 'capturing', 'completed': completed, 'total': steps, 'wall_seconds': wall,
                   'simulated_seconds': completed*runtime.simulation_dt, 'source_future_feedback': False}
        _write_json(status, payload)
        print(json.dumps(payload), flush=True)
    with patch.object(flow, 'velocity', wraps=flow.velocity) as velocity:
        rows, durations, wall = capture_frames(runtime, args.output_dir/'frames', steps,
                                               overlay=not args.no_overlay, progress=progress)
        velocity_calls = velocity.call_count
    if velocity_calls != steps:
        raise RuntimeError(f'Expected one Euler velocity evaluation per update: {velocity_calls} != {steps}')
    codec_after = _codec_hash(runtime)
    if codec_after != codec_before:
        raise RuntimeError('Frozen codec state changed during video inference')
    frame_root = args.output_dir/'frames'
    shutil.copyfile(frame_root/rows[len(rows)//2]['frame'], args.output_dir/'poster.png')
    _write_json(status, {'status': 'encoding', 'completed': steps, 'wall_seconds': wall})
    output = args.output_dir/'neural_game.mp4'
    command = encode_video(ffmpeg, frame_root, rows, durations, output, args.video_fps)
    report = runtime.report(wall, displayed_frames=round(sum(durations)*args.video_fps))
    report.update({'status': 'complete', 'video_path': str(output.resolve()),
                   'video_fps': args.video_fps, 'video_duration_seconds': sum(durations),
                   'playback_speed': 'measured_wall_clock_frame_ready_intervals; repeated frames; no temporal interpolation',
                   'video_origin': 'first generated frame ready; checkpoint loading and warmup excluded',
                   'recording_png_overhead_included': True, 'wall_clock_paced_max_simulation_hz': runtime.simulation_hz,
                   'headless_unpaced': False, 'all_predicted_states_memory_and_rgb_finite': True,
                   'navigation_only': True, 'experimental_preview': True,
                   'actual_renderer_velocity_evaluations': velocity_calls,
                   'one_euler_velocity_evaluation_per_step_verified': velocity_calls == steps,
                   'codec_all_state_dict_tensors_bitexact': codec_before == codec_after,
                   'codec_sha256': codec_after, 'action_timestamps': rows,
                   'display_durations_seconds': durations, 'encoding_command': command,
                   'video_file_bytes': output.stat().st_size})
    _write_json(args.output_dir/'report.json', report)
    _write_json(status, {key: report[key] for key in ('status', 'video_path', 'video_duration_seconds',
                                                     'simulated_seconds', 'codec_all_state_dict_tensors_bitexact')})
    if not args.keep_frames:
        # Only our newly created, generated frame directory is removed.
        resolved_frames = frame_root.resolve()
        if resolved_frames.parent != args.output_dir.resolve() or resolved_frames.name != 'frames':
            raise RuntimeError('Refusing cleanup outside the generated video frame directory')
        shutil.rmtree(resolved_frames)
    print(json.dumps({'status': 'complete', 'report': str((args.output_dir/'report.json').resolve()),
                      'video': str(output.resolve())}), flush=True)


if __name__ == '__main__':
    main()
