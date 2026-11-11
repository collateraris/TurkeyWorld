"""One input tape and clock for independent Source and neural game states."""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .engine import Game, STATE_DIM


def render_neural_pose(neural, state):
    """Render a pose without advancing or replacing the runtime's own h/z."""
    with torch.inference_mode():
        features = neural.model.vjepa_backend.latent_features(neural.memory)
        rgb = neural.model.render(state, features, neural.map_features)
        output = neural._resize(rgb)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError('Nonfinite paired neural display image')
        return output[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).contiguous().cpu().numpy()


class PairedRuntime:
    """Both games advance once from the same copied action and fixed dt.

    Autonomous mode displays the genuine neural state. Shared-camera mode is
    explicitly a renderer diagnostic: Source poses condition a second render,
    while the RSSM trajectory and memory continue independently and unchanged.
    """
    def __init__(self, neural, *, seed=None, source_size=256, horizon=450,
                 source_factory=Game, render_pose=None):
        if (not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1
                or not isinstance(source_size, int) or isinstance(source_size, bool)
                or not 32 <= source_size <= 2048):
            raise ValueError('Paired game requires a positive horizon and valid Source resolution')
        self.neural = neural
        self.seed = int(neural.args.seed if seed is None else seed)
        self.source_size, self.horizon = source_size, horizon
        self._source_factory = source_factory
        self._render_pose = render_pose or render_neural_pose
        self.mode = 'autonomous'
        self.resets = -1
        self.started_at = time.perf_counter()
        self.reset()

    def _new_source(self):
        config = self.neural.model.config
        source = self._source_factory(seed=self.seed, endless=config.endless,
            enemies_enabled=not getattr(config, 'navigation_only', False))
        reference = self.neural.initial_state[0].detach().float().cpu().numpy()
        actual = source.state_vector()
        if actual.shape != (STATE_DIM,) or not np.allclose(actual, reference, atol=1e-7, rtol=0):
            raise ValueError('Source and neural game must have the same initial numerical state')
        if hasattr(self.neural, 'initial_map') and not np.array_equal(source.map, self.neural.initial_map):
            raise ValueError('Source and neural game initial maps differ')
        initial_rgb = getattr(self.neural, 'initial_rgb', None)
        if initial_rgb is not None and not np.array_equal(
                source.render(size=self.neural.model.config.image_size, overlays=False), initial_rgb):
            raise ValueError('Source and neural game initial RGB observations differ')
        return source

    def reset(self):
        self.source = self._new_source()
        self.neural.reset_from_cached_initial_state()
        self.neural.steps = 0
        self.neural.samples.clear()
        self.neural.event_counts = dict.fromkeys(self.neural.event_counts, 0)
        self.neural.last_frame = None
        self.steps = 0
        self.actions = []
        self.shared_pose_renders = 0
        self.resets += 1
        self.source_frame = self.source.render(size=self.source_size, overlays=False)
        self._autonomous_frame = self._render_pose(self.neural, self.neural.state)
        self.neural_frame = self._autonomous_frame
        if self.mode == 'shared_camera':
            self._display_shared_camera()
        return self.source_frame, self.neural_frame

    @property
    def finished(self):
        return self.steps >= self.horizon

    @property
    def simulated_seconds(self):
        return self.steps*self.neural.simulation_dt

    @property
    def xy_drift_tiles(self):
        reference = self.source.state_vector()
        predicted = self.neural.state[0].detach().float().cpu().numpy()
        return float(np.linalg.norm((reference[:2]-predicted[:2])*16))

    def _display_shared_camera(self):
        pose = torch.as_tensor(self.source.state_vector().copy(), device=self.neural.state.device,
                               dtype=self.neural.state.dtype)[None]
        self.neural_frame = self._render_pose(self.neural, pose)
        self.shared_pose_renders += 1

    def set_mode(self, mode):
        if mode not in ('autonomous', 'shared_camera'):
            raise ValueError('Comparison mode must be autonomous or shared_camera')
        self.mode = mode
        if mode == 'shared_camera':
            self._display_shared_camera()
        else:
            self.neural_frame = self._autonomous_frame
        return self.source_frame, self.neural_frame

    def step(self, action):
        if self.finished:
            raise RuntimeError('Paired trial is complete; restart both games for another trial')
        controls = np.asarray(action, dtype=np.float32)
        if controls.shape != (4,) or not np.isfinite(controls).all():
            raise ValueError('Shared controls require four finite action values')
        controls = controls.copy()
        # A failed inference cannot leave the reference game a tick ahead.
        neural_frame = self.neural.step(controls.copy())
        self.source.step(controls.copy(), dt=self.neural.simulation_dt)
        self.steps += 1
        if self.neural.steps != self.steps:
            raise AssertionError('Source/neural game clocks diverged')
        self.actions.append(controls.tolist())
        self.source_frame = self.source.render(size=self.source_size, overlays=False)
        self._autonomous_frame = neural_frame
        self.neural_frame = neural_frame
        if self.mode == 'shared_camera':
            self._display_shared_camera()
        return self.source_frame, self.neural_frame

    def status(self):
        reference = self.source.state_vector()
        predicted = self.neural.state[0].detach().float().cpu().numpy()
        true_yaw = math.atan2(float(reference[2]), float(reference[3]))
        model_yaw = math.atan2(float(predicted[2]), float(predicted[3]))
        yaw_error = math.atan2(math.sin(model_yaw-true_yaw), math.cos(model_yaw-true_yaw))
        return {'steps': self.steps, 'horizon': self.horizon, 'finished': self.finished,
            'simulated_seconds': self.simulated_seconds, 'simulation_hz': self.neural.simulation_hz,
            'xy_drift_tiles': self.xy_drift_tiles, 'yaw_error_degrees': abs(math.degrees(yaw_error)),
            'mode': self.mode, 'source_size': self.source_size,
            'navigation_only': getattr(self.neural.model.config, 'navigation_only', False),
            'neural_training_step': self.neural.world_metadata.get('training_steps', 0),
            'shared_input_steps': len(self.actions), 'source_steps': self.steps,
            'neural_steps': self.neural.steps, 'engine_corrections': 0,
            'rssm_future_source_feedback': False,
            'reference_pose_for_display_only': self.mode == 'shared_camera',
            'shared_pose_renders': self.shared_pose_renders, 'trial_resets': self.resets,
            'state_finite': bool(np.isfinite(predicted).all()),
            'memory_finite': bool(torch.isfinite(self.neural.memory).all())}

    def report(self):
        return {**self.status(), 'actions': self.actions,
            'wall_seconds_including_pauses': time.perf_counter()-self.started_at,
            'neural_runtime': self.neural.report(time.perf_counter()-self.started_at),
            'display_mode_note': 'Source poses affect diagnostic display only; never RSSM state/memory'}
