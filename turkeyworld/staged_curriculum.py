"""Checkpointed, quality-gated navigation -> objects -> combat curriculum.

Training pools are immutable and split-separated. Snapshot pools deliberately
have no action or transition labels: they can only train the image generator.
"""
from __future__ import annotations

from .engine import GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS, STATE_DIM

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .jepa_training_utils import SequenceData


STAGES = (
    {'id': 'navigation', 'label': 'Карта и ходьба без врагов', 'enemy_count': 0,
     'renderer_only': False, 'major_stage': 0, 'minimum_updates': 600},
    {'id': 'static_enemy', 'label': 'Один неподвижный враг: изображение', 'enemy_count': 1,
     'renderer_only': True, 'major_stage': 1, 'minimum_updates': 200},
    {'id': 'moving1', 'label': 'Один движущийся враг: бой и спавн', 'enemy_count': 1,
     'renderer_only': False, 'major_stage': 2, 'minimum_updates': 300},
    {'id': 'full_game', 'label': 'Полная игра: 15 секунд / 450 переходов', 'enemy_count': 1,
     'renderer_only': False, 'major_stage': 3, 'minimum_updates': 200},
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def plan_fingerprint(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=True,
        separators=(',', ':')).encode()).hexdigest()


class StagedSequenceData(SequenceData):
    """Ordinary episode windows with exact, inline offline NPC annotations."""
    def _initialize_object_masks(self, directory, counterfactual_backgrounds=False):
        if directory is not None:
            return super()._initialize_object_masks(directory, counterfactual_backgrounds)
        super()._initialize_object_masks(None, False)
        self.object_masks_directory = self.directory.resolve()
        self.counterfactual_backgrounds = True
        self._inline_labels = OrderedDict()

    def _labels(self, episode):
        path = Path(episode['path'])
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        key = str(path.resolve())
        if key in self._inline_labels:
            masks, backgrounds, recorded = self._inline_labels[key]
            if identity != recorded:
                raise ValueError('Staged source episode changed after loading')
            self._inline_labels.move_to_end(key)
            return masks, backgrounds
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['split']) != self.split:
                raise ValueError('Staged annotation split mismatch')
            masks, backgrounds = saved['object_masks'], saved['background_frames']
            if (masks.dtype != np.bool_ or masks.shape != episode['frames'].shape[:3]
                    or backgrounds.dtype != np.uint8 or backgrounds.shape != episode['frames'].shape):
                raise ValueError('Invalid inline NPC mask/background dimensions')
        self._inline_labels[key] = (masks, backgrounds, identity)
        while len(self._inline_labels) > 16:
            self._inline_labels.popitem(last=False)
        return masks, backgrounds

    def _episode_object_masks(self, episode):
        if not hasattr(self, '_inline_labels'):
            return super()._episode_object_masks(episode)
        return self._labels(episode)[0]

    def _episode_object_backgrounds(self, episode):
        if not hasattr(self, '_inline_labels'):
            return super()._episode_object_backgrounds(episode)
        return self._labels(episode)[1]

    def clear_cache(self):
        super().clear_cache()
        if hasattr(self, '_inline_labels'):
            self._inline_labels.clear()


class SnapshotData:
    """Independent image queries, never masquerading as world transitions."""
    def __init__(self, directory, device='cpu', *, episode_paths=None):
        self.directory, self.device = Path(directory).resolve(), torch.device(device)
        self.split = self.directory.name
        if self.split not in ('train', 'val', 'test'):
            raise ValueError('Snapshot data requires an explicit split')
        paths = ([self.directory/name for name in episode_paths] if episode_paths is not None
                 else sorted(self.directory.glob('*.npz')))
        self.episodes, self._cache = [], OrderedDict()
        for path in paths:
            if path.resolve().parent != self.directory:
                raise ValueError('Snapshot path escapes its split')
            with np.load(path, allow_pickle=False) as saved:
                if (str(saved['dataset_kind']) != 'renderer_snapshots'
                        or bool(saved['world_transition_compatible'])
                        or str(saved['split']) != self.split or 'actions' in saved.files):
                    raise ValueError('Snapshot provenance/transition schema mismatch')
                count = len(saved['states'])
                if count < 1 or saved['states'].shape != (count, STATE_DIM):
                    raise ValueError('Invalid snapshot state shape')
            self.episodes.append({'path': str(path), 'count': count})
        if not self.episodes:
            raise ValueError('No renderer snapshots in split')
        self.files, self.count = len(self.episodes), sum(ep['count'] for ep in self.episodes)

    def _read(self, index):
        ep = self.episodes[index]
        path = Path(ep['path'])
        identity = (path.stat().st_size, path.stat().st_mtime_ns)
        if index in self._cache:
            arrays, old = self._cache[index]
            if identity != old:
                raise ValueError('Snapshot source changed after loading')
            self._cache.move_to_end(index)
            return arrays
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ('frames', 'states', 'map', 'object_masks', 'background_frames')}
        n = ep['count']
        frames = arrays['frames']
        if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != n
                or frames.shape[-1] != 3 or arrays['object_masks'].shape != frames.shape[:3]
                or arrays['object_masks'].dtype != np.bool_
                or arrays['background_frames'].shape != frames.shape
                or arrays['background_frames'].dtype != np.uint8
                or arrays['map'].shape != (16, 16)
                or not np.isfinite(arrays['states']).all()):
            raise ValueError('Invalid renderer snapshot arrays')
        self._cache[index] = (arrays, identity)
        while len(self._cache) > 16:
            self._cache.popitem(last=False)
        return arrays

    def batch(self, batch_size, *, rng=None, indices=None, **unused):
        rng = np.random.default_rng() if rng is None else rng
        if indices is None:
            indices = [(int(i), int(rng.integers(self.episodes[int(i)]['count'])))
                       for i in rng.integers(self.files, size=batch_size)]
        rows = []
        for index, frame in indices:
            arrays = self._read(int(index))
            rows.append({key: value if key == 'map' else value[int(frame)] for key, value in arrays.items()})
        return snapshot_tensors(rows, self.device)


def snapshot_tensors(rows, device):
    result = {}
    for key in rows[0]:
        value = torch.from_numpy(np.stack([row[key] for row in rows])).to(device)
        if key in ('frames', 'background_frames'):
            value = value.permute(0, 3, 1, 2).contiguous().float()/255
        elif key == 'object_masks':
            value = value[:, None].float()
        result[key] = value.long() if key == 'map' else value.float()
    return result


def as_snapshots(batch):
    if 'target_states' not in batch:
        return batch
    return {'states': batch['target_states'][:, -1], 'frames': batch['target_frames'][:, -1],
            'map': batch['map'], 'object_masks': batch['target_object_masks'][:, -1],
            'background_frames': batch['target_background_frames'][:, -1]}


def join_batches(batches):
    if len(batches) == 1:
        return batches[0]
    names = set(batches[0])
    if any(set(batch) != names for batch in batches):
        raise ValueError('Curriculum replay batches must share a tensor schema')
    return {key: torch.cat([batch[key] for batch in batches], 0) for key in names}


class StagedCurriculum:
    def __init__(self, manifest, *, start_step, previous=None):
        self.manifest_path = Path(manifest).resolve()
        plan = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if (plan.get('version') != 1 or plan.get('target_steps') != 450
                or plan.get('validation_every') != 100 or plan.get('required_passes', 2) < 1
                or set(plan.get('datasets', {})) != {stage['id'] for stage in STAGES}):
            raise ValueError('Incomplete staged curriculum manifest')
        self.plan, self.fingerprint = plan, plan_fingerprint(plan)
        previous = dict(previous or {})
        if previous and previous.get('fingerprint') != self.fingerprint:
            raise ValueError('Staged curriculum plan changed on resume')
        self.state = previous or {'version': 1, 'fingerprint': self.fingerprint,
            'manifest': str(self.manifest_path), 'stage_index': 0, 'stage_start_step': start_step,
            'start_step': start_step, 'history': [], 'validation_history': [], 'pass_streak': 0,
            'final_qualified': False, 'status': 'training', 'gate': {'status': 'pending'}}
        if not 0 <= self.state['stage_index'] < len(STAGES):
            raise ValueError('Invalid saved curriculum stage')
        self.sync(start_step)

    @property
    def stage(self):
        return STAGES[self.state['stage_index']]

    def sync(self, step):
        stage = self.stage
        dataset = self.plan['datasets'][stage['id']]
        self.state.update(stage=stage['id'], stage_label=stage['label'], stages=list(STAGES),
            renderer_only=stage['renderer_only'], stage_updates=step-self.state['stage_start_step'],
            validation_every=100, required_passes=self.plan.get('required_passes', 2),
            target_seconds=15, target_steps=450, source_engine_feedback=False,
            current_data={key: dataset.get(key) for key in
                ('directory', 'train_clips', 'validation_clips', 'train_transitions', 'train_snapshots')},
            replay_fraction=0. if self.state['stage_index'] == 0 else self.plan.get('replay_fraction', .25))
        return self.state

    def horizon(self, step):
        updates = step-self.state['stage_start_step']
        if self.stage['renderer_only']:
            return 1
        if self.stage['id'] == 'navigation':
            return 1 if updates < 100 else 4 if updates < 200 else 8 if updates < 300 else 16 if updates < 500 else 29
        return 4 if updates < 100 else 8 if updates < 200 else 29

    def validation_due(self, step):
        previous = self.state.get('gate', {}).get('step', -1)
        return previous < self.state['stage_start_step'] or step//100 > int(previous)//100

    def record_validation(self, report, step):
        self.sync(step)
        minimum = self.stage['minimum_updates']
        eligible = self.state['stage_updates'] >= minimum
        passed = report.get('passed') is True
        old_step = self.state.get('gate', {}).get('step')
        if old_step == step:
            raise ValueError('A validation step cannot advance the gate twice')
        streak = self.state['pass_streak']+1 if passed and eligible else 0
        self.state['pass_streak'] = streak
        gate = {'status': 'passed' if passed else 'failed', 'step': step,
            'metrics': report.get('metrics', {}), 'thresholds': report.get('thresholds', {}),
            'report': report.get('report_path'), 'quality_passed': passed,
            'minimum_updates': minimum, 'eligible_for_transition': eligible,
            'pass_streak': streak, 'required_passes': self.plan.get('required_passes', 2),
            'failures': report.get('failures', [])}
        self.state['gate'] = gate
        self.state['validation_history'].append({'stage': self.stage['id'], **gate})
        self.state['validation_history'] = self.state['validation_history'][-150:]
        advanced = eligible and streak >= self.plan.get('required_passes', 2)
        if advanced:
            self.state['history'].append({'stage': self.stage['id'], 'passed_at': utc_now(),
                'step': step, 'metrics': gate['metrics'], 'thresholds': gate['thresholds'],
                'validation_report': gate['report']})
            if self.stage['id'] == 'full_game':
                self.state.update(final_qualified=True, status='qualified')
            else:
                self.state.update(stage_index=self.state['stage_index']+1, stage_start_step=step,
                    pass_streak=0, gate={'status': 'pending'})
                self.sync(step)
        return advanced


class StagedTrainingPools:
    def __init__(self, curriculum, legacy, *, device='cpu', seed=173, cache_bytes=128*1024**2):
        self.curriculum, self.legacy = curriculum, legacy
        self.device, self.seed, self.cache_bytes = device, seed, cache_bytes
        self.pools = {}
        self.motion = {}

    def get(self, stage_id):
        if stage_id in self.pools:
            return self.pools[stage_id]
        if stage_id == 'legacy':
            return self.legacy
        info = self.curriculum.plan['datasets'][stage_id]
        root = Path(info['directory'])
        if info.get('kind') == 'legacy':
            pool = self.legacy
        elif info.get('kind') == 'renderer_snapshots':
            pool = SnapshotData(root/'snapshots/train', self.device,
                episode_paths=info.get('train_filenames'))
        else:
            pool = StagedSequenceData(root/'rgb/train', self.device, mode='stream', seed=self.seed,
                cache_bytes=self.cache_bytes, episode_paths=info.get('train_filenames'))
        self.pools[stage_id] = pool
        return pool

    def _dynamic_batch(self, stage_id, count, horizon, context, rng):
        pool = self.get(stage_id)
        # Navigation selects the actual wall/straight/slide/turn cases in the
        # newly recorded enemy-free corpus. No source label enters the model.
        if stage_id in ('navigation', 'moving1'):
            from .motion_curriculum import MotionCurriculum
            if stage_id not in self.motion:
                combat = None
                if stage_id == 'moving1':
                    from .combat_sampling import CombatSampling
                    combat = CombatSampling(pool, context)
                self.motion[stage_id] = MotionCurriculum(pool, context, 0, max_horizon=29,
                    combat_sampling=combat, combat_fraction=.40 if combat is not None else 0.)
            selector = self.motion[stage_id]
            offset = {1: 0, 4: 100, 8: 300, 16: 600, 29: 1000}[horizon]
            indices = selector.sample_indices(count, offset, rng)
            return pool.batch(count, context=context, horizon=horizon, rng=rng, indices=indices)
        return pool.batch(count, context=context, horizon=horizon, rng=rng)

    def batch(self, batch_size, step, context, rng):
        curriculum = self.curriculum
        stage_id, static = curriculum.stage['id'], curriculum.stage['renderer_only']
        horizon = curriculum.horizon(step)
        replay_count = int(round(batch_size*curriculum.state['replay_fraction']))
        current_count = batch_size-replay_count
        sample = (lambda name, n: self.get(name).batch(n, rng=rng)) if static else (
            lambda name, n: self._dynamic_batch(name, n, horizon, context, rng))
        batches = [sample(stage_id, current_count)]
        if replay_count:
            # Snapshots are never admitted into a dynamics minibatch. Keeping
            # 25% navigation in every later dynamics stage protects walking.
            replay = self._dynamic_batch('navigation', replay_count, horizon, context, rng)
            batches.append(as_snapshots(replay) if static else replay)
        return join_batches(batches)


def renderer_snapshot_loss(model, ema, batch, *, context=2, render=True, **unused):
    """Train Base/Flow only; detached map and no transition/predictor call."""
    from .object_losses import balanced_image_l1, balanced_edge_l1, foreground_contrastive_loss
    state, target = batch['states'].detach(), batch['frames'].detach()
    mask, background = batch['object_masks'].detach(), batch['background_frames'].detach()
    with torch.no_grad():
        fmap = model.encode_map(batch['map'])
    base = model.render_base(state, state.new_zeros(len(state), 128), fmap.detach())
    parts = {'base_image': balanced_image_l1(base, target, mask)+.2*balanced_edge_l1(base, target, mask),
             'foreground_base': .25*foreground_contrastive_loss(base, target, background, mask)}
    flow, fm = model.flow_renderer.loss(target, base.detach(), state, object_mask=mask,
        map_condition=batch['map'].detach())
    parts['flow'] = flow
    metrics = {name: float(value.detach()) for name, value in parts.items()}
    metrics.update({name: float(value) for name, value in fm.items()})
    metrics.update(renderer_only=True, dynamics_updated=False)
    from .object_losses import object_region_weights
    endpoint = {'base': base, 'state': state, 'target': target, 'object_mask': mask,
        'pixel_weights': object_region_weights(mask, 4.), 'background': background,
        'map_condition': batch['map'].detach()}
    return sum(parts.values()), metrics, endpoint
