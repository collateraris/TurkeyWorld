"""Measured held-out gates for a one-NPC, quality-gated game curriculum.

Recorded Source observations are targets only after initialization. Static
snapshots query the neural renderer directly and never call world dynamics.
Full-game qualification checks every one of450 own transitions/memories and
samples full neural RGB against prerecorded Source; no Game is instantiated.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .engine import BEHAVIOR_VERSION, ENTITY_DIM, ENTITY_SLOTS, GLOBAL_DIM, STATE_DIM
from .behavior_trajectory import _stats, evaluate_state_trajectories
from .behavior_heads import _Metrics, _measure
from .combat_sampling import classify_combat
from .runtime_checks import RolloutMonitor
from .training_pipeline import atomic_json
from .visual_quality import image_quality_metrics, strict_render_precision


STAGED_VALIDATION_VERSION = 'single_npc_staged_validation_v1'
STAGES = ('navigation', 'static_enemy', 'moving1', 'full_game')
DEFAULT_THRESHOLDS = {
    'player_xy_mean_max': .10, 'player_xy_p95_max': .25, 'wall_windows_max': 0,
    'rgb_ssim_min': .90, 'rgb_mae_max': .03,
    'npc_xy_mean_max': .25, 'npc_xy_p95_max': .50, 'npc_velocity_error_max': .25,
    'npc_mae_max': .05, 'background_mae_max': .03,
    'visible_frames_min': 8, 'contrast_fraction_min': .90,
    'event_positive_min': 8, 'hit_accuracy_min': .80, 'miss_accuracy_min': .80,
    'attack_recall_min': .80, 'attack_precision_min': .80,
    'spawn_accuracy_min': .80, 'spawn_xy_mean_max': .50,
    'full_event_agreement_min': .95,
}


@contextmanager
def _isolated_validation(model, data, seed):
    modes = [(module, module.training) for module in model.modules()]
    caches = [(module, name, getattr(module, name).detach().clone())
        for module in model.modules() for name in ('_spawn_walkable', '_map_condition')
        if torch.is_tensor(getattr(module, name, None))]
    numpy_state, python_state = np.random.get_state(), random.getstate()
    rng = getattr(data, 'rng', None)
    data_state = copy.deepcopy(rng.bit_generator.state) if isinstance(rng, np.random.Generator) else None
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), strict_render_precision(),\
                torch.autocast(device.type, enabled=False), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(seed)
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        for module, name, value in caches:
            setattr(module, name, value)
        np.random.set_state(numpy_state); random.setstate(python_state)
        if data_state is not None:
            rng.bit_generator.state = data_state


def _data(data_dir, stage, device, object_masks_directory):
    if hasattr(data_dir, 'batch') and hasattr(data_dir, 'episodes'):
        data = data_dir
    else:
        from .staged_curriculum import SnapshotData, StagedSequenceData
        root = Path(data_dir)
        kind = 'snapshots' if stage == 'static_enemy' else 'rgb'
        directory = root if root.name in ('val', 'test') else root/kind/'val'
        data = (SnapshotData(directory, device) if stage == 'static_enemy' else
                StagedSequenceData(directory, device, mode='stream', cache_bytes=128*1024**2,
                    object_masks_directory=object_masks_directory))
    if getattr(data, 'split', None) not in ('val', 'test'):
        raise ValueError('Staged validation requires explicit held-out val/test data')
    return data


def _data_identity(data, stage):
    """Refuse wrong stages, old dimensions, split mixing or source replacement."""
    identities, fingerprints, digest = [], set(), hashlib.sha256()
    directory = Path(data.directory).resolve()
    paths = [Path(episode['path']).resolve() for episode in data.episodes]
    if not paths or len(set(paths)) != len(paths) or any(path.parent != directory for path in paths):
        raise ValueError('Staged validation requires unique episodes inside its explicit held-out directory')
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            required = {'stage', 'split', 'behavior_version', 'state_dim', 'entity_slots',
                'dataset_kind', 'world_transition_compatible', 'staged_config_fingerprint',
                'states', 'object_masks', 'background_frames'}
            if not required.issubset(saved.files):
                raise ValueError(f'{path.name}: incomplete staged validation provenance')
            static = stage == 'static_enemy'
            if (str(saved['stage']) != stage or str(saved['split']) != data.split
                    or str(saved['behavior_version']) != BEHAVIOR_VERSION
                    or int(saved['state_dim']) != STATE_DIM or int(saved['entity_slots']) != ENTITY_SLOTS
                    or saved['states'].shape[-1] != STATE_DIM
                    or str(saved['dataset_kind']) != ('renderer_snapshots' if static else 'world_transitions')
                    or bool(saved['world_transition_compatible']) == static
                    or stage == 'full_game' and (
                        not {'initialization_source', 'frame_source', 'engine_semantics_modified', 'ai_freeze'}.issubset(saved.files)
                        or str(saved['initialization_source']) not in ('normal_game_start', 'seeded_valid_continuous_start')
                        or str(saved['frame_source']) != 'reference_engine')
                    or any(field in saved.files and bool(saved[field]) for field in ('engine_semantics_modified', 'ai_freeze'))
                    or static and ('actions' in saved.files or 'events' in saved.files)):
                raise ValueError(f'{path.name}: wrong stage/split/state/behavior or static transition claim')
            fingerprints.add(str(saved['staged_config_fingerprint']))
        file_hash = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                file_hash.update(chunk)
        identity = {'name': path.name, 'bytes': path.stat().st_size, 'sha256': file_hash.hexdigest()}
        identities.append(identity)
        digest.update(json.dumps(identity, sort_keys=True).encode())
    if len(fingerprints) != 1:
        raise ValueError('Staged validation refuses mixed source configuration fingerprints')
    return {'directory': str(directory), 'split': data.split, 'stage': stage,
        'source_config_fingerprint': next(iter(fingerprints)), 'files': identities,
        'fingerprint': digest.hexdigest()}


def _panel(data, context, horizon, count, seed):
    rng = np.random.default_rng(seed)
    eligible = sorted((Path(ep['path']).name, index, len(ep['frames']))
        for index, ep in enumerate(data.episodes) if len(ep['frames']) >= context+horizon)
    actual = min(count, len(eligible))
    if not actual:
        return []
    chosen = [eligible[int(rng.choice(group))] for group in np.array_split(np.arange(len(eligible)), actual)]
    return [(index, int(rng.integers(frames-context-horizon+1))) for _, index, frames in chosen]


def _image_rows(prediction, target, mask, background):
    measured = image_quality_metrics(prediction, target)
    mask = mask.detach().float()
    if mask.ndim == 3:
        mask = mask[:, None]
    error, gap = (prediction-target).abs(), (target-background).abs()
    result = []
    for row in range(len(prediction)):
        foreground = mask[row].expand_as(error[row]); mass = float(foreground.sum())
        bg = 1-foreground; bg_mass = float(bg.sum())
        valid = bool(measured['finite'][row])
        npc = float((error[row]*foreground).sum()/mass) if mass and valid else None
        reference_gap = float((gap[row]*foreground).sum()/mass) if mass else None
        result.append({name: float(value[row]) if value.dtype != torch.bool else bool(value[row])
                       for name, value in measured.items()})
        result[-1].update(npc_mae=npc,
            background_mae=float((error[row]*bg).sum()/bg_mass) if bg_mass and valid else None,
            npc_reference_gap=reference_gap, visible=mass > 0,
            contrast_retained=(bool(valid and npc <= .5*reference_gap) if mass else None))
    return result


def _image_summary(rows):
    def mean(name, selected=rows):
        values = [row[name] for row in selected if row.get(name) is not None]
        return float(np.mean(values)) if values else None
    visible = [row for row in rows if row['visible']]
    return {'rgb_frames': len(rows), 'rgb_ssim': mean('ssim'), 'rgb_mae': mean('mae'),
        'rgb_finite': all(row['finite'] and row['in_range'] for row in rows) and bool(rows),
        'npc_mae': mean('npc_mae', visible), 'background_mae': mean('background_mae'),
        'visible_frames': len(visible), 'contrast_fraction': mean('contrast_retained', visible)}


def _static_images(model, data, count, batch_size, seed, device):
    rng = np.random.default_rng(seed)
    candidates = [(episode, frame) for episode, item in enumerate(data.episodes) for frame in range(item['count'])]
    if not candidates:
        raise ValueError('Empty held-out snapshot panel')
    indices = sorted(rng.choice(len(candidates), size=min(count, len(candidates)), replace=False))
    panel, images = [candidates[int(index)] for index in indices], []
    for first in range(0, len(panel), batch_size):
        chosen = panel[first:first+batch_size]
        batch = {key: value.to(device) for key, value in data.batch(len(chosen), indices=chosen).items()}
        features = model.encode_map(batch['map'])
        # Direct pose snapshots are not a trajectory or an observed image past.
        predicted = model.render(batch['states'], batch['states'].new_zeros(len(chosen), 128), features)
        images.extend(_image_rows(predicted, batch['frames'], batch['object_masks'], batch['background_frames']))
    return _image_summary(images), panel


def _rollout_images_and_audit(model, data, panel, context, horizon, batch_size, render_every, device, full):
    images, audits, last_npc_errors, event_matches, npc_velocity_errors = [], [], [], [], []
    life_mismatches = 0
    for first in range(0, len(panel), batch_size):
        chosen = panel[first:first+batch_size]
        batch = {name: value.to(device) if torch.is_tensor(value) else value for name, value
                 in data.batch(len(chosen), context=context, horizon=horizon, indices=chosen).items()}
        maps = model.encode_map(batch['map'])
        current = batch['states'][:, context-1].float()
        memory = model.initialize_from_image(batch['context_frames'], batch['states'][:, :context], maps)
        shape = tuple(memory.shape[1:])
        monitors = [RolloutMonitor(grid.cpu().numpy(), 'mixed', horizon,
            dt=1/float(model.config.simulation_hz), endless=bool(model.config.endless), max_examples=5)
            for grid in batch['map']] if full else []
        for offset in range(horizon):
            previous = current
            result = model.transition(current, batch['actions'][:, offset], maps, memory)
            current, memory = result['state'], result['memory']
            if (current.shape != (len(chosen), STATE_DIM) or not bool(torch.isfinite(current).all())
                    or tuple(memory.shape) != (len(chosen), *shape) or not bool(torch.isfinite(memory).all())):
                raise FloatingPointError('Invalid own state/memory in staged free rollout')
            if full:
                for row, monitor in enumerate(monitors):
                    monitor.observe(previous[row].cpu().numpy(), current[row].cpu().numpy(),
                        batch['actions'][row, offset].cpu().numpy(), result['events'][row].cpu().numpy(),
                        batch['previous_states'][row, offset].cpu().numpy(), batch['target_states'][row, offset].cpu().numpy())
                event_matches.extend((result['events'] >= .5).eq(batch['events'][:, offset] >= .5).all(-1).cpu().tolist())
                expected = batch['target_states'][:, offset, GLOBAL_DIM:-2].reshape(-1, ENTITY_SLOTS, ENTITY_DIM)
                predicted = current[:, GLOBAL_DIM:-2].reshape_as(expected)
                before_source = batch['previous_states'][:, offset, GLOBAL_DIM:-2].reshape_as(expected)
                continuing = (before_source[..., 5] >= .5) & (expected[..., 5] >= .5)
                npc_velocity_errors.extend(((predicted[..., 2:4]-expected[..., 2:4]).norm(dim=-1)*2)[continuing].cpu().tolist())
                life_mismatches += int(((expected[..., 5] >= .5) != (predicted[..., 5] >= .5)).sum())
                if offset == horizon-1:
                    last_npc_errors.extend(((predicted[..., :2]-expected[..., :2]).norm(dim=-1)*16).cpu().flatten().tolist())
            if offset == 0 or (offset+1) % render_every == 0 or offset == horizon-1:
                prediction = model.render(current, result['features'], maps).float()
                images.extend(_image_rows(prediction, batch['target_frames'][:, offset],
                    batch['target_object_masks'][:, offset], batch['target_background_frames'][:, offset]))
        audits.extend(monitor.report() for monitor in monitors)
    metrics = _image_summary(images)
    if full:
        metrics.update(full_rollout_evaluated=True, full_rollout_steps=horizon, full_rollout_windows=len(panel),
            full_integrity_pass=all(item['integrity_pass'] for item in audits),
            full_active_pass=all(item['state_integrity_active_requested_pass'] for item in audits),
            full_event_agreement=float(np.mean(event_matches)) if event_matches else None,
            full_life_mismatches=life_mismatches, full_npc_xy=_stats(last_npc_errors),
            full_npc_velocity_error=_stats(npc_velocity_errors),
            engine_corrections=0, model_resets=0)
    return metrics, audits


def _head_metrics(model, data, context, batch_size, count, seed, device):
    random = np.random.default_rng(seed)
    cases = {name: [] for name in ('hit', 'miss', 'attack', 'spawn')}
    for index, episode in enumerate(data.episodes):
        masks = classify_combat(episode['states'], episode['actions'], episode['events'],
            episode['hit_ids'], episode['attack_flags'], episode['map'], float(episode['dt']))
        for name in cases:
            cases[name].extend((index, int(offset)-context+1) for offset in np.flatnonzero(masks[name]) if offset >= context-1)
    reports = {}
    for name, pairs in cases.items():
        selected = random.choice(len(pairs), size=min(count, len(pairs)), replace=False) if pairs else []
        panel = [pairs[int(index)] for index in selected]
        metrics = _Metrics()
        for first in range(0, len(panel), batch_size):
            chosen = panel[first:first+batch_size]
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value
                     in data.batch(len(chosen), context=context, horizon=1, indices=chosen).items()}
            maps = model.encode_map(batch['map'])
            memory = model.initialize_from_image(batch['context_frames'], batch['states'][:, :context], maps)
            result = model.transition(batch['states'][:, context-1], batch['actions'][:, 0], maps, memory)
            _measure(result, batch, context, metrics, float(model.config.simulation_hz), bool(model.config.endless),
                     game_training_v5=getattr(model.config, 'game_training_v5', False))
        reports[name] = {**metrics.report(), 'windows': len(panel), 'available_cases': len(pairs)}
    result = {}
    for name in cases:
        result[name+'_source_cases'] = reports[name]['windows']
    for name, key, alias in (('hit', 'hit_target_accuracy', 'hit_accuracy'),
                            ('miss', 'miss_target_none_accuracy', 'miss_accuracy'),
                            ('spawn', 'spawn_decoded_top1_tile_accuracy', 'spawn_accuracy'),
                            ('spawn', 'spawn_decoded_xy_error_tiles', 'spawn_xy_mean')):
        metric = reports[name]['metrics'].get(key, {})
        result[alias] = metric.get('mean'); result[alias+'_cases'] = metric.get('denominator', 0)
    attack = reports['attack']['binary'].get('attack_accepted', {})
    result.update(attack_recall=attack.get('recall'), attack_precision=attack.get('precision'),
                  attack_cases=attack.get('source_positives', 0))
    return result, reports


def _failures(metrics, stage, limits):
    failures = []
    def check(metric, threshold, minimum=False):
        value, bound = metrics.get(metric), limits[threshold]
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f'{metric}: no eligible measured cases')
        elif (value < bound if minimum else value > bound):
            failures.append(f'{metric}={value:.6g} {"<" if minimum else ">"} {bound:.6g}')
    if metrics.get('rgb_finite') is not True:
        failures.append('RGB contains invalid values or no evaluated frames')
    if stage != 'static_enemy':
        for metric, bound in (('player_xy_mean', 'player_xy_mean_max'), ('player_xy_p95', 'player_xy_p95_max'),
                              ('wall_windows', 'wall_windows_max'), ('rgb_mae', 'rgb_mae_max')):
            check(metric, bound)
        check('rgb_ssim', 'rgb_ssim_min', True)
    if stage != 'navigation':
        for metric, bound in (('npc_mae', 'npc_mae_max'), ('background_mae', 'background_mae_max')):
            check(metric, bound)
        check('visible_frames', 'visible_frames_min', True)
        check('contrast_fraction', 'contrast_fraction_min', True)
    if stage in ('moving1', 'full_game'):
        for metric, bound in (('npc_xy_mean', 'npc_xy_mean_max'), ('npc_xy_p95', 'npc_xy_p95_max'),
                              ('npc_velocity_error', 'npc_velocity_error_max')):
            check(metric, bound)
        for name, threshold in (('hit_accuracy', 'hit_accuracy_min'), ('miss_accuracy', 'miss_accuracy_min'),
                                ('attack_recall', 'attack_recall_min'), ('attack_precision', 'attack_precision_min'),
                                ('spawn_accuracy', 'spawn_accuracy_min'), ('spawn_xy_mean', 'spawn_xy_mean_max')):
            count = metrics.get('attack_cases' if name.startswith('attack_') else
                                'spawn_accuracy_cases' if name.startswith('spawn_') else name+'_cases', 0)
            source_case = 'attack' if name.startswith('attack_') else 'spawn' if name.startswith('spawn_') else name.split('_', 1)[0]
            source_count = metrics.get(source_case+'_source_cases', 0)
            if stage == 'full_game' and name != 'miss_accuracy' and count < limits['event_positive_min']:
                failures.append(f'{name}: fewer than {limits["event_positive_min"]:g} Source-positive cases ({count})')
            if count or source_count:
                check(name, threshold, minimum=name != 'spawn_xy_mean')
    if stage == 'full_game':
        if metrics.get('full_rollout_evaluated') is not True or metrics.get('full_rollout_steps') != 450:
            failures.append('Full450-transition own-state rollout was not evaluated')
        if metrics.get('full_integrity_pass') is not True:
            failures.append('Full rollout state-integrity check did not pass')
        if metrics.get('full_active_pass') is not True:
            failures.append('Full rollout motion/combat/respawn coverage did not qualify')
        if metrics.get('full_life_mismatches', 1) != 0:
            failures.append('Persistent NPC life/respawn states disagree with recorded Source')
        check('full_event_agreement', 'full_event_agreement_min', True)
    return failures


def evaluate_staged_model(model, data_dir, stage, device='cpu', *, context=2, horizon=29,
                          count=32, batch_size=4, seed=173, thresholds=None, output_dir=None,
                          object_masks_directory=None, final_rollout=True, full_probe_count=4, render_every=30):
    """Return a JSON gate; root owns consecutive passes/minimum-stage updates."""
    if isinstance(stage, dict):
        stage = stage.get('id')
    if stage not in STAGES:
        raise ValueError('Unknown one-enemy curriculum stage')
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in (context, horizon, count, batch_size, full_probe_count, render_every)):
        raise ValueError('Validation context/horizon/count/batch/probe/render intervals must be positive integers')
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('Staged held-out validation requires FP32 model parameters')
    target_device = next(model.parameters()).device
    requested_device = torch.device(device)
    if (requested_device.type != target_device.type or requested_device.index is not None
            and requested_device.index != target_device.index):
        raise ValueError('Move the model to the requested validation device before evaluation')
    supplied = dict(thresholds or {})
    if set(supplied)-set(DEFAULT_THRESHOLDS) or any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 for value in supplied.values()):
        raise ValueError('Unknown or invalid staged gate threshold')
    limits = {**DEFAULT_THRESHOLDS, **supplied}
    data = _data(data_dir, stage, target_device, object_masks_directory)
    identity = _data_identity(data, stage)
    temporary = tempfile.TemporaryDirectory() if output_dir is None else None
    output = Path(temporary.name if temporary is not None else output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output/'staged_validation.json'
    if report_path.exists():
        old = json.loads(report_path.read_text(encoding='utf-8'))
        if old.get('stage') != stage or old.get('data_fingerprint') != identity['fingerprint'] or old.get('thresholds') != limits:
            raise ValueError('Existing staged report belongs to different data/stage/thresholds')
    metrics = {'full_rollout_evaluated': False, 'source_future_feedback': False,
               'engine_corrections': 0, 'model_resets': 0}
    details, panel, error = {}, [], None
    try:
        with _isolated_validation(model, data, seed):
            if stage == 'static_enemy':
                image_metrics, panel = _static_images(model, data, count, batch_size, seed, target_device)
                metrics.update(image_metrics, renderer_only=True, transitions_evaluated=0)
            else:
                steps = 450 if stage == 'full_game' and final_rollout else horizon
                requested = min(count, full_probe_count) if stage == 'full_game' else count
                panel = _panel(data, context, steps, requested, seed)
                if not panel:
                    error = f'No held-out episode fits context{context}+horizon{steps}'
                else:
                    trajectory = evaluate_state_trajectories(model, data, output/'trajectory', context=context,
                        horizon=steps, count=len(panel), batch_size=batch_size, seed=seed)
                    last = trajectory['offsets'][str(steps)]
                    metrics.update(player_xy_mean=last['player_xy_error_tiles']['mean'],
                        player_xy_p95=last['player_xy_error_tiles']['p95'],
                        npc_xy_mean=last['npc_xy_error_tiles']['mean'], npc_xy_p95=last['npc_xy_error_tiles']['p95'],
                        npc_velocity_error=last['npc_velocity_field_error_tiles_s']['mean'],
                        wall_windows=trajectory['windows_with_wall_collision'], transitions_evaluated=len(panel)*steps,
                        trajectory_horizon=steps, trajectory_windows=len(panel), renderer_only=False,
                        all_states_finite=trajectory['all_states_finite'],
                        all_memory_finite=trajectory['all_memory_finite_and_shape_valid'])
                    # The identical seeded panel performs a fresh own rollout
                    # solely to expose render features/events; no GT resets.
                    image_metrics, audits = _rollout_images_and_audit(model, data, panel, context, steps,
                        batch_size, render_every if stage == 'full_game' else min(4, render_every),
                        target_device, stage == 'full_game' and final_rollout)
                    metrics.update(image_metrics)
                    if stage == 'full_game' and final_rollout:
                        metrics.update(npc_xy_mean=image_metrics['full_npc_xy']['mean'],
                                       npc_xy_p95=image_metrics['full_npc_xy']['p95'],
                                       npc_velocity_error=image_metrics['full_npc_velocity_error']['mean'])
                    details['rollout_audits'] = audits
                    details['trajectory_fingerprint'] = trajectory['fingerprint']
                    if stage in ('moving1', 'full_game'):
                        heads, reports = _head_metrics(model, data, context, batch_size, count, seed, target_device)
                        metrics.update(heads); details['head_cohorts'] = reports
    except FloatingPointError as failure:
        error = str(failure)
    except BaseException:
        if temporary is not None:
            temporary.cleanup()
        raise
    finally:
        if hasattr(data, 'clear_cache'):
            data.clear_cache()
    failures = _failures(metrics, stage, limits)
    if error is not None:
        failures.insert(0, error)
    report = {'version': STAGED_VALIDATION_VERSION, 'stage': stage, 'passed': not failures,
        'metrics': metrics, 'thresholds': limits, 'failures': failures,
        'data_fingerprint': identity['fingerprint'], 'source': identity,
        'panel': [list(map(int, pair)) for pair in panel], 'requested_count': count,
        'actual_windows': len(panel), 'context': context, 'source_held_out': True,
        'future_source_feedback': False, 'engine_corrections': 0, 'model_resets': 0,
        'final_qualified': stage == 'full_game' and not failures,
        'report_path': str(report_path.resolve()) if temporary is None else None,
        'scope': 'Direct renderer snapshots only' if stage == 'static_enemy' else
                 'Full450 own-state/own-memory held-out gameplay' if stage == 'full_game' and final_rollout else
                 'Short held-out own trajectories; not15-second qualification', **details}
    atomic_json(report_path, report)
    if temporary is not None:
        temporary.cleanup()
    return report
