"""One-step held-out head diagnostics; no renderer, training or engine calls.

Each query initializes only its observed context. Uniform and event-enriched
cohorts stay separate; event-enriched averages are not population estimates.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from turkeyworld.combat_objectives import combat_transition_losses
from turkeyworld.combat_sampling import classify_combat
from turkeyworld.spatial_entities import entity_motion_basis, world_to_local
from turkeyworld.structured_state import SHOT_DAMAGE
from turkeyworld.visual_quality import strict_render_precision


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(41):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 40:
                raise
            time.sleep(.05)


def _cohorts(data, context, seed):
    random = np.random.default_rng(seed)
    candidates = {key: [] for key in ('hit', 'miss', 'attack', 'death', 'spawn', 'wall_negative')}
    for episode_index, episode in enumerate(data.episodes):
        labels = classify_combat(episode['states'], episode['actions'], episode['events'],
            episode['hit_ids'], episode['attack_flags'], episode['map'], float(episode['dt']))
        for transition in range(context-1, len(episode['actions'])):
            pair = (episode_index, transition-context+1)
            for key in candidates:
                if labels[key][transition]:
                    candidates[key].append(pair)
    panels = {'uniform': [tuple(map(int, pair)) for pair in data.fixed_panel(
        count=128, context=context, horizon=1, seed=seed)]}
    for key, values in candidates.items():
        chosen = random.choice(len(values), size=min(32, len(values)), replace=False) if values else []
        panels[key] = sorted(values[int(index)] for index in chosen)
    return panels, {key: len(values) for key, values in candidates.items()}


class _Metrics:
    def __init__(self):
        self.means = defaultdict(lambda: [0., 0])
        self.confusions = defaultdict(lambda: np.zeros(4, dtype=np.int64))
        self.counts = defaultdict(int)
        self.attack_class_sums = {'positive': [0., 0], 'negative': [0., 0]}

    def add(self, name, values):
        values = torch.as_tensor(values).detach().double().reshape(-1).cpu()
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Nonfinite head metric: ' + name)
        self.means[name][0] += float(values.sum())
        self.means[name][1] += values.numel()

    def weighted(self, name, mean, denominator):
        denominator = int(denominator)
        entry = self.means[name]
        if denominator:
            entry[0] += float(mean)*denominator
            entry[1] += denominator

    def binary(self, name, predicted, target):
        p = torch.as_tensor(predicted).detach().bool().reshape(-1).cpu().numpy()
        t = torch.as_tensor(target).detach().bool().reshape(-1).cpu().numpy()
        self.confusions[name] += [int((p&t).sum()), int((p&~t).sum()),
                                 int((~p&t).sum()), int((~p&~t).sum())]

    def report(self):
        ratios = {name: {'mean': total/count if count else None,
                         'numerator': total, 'denominator': count}
                  for name, (total, count) in self.means.items()}
        positive, negative = self.attack_class_sums['positive'], self.attack_class_sums['negative']
        if positive[1] or negative[1]:
            present = 4*int(positive[1] > 0)+int(negative[1] > 0)
            balanced = (4*positive[0]/max(1, positive[1])+negative[0]/max(1, negative[1]))/present
            ratios['training_weighted_combat_attack'] = {'mean': balanced,
                'normalization': 'global_separate_positive_negative_means_4_to_1',
                'positive_count': positive[1], 'negative_count': negative[1]}
        binary = {}
        for name, values in self.confusions.items():
            tp, fp, fn, tn = map(int, values)
            binary[name] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
                source_positives=tp+fn, predicted_positives=tp+fp,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                accuracy=(tp+tn)/(tp+fp+fn+tn) if tp+fp+fn+tn else None)
        return {'metrics': ratios, 'binary': binary, 'counts': dict(self.counts)}


def _measure(result, batch, context, metrics, hz, endless, game_training_v5=False):
    previous, target = batch['states'][:, context-1], batch['target_states'][:, 0]
    action, events = batch['actions'][:, 0], batch['events'][:, 0]
    hit_ids, attacks = batch['hit_ids'][:, 0], batch['attack_flags'][:, 0]
    predicted, logits, attack_logits = result['state'], result['target_logits'], result['attack_logits']
    if not all(bool(torch.isfinite(value).all()) for value in (predicted, logits, attack_logits)):
        raise FloatingPointError('Nonfinite state or combat heads')
    size = len(previous)
    old, expected, actual = [value[:, 8:88].reshape(size, 8, 10) for value in (previous, target, predicted)]
    old_alive, next_alive = old[:, :, 5] >= .5, expected[:, :, 5] >= .5
    revived, continuing = ~old_alive & next_alive, old_alive & next_alive
    labels = torch.where(hit_ids < 0, 8, hit_ids).long()
    target_class = logits.argmax(-1)
    metrics.counts['transitions'] += size
    metrics.counts['hit_positives'] += int((hit_ids >= 0).sum())
    metrics.counts['misses_with_source_shot'] += int(((events[:, 0] >= .5) & (hit_ids < 0)).sum())
    metrics.counts['attack_positive_slots'] += int((attacks >= .5).sum())
    metrics.counts['npc_deaths'] += int((old_alive & ~next_alive).sum())
    metrics.counts['npc_revived_slots'] += int(revived.sum())
    metrics.add('raw_target_ce', F.cross_entropy(logits.float(), labels, reduction='none'))
    metrics.add('target_accuracy_all', target_class == labels)
    for name, mask in {'hit_target_accuracy': hit_ids >= 0, 'none_target_accuracy': hit_ids < 0,
                      'miss_target_none_accuracy': (events[:, 0] >= .5) & (hit_ids < 0)}.items():
        metrics.add(name, (target_class == labels)[mask])
    metrics.add('raw_attack_bce', F.binary_cross_entropy_with_logits(attack_logits.float(), attacks, reduction='none'))
    metrics.binary('attack_raw_head', attack_logits >= 0, attacks >= .5)
    if 'accepted_attacks' in result:
        metrics.binary('attack_accepted', result['accepted_attacks'], attacks >= .5)
    if 'event_logits' in result:
        event_logits = result['event_logits'].float()
        metrics.add('event_logits_bce', F.binary_cross_entropy_with_logits(event_logits, events, reduction='none'))
        for index, name in enumerate(('shot', 'hit', 'kill', 'player_hit')):
            metrics.binary('event_probability_'+name, event_logits[:, index] >= 0, events[:, index] >= .5)
    if torch.is_tensor(result.get('events')):
        for index, name in enumerate(('shot', 'hit', 'kill', 'player_hit')):
            metrics.binary('event_accepted_'+name, result['events'][:, index] >= .5, events[:, index] >= .5)
    # Genuine source legal-shot opportunities; unshootable/dead slots are excluded.
    killable = old_alive & (old[:, :, 4] > 0) & (old[:, :, 4] <= SHOT_DAMAGE+1e-6) & (events[:, :1] >= .5)
    slot = torch.arange(8, device=previous.device)[None]
    deaths = old_alive & ~next_alive
    metrics.binary('npc_death_head_on_killable', (target_class[:, None] == slot)[killable], deaths[killable])
    metrics.binary('npc_death_accepted_on_killable', (actual[:, :, 5] < .5)[killable], deaths[killable])
    metrics.counts['killable_source_legal_shot_slots'] += int(killable.sum())
    metrics.add('npc_hp_abs_error_hp', ((actual[:, :, 4]-expected[:, :, 4]).abs()*100)[old_alive])
    metrics.add('player_hp_abs_error_hp', (predicted[:, 4]-target[:, 4]).abs()*100)
    if 'spawn_logits' in result:
        tiles = (expected[:, :, :2]*16).long().clamp(0, 15)
        tile_labels = tiles[:, :, 0]+16*tiles[:, :, 1]
        spawn_logits = result['spawn_logits'].float()
        if 'spawn_anchor_logits' in result:
            from .aligned_training import spawn_supervision_statistics
            statistics = spawn_supervision_statistics(result, previous, target)
            count = int(statistics['spawn_count'])
            metrics.add('spawn_ce_on_revived', spawn_logits.new_empty(0))
            metrics.weighted('spawn_ce_on_revived', statistics['spawn_sum']/max(1, count), count)
        else:
            ce = F.cross_entropy(spawn_logits.reshape(-1, 256), tile_labels.flatten(), reduction='none').reshape(size, 8)
            metrics.add('spawn_ce_on_revived', ce[revived])
        metrics.add('spawn_raw_top1_tile_accuracy', (spawn_logits.argmax(-1) == tile_labels)[revived])
        positions = result['spawn_positions'].float()
        decoded = (positions*16).long().clamp(0, 15)
        metrics.add('spawn_decoded_top1_tile_accuracy', (decoded[:, :, 0]+16*decoded[:, :, 1] == tile_labels)[revived])
        metrics.add('spawn_decoded_xy_error_tiles', torch.linalg.vector_norm((positions-expected[:, :, :2])*16, dim=-1)[revived])
        metrics.add('spawn_state_xy_error_tiles', torch.linalg.vector_norm((actual[:, :, :2]-expected[:, :, :2])*16, dim=-1)[revived])
        metrics.add('spawn_decoder_valid_fraction', result['spawn_valid'][revived])
    else:
        metrics.counts['missing_spawn_head_transitions'] += size
    deltas = [torch.cat([(value[:, :2]-previous[:, :2])[:, None],
                         value[:, 8:88].reshape(size, 8, 10)[:, :, :2]-old[:, :, :2]], 1)*16
              for value in (predicted, target)]
    error = deltas[0]-deltas[1]
    basis = entity_motion_basis(previous, action=action, dt=1/hz)
    local_error = world_to_local(error, basis)
    metrics.add('player_delta_l2_tiles', torch.linalg.vector_norm(error[:, 0], dim=-1))
    metrics.add('npc_delta_l2_tiles', torch.linalg.vector_norm(error[:, 1:], dim=-1)[continuing])
    for index, axis in enumerate(('forward', 'side')):
        metrics.add('player_delta_local_'+axis+'_abs_tiles', local_error[:, 0, index].abs())
        metrics.add('npc_delta_local_'+axis+'_abs_tiles', local_error[:, 1:, index].abs()[continuing])
    angle_error = torch.atan2(predicted[:, 2], predicted[:, 3])-torch.atan2(target[:, 2], target[:, 3])
    metrics.add('player_heading_abs_degrees', torch.atan2(angle_error.sin(), angle_error.cos()).abs()*180/math.pi)
    terms = combat_transition_losses(result, previous, target, hit_ids, attacks, events,
        action=action, dt=1/hz, endless=endless, game_training_v5=game_training_v5)
    if game_training_v5:
        for name in ('positive', 'negative'):
            metrics.attack_class_sums[name][0] += float(terms[f'attack_{name}_sum'])
            metrics.attack_class_sums[name][1] += int(terms[f'attack_{name}_count'])
    valid = old_alive & ~result.get('respawn_flags', torch.zeros_like(old_alive)).bool()
    own_shot = result['shot'].reshape(-1, 1) > .5
    denominators = {'events': size*4, 'target': size, 'attack': size*8,
        'damage_npc': int(valid.sum()), 'damage_player': int((previous[:, 4] > 0).sum()),
        'death_npc': int((valid & (old[:, :, 4] <= SHOT_DAMAGE+1e-6) & own_shot).sum())}
    for name in denominators:
        if name == 'attack' and game_training_v5:
            continue
        metrics.weighted('training_weighted_combat_'+name, terms[name], denominators[name])


def evaluate_heads(model, data, output_dir, context=2, batch_size=4, seed=173):
    """Return/write independent one-step head reports on explicit held-out data."""
    if getattr(data, 'split', None) not in ('val', 'test'):
        raise ValueError('Head diagnostics require explicit held-out val/test data')
    if context < 1 or batch_size < 1:
        raise ValueError('Positive context and batch_size required')
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('Head validation requires an FP32 model')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    panels, available = _cohorts(data, context, seed)
    total = sum(len(panel) for panel in panels.values())
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    started, done, reports = time.perf_counter(), 0, {}
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        model.eval()
        with strict_render_precision(), torch.random.fork_rng(devices=devices), torch.inference_mode(), torch.autocast(device.type, enabled=False):
            torch.random.default_generator.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(seed)
            for name, panel in panels.items():
                metrics = _Metrics()
                for first in range(0, len(panel), batch_size):
                    chosen = panel[first:first+batch_size]
                    batch = {key: value.to(device) if torch.is_tensor(value) else value
                             for key, value in data.batch(len(chosen), context=context, horizon=1,
                                 rng=np.random.default_rng(seed), indices=chosen).items()}
                    if 'dt' in batch and not bool(torch.allclose(batch['dt'],
                            torch.full_like(batch['dt'], 1/float(model.config.simulation_hz)), atol=1e-7, rtol=0)):
                        raise ValueError('Source and model timestep mismatch')
                    features = model.encode_map(batch['map'].long())
                    memory = model.initialize_from_image(batch['context_frames'].float(),
                        batch['states'][:, :context].float(), features)
                    result = model.transition(batch['states'][:, context-1].float(),
                        batch['actions'][:, 0].float(), features, memory)
                    _measure(result, batch, context, metrics, float(model.config.simulation_hz), bool(model.config.endless),
                             game_training_v5=getattr(model.config, 'game_training_v5', False))
                    done += len(chosen)
                    progress = dict(status='running', cohort=name, completed=done, total=total,
                                    elapsed_seconds=round(time.perf_counter()-started, 3))
                    _write(output/'head_validation_status.json', progress)
                    print(json.dumps({'head_validation_progress': progress}), flush=True)
                rows = [{'episode': Path(data.episodes[episode]['path']).name, 'start': start,
                         'source_transition': start+context-1, 'target_frame': start+context}
                        for episode, start in panel]
                reports[name] = {**metrics.report(), 'windows': len(panel),
                    'available_cases': available.get(name), 'missing_cases': not bool(panel),
                    'sampling': 'fixed_panel' if name == 'uniform' else 'seeded_event_enriched', 'panel': rows}
    finally:
        for module, training in modes:
            module.training = training
    fingerprint = hashlib.sha256(json.dumps({name: value['panel'] for name, value in reports.items()},
                                            sort_keys=True).encode()).hexdigest()
    report = {'kind': 'held_out_one_step_heads', 'split': data.split, 'context': context, 'horizon': 1,
        'seed': seed, 'batch_size': batch_size, 'network_precision': 'fp32', 'future_ground_truth_feedback': False,
        'renderer_evaluated': False, 'autonomous_900_step_qualified': False,
        'generated_at': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter()-started,
        'source_directory': str(data.directory), 'panel_fingerprint': fingerprint, 'cohorts': reports,
        'notes': ['Only observed context RGB/state initializes each independent one-step query.',
            'Uniform and event-enriched cohorts overlap and must not be pooled as population estimates.',
            'Training-weighted combat components are not the full training/validation objective.',
            'A null rate/mean denotes no eligible examples; absent positives do not establish accuracy.',
            'Wall-negative uses the training classifier: a recorded source shot miss with an NPC behind a blocking static wall.',
            'Event logits are structured probabilities derived from target/attack heads and known gates, not an independent event network.',
            'NPC motion excludes death/respawn transitions; death scores require genuine source legal-shot killable slots.',
            'No engine, RGB renderer, optimizer update, GT future-state correction or runtime reset is used.']}
    _write(output/'heldout_head_validation.json', report)
    from .behavior_trajectory import evaluate_state_trajectories
    report['heldout_trajectory'] = evaluate_state_trajectories(model, data, output,
        context=context, horizon=16, count=32, batch_size=batch_size, seed=seed)
    _write(output/'heldout_head_validation.json', report)
    _write(output/'head_validation_status.json', dict(status='complete', completed=done, total=total,
        elapsed_seconds=report['elapsed_seconds'], report=str(output/'heldout_head_validation.json')))
    return report
