"""Independent CPU death-clock coverage audit over already sampled Source tapes.

No model/checkpoint/GPU, training process or Source corpus is changed. Legacy
predicates are a frozen mathematical reference; revised calls use production
helpers and an actual Game step at each selected birth. Pose/anchor checks are
controlled Source-pose references, not a claimed post-fix neural rollout.
"""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turkeyworld.engine import Game
from turkeyworld.structured_state import structured_transition
from turkeyworld.simulation_clock import advance_normalized_timer, normalized_timer_ready
from turkeyworld.spawn import SPAWN_ANCHORS
from turkeyworld.staged_curriculum import StagedSequenceData


def clock_birth(initial, horizon, *, revised):
    clock = torch.tensor(float(initial), dtype=torch.float32)
    for tick in range(horizon):
        if revised:
            typed = bool(normalized_timer_ready(clock, duration=.6, dt=1/30,
                                                increasing=True, deadline=3.))
            from turkeyworld.simulation_clock import advance_timer_seconds
            source = advance_timer_seconds(float(clock)*.6, duration=.6, dt=1/30,
                                           increasing=True, maximum=3.) >= 3.-1e-8
        else:
            typed = bool(clock*.6+1/30 >= 3.-1e-6)
            source = float(clock)*.6+1/30 >= 3.-1e-8
        if typed:
            return {'offset': tick+1, 'clock_before': float(clock), 'source_label': bool(source)}
        clock = (advance_normalized_timer(clock, duration=.6, dt=1/30, increasing=True, maximum=3.)
                 if revised else (clock+1/30/.6).clamp_max(5))
    return None


def canonical_probe(*, context=1, offset=1, badq=False, ticks=4):
    game = Game(seed=0, endless=True)
    state = game.state_vector(); state[2:4] = [0, 1]; state[12:14] = 0
    state[16] = np.float32((3.-(offset+context-1)/30)/.6)
    if ticks == 90:
        state[16] = 0
    if badq:
        state[16] = np.float32(4.944444179534912)
    game.set_state_vector(state)
    assert np.array_equal(game.state_vector(), state), 'State adapter changed a canonical FP32 value'
    for _ in range(context-1):
        game.step(np.zeros(4, np.float32))
    previous = torch.from_numpy(game.state_vector().copy())[None]
    source_births, typed_births = [], []
    for tick in range(1, ticks+1):
        before = game.state_vector()
        game.step(np.zeros(4, np.float32))
        target = game.state_vector()
        born = not (before[13] >= .5 and before[12] > 0) and target[13] >= .5 and target[12] > 0
        source_births.append(bool(born))
        candidate = torch.from_numpy(target.copy())[None]
        result = structured_transition(previous, candidate, torch.zeros(1, 4),
            torch.tensor([[-50., 50.]]), torch.tensor([[-50.]]), dt=1/30,
            endless=True, training=False, spawn_positions=candidate[:, None, 8:10],
            spawn_valid=torch.ones(1, 1, dtype=torch.bool))
        typed_births.append(bool(result['respawn_flags'][0, 0]))
        assert typed_births[-1] == source_births[-1], (tick, previous[0, 16], target[16])
        assert torch.equal(result['state'][:, 16], candidate[:, 16]), 'Death timer encode differs'
        previous = result['state']
    assert source_births == typed_births
    expected = 90 if ticks == 90 else offset
    assert [i+1 for i, value in enumerate(source_births) if value] == [expected]
    return {'context': context, 'expected_offset': expected, 'ticks': ticks,
            'Source_birth_offsets': [i+1 for i, value in enumerate(source_births) if value],
            'typed_birth_offsets': [i+1 for i, value in enumerate(typed_births) if value],
            'canonical_roundtrip_exact': True, 'badq': badq, 'passed': True}


def corpus_audit(run_dir, *, first_step=5438, last_step=5453):
    root = Path(run_dir)
    plan = json.loads((ROOT/'training_runs/rssm_moe_multitask_20261112/moe_training_plan.json').read_text(encoding='utf-8'))
    data = StagedSequenceData(Path(plan['data_roots']['recovery'])/'moving1/rgb/train',
        torch.device('cpu'), mode='stream', cache_bytes=32*1024**2)
    rows = []
    for line in (root/'metrics.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
            if first_step <= row.get('step', 0) <= last_step:
                rows.append(row)
        except ValueError:
            pass
    if [row['step'] for row in rows] != list(range(first_step, last_step+1)):
        raise ValueError('Coverage audit needs every committed update in the fixed metric window')
    cache = {}; game = Game(seed=0, endless=True)
    counts = Counter(); per_update = []; before_offsets = Counter(); after_offsets = Counter()
    recorded_anchors = Counter(); before_anchors = Counter(); after_anchors = Counter()
    anchor_by_cell = {tuple(cell): i for i, cell in enumerate(SPAWN_ANCHORS)}
    def anchor(state):
        return str(anchor_by_cell.get(tuple(np.floor(state[8:10]*16).astype(int)), 8))
    for row in rows:
        if len(row.get('microbatch_metrics', [])) != 4 or len(row.get('combat_sampling', [])) != 4:
            raise ValueError('Coverage audit requires all four recorded microbatches')
        local = Counter(); local['step'] = row['step']
        for micro, receipt in zip(row['microbatch_metrics'], row['combat_sampling']):
            if len(receipt['pairs']) != 64 or micro['moe_auxiliary']['horizon'] != 4:
                raise ValueError('Fixed coverage comparison requires the actual64x4-step auxiliary tape')
            local['logged_pre_fix_own_Source_spawns'] += int(micro['moe_auxiliary']['combat_spawn_event_count'])
            for episode, start in receipt['pairs']:
                path = data.episodes[episode]['path']
                if path not in cache:
                    with np.load(path, allow_pickle=False) as saved:
                        cache[path] = {key: saved[key] for key in ('states', 'map', 'actions')}
                values = cache[path]; tape = values['states'][start:start+5]
                alive = (tape[:, 13] >= .5) & (tape[:, 12] > 0)
                source_births = np.flatnonzero(~alive[:-1] & alive[1:])
                local['sampled_aux_windows'] += 1
                local['recorded_Source_spawns'] += len(source_births)
                for tick in source_births:
                    recorded_anchors[anchor(tape[tick+1])] += 1
                if alive[0] or tape[0, 4] <= 0:
                    continue
                for revised in (False, True):
                    birth = clock_birth(tape[0, 16], 4, revised=revised)
                    if birth is None:
                        continue
                    prefix = 'after' if revised else 'before'
                    local[prefix+'_typed_clock_births'] += 1
                    if not birth['source_label']:
                        local[prefix+'_teacher_timing_mismatches'] += 1
                        continue
                    local[prefix+'_same_tick_teacher_births'] += 1
                    (after_offsets if revised else before_offsets)[str(birth['offset'])] += 1
                    # Controlled Source-pose label at this same own timer. This
                    # is not a neural prediction, and never enters training.
                    tick = birth['offset']-1
                    previous = tape[tick].copy(); previous[12:14] = 0
                    previous[16] = birth['clock_before']
                    game.map = values['map'].copy()
                    game.map_height, game.map_width = game.map.shape
                    game.set_state_vector(previous)
                    assert np.array_equal(game.state_vector(), previous)
                    game.step(values['actions'][start+tick])
                    target = game.state_vector()
                    assert target[13] >= .5 and target[12] > 0
                    (after_anchors if revised else before_anchors)[anchor(target)] += 1
                    if revised:
                        local['fixed_clock_matches_recorded_birth_offset'] += int(tick in source_births)
                        local['fixed_clock_matches_recorded_anchor'] += int(
                            tick in source_births and anchor(target) == anchor(tape[tick+1]))
        for key, value in local.items():
            if key != 'step': counts[key] += value
        per_update.append(dict(local))
    assert counts['before_same_tick_teacher_births'] == counts['logged_pre_fix_own_Source_spawns']
    assert counts['after_teacher_timing_mismatches'] == 0
    assert counts['after_same_tick_teacher_births'] == counts['recorded_Source_spawns']
    counts['after_teacher_timing_mismatches'] = 0
    return {'passed': True, 'fixed_metric_window': [first_step, last_step], 'macro_updates': len(rows),
        'counts': dict(counts), 'per_update': per_update,
        'birth_offsets': {'before': dict(before_offsets), 'after': dict(after_offsets)},
        'anchor_labels': {'recorded_Source': dict(recorded_anchors),
                          'before_controlled_Source_pose': dict(before_anchors),
                          'after_controlled_Source_pose': dict(after_anchors)},
        'scope': 'Clock and controlled Source-pose labels only; actual post-fix neural coverage/loss requires separate validation',
        'models_or_checkpoints_loaded': False, 'GPU_used': False,
        'Source_corpus_or_training_process_changed': False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    torch.set_num_threads(2)
    probes = [canonical_probe(context=c, offset=o) for c in (1, 2) for o in (1, 2, 3, 4)]
    probes += [canonical_probe(badq=True), canonical_probe(ticks=90)]
    report = {'protocol': 'independent_respawn_clock_coverage_regression_v1',
              'canonical_probes': probes, 'corpus': corpus_audit(args.run_dir), 'passed': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: report['corpus'][key] for key in ('passed', 'counts', 'birth_offsets', 'anchor_labels')}, indent=2))


if __name__ == '__main__': main()
