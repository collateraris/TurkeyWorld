"""Label valid states visited by a frozen neural model, for training only.

Each main rollout receives one randomized valid source initialization, its RGB
image and a prerecorded action tape. All later inputs are the model's own state
and visual memory. Separate oracle branches start from an unmodified valid own
state after that rollout ends; their labels never correct the model trajectory.
Invalid geometry is skipped, without projection or reference-state fallback.
"""
from __future__ import annotations

from .engine import GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS, STATE_DIM

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .engine import ENTITY_SLOTS, EVENT_NAMES, Game
from .geometry_targets import build_geometry_targets
from .models import load_world_checkpoint
from .prepare_training_data import ClipPolicy, FORMAT_VERSION, _atomic_json, _random_initial_state
from .training_pipeline import model_fingerprint


COLLECTION_KIND = 'model_visited_state_oracle_branch'


def _restore_unmodified_candidate(game, vector):
    """Reject adapters that would clip/repair anything beyond float32 decoding."""
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (STATE_DIM,) or not np.isfinite(vector).all() or vector[4] <= 0:
        raise ValueError('A candidate must be a finite living model state')
    game.set_state_vector(vector)
    decoded = game.state_vector()
    if not np.allclose(decoded, vector, rtol=0, atol=2e-6):
        raise ValueError('Restoring this model state would require a numeric state repair')
    return decoded


def _record_branch(game, *, seed, steps, size, hz, policy_name, visited_vector,
                   visited_step, fingerprint):
    initial_state = json.dumps(game.get_state(), ensure_ascii=False)
    policy = ClipPolicy(seed+5000, policy_name, hz, steps)
    states, frames, depth, visibility = [], [], [], []
    actions, events, hit_ids, attack_flags, respawn_flags = [], [], [], [], []

    def observe():
        state = game.state_vector()
        states.append(state)
        frames.append(game.render(size=size, overlays=False))
        geometry = build_geometry_targets(game.map, state)
        depth.append(geometry['depth'])
        visibility.append(geometry['visibility'])

    observe()
    for index in range(steps):
        action = np.asarray(policy(game, index), dtype=np.float32)
        before_alive = np.array([enemy['alive'] for enemy in game.enemies])
        event = game.step(action, dt=1/hz)
        actions.append(action.copy())
        events.append([float(bool(event.get(name, False))) for name in EVENT_NAMES])
        hit_ids.append(int(event.get('hit_id', -1)))
        attacks = np.zeros(ENTITY_SLOTS, dtype=np.float32)
        attacks[event.get('attacker_ids', [])] = 1.
        attack_flags.append(attacks)
        respawn_flags.append(~before_alive & np.array([enemy['alive'] for enemy in game.enemies]))
        observe()
        if game.player['hp'] <= 0:
            break
    return {
        'format_version': np.asarray(FORMAT_VERSION), 'map': game.map.astype(np.int8),
        'states': np.asarray(states, dtype=np.float32), 'frames': np.asarray(frames, dtype=np.uint8),
        'actions': np.asarray(actions, dtype=np.float32), 'events': np.asarray(events, dtype=np.float32),
        'hit_ids': np.asarray(hit_ids, dtype=np.int64),
        'attack_flags': np.asarray(attack_flags, dtype=np.float32),
        'respawn_flags': np.asarray(respawn_flags, dtype=np.float32),
        'depth': np.asarray(depth, dtype=np.float32), 'visibility': np.asarray(visibility, dtype=np.float32),
        'initial_state': np.asarray(initial_state), 'seed': np.asarray(seed), 'dt': np.asarray(1/hz),
        'policy': np.asarray(policy_name), 'split': np.asarray('train'), 'endless': np.asarray(game.endless),
        'warmup_steps': np.asarray(0), 'randomized_initialization': np.asarray(False),
        'requested_transitions': np.asarray(steps), 'collection_kind': np.asarray(COLLECTION_KIND),
        'teacher_fingerprint': np.asarray(fingerprint), 'visited_step': np.asarray(visited_step),
        'model_visited_state': np.asarray(visited_vector, dtype=np.float32),
        'future_oracle_feedback': np.asarray(False), 'model_state_repairs': np.asarray(0),
    }


def _existing_record(path, fingerprint, steps, max_model_steps):
    with np.load(path, allow_pickle=False) as data:
        if (str(data['collection_kind']) != COLLECTION_KIND or str(data['teacher_fingerprint']) != fingerprint
                or str(data['split']) != 'train' or int(data['requested_transitions']) != steps
                or int(data['max_model_steps']) != max_model_steps):
            raise ValueError(f'Existing visited-state clip belongs to another collection: {path}')
        transitions = len(data['actions'])
        if (data['states'].shape != (transitions+1, STATE_DIM) or data['frames'].shape[0] != transitions+1
                or data['actions'].shape != (transitions, 4) or data['events'].shape != (transitions, 4)
                or transitions < 1 or not np.allclose(data['states'][0], data['model_visited_state'],
                                                     rtol=0, atol=2e-6)):
            raise ValueError(f'Invalid existing visited-state clip: {path}')
        totals = data['events'].sum(0).astype(int)
        return {'path': str(path.resolve()), 'seed': int(data['seed']), 'split': 'train',
            'visited_step': int(data['visited_step']), 'transitions': transitions,
            'events': {name: int(totals[i]) for i, name in enumerate(EVENT_NAMES)},
            'respawns': int(data['respawn_flags'].sum()), 'resumed': True}


def collect_model_visited_clips(checkpoint125, outputdir_train, *, count=256, seed=20_000_000,
                               steps=32, max_model_steps=32, device='cuda', candidate_interval=4,
                               max_attempts=None):
    """Return an honest manifest, including incomplete collections with no valid states.

    The default budget is four independent source initializations per requested
    clip. The latest valid sampled own state is selected; its validation is not
    supplied to the completed main rollout. Only the training split is allowed.
    """
    if min(count, steps, max_model_steps, candidate_interval) < 1 or seed < 0:
        raise ValueError('Collection counts must be positive and seed nonnegative')
    output = Path(outputdir_train).resolve()
    if output.name.lower() != 'train':
        raise ValueError('Model-visited oracle labels may only be written to a train directory')
    max_attempts = count*4 if max_attempts is None else max_attempts
    if max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    model, _ = load_world_checkpoint(checkpoint125, 'cpu')
    if model.config.architecture != 'vjepa2_ac' or model.config.kind != 'teacher':
        raise ValueError('Visited-state collection requires the selected V-JEPA teacher')
    fingerprint = model_fingerprint(model)
    model = model.to(device).eval().requires_grad_(False)
    hz, size = float(model.config.simulation_hz), model.config.image_size
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'model_visited_manifest.json'
    configuration = {'checkpoint': str(Path(checkpoint125).resolve()), 'teacher_fingerprint': fingerprint,
                     'requested_clips': count, 'seed': seed, 'steps': steps,
                     'max_model_steps': max_model_steps, 'candidate_interval': candidate_interval,
                     'max_attempts': max_attempts, 'split': 'train', 'collection_kind': COLLECTION_KIND}
    records = []
    counters = {'attempts': 0, 'valid_candidates': 0, 'invalid_candidates': 0,
                'attempts_without_valid_state': 0, 'source_initialization_failures': 0,
                'main_model_transitions': 0, 'source_image_initializations': 0}
    started = time.perf_counter()

    def save_manifest():
        manifest = dict(configuration, **counters, completed_clips=len(records),
            complete=len(records) >= count, status='complete' if len(records) >= count else 'incomplete',
            records=records, seconds=time.perf_counter()-started,
            future_oracle_feedback=False, model_state_repairs=0,
            selector='latest_valid_sampled_own_state', source_split='new_training_seeds',
            created_at=datetime.now(timezone.utc).isoformat())
        _atomic_json(manifest_path, manifest)
        return manifest

    # Deterministic attempt-index filenames permit resumable skips without
    # changing which random initialization belongs to a previously saved clip.
    policies = ('combat', 'misses', 'attack', 'sweep')
    for attempt in range(max_attempts):
        if len(records) >= count:
            break
        episode_seed = seed+attempt
        path = output/f'visited_{episode_seed:09d}.npz'
        counters['attempts'] += 1
        if path.exists():
            records.append(_existing_record(path, fingerprint, steps, max_model_steps))
            save_manifest()
            continue
        policy_name = policies[attempt % len(policies)]
        source = Game(seed=episode_seed, endless=model.config.endless)
        try:
            _random_initial_state(source, np.random.default_rng(episode_seed+77), policy_name)
        except ValueError:
            counters['source_initialization_failures'] += 1
            continue
        initial_vector, initial_snapshot = source.state_vector(), source.get_state()
        initial_rgb = source.render(size=size, overlays=False)
        # The controller sees the source trajectory only while making this fixed
        # tape, before the neural trajectory begins. No tape label enters it.
        policy = ClipPolicy(episode_seed+5000, policy_name, hz, max_model_steps)
        tape = []
        for index in range(max_model_steps):
            action = np.asarray(policy(source, index), dtype=np.float32).copy()
            tape.append(action)
            source.step(action, dt=1/hz)
            if source.player['hp'] <= 0:
                break
        candidates = []
        with torch.inference_mode():
            state = torch.as_tensor(initial_vector, device=device)[None]
            rgb = torch.as_tensor(initial_rgb, device=device).permute(2, 0, 1)[None].float()/255
            fmap = model.encode_map(torch.as_tensor(source.map, device=device, dtype=torch.long)[None])
            memory = model.initialize_from_image(rgb, state, fmap)
            counters['source_image_initializations'] += 1
            for index, action in enumerate(tape, start=1):
                result = model.transition(state, torch.as_tensor(action, device=device)[None], fmap, memory)
                state, memory = result['state'], result['memory']
                counters['main_model_transitions'] += 1
                if index % candidate_interval == 0 or index == len(tape):
                    candidates.append((index, state[0].detach().cpu().numpy().copy()))
                if not bool(torch.isfinite(state).all()) or float(state[0, 4]) <= 0:
                    break
        # The main rollout is complete before any oracle candidate validation.
        # Never replace `state`/`memory`, or continue it from an oracle result.
        selected = None
        for visited_step, candidate in candidates:
            oracle = Game(seed=episode_seed, endless=model.config.endless)
            oracle.set_state(initial_snapshot)
            try:
                _restore_unmodified_candidate(oracle, candidate)
            except ValueError:
                counters['invalid_candidates'] += 1
                continue
            counters['valid_candidates'] += 1
            selected = (oracle, visited_step, candidate)
        if selected is None:
            counters['attempts_without_valid_state'] += 1
            save_manifest()
            continue
        oracle, visited_step, candidate = selected
        payload = _record_branch(oracle, seed=episode_seed, steps=steps, size=size, hz=hz,
            policy_name=policy_name, visited_vector=candidate, visited_step=visited_step,
            fingerprint=fingerprint)
        payload['max_model_steps'] = np.asarray(max_model_steps)
        payload['source_initial_state'] = np.asarray(json.dumps(initial_snapshot, ensure_ascii=False))
        payload['candidate_interval'] = np.asarray(candidate_interval)
        temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
        record = _existing_record(path, fingerprint, steps, max_model_steps)
        record['resumed'] = False
        records.append(record)
        save_manifest()
        print(json.dumps({'collection_kind': COLLECTION_KIND, 'completed_clips': len(records),
                          'requested_clips': count, 'attempts': counters['attempts'],
                          'visited_step': visited_step, 'path': str(path)}, ensure_ascii=True), flush=True)
    return save_manifest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Destination rgb/train directory')
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20_000_000)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--max-model-steps', type=int, default=32)
    parser.add_argument('--candidate-interval', type=int, default=4)
    parser.add_argument('--max-attempts', type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    manifest = collect_model_visited_clips(args.checkpoint, args.output, count=args.count, seed=args.seed,
        steps=args.steps, max_model_steps=args.max_model_steps, device=args.device,
        candidate_interval=args.candidate_interval, max_attempts=args.max_attempts)
    print(json.dumps({key: value for key, value in manifest.items() if key != 'records'}, ensure_ascii=True), flush=True)
    return 0 if manifest['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
