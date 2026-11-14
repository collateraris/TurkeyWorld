"""Collect genuine TRAIN Source episodes with a frozen learned policy controller.

The controller observes each current Source RGB/state through all three RSSM
posteriors, preserving their memory. This is real-environment data collection,
not prior imagination; the actor's imagined training path is untouched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .dreamer_actor_critic import ActorCriticConfig, FactorizedActor
from .dreamer_policy_world import RSSMImaginationAdapter
from .engine import Game, BEHAVIOR_VERSION, ENTITY_SLOTS
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .rssm_combat_protocol import environment_profile
from .training_pipeline import atomic_json, now

PROTOCOL = 'rssm_motion_only_Source_cycle_corpus_v1'
POLICY_PROTOCOL = 'dreamer_actor_critic_imagination_v1'
EXPLORATION_ACTIONS = np.asarray([(*values, 0.) for values in itertools.product((-1., 0., 1.), repeat=3)
    if any(values)], np.float32)


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Policy Source collection immutable plan changed')
    else:
        atomic_json(path, value)


def _pin(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists(): os.link(source, destination)
    if not destination.is_file(): raise ValueError('Missing pinned collection checkpoint')
    if file_sha(source) != file_sha(destination):
        raise ValueError('An existing cycle cannot silently switch to different policy/world checkpoint bytes')
    return dict(path=str(destination), sha256=file_sha(destination), original_path=str(source))


def _initial_source(seed, stream, *, min_enemy_distance=8., source_factory=Game):
    game = source_factory(seed=seed, endless=True, enemies_enabled=stream == 'enemy_motion')
    rng = np.random.default_rng(seed)
    floor = np.argwhere(game.map == 0)[:, ::-1].astype(float)+.5
    pairs = []
    for _ in range(256):
        player = floor[int(rng.integers(len(floor)))]
        candidates = np.flatnonzero(np.linalg.norm(floor-player, axis=1) >= min_enemy_distance)
        if stream == 'navigation' or len(candidates):
            enemy = floor[int(rng.choice(candidates))] if len(candidates) else None
            pairs.append((player, enemy)); break
    if not pairs: raise ValueError('No legal distant Source initialization for quiet enemy')
    player, enemy = pairs[0]
    snapshot = game.get_state()
    snapshot['player'].update(x=float(player[0]), y=float(player[1]), angle=float(rng.uniform(-math.pi, math.pi)))
    snapshot['time'] = float(rng.integers(1800))/30
    if stream == 'enemy_motion':
        snapshot['enemies'][0].update(x=float(enemy[0]), y=float(enemy[1]), vx=0., vy=0.)
    game.set_state(snapshot)
    return game


def _rgb_tensor(images, device):
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)/255


def _cohort_valid(states, actions, stream):
    states, actions = np.asarray(states), np.asarray(actions)
    if not np.isfinite(states).all() or not np.isfinite(actions).all(): return False, 'nonfinite_Source'
    if np.any(actions[:, 3] != 0): return False, 'fire_mask_failed'
    if np.any(states[:, 4] <= 0) or not np.all(states[:, 4] == states[0, 4]): return False, 'player_HP_changed'
    if stream == 'navigation':
        if np.any(states[:, 12:14] != 0): return False, 'navigation_NPC_present'
    elif (np.any(states[:, 12] <= 0) or np.any(states[:, 13] < .5)
            or not np.all(states[:, 12] == states[0, 12])):
        return False, 'quiet_enemy_HP_or_life_changed'
    return True, None


def _episode_record(path):
    with np.load(path, allow_pickle=False) as saved:
        stream, steps = str(saved['motion_stream']), len(saved['actions'])
        valid, reason = _cohort_valid(saved['states'], saved['actions'], stream)
        if (not valid or str(saved['split']) != 'train' or str(saved['collection_protocol']) != PROTOCOL
                or saved['states'].shape != (steps+1, 20) or saved['frames'].shape != (steps+1, 32, 32, 3)):
            raise ValueError('Invalid collected Source episode: '+str(reason))
        return dict(path=str(Path(path).resolve()), sha256=file_sha(path), split='train', stream=stream,
            seed=int(saved['seed']), episode_index=int(saved['episode_index']), attempt=int(saved['collection_attempt']),
            transitions=steps, observations=steps+1, policy_checkpoint_sha256=str(saved['policy_checkpoint_sha256']),
            world_checkpoint_sha256=str(saved['world_checkpoint_sha256']),
            exploration_steps=int(saved['exploration_flags'].sum()), fire_masked_steps=int(saved['fire_masked_flags'].sum()),
            full_initial_snapshot=True, Source_state_repairs=0, behavior_version=str(saved['behavior_version']))


def _write_episode(path, tape, *, stream, index, attempt, seed, policy_pin, world_pin, plan):
    initial_json = json.dumps(tape['initial'], sort_keys=True, allow_nan=False)
    payload = dict(map=tape['map'].astype(np.int8), states=np.asarray(tape['states'], np.float32),
        frames=np.asarray(tape['frames'], np.uint8), actions=np.asarray(tape['actions'], np.float32),
        events=np.asarray(tape['events'], np.float32), hit_ids=np.asarray(tape['hit_ids'], np.int64),
        attack_flags=np.asarray(tape['attack_flags'], np.float32), respawn_flags=np.asarray(tape['respawn_flags'], np.float32),
        exploration_flags=np.asarray(tape['exploration'], bool), fire_masked_flags=np.asarray(tape['fire_masked'], bool),
        initial_state=np.asarray(initial_json), initial_state_sha256=np.asarray(hashlib.sha256(initial_json.encode()).hexdigest()),
        initial_rgb_sha256=np.asarray(hashlib.sha256(tape['frames'][0].tobytes()).hexdigest()),
        seed=np.asarray(seed), split=np.asarray('train'), stage=np.asarray('navigation' if stream == 'navigation' else 'full_game'),
        case=np.asarray('policy_no_enemy' if stream == 'navigation' else 'policy_quiet_enemy'),
        motion_stream=np.asarray(stream), episode_index=np.asarray(index), collection_attempt=np.asarray(attempt),
        enemies_enabled=np.asarray(stream == 'enemy_motion'), navigation_only=np.asarray(stream == 'navigation'),
        dt=np.asarray(1/30), hz=np.asarray(30), state_dim=np.asarray(20), entity_slots=np.asarray(ENTITY_SLOTS),
        behavior_version=np.asarray(BEHAVIOR_VERSION), state_schema_version=np.asarray('single_enemy_v1'),
        world_transition_compatible=np.asarray(True), format_version=np.asarray(4),
        collection_protocol=np.asarray(PROTOCOL), cycle_index=np.asarray(plan['cycle_index']),
        policy=np.asarray('current_Source_posterior_argmax_with_held_epsilon_blocks'),
        controller_observation=np.asarray(plan['controller_observation']),
        policy_checkpoint_sha256=np.asarray(policy_pin['sha256']), world_checkpoint_sha256=np.asarray(world_pin['sha256']))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise ValueError('Refusing to overwrite an accepted immutable Source episode')
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)
    return _episode_record(path)


@torch.no_grad()
def _collect_batch(actor, adapter, jobs, *, stream, transitions, exploration, device,
                   progress_callback, output_dir):
    games = [_initial_source(job['seed'], stream) for job in jobs]
    tapes = [dict(initial=game.get_state(), map=game.map.copy(), states=[game.state_vector().copy()],
        frames=[game.render(size=32, overlays=False)], actions=[], events=[], hit_ids=[], attack_flags=[],
        respawn_flags=[], exploration=[], fire_masked=[]) for game in games]
    rngs = [np.random.default_rng(job['seed']+173) for job in jobs]
    remaining = [0]*len(jobs); exploring = [False]*len(jobs)
    held = [np.zeros(4, np.float32) for _ in jobs]
    state = torch.as_tensor(np.stack([tape['states'][0] for tape in tapes]), device=device)
    grid = torch.as_tensor(np.stack([tape['map'] for tape in tapes]), device=device)
    with environment_profile(adapter.world, navigation_only=stream == 'navigation'):
        current = adapter.initialize(state, grid, _rgb_tensor([tape['frames'][0] for tape in tapes], device))
        for tick in range(transitions):
            if (Path(output_dir)/'stop.request').exists(): raise InterruptedError('Source collection stop requested')
            proposed = actor.act(adapter.features(current), deterministic=True).detach().cpu().numpy()
            applied = []
            for index, (game, tape, rng) in enumerate(zip(games, tapes, rngs)):
                if remaining[index] == 0:
                    remaining[index] = int(rng.integers(4, 13))
                    exploring[index] = bool(rng.random() < exploration)
                    if exploring[index]: held[index] = EXPLORATION_ACTIONS[int(rng.integers(len(EXPLORATION_ACTIONS)))].copy()
                control = held[index].copy() if exploring[index] else proposed[index].astype(np.float32).copy()
                masked = bool(control[3] != 0)
                control[3] = 0.
                before = np.asarray([enemy['alive'] for enemy in game.enemies])
                event = game.step(control, dt=1/30)
                following = game.state_vector().copy()
                image = game.render(size=32, overlays=False)
                flags = np.zeros(ENTITY_SLOTS, np.float32); flags[event.get('attacker_ids', [])] = 1
                tape['actions'].append(control); tape['states'].append(following); tape['frames'].append(image)
                tape['events'].append(Game.event_vector(event)); tape['hit_ids'].append(event.get('hit_id', -1))
                tape['attack_flags'].append(flags)
                tape['respawn_flags'].append((~before & np.asarray([enemy['alive'] for enemy in game.enemies])).astype(np.float32))
                tape['exploration'].append(exploring[index]); tape['fire_masked'].append(masked)
                applied.append(control); remaining[index] -= 1
            # Actual matching Source posteriors, not reset memory or own-prior
            # states disguised as observations. The next actor sees this RGB.
            current = adapter.observe(current, torch.as_tensor(np.stack(applied), device=device),
                torch.as_tensor(np.stack([tape['states'][-1] for tape in tapes]), device=device),
                _rgb_tensor([tape['frames'][-1] for tape in tapes], device))
            if progress_callback and (tick % 15 == 0 or tick+1 == transitions):
                progress_callback(dict(stream=stream, transitions_completed=tick+1,
                    horizon=transitions, batch_episodes=len(jobs)))
    return tapes


def collect_policy_motion(policy_checkpoint, world_checkpoint, output_dir, *, cycle_index,
                          episodes_per_stream=16, transitions=180, seedbase=1200000000,
                          exploration=.3, max_attempts=16, inference_batch_size=16,
                          device='cuda', progress_callback=None):
    if (type(cycle_index) is not int or cycle_index < 0
            or any(type(value) is not int or value < 1 for value in (episodes_per_stream, transitions, max_attempts, inference_batch_size))
            or transitions > 180 or episodes_per_stream > 128 or max_attempts > 128
            or type(seedbase) is not int or seedbase < 0
            or type(exploration) not in (int, float) or not math.isfinite(exploration) or not 0 <= exploration <= 1):
        raise ValueError('Invalid bounded policy-motion Source collection contract')
    device = torch.device(device)
    if device.type == 'cuda': torch.cuda.init()
    torch.set_num_threads(2)
    root = Path(output_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
    policy_pin = _pin(policy_checkpoint, root/'assets/policy_source.pt')
    world_pin = _pin(world_checkpoint, root/'assets/world_source.pt')
    plan = _json(dict(protocol=PROTOCOL, cycle_index=cycle_index, split='train', validation_unchanged=True,
        policy_checkpoint=policy_pin, world_checkpoint=world_pin, episodes_per_stream=episodes_per_stream,
        transitions=transitions, seedbase=seedbase, exploration_block_probability=exploration,
        exploration_hold_ticks=[4, 12], exploration_zero_vector_allowed=False,
        policy_action_mode='categorical_argmax_recomputed_each_Source_tick',
        fire_masked_during_collection=True, fire_mask_changes_actor_weights=False,
        controller_observation='causal_current_Source_RGB32_state20_and_all_three_RSSM_posteriors',
        initialization_observation_only=False, posterior_each_real_Source_transition=True,
        memory_reset_after_initialization=False, imagination_training_Source_feedback=False,
        max_attempts_per_episode=max_attempts, inference_batch_size=inference_batch_size,
        quiet_HP_change_rejection=True, initial_enemy_min_distance_tiles=8., Source_state_repairs=0,
        behavior_version=BEHAVIOR_VERSION, real_Source_dt=1/30, native_size=32))
    _immutable_json(root/'collection_plan.json', plan)
    receipt_path = root/'collection_receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['collection_plan_sha256'] != file_sha(root/'collection_plan.json'):
            raise ValueError('Completed Source receipt and immutable collection plan disagree')
        for row in receipt['episodes']:
            if file_sha(row['path']) != row['sha256']: raise ValueError('Accepted Source episode changed')
        return receipt
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        policy_saved = torch.load(policy_pin['path'], map_location='cpu', weights_only=False, mmap=True)
        if (policy_saved.get('protocol') != POLICY_PROTOCOL or type(policy_saved.get('step')) is not int
                or type(policy_saved.get('actor_updates')) is not int or policy_saved['actor_updates'] < 1):
            raise ValueError('Source controller requires the pinned trained Dreamer actor checkpoint')
        config = ActorCriticConfig.from_dict(policy_saved['actor_critic_config'])
        actor = FactorizedActor(config).to(device).eval().requires_grad_(False)
        actor.load_state_dict({name[len('actor.'):]: value for name, value in policy_saved['policy'].items()
            if name.startswith('actor.')}, strict=True)
        world, world_saved = load_world_checkpoint(world_pin['path'], device)
        if (world_saved.get('training_steps') != world_saved.get('global_training_step')
                or world_saved.get('training_steps') != world_saved.get('run_state', {}).get('global_training_step')):
            raise ValueError('Source collector world is not a complete optimizer boundary')
        adapter = RSSMImaginationAdapter(world, heads=None, sample_latents=False)
        if config.feature_dim != adapter.feature_dim: raise ValueError('Source actor and refreshed world feature schemas differ')
        world_parameters = _hash_tensors(dict(world.named_parameters()))
        actor_parameters = _hash_tensors(actor.state_dict())
        episodes = []
        rejected_path = root/'rejected_attempts.json'
        rejected = json.loads(rejected_path.read_text(encoding='utf-8')) if rejected_path.exists() else []
        directories = {stream: root/stream/'rgb/train' for stream in ('navigation', 'enemy_motion')}
        started = time.perf_counter()
        for role, stream in enumerate(('navigation', 'enemy_motion')):
            directories[stream].mkdir(parents=True, exist_ok=True)
            accepted = {}
            for path in sorted(directories[stream].glob('*.npz')):
                record = _episode_record(path)
                if (record['policy_checkpoint_sha256'] != policy_pin['sha256']
                        or record['world_checkpoint_sha256'] != world_pin['sha256']
                        or record['transitions'] != transitions or not 0 <= record['episode_index'] < episodes_per_stream
                        or record['episode_index'] in accepted):
                    raise ValueError('Existing Source episode disagrees with immutable collection plan')
                accepted[record['episode_index']] = record
            attempts = {index: 0 for index in range(episodes_per_stream) if index not in accepted}
            for row in rejected:
                if row['stream'] == stream and row['index'] in attempts:
                    attempts[row['index']] = max(attempts[row['index']], row['attempt']+1)
            while len(accepted) < episodes_per_stream:
                pending = sorted(index for index in attempts if index not in accepted)[:inference_batch_size]
                jobs = []
                for index in pending:
                    attempt = attempts[index]
                    if attempt >= max_attempts: raise RuntimeError('Bounded quiet/no-enemy Source collection exhausted retries')
                    seed = seedbase+cycle_index*10000000+role*1000000+index*1000+attempt
                    jobs.append(dict(index=index, attempt=attempt, seed=seed)); attempts[index] += 1
                def progress(value):
                    value = dict(value, protocol=PROTOCOL, cycle_index=cycle_index, status='collecting',
                        accepted=len(episodes)+len(accepted), total_episodes=2*episodes_per_stream,
                        rejected_attempts=len(rejected), elapsed_seconds=time.perf_counter()-started, updated_at=now())
                    atomic_json(root/'collection_progress.json', value)
                    if progress_callback: progress_callback(value)
                tapes = _collect_batch(actor, adapter, jobs, stream=stream, transitions=transitions,
                    exploration=exploration, device=device, progress_callback=progress, output_dir=root)
                for job, tape in zip(jobs, tapes):
                    valid, reason = _cohort_valid(tape['states'], tape['actions'], stream)
                    if not valid:
                        rejected.append(dict(stream=stream, **job, reason=reason)); continue
                    path = directories[stream]/f'episode_{job["seed"]:012d}.npz'
                    accepted[job['index']] = _write_episode(path, tape, stream=stream, index=job['index'],
                        attempt=job['attempt'], seed=job['seed'], policy_pin=policy_pin, world_pin=world_pin, plan=plan)
                atomic_json(root/'rejected_attempts.json', rejected)
            episodes.extend(accepted[index] for index in sorted(accepted))
        if (_hash_tensors(dict(world.named_parameters())) != world_parameters
                or _hash_tensors(actor.state_dict()) != actor_parameters):
            raise AssertionError('Source collection modified frozen policy/world parameters')
        receipt = dict(kind=PROTOCOL, protocol=PROTOCOL, split='train', validation_unchanged=True,
            cycle_index=cycle_index, policy_checkpoint_sha256=policy_pin['sha256'], world_checkpoint_sha256=world_pin['sha256'],
            policy_checkpoint=policy_pin['path'], world_checkpoint=world_pin['path'],
            policy_step=policy_saved['step'], policy_actor_updates=policy_saved['actor_updates'],
            world_step=world_saved['training_steps'], nav_train_directory=str(directories['navigation']),
            enemy_train_directory=str(directories['enemy_motion']), episodes=episodes,
            episodes_per_stream=episodes_per_stream, transitions_per_episode=transitions,
            total_real_Source_transitions=2*episodes_per_stream*transitions,
            collection_plan_sha256=file_sha(root/'collection_plan.json'), controller_observation=plan['controller_observation'],
            source_policy_world_identity=deepcopy(policy_saved.get('world_identity')),
            collected_current_posterior=True, imagination_Source_feedback=False, fire_mask=True,
            Source_state_repairs=0, independent_Source_episodes=True, rejected_attempts=rejected,
            world_parameters_bitexact=True, policy_parameters_bitexact=True,
            actor_RNG_preserved_outside_collection=True, duration_seconds=time.perf_counter()-started, completed_at=now())
        atomic_json(receipt_path, receipt)
        atomic_json(root/'manifest.json', dict(receipt, status='complete'))
        atomic_json(root/'collection_progress.json', dict(status='complete', accepted=len(episodes),
            total_episodes=2*episodes_per_stream, updated_at=now()))
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('policy-checkpoint', 'world-checkpoint', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cycle-index', type=int, required=True)
    parser.add_argument('--episodes-per-stream', type=int, default=16)
    parser.add_argument('--transitions', type=int, default=180)
    parser.add_argument('--seedbase', type=int, default=1200000000)
    parser.add_argument('--exploration', type=float, default=.3)
    parser.add_argument('--max-attempts', type=int, default=16)
    parser.add_argument('--inference-batch-size', type=int, default=16)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args(argv)
    if args.device == 'cuda': torch.cuda.init()
    torch.set_num_threads(2)
    with KeepAwake():
        result = collect_policy_motion(**vars(args), progress_callback=lambda value: print(json.dumps(value), flush=True))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__': main()
