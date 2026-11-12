"""Discarded-copy B64/450 memory/backward probe with the real saved Adam."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import psutil
import torch

from .gradient_safety import finite_gradient_norm
from .models import load_world_checkpoint
from .probe_rssm_components import _json_ready, _rng_identity
from .renderer_student import _hash_tensors
from .rssm_combat_branches import normalize_combat_branch_protocol
from .rssm_component_balance import ComponentLossBalancer
from .rssm_migration import configure_rssm_training
from .rssm_objectives import rssm_dynamics_backward
from .rssm_spawn_curriculum import RecoveryCombatSampler, file_sha
from .staged_curriculum import StagedSequenceData
from .training_pipeline import atomic_json


def optimizer_sha(state):
    digest = hashlib.sha256(json.dumps(state['param_groups'], sort_keys=True).encode())
    for identifier, values in sorted(state['state'].items()):
        for name, value in sorted(values.items()):
            digest.update(str((identifier, name)).encode())
            if torch.is_tensor(value):
                value = value.detach().cpu().contiguous()
                digest.update(str((str(value.dtype), list(value.shape))).encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            else: digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def run(args):
    torch.set_num_threads(args.threads)
    torch.cuda.init()  # Before loading any checkpoint, as in actual launchers.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.run_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    source = args.checkpoint.resolve(strict=True); source_root = source.parent.parent
    source_status = json.loads((source_root/'status.json').read_text(encoding='utf-8'))
    pidfile = source_root/'training.pid'
    source_pid = int(pidfile.read_text().strip()) if pidfile.exists() else None
    if source_status.get('status') != 'stopped' or source_pid and psutil.pid_exists(source_pid):
        raise ValueError('Wait for a durable stopped Source checkpoint and exited worker')
    started = time.perf_counter()
    status = {'status': 'initializing', 'optimizer_updated': False, 'optimizer_steps': 0,
        'microbatches_probed_per_batch': 1, 'production_accumulation_steps': 4, 'full_horizon': 450}
    def publish(**values):
        status.update(values, elapsed_seconds=time.perf_counter()-started)
        atomic_json(root/'probe_status.json', status)
        print(json.dumps(_json_ready(status)), flush=True)
    publish(phase='archive_stopped_checkpoint')
    source_sha = file_sha(source)
    step = source_status['current']
    archive = root/f'source_g{step}.pt'
    if archive.exists():
        if file_sha(archive) != source_sha: raise ValueError('Existing archive has a different immutable source')
    else:
        temporary = archive.with_suffix('.copying')
        shutil.copy2(source, temporary)
        if file_sha(temporary) != source_sha or file_sha(source) != source_sha:
            raise RuntimeError('Source changed while making its separate immutable archive')
        temporary.replace(archive)
    publish(phase='model_and_all_Adam_setup', source_step=step, source_checkpoint_sha256=source_sha)
    model, saved = load_world_checkpoint(archive, 'cpu')
    if saved['training_steps'] != step or model.config.navigation_only:
        raise ValueError('Expected the same stopped one-enemy combat source boundary')
    training = saved['run_state']['rssm_training']
    weights_sha = _hash_tensors(model.state_dict())
    model.to('cuda').float(); configure_rssm_training(model)
    parameters = dict(model.named_parameters()); names = saved['optimizer_parameter_names']
    canonical = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if names != canonical: raise ValueError('Actual saved optimizer parameter order changed')
    groups = [{**{key: value for key, value in group.items() if key != 'params'},
        'params': [parameters[names[index]] for index in group['params']]}
        for group in saved['optimizer']['param_groups']]
    optimizer = torch.optim.AdamW(groups)
    optimizer.load_state_dict(saved['optimizer'])
    adam_sha = optimizer_sha(optimizer.state_dict())
    if adam_sha != optimizer_sha(saved['optimizer']):
        raise AssertionError('Real loaded Adam tensors/options differ from the source')
    recovery = json.loads(Path(training['combat_recovery_revision']['plan']).read_text(encoding='utf-8'))
    launch = json.loads((source_root/'launch_command.json').read_text(encoding='utf-8'))
    argv = launch['argv']; combat_root = Path(argv[argv.index('--combat-data-root')+1])
    short = StagedSequenceData(Path(recovery['data_root'])/'moving1/rgb/train', 'cuda',
        mode='stream', cache_bytes=64*1024**2)
    long = StagedSequenceData(combat_root/'full_game/rgb/train', 'cuda',
        mode='stream', cache_bytes=64*1024**2)
    sampler = RecoveryCombatSampler(short, context=1, horizon=16, target_distance_balanced=True,
        attack_positive_balanced=True, cache_path=root/'sampler_cache.json')
    sampler_source = copy.deepcopy(training['combat_recovery_revision']['sampler_state'])
    if sampler_source.get('version') != 2:
        raise ValueError('B64 probe requires the complete actual positive-balanced v2 sampler ledger')
    context = 1+(training['combat_completed']//16)%2
    mode = ['argmax', 'argmax', 'sample', 'argmax'][(training['combat_completed']//4)%4]
    contact = saved['run_state']['navigation_training_history']['contact_revision']['loss_settings']
    actor_source = copy.deepcopy(training['loss_balance_state'])
    balancer = ComponentLossBalancer(actor_source)
    results = []
    for count in ([32, 64] if args.baseline_b32 else [64]):
        optimizer.zero_grad(set_to_none=True)
        sampler.restore_state(sampler_source)
        model.core.train(mode == 'sample')
        rng = np.random.default_rng(); rng.bit_generator.state = copy.deepcopy(saved['rng_state'])
        torch.set_rng_state(saved['torch_rng_state'].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in saved['cuda_rng_state']])
        restored = _rng_identity({'rng_state': rng.bit_generator.state,
            'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all()})
        if restored != _rng_identity(saved): raise AssertionError('Source RNG restoration differs')
        pairs = np.column_stack([rng.integers(len(long.episodes), size=count), np.zeros(count, np.int64)])
        batch = long.batch(count, context=context, horizon=450, indices=pairs)
        auxiliary_pairs = sampler.sample(count, rng)
        auxiliary = short.batch(count, context=1, horizon=16, indices=auxiliary_pairs)
        protocol = normalize_combat_branch_protocol({'version': 2, **(
            {'max_branches': 16, 'max_source_probes': 128, 'max_direct_source_queries': 256} if count == 64 else {})})
        accumulator = balancer.new_microbatch('combat')
        atomic_json(root/f'input_b{count}.json', _json_ready({'source_step': step,
            'source_checkpoint_sha256': source_sha, 'context': context, 'mode': mode,
            'source_rng_sha256': restored, 'long_indices': pairs.tolist(), 'auxiliary_indices': auxiliary_pairs.tolist(),
            'long_files': [str(long.episodes[int(index)]['path']) for index, _ in pairs],
            'auxiliary_files': [str(short.episodes[int(index)]['path']) for index, _ in auxiliary_pairs],
            'sampler_parent': sampler_source, 'sampler_after_private_draw': sampler.state_dict(),
            'sampling_report': sampler.last_sample, 'branch_protocol': protocol,
            'component_coefficients': accumulator.report()}))
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base_allocated, base_reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        measured_start = time.perf_counter()
        def progress(processed, horizon):
            publish(status='running', phase='full450_backward', batch=count,
                transitions_processed=processed, horizon=horizon)
        metrics = rssm_dynamics_backward(model, batch, auxiliary_batch=auxiliary,
            context=context, observed_context=context, auxiliary_context=1, auxiliary_observed_context=1,
            expected_horizon=450, chunk_size=16, amp=nullcontext, progress_callback=progress,
            contact_protocol=contact, combat_protocol={'npc_motion_weight': 5., 'spawn_weight': 1.,
                'game_training_v5': True, 'target_supervision': 'shot_distance_balanced'},
            combat_branch_protocol=protocol, numeric_state_feedback='detached', loss_groups=accumulator)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-measured_start
        norm = float(finite_gradient_norm(model.named_parameters(), context=f'B{count} discarded-copy full450'))
        row = {'batch': count, 'horizon': 450, 'auxiliary_horizon': 16, 'context': context, 'latent_mode': mode,
            'elapsed_seconds': elapsed, 'train_transitions_per_second': count*450/elapsed,
            'baseline_memory_allocated_bytes': base_allocated, 'baseline_memory_reserved_bytes': base_reserved,
            'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
            'max_memory_reserved_bytes': torch.cuda.max_memory_reserved(), 'gradient_norm': norm,
            'all_gradients_finite': True, 'optimizer_steps': 0, 'metrics': metrics, 'branch_protocol': protocol}
        results.append(row)
        atomic_json(root/f'batch{count}_measurement.json', _json_ready(row))
        del batch, auxiliary, accumulator, metrics
        gc.collect()
    publish(phase='immutable_weights_Adam_and_Source_proofs')
    weights_after, adam_after = _hash_tensors(model.state_dict()), optimizer_sha(optimizer.state_dict())
    if weights_after != weights_sha or adam_after != adam_sha:
        raise AssertionError('Probe changed model buffers/weights or an actual Adam state')
    if file_sha(source) != source_sha or file_sha(archive) != source_sha:
        raise AssertionError('Original or archived source checkpoint changed')
    if balancer.state_dict() != actor_source: raise AssertionError('Probe committed a production component EMA')
    report = {'passed': True, 'source_step': step, 'source_checkpoint': str(archive),
        'source_checkpoint_sha256': source_sha, 'source_checkpoint_unchanged': True,
        'source_sampler_state_unchanged_in_checkpoint': True, 'model_weights_and_buffers_unchanged': True,
        'model_weights_sha256': weights_sha, 'all_Adam_tensors_and_options_unchanged': True,
        'Adam_sha256': adam_sha, 'Adam_parameter_states': len(optimizer.state),
        'optimizer_steps': 0, 'optimizer_updated': False, 'component_EMA_updates': 0,
        'microbatch_size': 64, 'all_gradients_finite': all(row['all_gradients_finite'] for row in results),
        'full_horizon': 450, 'tbptt_chunk': 16,
        'precision': 'fp32', 'tf32_enabled': False, 'production_accumulation_steps': 4,
        'microbatches_probed_per_batch': 1, 'measurements': results,
        'scope': 'Training backward memory/speed on a discarded model+Adam; not gameplay FPS or a committed256-fragment update',
        'elapsed_seconds': time.perf_counter()-started}
    atomic_json(root/'batch64_probe.json', _json_ready(report))
    publish(status='completed', phase='completed', passed=True, report=str(root/'batch64_probe.json'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--baseline-b32', action='store_true')
    args = parser.parse_args()
    try: run(args)
    except Exception as error:
        atomic_json(args.run_dir/'probe_error.json', {'status': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__': main()
