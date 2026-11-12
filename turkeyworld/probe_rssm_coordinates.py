"""Discarded real B64 short16/full450 coordinate and memory backward probes."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from .gradient_safety import finite_gradient_norm
from .models import load_world_checkpoint
from .probe_rssm_batch64 import optimizer_sha
from .probe_rssm_components import _json_ready, _rng_identity
from .renderer_student import _hash_tensors
from .rssm_combat_branches import normalize_combat_branch_protocol
from .rssm_component_balance import ComponentLossBalancer
from .rssm_coordinate_gradients import COORDINATE_GRADIENT_PROTOCOL
from .rssm_kinematic_losses import KINEMATIC_PROTOCOL_V1
from .rssm_memory_overshooting import MEMORY_OVERSHOOTING_PROTOCOL
from .rssm_migration import configure_rssm_training
from .rssm_objectives import rssm_dynamics_backward
from .rssm_spawn_curriculum import RecoveryCombatSampler, file_sha
from .staged_curriculum import StagedSequenceData
from .training_pipeline import atomic_json


def run(args):
    torch.set_num_threads(args.threads)
    torch.cuda.init()  # Same ordering as the real launcher, before torch.load.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root/'coordinates_probe.json'
    if report_path.exists():
        raise FileExistsError('Preserve the existing coordinate probe receipt')
    source = args.checkpoint.resolve(strict=True)
    source_root = source.parent.parent
    source_status = json.loads((source_root/'status.json').read_text(encoding='utf-8'))
    pidfile = source_root/'training.pid'
    source_pid = int(pidfile.read_text().strip()) if pidfile.exists() else None
    if source_status.get('status') != 'stopped' or source_pid and psutil.pid_exists(source_pid):
        raise ValueError('Coordinate probe requires a durable stopped Source and exited worker')
    started = time.perf_counter()
    status = {'status': 'initializing', 'optimizer_steps': 0, 'EMA_commits': 0,
              'batch': 64, 'horizons': [16, 450], 'production_accumulation_steps': 4}
    def publish(**values):
        status.update(values, elapsed_seconds=time.perf_counter()-started)
        atomic_json(root/'coordinates_probe_status.json', status)
        print(json.dumps(_json_ready(status)), flush=True)
    publish(phase='load_discarded_model_and_complete_Adam')
    source_sha = file_sha(source)
    model, saved = load_world_checkpoint(source, 'cpu')
    step = saved['training_steps']
    if (step != args.source_step or source_status['current'] != step
            or saved['global_training_step'] != step or model.config.navigation_only):
        raise ValueError('Coordinate probe must retain the exact stopped one-enemy Source boundary')
    training = saved['run_state']['rssm_training']
    source_training_sha = json.dumps(_json_ready(training), sort_keys=True)
    weights_sha = _hash_tensors(model.state_dict())
    model.to('cuda').float()
    configure_rssm_training(model)
    named = dict(model.named_parameters())
    names = saved['optimizer_parameter_names']
    if names != [name for name, value in model.named_parameters() if value.requires_grad]:
        raise ValueError('Saved Adam parameter names or order changed')
    groups = [{**{key: value for key, value in group.items() if key != 'params'},
               'params': [named[names[index]] for index in group['params']]}
              for group in saved['optimizer']['param_groups']]
    optimizer = torch.optim.AdamW(groups)
    optimizer.load_state_dict(saved['optimizer'])
    adam_sha = optimizer_sha(optimizer.state_dict())
    if adam_sha != optimizer_sha(saved['optimizer']) or len(optimizer.state) != 165 or len(names) != 165:
        raise AssertionError('Coordinate probe did not preserve all165 Source Adam parameter states/options')
    recovery = json.loads(Path(training['combat_recovery_revision']['plan']).read_text(encoding='utf-8'))
    argv = json.loads((source_root/'launch_command.json').read_text(encoding='utf-8'))['argv']
    combat_root = Path(argv[argv.index('--combat-data-root')+1])
    short = StagedSequenceData(Path(recovery['data_root'])/'moving1/rgb/train', 'cuda',
                               mode='stream', cache_bytes=64*1024**2)
    long = StagedSequenceData(combat_root/'full_game/rgb/train', 'cuda',
                              mode='stream', cache_bytes=64*1024**2)
    sampler = RecoveryCombatSampler(short, context=1, horizon=16, target_distance_balanced=True,
        attack_positive_balanced=True, cache_path=root/'coordinates_probe_sampler_cache.json')
    sampler_source = copy.deepcopy(training['combat_recovery_revision']['sampler_state'])
    if sampler_source.get('version') != 2:
        raise ValueError('Coordinate probes require the actual positive-balanced v2 sampler state')
    actor_source = copy.deepcopy(training['loss_balance_state'])
    balancer = ComponentLossBalancer(actor_source)
    context = 1+(training['combat_completed']//16)%2
    mode = ['argmax', 'argmax', 'sample', 'argmax'][(training['combat_completed']//4)%4]
    contact = saved['run_state']['navigation_training_history']['contact_revision']['loss_settings']
    branches = normalize_combat_branch_protocol({'version': 2, 'max_branches': 16,
        'max_source_probes': 128, 'max_direct_source_queries': 256})
    combat = {'npc_motion_weight': 5., 'spawn_weight': 1., 'game_training_v5': True,
              'target_supervision': 'shot_distance_balanced', 'kinematic_protocol': KINEMATIC_PROTOCOL_V1}
    rng = np.random.default_rng()
    source_rng_sha = _rng_identity(saved)
    def restore_rng():
        rng.bit_generator.state = copy.deepcopy(saved['rng_state'])
        torch.set_rng_state(saved['torch_rng_state'].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in saved['cuda_rng_state']])
        identity = _rng_identity({'rng_state': rng.bit_generator.state,
            'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all()})
        if identity != source_rng_sha:
            raise AssertionError('Source NumPy/CPU/CUDA RNG restoration differs')
        return identity
    measurements = []
    for horizon in (16, 450):
        optimizer.zero_grad(set_to_none=True)
        sampler.restore_state(sampler_source)
        model.core.train(mode == 'sample')
        restore_rng()
        pairs = np.column_stack([rng.integers(len(long.episodes), size=64), np.zeros(64, np.int64)])
        batch = long.batch(64, context=context, horizon=horizon, indices=pairs)
        auxiliary_pairs = sampler.sample(64, rng)
        auxiliary = short.batch(64, context=1, horizon=16, indices=auxiliary_pairs)
        accumulator = balancer.new_microbatch('combat')
        atomic_json(root/f'coordinates_probe_input_h{horizon}.json', _json_ready({
            'source_step': step, 'source_checkpoint_sha256': source_sha,
            'source_rng_sha256': source_rng_sha, 'context': context, 'latent_mode': mode,
            'main_indices': pairs.tolist(), 'auxiliary_indices': auxiliary_pairs.tolist(),
            'main_files': [str(long.episodes[int(index)]['path']) for index, _ in pairs],
            'auxiliary_files': [str(short.episodes[int(index)]['path']) for index, _ in auxiliary_pairs],
            'sampler_parent': sampler_source, 'sampler_after_private_draw': sampler.state_dict(),
            'sampling_report': sampler.last_sample, 'branch_protocol': branches,
            'actor_component_coefficients': accumulator.report()}))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        measured_start = time.perf_counter()
        def progress(processed, total):
            publish(status='running', phase=f'h{horizon}_forward_backward', horizon=horizon,
                    transitions_processed=processed, total_transitions=total)
        metrics = rssm_dynamics_backward(model, batch, auxiliary_batch=auxiliary,
            context=context, observed_context=context, auxiliary_context=1, auxiliary_observed_context=1,
            expected_horizon=horizon, chunk_size=16, amp=nullcontext, progress_callback=progress,
            contact_protocol=contact, combat_protocol=combat, combat_branch_protocol=branches,
            coordinate_gradient_protocol=COORDINATE_GRADIENT_PROTOCOL,
            memory_overshooting_protocol=MEMORY_OVERSHOOTING_PROTOCOL,
            numeric_state_feedback='detached', loss_groups=accumulator)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-measured_start
        norm = float(finite_gradient_norm(model.named_parameters(), context=f'B64 coordinate h{horizon} discarded probe'))
        if (not metrics['rssm_integrated_xy_gradients']
                or metrics.get('rssm_aux_combat_kinematic_npc_speed_count', 0) <= 0
                or metrics.get('rssm_memory_overshooting_backward_calls') != 1
                or metrics.get('rssm_memory_overshooting_no_future_feedback') is not True
                or metrics.get('rssm_combat_branch_branches', 0) < 1):
            raise AssertionError('Probe omitted a required coordinate, kinematic, memory or branch objective')
        row = {'batch': 64, 'horizon': horizon, 'auxiliary_horizon': 16, 'context': context,
            'latent_mode': mode, 'elapsed_seconds': elapsed, 'gradient_norm': norm,
            'all_gradients_finite': True, 'optimizer_steps': 0,
            'baseline_memory_allocated_bytes': allocated, 'baseline_memory_reserved_bytes': reserved,
            'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved(), 'metrics': metrics}
        measurements.append(row)
        atomic_json(root/f'coordinates_probe_h{horizon}.json', _json_ready(row))
        del batch, auxiliary, accumulator, metrics
        gc.collect()
    publish(phase='prove_model_Adam_EMA_Source_and_RNG_preservation')
    restore_rng()
    sampler.restore_state(sampler_source)
    if _hash_tensors(model.state_dict()) != weights_sha or optimizer_sha(optimizer.state_dict()) != adam_sha:
        raise AssertionError('Discarded probe changed model buffers/weights or actual Adam states/options')
    if (file_sha(source) != source_sha or balancer.state_dict() != actor_source
            or sampler.state_dict() != sampler_source
            or json.dumps(_json_ready(training), sort_keys=True) != source_training_sha):
        raise AssertionError('Source checkpoint, actor/component EMA, sampler or training ledger changed')
    report = {'passed': True, 'source_step': step, 'source_checkpoint': str(source),
        'source_checkpoint_sha256': source_sha, 'source_checkpoint_unchanged': True,
        'discarded_model_copy': True, 'model_weights_and_buffers_unchanged': True,
        'model_weights_sha256': weights_sha, 'all_Adam_tensors_and_options_unchanged': True,
        'Adam_sha256': adam_sha, 'Adam_parameter_states': len(optimizer.state),
        'source_rng_restored_exactly': True, 'source_rng_sha256': source_rng_sha,
        'source_sampler_restored_exactly': True, 'actor_and_component_EMA_unchanged': True,
        'optimizer_steps': 0, 'component_EMA_updates': 0,
        'batch': 64, 'horizons': [16, 450], 'auxiliary_horizon': 16, 'tbptt_chunk': 16,
        'production_accumulation_steps': 4, 'microbatches_probed_per_horizon': 1,
        'coordinate_gradient_protocol': COORDINATE_GRADIENT_PROTOCOL,
        'kinematic_protocol': KINEMATIC_PROTOCOL_V1,
        'memory_overshooting_protocol': MEMORY_OVERSHOOTING_PROTOCOL,
        'branch_protocol': branches, 'all_gradients_finite': True,
        'precision': 'fp32', 'tf32_enabled': False, 'measurements': measurements,
        'elapsed_seconds': time.perf_counter()-started,
        'scope': 'Two discarded training backward probes; no committed effective256 update or gameplay qualification'}
    atomic_json(report_path, _json_ready(report))
    publish(status='completed', phase='completed', passed=True, report=str(report_path))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-step', type=int, default=5014)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        atomic_json(args.run_dir/'coordinates_probe_error.json', {'status': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
