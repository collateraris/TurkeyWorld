"""One discarded FP32 B32/450 backward; measure before changing any weights."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .models import load_world_checkpoint
from .renderer_student import _hash_tensors
from .rssm_combat_branches import normalize_combat_branch_protocol
from .rssm_component_balance import SETTINGS
from .rssm_gradient_calibration import ComponentGradientMeter
from .rssm_loss_balance import GroupLossAccumulator, GroupLossBalancer
from .rssm_migration import configure_rssm_training
from .rssm_objectives import rssm_dynamics_backward
from .rssm_spawn_curriculum import RecoveryCombatSampler, upgrade_attack_balanced_sampler_state, file_sha
from .staged_curriculum import StagedSequenceData
from .training_pipeline import atomic_json


def _rng_identity(saved):
    digest = hashlib.sha256()
    digest.update(json.dumps(saved['rng_state'], sort_keys=True).encode())
    digest.update(saved['torch_rng_state'].cpu().numpy().tobytes())
    for value in saved['cuda_rng_state']: digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value, dict): return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_json_ready(item) for item in value]
    return value


def run(args):
    torch.set_num_threads(args.threads)
    torch.cuda.init()  # Before any torch.load, matching production launch ordering.
    root = args.run_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {'status': 'initializing', 'optimizer_updated': False, 'microbatches': 1,
        'microbatch_size': 32, 'gradient_accumulation_steps_probed': 1, 'horizon': 450}
    def publish(**values):
        status.update(values, elapsed_seconds=time.perf_counter()-started)
        atomic_json(root/'probe_status.json', status)
        print(json.dumps(status), flush=True)
    publish(phase='archive_stopped_checkpoint')
    source_sha = file_sha(args.checkpoint)
    archive = root/'source_g4944.pt'
    if archive.exists():
        if file_sha(archive) != source_sha: raise ValueError('Existing probe archive differs from the supplied source')
    else:
        temporary = archive.with_suffix('.copying')
        shutil.copy2(args.checkpoint, temporary)
        if file_sha(temporary) != source_sha or file_sha(args.checkpoint) != source_sha:
            raise RuntimeError('Stopped source changed while copying; no probe was run')
        temporary.replace(archive)
    publish(phase='load_discarded_model')
    model, saved = load_world_checkpoint(archive, 'cpu')
    if saved['training_steps'] != 4944 or model.config.navigation_only:
        raise ValueError('Calibration must use the completed combat4944 source boundary')
    before = _hash_tensors(model.state_dict())
    state = saved['run_state']; training = state['rssm_training']
    recovery = json.loads(Path(training['combat_recovery_revision']['plan']).read_text(encoding='utf-8'))
    short_root = Path(recovery['data_root'])/'moving1/rgb/train'
    launch = json.loads((args.checkpoint.parent.parent/'launch_command.json').read_text(encoding='utf-8'))
    argv = launch['argv']; combat_root = Path(argv[argv.index('--combat-data-root')+1])
    short = StagedSequenceData(short_root, 'cuda', mode='stream', cache_bytes=64*1024**2)
    long = StagedSequenceData(combat_root/'full_game/rgb/train', 'cuda', mode='stream', cache_bytes=64*1024**2)
    sampler = RecoveryCombatSampler(short, context=1, horizon=16, target_distance_balanced=True,
        attack_positive_balanced=True, cache_path=root/'sampler_cache.json')
    prior_sampler = training['combat_recovery_revision']['sampler_state']
    upgraded = upgrade_attack_balanced_sampler_state(prior_sampler)
    sampler.restore_state(upgraded)
    completed = training['combat_completed']
    context = 1+(completed//16)%2
    mode = ['argmax', 'argmax', 'sample', 'argmax'][(completed//4)%4]
    model.to('cuda').float(); configure_rssm_training(model); model.core.train(mode == 'sample')
    rng = np.random.default_rng(); rng.bit_generator.state = copy.deepcopy(saved['rng_state'])
    torch.set_rng_state(saved['torch_rng_state'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in saved['cuda_rng_state']])
    restored_rng_sha = _rng_identity({'rng_state': rng.bit_generator.state,
        'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all()})
    if restored_rng_sha != _rng_identity(saved): raise AssertionError('Probe RNG restoration differs')
    pairs = np.column_stack([rng.integers(len(long.episodes), size=32), np.zeros(32, np.int64)])
    batch = long.batch(32, context=context, horizon=450, indices=pairs)
    auxiliary_pairs = sampler.sample(32, rng)
    auxiliary = short.batch(32, context=1, horizon=16, indices=auxiliary_pairs)
    inputs = {'source_step': 4944, 'source_checkpoint_sha256': source_sha,
        'context': context, 'observed_context': context, 'latent_mode': mode,
        'long_directory': str(long.directory), 'short_directory': str(short.directory),
        'long_indices': pairs.tolist(), 'auxiliary_indices': auxiliary_pairs.tolist(),
        'long_files': [str(long.episodes[int(i)]['path']) for i, _ in pairs],
        'auxiliary_files': [str(short.episodes[int(i)]['path']) for i, _ in auxiliary_pairs],
        'rng_exact_source_boundary': True, 'source_rng_sha256': restored_rng_sha,
        'sampler_parent': prior_sampler, 'sampler_upgraded': upgraded,
        'sampler_after_draw': sampler.state_dict(), 'sampling_report': sampler.last_sample}
    atomic_json(root/'probe_inputs.json', inputs)
    actor = GroupLossBalancer(training['loss_balance_state']).new_microbatch('combat')
    coefficients = {name: actor.priorities[name]/actor.scales[name] for name in actor.scales}
    meter = ComponentGradientMeter(model.named_parameters(), coefficients)
    accumulator = GroupLossAccumulator(actor.scales, actor.priorities, gradient_meter=meter)
    contact = state['navigation_training_history']['contact_revision']['loss_settings']
    protocol = normalize_combat_branch_protocol({'version': 2})
    combat = {'npc_motion_weight': 5., 'spawn_weight': 1., 'game_training_v5': True,
        'target_supervision': 'shot_distance_balanced'}
    publish(status='running', phase='full450_backward', source_step=4944,
        context=context, latent_mode=mode, source_checkpoint_sha256=source_sha)
    def progress(processed, horizon):
        publish(phase='full450_backward', transitions_processed=processed, horizon=horizon)
    metrics = rssm_dynamics_backward(model, batch, auxiliary_batch=auxiliary,
        context=context, observed_context=context, auxiliary_context=1, auxiliary_observed_context=1,
        expected_horizon=450, chunk_size=16, amp=nullcontext, progress_callback=progress,
        contact_protocol=contact, combat_protocol=combat, combat_branch_protocol=protocol,
        numeric_state_feedback='detached', loss_groups=accumulator)
    publish(phase='gradient_decomposition_and_frozen_proof')
    measurement = meter.finish()
    atomic_json(root/'gradient_measurement.json', measurement)
    after = _hash_tensors(model.state_dict())
    if before != after: raise AssertionError('Discarded backward changed model tensors or buffers')
    report = {'passed': True, 'source_step': 4944, 'source_checkpoint': str(archive),
        'source_checkpoint_sha256': source_sha, 'settings': SETTINGS, 'components': measurement['components'],
        **{k: v for k, v in measurement.items() if k != 'components'},
        'actor_coefficients_preserved': coefficients, 'model_weights_sha256': before,
        'model_weights_and_buffers_unchanged': True, 'optimizer_updated': False, 'optimizer_steps': 0,
        'microbatches_probed': 1, 'production_accumulation_steps': 4, 'microbatch_size': 32,
        'full_horizon': 450, 'tbptt_chunk': 16, 'precision': 'fp32',
        'numeric_state_feedback': 'detached', 'source_rng_restored_exactly': True,
        'branch_protocol': protocol, 'branch_label_coverage': {
            key: value for key, value in metrics.items() if key.startswith('rssm_combat_branch_')},
        'metrics': metrics, 'input_receipt': str(root/'probe_inputs.json'),
        'elapsed_seconds': time.perf_counter()-started, 'gameplay_qualified': False,
        'scope': 'One real B32 combat microbatch before any component weights or optimizer updates'}
    atomic_json(root/'calibration.json', _json_ready(report))
    publish(status='completed', phase='completed', passed=True,
        calibration=str(root/'calibration.json'), base_gradient_norm=measurement['base_gradient_norm'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    try: run(args)
    except Exception as error:
        atomic_json(args.run_dir/'probe_error.json', {'status': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__': main()
