"""Discarded real B64 XY-only short/scout and450 backward checks."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import json
import math
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
from .rssm_coordinate_curriculum import coordinate_horizon
from .rssm_coordinate_gradients import COORDINATE_GRADIENT_PROTOCOL
from .rssm_drift_only_objective import drift_only_backward, DISABLED_OBJECTIVES
from .rssm_drift_only_protocol import (DRIFT_ONLY_PROTOCOL, configure_drift_only_training,
    bind_drift_only, frozen_balance_values)
from .rssm_late_prefix_recovery import LATE_PREFIX_RECOVERY_PROTOCOL
from .rssm_migration import configure_rssm_training
from .rssm_replay_curriculum import replay_training_horizon
from .rssm_spawn_curriculum import RecoveryCombatSampler, file_sha
from .staged_curriculum import StagedSequenceData
from .training_pipeline import atomic_json


def run(args):
    torch.set_num_threads(args.threads); torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    root = args.run_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    gated = getattr(args,'loss_gated',False)
    prefix = 'loss_gated_probe' if gated else 'drift_only_probe'
    report_path = root/(prefix+'.json')
    if report_path.exists(): raise FileExistsError('Preserve the existing pure XY probe receipt')
    source = args.checkpoint.resolve(strict=True); source_root = args.source_run.resolve()
    status_before = json.loads((source_root/'status.json').read_text(encoding='utf-8'))
    source_pid = int((source_root/'training.pid').read_text().strip())
    if status_before.get('status') != 'stopped' or psutil.pid_exists(source_pid):
        raise ValueError('Pure XY probe requires a durable stopped Source and exited worker')
    started = time.perf_counter(); status = {'status':'initializing', 'optimizer_steps':0, 'EMA_commits':0}

    def publish(**values):
        status.update(values, elapsed_seconds=time.perf_counter()-started)
        atomic_json(root/(prefix+'_status.json'), status)
        print(json.dumps(_json_ready(status)), flush=True)

    publish(phase='load_discarded_model_and_complete_Adam')
    source_sha = file_sha(source); model, saved = load_world_checkpoint(source, 'cpu')
    step = saved['training_steps']; training = saved['run_state']['rssm_training']
    if step != args.source_step or status_before['current'] != step or saved['global_training_step'] != step:
        raise ValueError('Pure XY probe must use the exact stopped optimizer boundary')
    source_training_identity = json.dumps(_json_ready(training), sort_keys=True)
    source_balance = copy.deepcopy(training['loss_balance_state'])
    before_weights = _hash_tensors(model.state_dict()); model.to('cuda').float()
    configure_rssm_training(model)
    names = saved['optimizer_parameter_names']; named = dict(model.named_parameters())
    if len(names) != 165 or names != [name for name,p in named.items()if p.requires_grad]:
        raise ValueError('Preserve all165 canonical Source Adam names before freezing')
    groups = [{**{key:value for key,value in group.items() if key!='params'},
        'params':[named[names[index]] for index in group['params']]}
        for group in saved['optimizer']['param_groups']]
    optimizer = torch.optim.AdamW(groups); optimizer.load_state_dict(saved['optimizer'])
    before_adam = optimizer_sha(optimizer.state_dict())
    if before_adam != optimizer_sha(saved['optimizer']) or len(optimizer.state) != 165:
        raise AssertionError('Frozen and active Adam states/options must all load exactly')
    policy = configure_drift_only_training(model, optimizer_parameter_names=names)
    allowed = set(policy['trainable_parameter_names'])
    filename = 'loss_gated_curriculum_plan.json' if gated else 'drift_only_plan.json'
    plan_path = (args.plan or root/filename).resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding='utf-8')); plan_sha = file_sha(plan_path)
    private = copy.deepcopy(saved['run_state'])
    if gated:
        from .rssm_loss_gated_curriculum import (bind_loss_gated_curriculum,
            loss_gated_training_horizon, LOSS_GATE_PROTOCOL, HORIZONS)
        bind_loss_gated_curriculum(private,plan,path=plan_path,sha256=plan_sha,
            source_checkpoint_sha256=source_sha,loaded_checkpoint_sha256=source_sha,
            base_plan_sha256=saved['plan_sha256'])
        gate_before=copy.deepcopy(private['rssm_training']['loss_gated_curriculum'])
        next_training=loss_gated_training_horizon(private)
        if next_training['horizon']!=16 or next_training['forced_long_replay']is not False:
            raise AssertionError('A fresh loss-gated phase must start16 without a forced450 update')
        horizons=[16,96,450]  # Discarded boundary probes, not training schedule decisions.
    else:
        bind_drift_only(private, plan, path=plan_path, sha256=plan_sha,
            source_checkpoint_sha256=source_sha, loaded_checkpoint_sha256=source_sha,
            base_plan_sha256=saved['plan_sha256'], optimizer_parameter_names=names)
        short_horizon = coordinate_horizon(training['coordinate_curriculum']['updates'])
        horizons = [short_horizon,450] if short_horizon != 450 else [450]
        next_training = replay_training_horizon(saved['run_state'])
    private_balance_before = copy.deepcopy(private['rssm_training']['loss_balance_state'])
    if frozen_balance_values(private_balance_before) != frozen_balance_values(source_balance):
        raise AssertionError('Pure XY binding changed a preserved EMA value')
    argv = json.loads((source_root/'launch_command.json').read_text(encoding='utf-8'))['argv']
    combat_root = Path(argv[argv.index('--combat-data-root')+1])
    recovery = json.loads(Path(training['combat_recovery_revision']['plan']).read_text(encoding='utf-8'))
    long = StagedSequenceData(combat_root/'full_game/rgb/train','cuda',mode='stream',cache_bytes=64*1024**2)
    # This index object only reserves the historical RNG/sampler trace. No
    # short/auxiliary batch is constructed or passed into any objective.
    index_data = StagedSequenceData(Path(recovery['data_root'])/'moving1/rgb/train','cuda',
        mode='stream',cache_bytes=64*1024**2)
    sampler = RecoveryCombatSampler(index_data,context=1,horizon=16,target_distance_balanced=True,
        attack_positive_balanced=True,cache_path=root/(prefix+'_sampler_cache.json'))
    sampler_source = copy.deepcopy(training['combat_recovery_revision']['sampler_state'])
    if sampler_source.get('version') != 2: raise ValueError('Preserve the Source v2 sampler ledger')
    context = 1+(training['combat_completed']//16)%2
    mode = ['argmax','argmax','sample','argmax'][(training['combat_completed']//4)%4]
    branches = normalize_combat_branch_protocol({'version':2,'max_branches':16,
        'max_source_probes':128,'max_direct_source_queries':256})
    rng = np.random.default_rng(); source_rng = _rng_identity(saved)

    def restore_rng():
        rng.bit_generator.state = copy.deepcopy(saved['rng_state'])
        torch.set_rng_state(saved['torch_rng_state'].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in saved['cuda_rng_state']])
        current = _rng_identity({'rng_state':rng.bit_generator.state,'torch_rng_state':torch.get_rng_state(),
            'cuda_rng_state':torch.cuda.get_rng_state_all()})
        if current != source_rng: raise AssertionError('Source NumPy/CPU/CUDA RNG restore differs')

    measurements=[]
    for horizon in horizons:
        optimizer.zero_grad(set_to_none=True); sampler.restore_state(sampler_source)
        model.core.train(mode=='sample'); restore_rng()
        pairs=np.column_stack([rng.integers(len(long.episodes),size=64),np.zeros(64,np.int64)])
        batch=long.batch(64,context=context,horizon=horizon,indices=pairs)
        reserved=sampler.sample(64,rng)  # Index-only compatibility reservation.
        late={'late_prefix_protocol':LATE_PREFIX_RECOVERY_PROTOCOL}
        if horizon < 450:
            controls=long.batch(8,context=context,horizon=450,indices=pairs[:8])
            late['late_prefix_batch']={'actions':controls['actions'].detach(),'map':controls['map'].detach()}
            del controls
        atomic_json(root/f'{prefix}_input_h{horizon}.json',_json_ready({
            'source_step':step,'source_checkpoint_sha256':source_sha,'source_rng_sha256':source_rng,
            'context':context,'latent_mode':mode,'main_indices':pairs.tolist(),
            'main_files':[str(long.episodes[int(index)]['path'])for index,_ in pairs],
            'reserved_auxiliary_indices_not_trained':reserved.tolist(),'used_aux_fragments':0,
            'source_posterior_calls':0,'sampler_parent':sampler_source,
            'sampler_after_private_reservation':sampler.state_dict(),
            'parameter_policy':policy,'branch_protocol':branches,
            'late_prefix_protocol':LATE_PREFIX_RECOVERY_PROTOCOL}))
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        measured=time.perf_counter()

        def progress(processed,total):
            publish(status='running',phase=f'h{horizon}_XY_only_forward_backward',horizon=horizon,
                transitions_processed=processed,total_transitions=total)

        metrics=drift_only_backward(model,batch,context=context,observed_context=context,
            expected_horizon=horizon,chunk_size=16,amp=nullcontext,progress_callback=progress,
            coordinate_gradient_protocol=COORDINATE_GRADIENT_PROTOCOL,
            combat_branch_protocol=branches,**late)
        json.dumps(metrics,ensure_ascii=False,allow_nan=False)
        torch.cuda.synchronize();elapsed=time.perf_counter()-measured
        norm=float(finite_gradient_norm(model.named_parameters(),context=f'pureXYh{horizon} discardedprobe'))
        forbidden=[name for name,p in named.items()if name not in allowed and p.grad is not None]
        active=[name for name,p in named.items()if name in allowed and p.grad is not None]
        if forbidden or not active: raise AssertionError(f'Invalid pure XY gradient policy: {forbidden}')
        both_cores={prefix:any(name.startswith(prefix)and bool(torch.count_nonzero(named[name].grad))for name in active)
            for prefix in ['vjepa_backend.core.','vjepa_backend.enemy_rssm.core.']}
        if not all(both_cores.values()): raise AssertionError('Both RSSM cores need real nonzero XY gradients')
        zeros=['rssm_kl_minimum_floor','rssm_source_posterior_calls','rssm_decoder_calls',
            'rssm_renderer_calls','rssm_optimizer_steps','rssm_ema_updates','rssm_prior_future_rgb_inputs',
            'rssm_prior_numerical_resets','rssm_combat_branch_combat_loss_calls',
            'rssm_combat_branch_direct_source_queries']
        if (metrics.get('rssm_drift_only')is not True or any(metrics.get(key)!=0 for key in zeros)
                or metrics.get('rssm_disabled_objectives')!=list(DISABLED_OBJECTIVES)
                or metrics.get('rssm_combat_branch_xy_only')is not True
                or metrics.get('rssm_combat_branch_late_prefix_selected_count',0)<1
                or metrics.get('rssm_player_xy_supervised_transitions',0)<1
                or metrics.get('rssm_npc_xy_supervised_transitions',0)<1):
            raise AssertionError('Pure XY probe called a disabled objective or omitted real coordinate supervision')
        if not math.isclose(metrics['train_loss'],metrics['rssm_main_loss']+metrics['rssm_combat_branch_loss'],rel_tol=1e-6,abs_tol=1e-7):
            raise AssertionError('Pure XY total includes a non-coordinate objective')
        if any(metrics['loss_groups'][name]['scale']!=1 or metrics['loss_groups'][name]['kl_floor']!=0
               for name in ['player','enemy','interaction']):
            raise AssertionError('Pure XY may not use EMA normalization or a KL floor')
        if horizon<450:
            if (metrics.get('rssm_late_prefix_scout_start_step')!=horizon
                    or metrics.get('rssm_late_prefix_scout_end_step')!=450
                    or metrics.get('rssm_late_prefix_scout_actors')!=8
                    or metrics.get('rssm_late_prefix_scout_memory_resets')!=0
                    or metrics.get('rssm_late_prefix_scout_grad_enabled')is not False):
                raise AssertionError('Short XY must continue its exact own8prefixes without reset or gradients')
        elif any(key.startswith('rssm_late_prefix_scout_')for key in metrics):
            raise AssertionError('Full450 XY must reuse its actual main prefixes without an extra scout')
        row={'batch':64,'horizon':horizon,'context':context,'latent_mode':mode,'elapsed_seconds':elapsed,
            'gradient_norm':norm,'all_gradients_finite':True,'disabled_parameter_gradients_all_none':True,
            'active_gradient_parameter_names':active,'both_RSSM_cores_have_nonzero_XY_gradients':both_cores,
            'peak_memory_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_memory_reserved_bytes':torch.cuda.max_memory_reserved(),'optimizer_steps':0,
            'EMA_updates':0,'used_aux_fragments':0,'reserved_auxiliary_indices':64,'metrics':metrics}
        measurements.append(row);atomic_json(root/f'{prefix}_h{horizon}.json',_json_ready(row))
        del batch,late,metrics;gc.collect()
    publish(phase='prove_source_model_Adam_frozenEMA_and_RNG_unchanged')
    restore_rng();sampler.restore_state(sampler_source)
    if _hash_tensors(model.state_dict())!=before_weights or optimizer_sha(optimizer.state_dict())!=before_adam:
        raise AssertionError('Discarded XY probe changed model weights/buffers or any active/frozen Adam state')
    if (file_sha(source)!=source_sha or file_sha(plan_path)!=plan_sha
            or private['rssm_training']['loss_balance_state']!=private_balance_before
            or sampler.state_dict()!=sampler_source
            or json.dumps(_json_ready(training),sort_keys=True)!=source_training_identity):
        raise AssertionError('XY probe changed Source bytes, plan, EMA, sampler or training ledgers')
    if gated and private['rssm_training']['loss_gated_curriculum']!=gate_before:
        raise AssertionError('Discarded GPU probes must not advance the actual loss gate')
    report={'passed':True,'source_step':step,'source_checkpoint':str(source),'source_checkpoint_sha256':source_sha,
        'source_checkpoint_unchanged':True,
        'drift_only_plan_sha256':training['drift_only_revision']['sha256']if gated else plan_sha,
        'objective_protocol':DRIFT_ONLY_PROTOCOL,'discarded_model_copy':True,
        'model_weights_and_buffers_unchanged':True,'model_weights_sha256':before_weights,
        'all_Adam_tensors_and_options_unchanged':True,'Adam_sha256':before_adam,'Adam_parameter_states':165,
        'source_rng_restored_exactly':True,'source_rng_sha256':source_rng,
        'source_sampler_restored_exactly':True,'actor_and_component_EMA_unchanged':True,
        'ema_values_unchanged':True,'disabled_parameter_gradients_none':True,
        'disabled_parameter_gradients_all_none':True,'only_allowed_coordinate_gradients':True,
        'all_gradients_finite':True,'parameter_policy':policy,'optimizer_steps':0,'EMA_updates':0,
        'used_aux_fragments':0,'auxiliary_batches_constructed':0,
        'auxiliary_source_queries':0,'disabled_objectives':list(DISABLED_OBJECTIVES),
        'batch':64,'horizons':horizons,'source_next_training_horizon':next_training,
        'production_accumulation_steps':4,'microbatches_probed_per_horizon':1,
        'coordinate_gradient_protocol':COORDINATE_GRADIENT_PROTOCOL,
        'late_prefix_protocol':LATE_PREFIX_RECOVERY_PROTOCOL,'branch_protocol':branches,
        'source_future_feedback':False,'precision':'fp32','tf32_enabled':False,
        'measurements':measurements,'elapsed_seconds':time.perf_counter()-started,
        'scope':'Discarded one-micro pureXYbackward perhorizon; noSGD/EMA or four-micro commit/qualityclaim'}
    if gated:
        report.update(curriculum_protocol=LOSS_GATE_PROTOCOL,loss_gated_plan_sha256=plan_sha,
            approved_horizon_path=list(HORIZONS),forced_long_replay=False,
            forced_long_updates_disabled=True,gate_state_unchanged=True,
            required_consecutive_passes=plan['settings']['required_consecutive_passes'],
            initial_gate_consecutive_below_threshold=gate_before['consecutive_below_threshold'],
            actual_next_training_horizon=next_training['horizon'],
            h450_probe_is_discarded_capacity_check_not_forced_training=True)
    atomic_json(report_path,_json_ready(report));publish(status='completed',phase='completed',passed=True,report=str(report_path))
    return report


def main(argv=None, *, loss_gated=False):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--source-step',type=int,default=5079)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args(argv);args.loss_gated=loss_gated
    try:run(args)
    except Exception as error:
        prefix='loss_gated_probe'if loss_gated else'drift_only_probe'
        atomic_json(args.run_dir/(prefix+'_error.json'),{'status':'failed','error':repr(error)})
        raise


if __name__=='__main__':main()
