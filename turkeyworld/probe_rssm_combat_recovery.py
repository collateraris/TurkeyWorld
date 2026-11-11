"""Discarded-copy full450 backward preflight for the explicit combat revision."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import torch

from .gradient_safety import finite_gradient_norm
from .models import load_world_checkpoint
from .renderer_student import _hash_tensors
from .rssm_combat_branches import normalize_combat_branch_protocol
from .rssm_migration import configure_rssm_training
from .rssm_objectives import rssm_dynamics_backward
from .rssm_spawn_curriculum import RecoveryCombatSampler
from .staged_curriculum import StagedSequenceData
from .training_pipeline import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2);torch.cuda.init()
    started=time.perf_counter();model,saved=load_world_checkpoint(args.checkpoint,'cuda');model.float()
    if model.config.navigation_only:raise ValueError('Probe requires one-enemy combat')
    configure_rssm_training(model)
    before=_hash_tensors(model.state_dict())
    completed=saved['run_state']['rssm_training']['combat_completed'];context=1+(completed//16)%2
    model.core.train(['argmax','argmax','sample','argmax'][(completed//4)%4]=='sample')
    short=StagedSequenceData(args.data_root/'moving1/rgb/train','cuda',mode='stream',cache_bytes=64*1024**2)
    long=StagedSequenceData(args.data_root/'full_game/rgb/train','cuda',mode='stream',cache_bytes=64*1024**2)
    sampler=RecoveryCombatSampler(short)
    rng=np.random.default_rng();rng.bit_generator.state=saved['rng_state']
    torch.set_rng_state(saved['torch_rng_state'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng_state']])
    pairs=np.column_stack([rng.integers(len(long.episodes),size=16),np.zeros(16,np.int64)])
    batch=long.batch(16,context=context,horizon=450,indices=pairs)
    auxiliary=short.batch(16,context=1,horizon=16,indices=sampler.sample(16,rng))
    settings=saved['run_state']['navigation_training_history']['contact_revision']['loss_settings']
    protocol=normalize_combat_branch_protocol({})
    metrics=rssm_dynamics_backward(model,batch,auxiliary_batch=auxiliary,context=context,observed_context=context,
        auxiliary_context=1,auxiliary_observed_context=1,expected_horizon=450,amp=nullcontext,
        contact_protocol=settings,combat_protocol={'npc_motion_weight':5.,'spawn_weight':1.,'game_training_v5':True},
        numeric_state_feedback='detached',combat_branch_protocol=protocol)
    norm=float(finite_gradient_norm(model.named_parameters(),context='Recovery discarded-copy full450 preflight'))
    after=_hash_tensors(model.state_dict())
    if before!=after:raise AssertionError('Backward preflight changed model tensors without an optimizer step')
    if metrics['rssm_combat_branch_branches']<1:raise AssertionError('No actual own-memory branch was trained')
    if metrics['rssm_aux_combat_attack_negative_count']<1:raise AssertionError('New sampler supplied no eligible negative')
    report={'source_step':saved['training_steps'],'passed':True,'elapsed_seconds':time.perf_counter()-started,
        'discarded_model_copy':True,'optimizer_steps':0,'main_worker_untouched':True,'full_horizon':450,
        'batch':16,'context':context,'all_weights_and_buffers_bitexact':True,'gradient_norm':norm,
        'model_tensor_sha256':before,'sampler':sampler.last_sample,'metrics':metrics}
    atomic_json(args.output,report)
    print(json.dumps({key:report[key] for key in ['source_step','passed','elapsed_seconds','gradient_norm']}),flush=True)
    print(json.dumps({key:value for key,value in metrics.items() if key.startswith('rssm_combat_branch_') or
        key in ('rssm_aux_combat_attack_negative_count','rssm_aux_combat_spawn_event_count')}),flush=True)


if __name__=='__main__':main()
