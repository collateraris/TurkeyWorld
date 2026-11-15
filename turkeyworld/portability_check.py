"""Read-only Windows/Ubuntu transfer checks. Never starts or resumes training.

Use --resume to check the paused motion cycle, immutable Source/VAL/preview
files and the saved world/policy optimizer boundaries. Run with the bootstrap
on a CUDA training host; tensors are inspected via CPU mmap, without SGD.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import platform
from pathlib import Path
import sys

from .workspace_paths import ROOT,resolve_path,identity_path,load_workspace_paths

WORLD='training_runs/rssm_moe_nav15_enemy25_cap20_20261113'
POLICY='training_runs/dreamer_actor_critic_20261114'
CYCLE='training_runs/dreamer_motion_cycle_20261114'
HANDOFF='training_runs/ubuntu_handoff_20261115'


def read(path):return json.loads(resolve_path(path,must_exist=True).read_text(encoding='utf-8-sig'))


def file_sha(path):
    with resolve_path(path,must_exist=True).open('rb')as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def check_snapshot_manifest(path):
    value=read(path);checks={}
    for name,asset in value['assets'].items():
        for key in('source','snapshot'):
            actual=resolve_path(asset[key],must_exist=True)
            checks[name+'_'+key]=(actual.stat().st_size==asset['bytes']and file_sha(actual)==asset['sha256'])
    if not all(checks.values()):raise ValueError('A paused checkpoint changed: '+str(checks))
    return checks


def check_resume():
    # Importing this check module alone does not import torch. CPU loading here
    # reads only trusted local training checkpoints, never untrusted downloads.
    import torch
    from .rssm_motion_only_revision import bind_motion_only,validate_motion_only
    from .rssm_motion_only_validation import verify_motion_validation_plan
    from .rssm_preview_assets import resolve_preview_assets_plan
    from .dreamer_policy_validation import validate_panel
    from .dreamer_motion_cycle import MotionCycle
    from types import SimpleNamespace

    receipt=read(HANDOFF+'/pause_receipt.json')
    cycle=read(CYCLE+'/status.json')
    # Constructor validates the entire existing immutable cycle plan, not just
    # file accessibility. No run()/publish()/PID adoption is performed here.
    controller=MotionCycle(SimpleNamespace(run_dir=ROOT/CYCLE,world_run=ROOT/WORLD,
        policy_run=ROOT/POLICY,motion_plan=ROOT/WORLD/'motion_only_plan.json',
        policy_updates=50,episodes_per_stream=16))
    world=resolve_path(WORLD+'/checkpoints/rssm_world.pt',must_exist=True)
    saved=torch.load(world,map_location='cpu',weights_only=False,mmap=True)
    state=deepcopy(saved['run_state']);before=deepcopy(state)
    plan_path=resolve_path(WORLD+'/motion_only_plan.json');plan=read(plan_path)
    bind_motion_only(state,plan,path=plan_path,loaded_sha=file_sha(world))
    validate_motion_only(state)
    if state!=before:raise ValueError('Read-only boundary binding unexpectedly changed the saved motion clock')
    revision=state['rssm_training']['motion_only_revision'];gate=revision['gate']
    expected=receipt['world_step']
    checks=dict(world_step=saved['training_steps']==saved['global_training_step']==state['global_training_step']==expected,
        world_model_count=len(saved['model'])==1599,
        world_Adam_count=len(saved['optimizer']['state'])==442,
        world_Adam_groups=len(saved['optimizer']['param_groups'])==7,
        world_named_optimizer=bool(saved['optimizer_parameter_names']),
        world_rng_saved=all(key in saved for key in('rng_state','torch_rng_state','cuda_rng_state')),
        same_round=gate['round_index']==cycle['round_index']==receipt['world_round_index'],
        same_horizon=gate['current_horizon']==receipt['world_stage_horizon'],
        same_stage_updates=gate['stage_updates']==receipt['world_stage_updates'],
        same_round_updates=gate['updates']==receipt['world_round_updates'],
        cycle_settings=read(CYCLE+'/cycle_plan.json')['world_run']==identity_path(resolve_path(WORLD)))
    checks['cycle_constructor']=controller.state['round_index']==cycle['round_index']and controller.state['phase']==cycle['phase']
    # Saved VAL mtimes intentionally remain part of their old identity. A copy
    # that loses timestamps fails here before any optimizer is created.
    checks['heldout_validation']=verify_motion_validation_plan(state['motion_only_validation_plan'])
    original=ROOT/'training_runs/rssm25m_combat_20261111/preview_assets_batch128_g4398.json'
    latest=ROOT/WORLD/'preview_assets_revision_npr32317_decoder4125_sr_detail3000_v5.json'
    resolve_preview_assets_plan(original,latest,state.get('preview_assets_revision'))
    checks['preview_SHA_chain']=True
    episodes=0
    for corpus in revision['corpus_revisions']:
        for row in corpus['receipt']['episodes']:
            if row.get('split')!='train'or file_sha(row['path'])!=row['sha256']:
                raise ValueError('Collected TRAIN Source file changed')
            episodes+=1
    checks['source_episodes']=episodes==receipt['collected_Source_episodes']
    del saved,state,before
    policy_path=resolve_path(POLICY+'/checkpoints/policy.pt',must_exist=True)
    policy=torch.load(policy_path,map_location='cpu',weights_only=False,mmap=True)
    checks['policy_clocks']=policy['step']==receipt['policy_step']and policy['actor_updates']==receipt['policy_actor_updates']
    checks['policy_world_pin']=file_sha(policy['world_identity']['path'])==policy['world_identity']['sha256']
    policy_plan=read(POLICY+'/training_plan.json')
    panel=read(POLICY+'/validation_panel.json')
    validate_panel(panel)
    checks['policy_VAL_panel']=panel['fingerprint']==policy_plan['validation_panel_fingerprint']
    checks['policy_optimizers']=all(key in policy for key in('actor_optimizer','critic_optimizer','head_optimizer'))
    checks['policy_rng_saved']=all(key in policy for key in('torch_rng','cuda_rng','numpy_rng','python_rng','replay'))
    checks['cycle_stopped']=resolve_path(CYCLE+'/stop.request').is_file()
    checks['world_stopped']=resolve_path(WORLD+'/stop_training').is_file()
    if not all(checks.values()):raise ValueError('Paused resume preflight failed: '+str(checks))
    return checks


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true',help='Verify existing paused runs; requires copied training_runs and PyTorch.')
    parser.add_argument('--snapshots',action='store_true',help='SHA-check all four paused assets and their copies (about 3 GB of reads).')
    args=parser.parse_args(argv)
    result=dict(platform=platform.system(),python=sys.version.split()[0],workspace=str(ROOT),
        mapping_configured=load_workspace_paths()is not None,training_started=False)
    try:
        if args.snapshots:result['snapshot_checks']=check_snapshot_manifest(HANDOFF+'/source_manifest.json')
        if args.resume:result['resume_checks']=check_resume()
        result['passed']=True
    except (OSError,ValueError,KeyError,TypeError,ImportError)as error:
        result.update(passed=False,error=f'{type(error).__name__}: {error}')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['passed']else 1


if __name__=='__main__':raise SystemExit(main())
