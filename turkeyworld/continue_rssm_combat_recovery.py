"""Archive a stopped full Adam boundary and launch the opt-in recovery phase."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .rssm_combat_branches import normalize_combat_branch_protocol
from .rssm_spawn_curriculum import file_sha
from .train_rssm_dynamics import verify_data
from .training_pipeline import atomic_json,now
from .vqgan_pipeline import process_alive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--negative-root',type=Path,required=True)
    args=parser.parse_args();root=args.run_dir.resolve()
    status=json.loads((root/'status.json').read_text(encoding='utf-8'))
    if status['status']!='stopped':raise ValueError('Wait for graceful complete-Adam stop')
    oldpid=int((root/'training.pid').read_text())
    if process_alive(oldpid):raise ValueError('Old worker must finish before replacing the checkpoint')
    import torch
    torch.cuda.init()
    checkpoint=root/'checkpoints/rssm_world.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    step=saved['training_steps']
    if (step!=status['global_training_step'] or saved['global_training_step']!=step
            or saved['run_state']['global_training_step']!=step or saved['config']['navigation_only']
            or not saved['run_state']['rssm_training'].get('feedback_revision')
            or saved['run_state']['rssm_training'].get('combat_recovery_revision')):
        raise ValueError('Expected a complete, previously unrevised combat Adam boundary')
    archive=root/'checkpoints'/f'combat_before_recovery_v3_g{step}.pt'
    if archive.exists():raise FileExistsError('Never overwrite an archived recovery boundary')
    shutil.copy2(checkpoint,archive)
    source_sha=file_sha(archive)
    if file_sha(checkpoint)!=source_sha:raise ValueError('Saved boundary changed during its archive')
    shutil.copy2(root/'status.json',root/f'status_before_recovery_v3_g{step}.json')
    shutil.copy2(root/'launch_command.json',root/'launch_command_before_recovery_v3.json')
    shutil.copy2(root/'metrics.jsonl',root/f'metrics_before_recovery_v3_g{step}.jsonl')
    revision=root/'combat_recovery_revision.json'
    if revision.exists():raise FileExistsError('Recovery plan must have a new immutable identity')
    data_root=args.data_root.resolve();negative_root=args.negative_root.resolve()
    plan={'version':3,'created_at':now(),'reason':'user_requested_own_branches_balanced_spawn_melee_negatives',
        'start_optimizer_step':step,'source_checkpoint':str(archive),'source_checkpoint_sha256':source_sha,
        'base_plan_sha256':file_sha(root/'combat_training_plan.json'),
        'feedback_revision_sha256':file_sha(root/'numeric_feedback_revision.json'),
        'data_root':str(data_root),'data_manifest_sha256':verify_data(data_root),
        'negative_root':str(negative_root),'negative_manifest_sha256':verify_data(negative_root),
        'numeric_state_feedback':'detached','source_future_feedback':False,
        'sampler':{'batch':16,'spawn':4,'attack_negative':4,'ordinary':8},
        'combat_branch_protocol':normalize_combat_branch_protocol({}),
        'expanded_spawn_validation':'all_long_val_events','validation_every':250,
        'main_horizon':450,'tbptt_chunk':16,'optimizer_reset':False,'rollback':False,
        'renderer_or_game_window_changed':False}
    atomic_json(revision,plan)
    launch=json.loads((root/'launch_command.json').read_text(encoding='utf-8'))
    argv=launch['argv']+['--combat-recovery-plan',str(revision)]
    if '--combat-recovery-plan' in launch['argv']:raise ValueError('Recovery was already launched')
    stop=root/'stop_training'
    if stop.exists():stop.unlink()
    stdout=(root/'recovery_v3.stdout.log').open('ab');stderr=(root/'recovery_v3.stderr.log').open('ab')
    try:
        child=subprocess.Popen(argv,cwd=Path(__file__).resolve().parents[1],stdout=stdout,stderr=stderr,
            stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)
    finally:stdout.close();stderr.close()
    launch.update(argv=argv,pid=child.pid,resumed_at=now(),resumed_from_global=step,
        combat_recovery_revision=3,recovery_plan_sha256=file_sha(revision),optimizer_preserved=True,
        no_rollback=True,game_window_untouched=True)
    atomic_json(root/'launch_command.json',launch)
    print(json.dumps({'pid':child.pid,'resumed_from_global':step,'revision':str(revision),
        'source_sha256':source_sha,'optimizer_reset':False}),flush=True)


if __name__=='__main__':main()
