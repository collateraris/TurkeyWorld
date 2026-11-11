"""Continue a complete dual-RSSM Adam boundary with explicit task normalization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .rssm_loss_balance import LOSS_PROTOCOL, SETTINGS
from .rssm_spawn_curriculum import file_sha
from .training_pipeline import atomic_json, now
from .vqgan_pipeline import process_alive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();source=args.source_run.resolve();root=args.run_dir.resolve()
    status=json.loads((source/'status.json').read_text(encoding='utf-8'))
    if status['status']!='stopped' or process_alive(int((source/'training.pid').read_text())):
        raise ValueError('Wait for the current32x4 update and graceful complete Adam save')
    if status.get('world_dynamics_protocol')!='dual_player_enemy_rssm_v1':
        raise ValueError('Separate loss normalization requires the two actual RSSMs')
    if (root/'checkpoints/rssm_world.pt').exists():
        raise FileExistsError('Never overwrite an existing task-balanced run')
    step=status['global_training_step'];checkpoint=source/'checkpoints/rssm_world.pt'
    archive=source/'checkpoints'/f'combat_before_group_balance_g{step}.pt'
    if archive.exists():raise FileExistsError('Never overwrite a source Adam boundary')
    shutil.copy2(checkpoint,archive);digest=file_sha(archive)
    if file_sha(checkpoint)!=digest:raise ValueError('Source checkpoint changed during archive')
    destination=root/'checkpoints/rssm_world.pt';destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(archive,destination)
    if file_sha(destination)!=digest:raise ValueError('Continuation checkpoint copy changed')
    shutil.copy2(source/'combat_training_plan.json',root/'combat_training_plan.json')
    shutil.copy2(source/'launch_command.json',root/'source_launch_command.json')
    training=status['rssm_training']
    plan={'version':1,'loss_protocol':LOSS_PROTOCOL,'settings':SETTINGS,
        'reason':'user_requested_player_enemy_interaction_separate_normalization_and_weights',
        'start_optimizer_step':step,'source_checkpoint':str(archive),'source_checkpoint_sha256':digest,
        'base_plan_sha256':file_sha(root/'combat_training_plan.json'),
        'batch_revision_sha256':training['batch_revision']['sha256'],
        'source_future_feedback':False,'optimizer_reset':False,'created_at':now(),
        'first_update':'calibrate scales from actual complete32x4 training update',
        'constant_kl_floor':'unscaled and excluded from EMA;1.1player+1.1enemy',
        'game_window_untouched':True,'new_run_separates_loss_units_from_previous_curve':True}
    path=root/'loss_balance_revision.json';atomic_json(path,plan)
    launch=json.loads((source/'launch_command.json').read_text(encoding='utf-8'))
    argv=list(launch['argv']);argv[argv.index('--run-dir')+1]=str(root)
    argv+=['--loss-balance-plan',str(path)]
    out=(root/'training.stdout.log').open('ab');err=(root/'training.stderr.log').open('ab')
    try:
        child=subprocess.Popen(argv,cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,
            stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)
    finally:out.close();err.close()
    launch.update(argv=argv,pid=child.pid,resumed_at=now(),resumed_from_global=step,
        source_run=str(source),loss_balance_plan=str(path),loss_protocol=LOSS_PROTOCOL,
        full_checkpoint_copied_bitexact=True,optimizer_preserved=True,no_rollback=True,
        user_game_window_untouched=True)
    atomic_json(root/'launch_command.json',launch)
    atomic_json(root/'initialization.json',{'source_run':str(source),'source_step':step,
        'source_checkpoint':str(archive),'source_sha256':digest,'copied_sha256':file_sha(destination),
        'complete_checkpoint_copied_bitexact':True,'pid':child.pid,'started_at':now(),
        'loss_balance_plan':str(path),'optimizer_reset':False,'game_window_untouched':True})
    print(json.dumps({'pid':child.pid,'source_step':step,'run_dir':str(root),'loss_protocol':LOSS_PROTOCOL}),flush=True)


if __name__=='__main__':main()
