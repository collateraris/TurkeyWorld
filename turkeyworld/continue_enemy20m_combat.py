"""Resize only the enemy RSSM and continue the preserved run on combat only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .rssm_spawn_curriculum import file_sha
from .rssm_training_schedule import COMBAT_ONLY_SCHEDULE
from .training_pipeline import atomic_json, now
from .vqgan_pipeline import process_alive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();source=args.source_run.resolve();root=args.run_dir.resolve()
    status=json.loads((source/'status.json').read_text(encoding='utf-8'))
    if status['status']!='stopped' or process_alive(int((source/'training.pid').read_text())):
        raise ValueError('Wait for the current complete32x4 optimizer update and Adam save')
    if status.get('world_dynamics_protocol')!='dual_player_enemy_rssm_v1':
        raise ValueError('Need the current dual player/enemy RSSM continuation')
    destination=root/'checkpoints/rssm_world.pt'
    if destination.exists():raise FileExistsError('Do not overwrite an existing enemy20M run')
    step=status['global_training_step'];checkpoint=source/'checkpoints/rssm_world.pt'
    archive=source/'checkpoints'/f'combat_before_enemy20m_g{step}.pt'
    if archive.exists():raise FileExistsError('Never overwrite the original learned enemy/player Adam boundary')
    shutil.copy2(checkpoint,archive);source_sha=file_sha(archive)
    if file_sha(checkpoint)!=source_sha:raise ValueError('Source checkpoint changed during archive')
    import torch
    torch.cuda.init()  # Initialize before deserializing GPU-authored checkpoints on Windows.
    from .enemy_rssm_resize import prepare_enemy_rssm_resize
    initial=root/'checkpoints'/f'enemy20m_initial_g{step}.pt'
    initial.parent.mkdir(parents=True,exist_ok=True)
    report=prepare_enemy_rssm_resize(archive,initial,run_id=root.name)
    initial_sha=file_sha(initial);shutil.copy2(initial,destination)
    if file_sha(destination)!=initial_sha:raise ValueError('Resized initial model/Adam copy changed')
    shutil.copy2(source/'combat_training_plan.json',root/'combat_training_plan.json')
    shutil.copy2(source/'launch_command.json',root/'source_launch_command.json')
    plan={'version':1,'training_schedule':COMBAT_ONLY_SCHEDULE,'update_cycle':['combat'],
        'navigation_replay_enabled':False,'navigation_validation_enabled':True,
        'optimizer_reset':False,'source_future_feedback':False,'start_optimizer_step':step,
        'source_checkpoint':str(initial),'source_checkpoint_sha256':initial_sha,
        'base_plan_sha256':file_sha(root/'combat_training_plan.json'),
        'loss_balance_plan_sha256':status['rssm_training']['loss_balance_revision']['sha256'],
        'reason':'user_requested_temporary_combat_only_training_and_enemy20M',
        'created_at':now(),'enemy_rssm_resize_report':str(initial.with_suffix('.resize.json')),
        'parent_original_checkpoint':str(archive),'parent_original_checkpoint_sha256':source_sha,
        'latent_and_context_clocks_reset':False,'navigation_EMA_reset':False,
        'validation_every':250,'preview_every':10,'game_window_untouched':True,
        'optimizer_scope':'preserve player and exact-shape enemy Adam; resized enemy tensors start fresh'}
    path=root/'combat_only_revision.json';atomic_json(path,plan)
    launch=json.loads((source/'launch_command.json').read_text(encoding='utf-8'))
    argv=list(launch['argv']);argv[argv.index('--run-dir')+1]=str(root)
    argv+=['--combat-only-plan',str(path)]
    out=(root/'training.stdout.log').open('ab');err=(root/'training.stderr.log').open('ab')
    try:
        child=subprocess.Popen(argv,cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,
            stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)
    finally:out.close();err.close()
    launch.update(argv=argv,pid=child.pid,resumed_at=now(),resumed_from_global=step,
        source_run=str(source),combat_only_plan=str(path),training_schedule=COMBAT_ONLY_SCHEDULE,
        enemy_rssm_resize=str(initial.with_suffix('.resize.json')),
        player_weights_and_Adam_preserved=True,enemy_resized_Adam_fresh=True,
        user_game_window_untouched=True)
    atomic_json(root/'launch_command.json',launch)
    atomic_json(root/'initialization.json',{'source_run':str(source),'source_step':step,
        'original_checkpoint':str(archive),'original_checkpoint_sha256':source_sha,
        'resized_initial_checkpoint':str(initial),'resized_initial_sha256':initial_sha,
        'copied_checkpoint_sha256':file_sha(destination),'copied_resized_checkpoint_bitexact':True,
        'enemy_rssm_resize_report':str(initial.with_suffix('.resize.json')),
        'combat_only_plan':str(path),'pid':child.pid,'started_at':now(),
        'player_weights_and_Adam_preserved':True,'global_clocks_and_EMA_preserved':True,
        'entire_old_checkpoint_bitexact_claimed':False,'game_window_untouched':True})
    print(json.dumps({'pid':child.pid,'source_step':step,'run_dir':str(root),
        'training_schedule':COMBAT_ONLY_SCHEDULE,'resize_report':str(initial.with_suffix('.resize.json'))}),flush=True)


if __name__=='__main__':main()
