"""Start separate player/enemy RSSMs from a stopped preserved Adam boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .dual_rssm_migration import prepare_dual_rssm_continuation
from .rssm_spawn_curriculum import file_sha
from .training_pipeline import atomic_json,now
from .vqgan_pipeline import process_alive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--preview-renderer',type=Path,required=True)
    parser.add_argument('--preview-sr',type=Path,required=True)
    args=parser.parse_args();source=args.source_run.resolve();root=args.run_dir.resolve()
    status=json.loads((source/'status.json').read_text(encoding='utf-8'))
    if status['status']!='stopped' or process_alive(int((source/'training.pid').read_text())):
        raise ValueError('Wait for the old worker to save a complete32x4 optimizer boundary')
    import torch
    torch.cuda.init()
    step=status['global_training_step'];old=source/'checkpoints/rssm_world.pt'
    archive=source/'checkpoints'/f'combat_before_dual_g{step}.pt'
    if archive.exists():raise FileExistsError('Do not overwrite an archived optimizer boundary')
    shutil.copy2(old,archive)
    if file_sha(old)!=file_sha(archive):raise ValueError('Source changed during archive')
    root.mkdir(parents=True,exist_ok=True)
    destination=root/'checkpoints/rssm_world.pt'
    proof=prepare_dual_rssm_continuation(archive,destination,run_id=root.name)
    shutil.copy2(source/'combat_training_plan.json',root/'combat_training_plan.json')
    shutil.copy2(source/'launch_command.json',root/'source_launch_command.json')
    selected=root/'preview_assets_npr6600_sr18100.json'
    selected_plan={'version':1,'kind':'frozen_preview_assets_v1','allow_unqualified_codec':True,
        'renderer':{'checkpoint':str(args.preview_renderer.resolve()),'file_sha256':file_sha(args.preview_renderer),
            'refinement_updates':6600},
        'sr':{'checkpoint':str(args.preview_sr.resolve()),'file_sha256':file_sha(args.preview_sr),
            'selected_local_update':18100}}
    atomic_json(selected,selected_plan)
    launch=json.loads((source/'launch_command.json').read_text(encoding='utf-8'))
    argv=list(launch['argv']);argv[argv.index('--run-dir')+1]=str(root)
    initial=Path(argv[argv.index('--preview-assets-plan')+1])
    revision=root/'preview_assets_revision_npr6600.json'
    atomic_json(revision,{'version':1,'kind':'preview_assets_revision_v1',
        'initial_plan_sha256':file_sha(initial),'selected_plan':str(selected),
        'selected_plan_sha256':file_sha(selected),'reason':'user_requested_updated_renderer_and_upscaler',
        'preview_only':True,'game_assets_changed':False})
    argv+=['--preview-assets-revision-plan',str(revision)]
    out=(root/'training.stdout.log').open('ab');err=(root/'training.stderr.log').open('ab')
    try:
        child=subprocess.Popen(argv,cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,
            stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)
    finally:out.close();err.close()
    launch.update(argv=argv,pid=child.pid,resumed_at=now(),resumed_from_global=step,
        source_run=str(source),dual_rssm_migration=str(destination.with_suffix('.migration.json')),
        player_weights_preserved=True,enemy_parameters=proof['added_parameter_count'],
        optimizer_old_groups_preserved=True,enemy_optimizer_group='fresh_only',
        preview_assets_revision_plan=str(revision),game_window_untouched=True)
    atomic_json(root/'launch_command.json',launch)
    atomic_json(root/'activation.json',{'pid':child.pid,'source_step':step,'source_archive':str(archive),
        'checkpoint':str(destination),'player_memory_size':proof['player_memory_size'],
        'enemy_memory_size':proof['enemy_memory_size'],'combined_memory_size':proof['combined_memory_size'],
        'enemy_parameters':proof['added_parameter_count'],'started_at':now(),'gameplay_qualified':False})
    print(json.dumps({'pid':child.pid,'run':str(root),'source_step':step,
        'enemy_parameters':proof['added_parameter_count'],'combined_memory_size':proof['combined_memory_size']}),flush=True)


if __name__=='__main__':main()
