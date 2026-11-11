"""Archive a complete Adam boundary and continue with32x4 and pinned previews."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess

from .rssm_spawn_curriculum import file_sha
from .training_pipeline import atomic_json,now
from .vqgan_pipeline import process_alive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--renderer',type=Path,required=True)
    parser.add_argument('--sr',type=Path,required=True)
    args=parser.parse_args();root=args.run_dir.resolve()
    status=json.loads((root/'status.json').read_text(encoding='utf-8'))
    if status['status']!='stopped' or process_alive(int((root/'training.pid').read_text())):
        raise ValueError('Wait for a complete graceful Adam stop')
    import torch
    torch.cuda.init()
    checkpoint=root/'checkpoints/rssm_world.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    step=saved['training_steps'];training=saved['run_state']['rssm_training']
    if (step!=status['global_training_step'] or step!=saved['global_training_step']
            or training.get('batch_revision') or not training.get('combat_recovery_revision')):
        raise ValueError('Need the complete unrevised combat-recovery Adam boundary')
    archive=root/'checkpoints'/f'combat_before_batch128_g{step}.pt'
    if archive.exists():raise FileExistsError('Never replace a saved Adam boundary')
    shutil.copy2(checkpoint,archive);source_sha=file_sha(archive)
    if file_sha(checkpoint)!=source_sha:raise ValueError('Source changed while archiving')
    for source,target in [('status.json',f'status_before_batch128_g{step}.json'),
        ('launch_command.json','launch_command_before_batch128.json'),
        ('metrics.jsonl',f'metrics_before_batch128_g{step}.jsonl')]:shutil.copy2(root/source,root/target)
    preview_path=root/f'preview_assets_batch128_g{step}.json'
    preview={'version':1,'kind':'frozen_preview_assets_v1','allow_unqualified_codec':True,
        'renderer':{'checkpoint':str(args.renderer.resolve()),'file_sha256':file_sha(args.renderer),
            'refinement_updates':6000},
        'sr':{'checkpoint':str(args.sr.resolve()),'file_sha256':file_sha(args.sr),
            'selected_local_update':18100}}
    atomic_json(preview_path,preview)
    plan={'version':4,'created_at':now(),'reason':'user_requested_effective128_and_distance_error_fix',
        'source_checkpoint':str(archive),'source_checkpoint_sha256':source_sha,
        'start_optimizer_step':step,'base_plan_sha256':file_sha(root/'combat_training_plan.json'),
        'parent_recovery_sha256':training['combat_recovery_revision']['sha256'],
        'feedback_revision_sha256':training['feedback_revision']['sha256'],
        'microbatch_size':32,'gradient_accumulation_steps':4,'effective_batch_size':128,
        'gradient_reduction':'mean_over_microbatches','same_environment_per_optimizer_step':True,
        'same_context_per_optimizer_step':True,'same_latent_mode_per_optimizer_step':True,
        'source_future_feedback':False,'optimizer_reset':False,'rollback':False,
        'combat_loss_overrides':{'target_supervision':'shot_distance_balanced'},
        'target_distance_balanced_sampling':True,
        'combat_microbatch_quotas':{'spawn':8,'attack_negative':8,'hit_near':3,'hit_mid':3,
            'hit_far':3,'fired_miss':3,'legacy':4},
        'combat_target_normalization':'Source shots only; hit/miss1:1; near/mid/far hit means equally weighted',
        'preview_assets_plan':str(preview_path),'preview_assets_sha256':file_sha(preview_path),
        'validation_every_optimizer_updates':250,'main_horizon':450,'tbptt_chunk':16,
        'user_game_window_untouched':True}
    path=root/'batch128_revision.json'
    if path.exists():raise FileExistsError('Do not overwrite an immutable batch revision')
    atomic_json(path,plan)
    launch=json.loads((root/'launch_command.json').read_text(encoding='utf-8'))
    argv=list(launch['argv']);argv[argv.index('--batch')+1]='32'
    argv+=['--accumulation-steps','4','--batch-revision-plan',str(path),
        '--preview-assets-plan',str(preview_path)]
    stop=root/'stop_training'
    if stop.exists():stop.unlink()
    out=(root/'batch128.stdout.log').open('ab');err=(root/'batch128.stderr.log').open('ab')
    try:
        child=subprocess.Popen(argv,cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,
            stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)
    finally:out.close();err.close()
    launch.update(argv=argv,pid=child.pid,resumed_at=now(),resumed_from_global=step,
        microbatch_size=32,gradient_accumulation_steps=4,effective_batch_size=128,
        batch_revision_plan_sha256=file_sha(path),preview_assets_plan=str(preview_path),
        weights_preserved=True,optimizer_preserved=True,no_rollback=True,user_game_window_untouched=True)
    atomic_json(root/'launch_command.json',launch)
    print(json.dumps({'pid':child.pid,'source_step':step,'effective_batch':128,
        'preview_assets':str(preview_path),'batch_revision':str(path)}),flush=True)


if __name__=='__main__':main()
