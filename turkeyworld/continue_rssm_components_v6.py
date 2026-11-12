"""Continue the saved combat-only run with calibrated late-event objectives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .rssm_component_balance import LOSS_PROTOCOL, SETTINGS
from .rssm_combat_branches import normalize_combat_branch_protocol
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
        raise ValueError('Wait for the complete32x4 source Adam boundary')
    if status.get('training_schedule')!='combat_only_v1':
        raise ValueError('V6 continues the existing combat-only player/enemy RSSMs')
    calibration_path=root/'calibration.json'
    calibration=json.loads(calibration_path.read_text(encoding='utf-8'))
    step=status['global_training_step'];archive=root/f'source_g{step}.pt'
    if (calibration.get('source_step')!=step or calibration.get('settings')!=SETTINGS
            or calibration.get('passed')is not True
            or calibration.get('passed_priority_budget_check')is not True
            or calibration.get('model_weights_and_buffers_unchanged')is not True
            or calibration.get('optimizer_updated')is not False):
        raise ValueError('Need a safe measured calibration at this exact source boundary')
    if not archive.exists():raise FileNotFoundError('Use the immutable checkpoint measured by the gradient probe')
    digest=file_sha(archive)
    if (digest!=calibration['source_checkpoint_sha256']
            or file_sha(source/'checkpoints/rssm_world.pt')!=digest):
        raise ValueError('Gradient measurement/source optimizer boundary differs')
    checkpoint=root/'checkpoints/rssm_world.pt'
    if checkpoint.exists():raise FileExistsError('Never overwrite a component-training run')
    checkpoint.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(archive,checkpoint)
    if file_sha(checkpoint)!=digest:raise ValueError('Copied complete source checkpoint differs')
    shutil.copy2(source/'combat_training_plan.json',root/'combat_training_plan.json')
    shutil.copy2(source/'launch_command.json',root/'source_launch_command.json')
    q=status['rssm_training']
    plan={'version':6,'loss_protocol':LOSS_PROTOCOL,'start_optimizer_step':step,
        'source_checkpoint':str(archive),'source_checkpoint_sha256':digest,
        'base_plan_sha256':file_sha(root/'combat_training_plan.json'),
        'parent_loss_balance_sha256':q['loss_balance_revision']['sha256'],
        'parent_batch_sha256':q['batch_revision']['sha256'],
        'parent_combat_only_sha256':q['combat_only_revision']['sha256'],
        'gradient_calibration':str(calibration_path),'gradient_calibration_sha256':file_sha(calibration_path),
        'component_settings':SETTINGS,'combat_branch_protocol':normalize_combat_branch_protocol({'version':2}),
        'attack_positive_balanced_sampling':True,'optimizer_reset':False,'source_future_feedback':False,
        'main_horizon':450,'tbptt_chunk':16,'validation_every':250,'preview_every':10,
        'control_validation_updates':[100,250],'reason':'explicit_user_request_four_combat_training_fixes',
        'created_at':now(),'game_window_untouched':True,'model_parameters_unchanged':True}
    path=root/'combat_v6_revision.json';atomic_json(path,plan)
    launch=json.loads((source/'launch_command.json').read_text(encoding='utf-8'))
    argv=list(launch['argv']);argv[argv.index('--run-dir')+1]=str(root)
    argv+=['--combat-v6-plan',str(path)]
    out=(root/'training.stdout.log').open('ab');err=(root/'training.stderr.log').open('ab')
    try:
        child=subprocess.Popen(argv,cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,
            stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)
    finally:out.close();err.close()
    launch.update(argv=argv,pid=child.pid,resumed_at=now(),resumed_from_global=step,
        source_run=str(source),combat_v6_plan=str(path),loss_protocol=LOSS_PROTOCOL,
        weights_and_optimizer_preserved=True,game_window_untouched=True)
    atomic_json(root/'launch_command.json',launch)
    atomic_json(root/'initialization.json',{'source_step':step,'source_checkpoint':str(archive),
        'source_checkpoint_sha256':digest,'copied_checkpoint_sha256':file_sha(checkpoint),
        'complete_checkpoint_copied_bitexact':True,'optimizer_reset':False,
        'gradient_calibration_sha256':file_sha(calibration_path),'combat_v6_plan':str(path),
        'pid':child.pid,'started_at':now(),'game_window_untouched':True})
    print(json.dumps({'pid':child.pid,'source_step':step,'run_dir':str(root),'protocol':LOSS_PROTOCOL}),flush=True)


if __name__=='__main__':main()
