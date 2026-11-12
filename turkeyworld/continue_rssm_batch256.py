"""Continue an immutable v6 checkpoint with the measured64x4 batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .rssm_batch256 import BATCH_PROTOCOL, scaled_branch_protocol
from .rssm_component_balance import LOSS_PROTOCOL
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
        raise ValueError('Wait for the current complete32x4 optimizer boundary')
    step=status['global_training_step'];archive=root/f'source_g{step}.pt'
    probe_path=root/'batch64_probe.json';probe=json.loads(probe_path.read_text(encoding='utf-8'))
    digest=file_sha(archive)
    if (probe.get('passed')is not True or probe.get('source_step')!=step
            or probe.get('source_checkpoint_sha256')!=digest
            or probe.get('microbatch_size')!=64 or probe.get('optimizer_updated')is not False
            or probe.get('model_weights_and_buffers_unchanged')is not True
            or probe.get('all_Adam_tensors_and_options_unchanged')is not True
            or probe.get('all_gradients_finite')is not True
            or file_sha(source/'checkpoints/rssm_world.pt')!=digest):
        raise ValueError('Need the passed full64-example backward probe at this exact saved source')
    checkpoint=root/'checkpoints/rssm_world.pt'
    if checkpoint.exists():raise FileExistsError('Never overwrite an existing batch256 run')
    checkpoint.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(archive,checkpoint)
    if file_sha(checkpoint)!=digest:raise ValueError('Copied full model/Adam/RNG checkpoint differs')
    shutil.copy2(source/'combat_training_plan.json',root/'combat_training_plan.json')
    shutil.copy2(source/'launch_command.json',root/'source_launch_command.json')
    training=status['rssm_training']
    plan={'version':7,'batch_protocol':BATCH_PROTOCOL,'loss_protocol':LOSS_PROTOCOL,
        'start_optimizer_step':step,'source_checkpoint':str(archive),'source_checkpoint_sha256':digest,
        'batch64_probe':str(probe_path),'batch64_probe_sha256':file_sha(probe_path),
        'parent_v6_sha256':training['combat_v6_revision']['sha256'],
        'parent_batch_sha256':training['batch_revision']['sha256'],
        'microbatch_size':64,'gradient_accumulation_steps':4,'effective_batch_size':256,
        'previous_effective_batch_size':128,'gradient_reduction':'mean_over_microbatches',
        'combat_branch_protocol':scaled_branch_protocol(status['training_protocol']['combat_branch_protocol']),
        'optimizer_reset':False,'source_future_feedback':False,
        'main_horizon':450,'tbptt_chunk':16,'validation_every':250,'preview_every':10,
        'v6_control_validation_clock_preserved':True,'component_priorities_and_EMA_preserved':True,
        'reason':'explicit_user_requested64x4','created_at':now(),'game_window_untouched':True}
    path=root/'batch256_revision.json';atomic_json(path,plan)
    launch=json.loads((source/'launch_command.json').read_text(encoding='utf-8'))
    argv=list(launch['argv']);argv[argv.index('--run-dir')+1]=str(root);argv[argv.index('--batch')+1]='64'
    argv+=['--batch256-plan',str(path)]
    out=(root/'training.stdout.log').open('ab');err=(root/'training.stderr.log').open('ab')
    try:
        child=subprocess.Popen(argv,cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,
            stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)
    finally:out.close();err.close()
    launch.update(argv=argv,pid=child.pid,resumed_at=now(),resumed_from_global=step,
        source_run=str(source),batch256_plan=str(path),batch_protocol=BATCH_PROTOCOL,
        microbatch_size=64,gradient_accumulation_steps=4,effective_batch_size=256,
        model_Adam_RNG_and_EMA_preserved=True,game_window_untouched=True)
    atomic_json(root/'launch_command.json',launch)
    atomic_json(root/'initialization.json',{'source_step':step,'source_checkpoint':str(archive),
        'source_checkpoint_sha256':digest,'copied_checkpoint_sha256':file_sha(checkpoint),
        'complete_checkpoint_copied_bitexact':True,'optimizer_reset':False,
        'batch64_probe_sha256':file_sha(probe_path),'batch256_plan':str(path),
        'pid':child.pid,'started_at':now(),'game_window_untouched':True})
    print(json.dumps({'pid':child.pid,'source_step':step,'run_dir':str(root),'batch':'64x4=256'}),flush=True)


if __name__=='__main__':main()
