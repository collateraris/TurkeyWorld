"""Continue pure coordinate training with measured-loss horizon advancement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .continue_rssm_replay import inherit_best
from .rssm_loss_gated_curriculum import loss_gated_curriculum_plan
from .rssm_spawn_curriculum import file_sha
from .training_pipeline import atomic_json, now
from .vqgan_pipeline import process_alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--start-prepared', action='store_true')
    args = parser.parse_args()
    source, root = args.source_run.resolve(), args.run_dir.resolve()
    state = json.loads((source/'status.json').read_text(encoding='utf-8'))
    if state['status'] != 'stopped' or process_alive(int((source/'training.pid').read_text())):
        raise ValueError('Need a stopped worker at a complete saved optimizer boundary')
    if state.get('objective_protocol') != 'rssm_coordinate_drift_only_v1':
        raise ValueError('Loss gating preserves the existing pure XY objective')
    checkpoint = source/'checkpoints/rssm_world.pt'
    digest = file_sha(checkpoint)
    destination = root/'checkpoints/rssm_world.pt'
    plan_path, init_path = root/'loss_gated_curriculum_plan.json', root/'initialization.json'
    if args.prepare_only:
        if destination.exists() or plan_path.exists() or init_path.exists():
            raise FileExistsError('Never overwrite an existing loss-gated continuation')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, destination)
        if file_sha(destination) != digest or file_sha(checkpoint) != digest:
            raise ValueError('Complete model/Adam/RNG source copy differs')
        shutil.copy2(source/'combat_training_plan.json', root/'combat_training_plan.json')
        shutil.copy2(source/'launch_command.json', root/'source_launch_command.json')
        plan = loss_gated_curriculum_plan(state, source_checkpoint=checkpoint,
            source_checkpoint_sha256=digest, base_plan_sha256=file_sha(root/'combat_training_plan.json'))
        atomic_json(plan_path, plan)
        best = inherit_best(source, root)
        launch = json.loads((source/'launch_command.json').read_text(encoding='utf-8'))
        argv = list(launch['argv']); argv[argv.index('--run-dir')+1] = str(root)
        argv += ['--loss-gated-curriculum-plan', str(plan_path)]
        launch.update(argv=argv, pid=None, source_run=str(source), status='prepared',
            loss_gated_curriculum_plan=str(plan_path), resumed_from_global=state['global_training_step'],
            full_model_Adam_RNG_preserved=True, forced_long_updates_disabled=True,
            game_window_untouched=True)
        atomic_json(root/'launch_command.json', launch)
        atomic_json(init_path, dict(source_step=state['global_training_step'], source_run=str(source),
            source_checkpoint=str(checkpoint), source_checkpoint_sha256=digest,
            copied_checkpoint_sha256=file_sha(destination), complete_checkpoint_copied_bitexact=True,
            loss_gated_plan_sha256=file_sha(plan_path), optimizer_reset=False,
            inherited_best_step=best['step'], prepared_at=now(), game_window_untouched=True))
        print(json.dumps(dict(status='prepared', source_step=state['global_training_step'], run_dir=str(root))))
        return
    init = json.loads(init_path.read_text(encoding='utf-8'))
    launch = json.loads((root/'launch_command.json').read_text(encoding='utf-8'))
    if launch.get('pid') and process_alive(int(launch['pid'])):
        raise ValueError('A loss-gated worker is already running')
    if (init['source_checkpoint_sha256'] != digest or file_sha(destination) != digest
            or init['loss_gated_plan_sha256'] != file_sha(plan_path)):
        raise ValueError('Prepared checkpoint or loss-gated plan changed')
    probe_path = root/'loss_gated_probe.json'
    probe = json.loads(probe_path.read_text(encoding='utf-8'))
    if (probe.get('passed') is not True or probe.get('source_checkpoint_sha256') != digest
            or probe.get('loss_gated_plan_sha256') != init['loss_gated_plan_sha256']
            or probe.get('optimizer_steps') != 0 or probe.get('horizons') != [16, 96, 450]
            or probe.get('all_gradients_finite') is not True
            or probe.get('disabled_parameter_gradients_none') is not True
            or probe.get('model_weights_and_buffers_unchanged') is not True
            or probe.get('all_Adam_tensors_and_options_unchanged') is not True
            or probe.get('ema_values_unchanged') is not True
            or probe.get('forced_long_updates_disabled') is not True):
        raise ValueError('Need measured pure-XY short/new96 backward probes and loss-gate checks')
    with (root/'training.stdout.log').open('ab') as out, (root/'training.stderr.log').open('ab') as err:
        child = subprocess.Popen(launch['argv'], cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), close_fds=True)
    launch.update(pid=child.pid, status='started', resumed_at=now(), probe_sha256=file_sha(probe_path))
    atomic_json(root/'launch_command.json', launch)
    atomic_json(root/'start.json', dict(pid=child.pid, source_step=init['source_step'], started_at=now(),
        probe_sha256=file_sha(probe_path), optimizer_reset=False, forced_long_updates_disabled=True))
    print(json.dumps(dict(pid=child.pid, source_step=init['source_step'], run_dir=str(root))))


if __name__ == '__main__':
    main()
