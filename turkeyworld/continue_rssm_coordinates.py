"""Preserve a stopped64x4 checkpoint and continue the pinned drift curriculum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .rssm_coordinate_curriculum import coordinate_curriculum_plan
from .rssm_kinematic_losses import KINEMATIC_PROTOCOL_V1
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
        raise ValueError('Need an exited worker and a complete saved source Adam boundary')
    checkpoint = source/'checkpoints/rssm_world.pt'
    digest = file_sha(checkpoint)
    destination = root/'checkpoints/rssm_world.pt'
    plan_path, init_path = root/'coordinate_curriculum_plan.json', root/'initialization.json'
    if args.prepare_only:
        from .rssm_memory_overshooting import MEMORY_OVERSHOOTING_PROTOCOL
        if destination.exists() or plan_path.exists() or init_path.exists():
            raise FileExistsError('Never overwrite an existing coordinate continuation')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, destination)
        if file_sha(destination) != digest or file_sha(checkpoint) != digest:
            raise ValueError('Complete source copy changed')
        shutil.copy2(source/'combat_training_plan.json', root/'combat_training_plan.json')
        shutil.copy2(source/'launch_command.json', root/'source_launch_command.json')
        plan = coordinate_curriculum_plan(state, source_checkpoint=checkpoint,
            source_checkpoint_sha256=digest, base_plan_sha256=file_sha(root/'combat_training_plan.json'),
            kinematic_protocol=KINEMATIC_PROTOCOL_V1, overshooting_protocol=MEMORY_OVERSHOOTING_PROTOCOL)
        atomic_json(plan_path, plan)
        launch = json.loads((source/'launch_command.json').read_text(encoding='utf-8'))
        argv = list(launch['argv'])
        argv[argv.index('--run-dir')+1] = str(root)
        argv += ['--coordinate-curriculum-plan', str(plan_path)]
        launch.update(argv=argv, pid=None, source_run=str(source), resumed_from_global=state['global_training_step'],
            coordinate_curriculum_plan=str(plan_path), model_Adam_RNG_and_EMA_preserved=True,
            game_window_untouched=True, status='prepared')
        atomic_json(root/'launch_command.json', launch)
        atomic_json(init_path, dict(source_checkpoint=str(checkpoint), source_checkpoint_sha256=digest,
            source_step=state['global_training_step'], copied_checkpoint_sha256=file_sha(destination),
            complete_checkpoint_copied_bitexact=True, optimizer_reset=False,
            coordinate_curriculum_plan_sha256=file_sha(plan_path), prepared_at=now(),
            source_run=str(source), game_window_untouched=True))
        print(json.dumps(dict(status='prepared', source_step=state['global_training_step'], run_dir=str(root))))
        return
    init = json.loads(init_path.read_text(encoding='utf-8'))
    launch = json.loads((root/'launch_command.json').read_text(encoding='utf-8'))
    if launch.get('pid') and process_alive(int(launch['pid'])):
        raise ValueError('Coordinate worker is already running')
    if (init['source_checkpoint_sha256'] != digest or file_sha(destination) != digest
            or init['coordinate_curriculum_plan_sha256'] != file_sha(plan_path)
            or state['global_training_step'] != init['source_step']):
        raise ValueError('Prepared model/Adam source or pinned curriculum changed')
    probe_path = root/'coordinates_probe.json'
    probe = json.loads(probe_path.read_text(encoding='utf-8'))
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if (probe.get('passed') is not True or probe.get('source_checkpoint_sha256') != digest
            or probe.get('optimizer_steps') != 0 or probe.get('component_EMA_updates') != 0
            or probe.get('horizons') != [16, 450]
            or probe.get('coordinate_gradient_protocol') != plan['coordinate_gradient_protocol']
            or probe.get('kinematic_protocol') != plan['kinematic_protocol']
            or probe.get('memory_overshooting_protocol') != plan['overshooting_protocol']
            or probe.get('model_weights_and_buffers_unchanged') is not True
            or probe.get('all_Adam_tensors_and_options_unchanged') is not True
            or probe.get('all_gradients_finite') is not True):
        raise ValueError('Need the passed short16 and full450 coordinate/memory backward probes')
    with (root/'training.stdout.log').open('ab') as out, (root/'training.stderr.log').open('ab') as err:
        child = subprocess.Popen(launch['argv'], cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), close_fds=True)
    launch.update(pid=child.pid, status='started', resumed_at=now(),
        coordinates_probe_sha256=file_sha(probe_path))
    atomic_json(root/'launch_command.json', launch)
    atomic_json(root/'start.json', dict(pid=child.pid, source_step=init['source_step'],
        started_at=now(), coordinates_probe_sha256=file_sha(probe_path), optimizer_reset=False))
    print(json.dumps(dict(pid=child.pid, source_step=init['source_step'], run_dir=str(root))))


if __name__ == '__main__':
    main()
