"""Continue the coordinate curriculum with pinned long/late-state replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from .rssm_replay_curriculum import replay_curriculum_plan
from .rssm_late_prefix_recovery import LATE_PREFIX_RECOVERY_PROTOCOL
from .rssm_spawn_curriculum import file_sha
from .training_pipeline import atomic_json, now
from .vqgan_pipeline import process_alive


def inherit_best(source, root):
    """Retain the selected historical weights without deploying or rolling back."""
    path = source/'best_drift.json'
    if not path.exists():
        raise FileNotFoundError('The source best-drift selector must be retained')
    value = json.loads(path.read_text(encoding='utf-8'))
    original = Path(value['checkpoint']).resolve(strict=True)
    directory = root/'checkpoints/best_drift_candidates'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/original.name
    try:
        os.link(original, target)
    except OSError:
        shutil.copy2(original, target)
    alias = root/'checkpoints/best_drift.pt'
    try:
        os.link(target, alias)
    except OSError:
        shutil.copy2(target, alias)
    if file_sha(original) != file_sha(target):
        raise ValueError('Inherited best checkpoint differs')
    value.update(checkpoint=str(target), checkpoint_alias=str(alias),
                 inherited_from_run=str(source), inherited_at=now())
    atomic_json(root/'best_drift.json', value)
    return value


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
        raise ValueError('Need a stopped worker at a complete source Adam boundary')
    checkpoint = source/'checkpoints/rssm_world.pt'
    digest = file_sha(checkpoint)
    destination = root/'checkpoints/rssm_world.pt'
    plan_path, init_path = root/'replay_curriculum_plan.json', root/'initialization.json'
    coordinate_path = Path(state['rssm_training']['coordinate_curriculum']['plan'])
    coordinate_plan = json.loads(coordinate_path.read_text(encoding='utf-8'))
    seed = Path(coordinate_plan['source_checkpoint']).resolve(strict=True)
    seed_sha = file_sha(seed)
    if seed_sha != coordinate_plan['source_checkpoint_sha256']:
        raise ValueError('Original long-horizon EMA seed checkpoint changed')
    if args.prepare_only:
        if destination.exists() or plan_path.exists() or init_path.exists():
            raise FileExistsError('Never overwrite an existing replay continuation')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, destination)
        if file_sha(destination) != digest or file_sha(checkpoint) != digest:
            raise ValueError('Complete source model/Adam copy differs')
        shutil.copy2(source/'combat_training_plan.json', root/'combat_training_plan.json')
        shutil.copy2(source/'launch_command.json', root/'source_launch_command.json')
        plan = replay_curriculum_plan(state, source_checkpoint=checkpoint,
            source_checkpoint_sha256=digest, long_seed_checkpoint=seed, long_seed_sha256=seed_sha,
            base_plan_sha256=file_sha(root/'combat_training_plan.json'),
            late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL)
        atomic_json(plan_path, plan)
        best = inherit_best(source, root)
        launch = json.loads((source/'launch_command.json').read_text(encoding='utf-8'))
        argv = list(launch['argv']); argv[argv.index('--run-dir')+1] = str(root)
        argv += ['--replay-curriculum-plan', str(plan_path)]
        launch.update(argv=argv, pid=None, source_run=str(source), status='prepared',
            replay_curriculum_plan=str(plan_path), resumed_from_global=state['global_training_step'],
            coordinate_updates_preserved=state['rssm_training']['coordinate_curriculum']['updates'],
            model_Adam_RNG_and_EMA_preserved=True, game_window_untouched=True)
        atomic_json(root/'launch_command.json', launch)
        atomic_json(init_path, dict(source_step=state['global_training_step'], source_run=str(source),
            source_checkpoint=str(checkpoint), source_checkpoint_sha256=digest,
            copied_checkpoint_sha256=file_sha(destination), complete_checkpoint_copied_bitexact=True,
            long_seed_checkpoint=str(seed), long_seed_checkpoint_sha256=seed_sha,
            replay_plan_sha256=file_sha(plan_path), optimizer_reset=False,
            inherited_best_step=best['step'], prepared_at=now(), game_window_untouched=True))
        print(json.dumps(dict(status='prepared', source_step=state['global_training_step'], run_dir=str(root))))
        return
    init = json.loads(init_path.read_text(encoding='utf-8'))
    launch = json.loads((root/'launch_command.json').read_text(encoding='utf-8'))
    if launch.get('pid') and process_alive(int(launch['pid'])):
        raise ValueError('Replay worker is already running')
    if (init['source_checkpoint_sha256'] != digest or file_sha(destination) != digest
            or init['replay_plan_sha256'] != file_sha(plan_path)
            or init['long_seed_checkpoint_sha256'] != seed_sha):
        raise ValueError('Prepared source or replay plan changed')
    probe_path = root/'replay_probe.json'
    probe = json.loads(probe_path.read_text(encoding='utf-8'))
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if (probe.get('passed') is not True or probe.get('source_checkpoint_sha256') != digest
            or probe.get('optimizer_steps') != 0 or probe.get('horizons') != [16, 450]
            or probe.get('replay_plan_sha256') != init['replay_plan_sha256']
            or probe.get('long_seed_checkpoint_sha256') != seed_sha
            or probe.get('late_prefix_protocol') != plan['late_prefix_protocol']
            or probe.get('both_horizon_bank_values_and_clocks_unchanged') is not True
            or probe.get('source_rng_restored_exactly') is not True
            or probe.get('all_gradients_finite') is not True
            or probe.get('model_weights_and_buffers_unchanged') is not True
            or probe.get('all_Adam_tensors_and_options_unchanged') is not True):
        raise ValueError('Need passed discarded short/scout and full450 replay backward probes')
    with (root/'training.stdout.log').open('ab') as out, (root/'training.stderr.log').open('ab') as err:
        child = subprocess.Popen(launch['argv'], cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), close_fds=True)
    launch.update(pid=child.pid, status='started', resumed_at=now(), replay_probe_sha256=file_sha(probe_path))
    atomic_json(root/'launch_command.json', launch)
    atomic_json(root/'start.json', dict(pid=child.pid, source_step=init['source_step'],
        started_at=now(), replay_probe_sha256=file_sha(probe_path), optimizer_reset=False))
    print(json.dumps(dict(pid=child.pid, source_step=init['source_step'], run_dir=str(root))))


if __name__ == '__main__':
    main()
