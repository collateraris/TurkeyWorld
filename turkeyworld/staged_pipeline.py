"""Durable collection monitor and single-enemy staged teacher supervisor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .engine import BEHAVIOR_VERSION, ENTITY_SLOTS, STATE_DIM
from .power_guard import KeepAwake
from .staged_curriculum import STAGES
from .training_pipeline import atomic_json, now
from .vqgan_pipeline import process_alive


def build_plan(data_root):
    root = Path(data_root).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('status') != 'complete'
            or manifest.get('config', {}).get('behavior_version') != BEHAVIOR_VERSION):
        raise ValueError('The single-enemy staged corpus is not complete')
    datasets = {}
    for stage in STAGES:
        name = stage['id']
        records = [record for record in manifest['records'] if record['stage'] == name]
        train = sorted([r for r in records if r['split'] == 'train'], key=lambda r: r['relative_path'])
        val = sorted([r for r in records if r['split'] == 'val'], key=lambda r: r['relative_path'])
        if not train or len(val) < 32:
            raise ValueError('Every curriculum stage needs train data and32 held-out episodes')
        for record in records:
            path = (root/record['relative_path']).resolve()
            if root not in path.parents or not path.is_file() or path.stat().st_size != record['size_bytes']:
                raise ValueError('Staged corpus record changed or escapes the data root')
        if name == 'full_game':
            train = [r for r in train if r['eligible_450']]
            val = [r for r in val if r['eligible_450']]
            if not train or len(val) < 32:
                raise ValueError('Full-game data needs complete uninterrupted450-future-step episodes')
        datasets[name] = {'directory': str(root/name),
            'kind': 'renderer_snapshots' if stage['renderer_only'] else 'world_transitions',
            'train_filenames': [Path(r['relative_path']).name for r in train],
            'val_filenames': [Path(r['relative_path']).name for r in val],
            'train_clips': len(train), 'validation_clips': len(val),
            'train_transitions': sum(r['transitions'] for r in train),
            'train_snapshots': sum(r['snapshots'] for r in train),
            'records': [{key: r[key] for key in ('relative_path', 'seed', 'size_bytes')}
                        for r in train+val]}
    return {'version': 1, 'entity_slots': ENTITY_SLOTS, 'state_dim': STATE_DIM,
        'behavior_version': BEHAVIOR_VERSION, 'target_seconds': 15, 'target_steps': 450,
        'validation_every': 100, 'required_passes': 2, 'replay_fraction': .25,
        'data_fingerprint': manifest['config']['fingerprint'], 'datasets': datasets}


def run(args):
    root, data = args.run_dir.resolve(), args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    pipeline_pid = root/'pipeline.pid'
    if pipeline_pid.exists() and process_alive(int(pipeline_pid.read_text())):
        raise ValueError('Staged supervisor is already running')
    atomic_json(pipeline_pid, os.getpid())
    with KeepAwake() as guard:
        while True:
            status = json.loads((root/'status.json').read_text(encoding='utf-8'))
            status['power_guard'] = {'active': guard.active, 'display_required': False,
                'screen_lock_allowed': True, 'error': guard.error}
            if (root/'stop_training').exists():
                status.update(status='stopped', updated_at=now())
                atomic_json(root/'status.json', status)
                return
            path = data/'status.json'
            progress = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status': 'preparing'}
            status.setdefault('staged_curriculum', {})['data_preparation'] = progress
            status.update(data_preparation=progress, updated_at=now())
            atomic_json(root/'status.json', status)
            if progress.get('status') == 'failed':
                raise RuntimeError('Staged data collection failed: '+str(progress.get('error')))
            if progress.get('status') == 'complete' and (root/'training_ready.json').exists():
                break
            time.sleep(2)
        plan = build_plan(data)
        plan_path = root/'curriculum_plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('The immutable curriculum plan changed')
        atomic_json(plan_path, plan)
        if not args.source.is_file():
            raise ValueError('Migrated single-enemy initialization is missing')
        command = [sys.executable, '-X', 'utf8', '-u', '-m', 'turkeyworld.train_staged_world',
            '--run-dir', str(root), '--source', str(args.source.resolve()),
            '--curriculum-manifest', str(plan_path), '--sr-checkpoint', str(args.sr_checkpoint.resolve()),
            '--device', args.device, '--batch', '16', '--gradient-accumulation-steps', '1',
            '--target-seconds', '15', '--updates', str(args.updates), '--cache-mb', '128']
        atomic_json(root/'launch_command.json', command)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONFAULTHANDLER='1')
        with (root/'training.stdout.log').open('ab') as stdout, (root/'training.stderr.log').open('ab') as stderr:
            child = subprocess.Popen(command, cwd=Path(__file__).resolve().parent.parent, env=env,
                stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, close_fds=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        atomic_json(root/'training.pid', child.pid)
        record = {'run_dir': str(root), 'pid': child.pid, 'updated_at': now(), 'native_image_size': 32}
        atomic_json(args.monitor_dir.resolve()/'active_run.json', record)
        atomic_json(root/'active_run.json', record)
        code = child.wait()
        if code:
            raise RuntimeError('Staged teacher exited with code'+str(code))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sr-checkpoint', type=Path, required=True)
    parser.add_argument('--monitor-dir', type=Path, default=Path('training_runs/vqgan_20261109'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--updates', type=int, default=5182)
    args = parser.parse_args(argv)
    try:
        run(args)
    except BaseException as error:
        path = args.run_dir/'status.json'
        if path.exists():
            status = json.loads(path.read_text(encoding='utf-8'))
            status.update(status='failed', error=str(error), updated_at=now())
            atomic_json(path, status)
        raise


if __name__ == '__main__':
    main()
