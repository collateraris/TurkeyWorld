"""Durable handoff: qualified scratch VQGAN -> latent teacher -> latent student.

The codec worker starts a detached world worker and exits, releasing its GPU
loss networks/optimizers. Both the local and hosted monitor follow active_run.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .training_pipeline import atomic_json, now
from .vq_world import migrate_teacher_to_vq, validate_frozen_codec


def process_alive(pid):
    if not isinstance(pid, int) or pid < 1:
        return False
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def link_data(destination, source):
    destination, source = Path(destination), Path(source).resolve(strict=True)
    if destination.exists():
        if not destination.samefile(source):
            raise ValueError('Existing world data directory refers to a different corpus')
        return
    if os.name == 'nt':
        # PowerShell literal single-quoted paths; no shell substitution.
        quote = lambda value: "'"+str(value).replace("'", "''")+"'"
        subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command',
            f"New-Item -ItemType Junction -Path {quote(destination)} -Value {quote(source)} | Out-Null"],
            check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        destination.symlink_to(source, target_is_directory=True)


def recover_migration(initial, resume, lineage, source, fingerprint):
    """Recover a crash between the resume, initial and lineage atomic writes."""
    import torch
    from .vq_world import _save_new
    candidate = initial if initial.exists() else resume
    saved = torch.load(candidate, map_location='cpu', mmap=True, weights_only=False)
    report = saved.get('vq_migration', {})
    if (report.get('codec_fingerprint') != fingerprint
            or Path(report.get('source_checkpoint', '')).resolve() != source.resolve()
            or Path(report.get('initial_checkpoint', '')).resolve() != initial.resolve()):
        raise ValueError('Incomplete migration belongs to a different source or frozen codec')
    if report.get('optimizer_preserved'):
        if not resume.is_file():
            raise RuntimeError('Incomplete migration lost its promised optimizer resume checkpoint')
        full = saved if candidate == resume else torch.load(resume, map_location='cpu', mmap=True, weights_only=False)
        if (full.get('pipeline_phase') != 'teacher500vq_world' or
                full.get('vq_migration') != report or 'optimizer' not in full):
            raise ValueError('Incomplete migration resume metadata differs from initial weights')
    if not initial.exists():
        if saved.get('pipeline_phase') != 'teacher500vq_world':
            raise ValueError('Orphan migration artifact is not a world pretraining resume')
        payload = {key: value for key, value in saved.items() if key not in
            ('optimizer', 'ema_target_encoder', 'rng_numpy', 'rng_torch', 'rng_cuda', 'phase_complete')}
        payload.update(training_steps=0, pipeline_phase='teacher500vq_initial')
        _save_new(initial, payload)
    atomic_json(lineage, report)
    return report


def continue_world_after_codec(codec_path, args):
    # Validate before touching the former world job or creating any process.
    info = validate_frozen_codec(codec_path)
    control = args.run_dir.resolve()
    source_root = args.world_source_run.resolve()
    world = Path(getattr(args, 'world_run_dir', None) or
        control.parent/'vjepa500_vqgan_20261109').resolve()
    if world in (control, source_root):
        raise ValueError('Latent world run needs a separate run directory')
    if (control/'stop_training').exists():
        return None
    old_pid_path = source_root/'training.pid'
    if old_pid_path.exists():
        old_pid = int(old_pid_path.read_text().strip())
        if process_alive(old_pid):
            (source_root/'stop_training').touch()
            deadline = time.monotonic()+180
            while process_alive(old_pid):
                if time.monotonic() > deadline:
                    raise RuntimeError('Previous teacher is still checkpointing; resume codec handoff later')
                time.sleep(.5)
    source = source_root/'checkpoints'/'teacher500_world.pt'
    if not source.is_file():
        raise FileNotFoundError(f'500M teacher source checkpoint is missing: {source}')
    world.mkdir(parents=True, exist_ok=True)
    link_data(world/'data', args.data_dir)
    initial = world/'checkpoints'/'teacher_vjepa500vq_initial.pt'
    resume = world/'checkpoints'/'teacher500vq_world.pt'
    lineage = world/'vqgan_lineage.json'
    if (initial.exists() or resume.exists()) and not lineage.exists():
        recover_migration(initial, resume, lineage, source, info['codec_fingerprint'])
    if initial.exists():
        # A durable lineage is written after both migration artifacts. Never
        # silently pair an existing world job with a newly trained codebook.
        report = json.loads(lineage.read_text(encoding='utf-8'))
        if (report['codec_fingerprint'] != info['codec_fingerprint'] or
                Path(report['source_checkpoint']).resolve() != source):
            raise ValueError('Existing world run uses a different frozen codec')
    else:
        report = migrate_teacher_to_vq(source, codec_path, initial, resume_output=resume)
        atomic_json(lineage, report)
    panel = source_root/'validation_panel500.json'
    frozen_panel = world/'validation_panel500vq.json'
    if panel.exists() and not frozen_panel.exists():
        shutil.copyfile(panel, frozen_panel)
    atomic_json(world/'codec_ready.json', info)
    pid_path = world/'training.pid'
    launch_pending = world/'launch_pending.json'
    if pid_path.exists() and process_alive(int(pid_path.read_text().strip())):
        live_pid = int(pid_path.read_text().strip())
        atomic_json(control/'active_run.json', {'run_dir': str(world),
            'codec_fingerprint': info['codec_fingerprint'], 'pid': live_pid, 'updated_at': now()})
        atomic_json(control/'world_handoff.json', {'run_dir': str(world), 'pid': live_pid,
            'migration': report, 'created_at': now()})
        launch_pending.unlink(missing_ok=True)
        return world
    if (world/'stop_training').exists():
        raise RuntimeError('Latent world run is stopped; remove its stop_training marker to resume explicitly')
    if launch_pending.exists():
        raise RuntimeError('A previous world launch is pending; wait for its worker PID before retrying')
    command = [sys.executable, '-X', 'utf8', '-u', '-m', 'turkeyworld.train_vq_world',
        '--run-dir', str(world), '--source', str(source), '--codec-checkpoint', str(Path(codec_path).resolve()),
        '--control-run-dir', str(control), '--device', args.device]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    workspace = Path(__file__).resolve().parent.parent
    atomic_json(world/'status.json', {'run_id': world.name, 'status': 'running',
        'stage': 'initializing', 'stage_label': 'Запуск учителя с общим замороженным VQGAN',
        'codec_ready': info, 'codec_validation': info['quality_gate'],
        'codec_training': {'frozen': True, 'from_scratch': True,
            'consecutive_passes': info['quality_gate']['consecutive_passes'],
            'required_passes': info['quality_gate']['required_passes']},
        'evaluation': {'status': 'pending', 'target_seconds': 300, 'passed': False}, 'updated_at': now()})
    with launch_pending.open('x', encoding='utf-8') as pending:
        json.dump({'created_at': now(), 'codec_fingerprint': info['codec_fingerprint']}, pending)
    try:
        with (world/'training.stdout.log').open('ab') as stdout, (world/'training.stderr.log').open('ab') as stderr:
            process = subprocess.Popen(command, cwd=workspace, env=env, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, close_fds=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                start_new_session=os.name != 'nt')
    except Exception:
        launch_pending.unlink(missing_ok=True)
        raise
    atomic_json(pid_path, process.pid)
    atomic_json(control/'active_run.json', {'run_dir': str(world),
        'codec_fingerprint': info['codec_fingerprint'], 'pid': process.pid, 'updated_at': now()})
    atomic_json(control/'world_handoff.json', {'run_dir': str(world), 'pid': process.pid,
        'migration': report, 'created_at': now()})
    launch_pending.unlink(missing_ok=True)
    return world
