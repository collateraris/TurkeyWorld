"""Fresh native-RGB codec -> learned x4 SR -> fresh world/CFM/GAN workers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .power_guard import KeepAwake
from .training_pipeline import atomic_json
from .training_subset import create_training_subset
from .vqgan_pipeline import link_data, process_alive


def now():
    return datetime.now(timezone.utc).isoformat()


def build_commands(args, subset, masks, *, codec_info=None):
    # A completed, supervisor-owned migration keeps the original launch config
    # reproducible while making subsequent resumes use the upgraded teacher.
    override_path = args.world_run_dir/'spatial_entities_migration.json'
    spatial = json.loads(override_path.read_text(encoding='utf-8')) if override_path.exists() else {}
    enabled = spatial.get('status') == 'complete' and spatial.get('spatial_entities') is True
    target_seconds = spatial.get('target_seconds', 30) if enabled else 30
    if enabled and (not isinstance(target_seconds, (int, float)) or not 0 < target_seconds <= 120
                    or not float(target_seconds).is_integer()):
        raise ValueError('Invalid spatial migration rollout duration')
    target_seconds = int(target_seconds)
    prefix = [sys.executable, '-X', 'utf8', '-u', '-m']
    codec = prefix+['turkeyworld.train_vqgan', '--run-dir', str(args.codec_run_dir),
        '--data-dir', str(args.data_dir), '--device', args.device,
        '--image-size', str(args.image_size), '--teacher-size', '175m_vqgan',
        '--target-seconds', str(target_seconds), '--world-from-scratch', '--world-run-dir', str(args.world_run_dir),
        '--batch', str(args.codec_batch), '--initial-files', '0', '--growth-files', '2048',
        '--disc-start', '250', '--disc-ramp', '500', '--validate-every', '100',
        '--save-every', '100', '--log-every', '10', '--validation-pairs', '64',
        '--validation-batch', '8', '--quality-psnr', '35', '--quality-ssim', '.98',
        '--quality-mae', str(2/255), '--quality-fraction', '.95', '--quality-passes', '3',
        '--max-steps', '0', '--seed', '273']
    sr = prefix+['turkeyworld.train_sr', '--run-dir', str(args.sr_run_dir),
        '--data-dir', str(args.data_dir), '--device', args.device,
        '--image-size', str(args.image_size), '--batch', str(args.sr_batch),
        '--steps', str(args.sr_steps), '--validate-every', '100', '--preview-every', '100',
        '--save-every', '100']
    world = prefix+['turkeyworld.train_vq_world', '--run-dir', str(args.world_run_dir),
        '--device', args.device,
        '--codec-checkpoint', str(args.codec_run_dir/'checkpoints/codec_for_renderer.pt'),
        '--control-run-dir', str(args.run_dir), '--image-size', str(args.image_size),
        '--teacher-size', '175m_vqgan', '--from-scratch', '--train-renderer-during-pretrain',
        '--target-seconds', str(target_seconds), '--batch', str(args.batch),
        '--gradient-accumulation-steps', str(args.gradient_accumulation_steps),
        '--preview-every', '10', '--minimum-train-clips', '4000', '--visited-clips', '0',
        '--training-subset-manifest', str(subset), '--optimizer-update-budget', '6000',
        '--log-every', '1', '--save-every', '10', '--adversarial-during-pretrain',
        '--pretrain-adversarial-weight', '.005', '--pretrain-adversarial-feature-weight', '.01',
        '--pretrain-adversarial-warmup', '32', '--pretrain-adversarial-ramp', '128',
        '--object-aware-rendering', '--npc-region-boost', '4', '--object-mask-directory', str(masks),
        '--detach-flow-base', '--gan-npc-negatives', '--npc-negative-weight', '1',
        '--npc-negative-min-fraction', '.005', '--sr-checkpoint', str(args.sr_run_dir/'checkpoints/sr_for_game.pt')]
    if codec_info is not None and codec_info.get('accepted_by_user') is True:
        world.append('--accept-current-codec')
    if enabled:
        world.extend(['--spatial-entities', '--motion-curriculum'])
        if spatial.get('sr_checkpoint'):
            sr_path = Path(spatial['sr_checkpoint'])
            if not sr_path.is_absolute() or not sr_path.is_file():
                raise ValueError('Migrated SR checkpoint must be an existing absolute file')
            world[world.index('--sr-checkpoint')+1] = str(sr_path)
        combat_path = args.world_run_dir/'combat_conditioning_migration.json'
        combat = json.loads(combat_path.read_text(encoding='utf-8')) if combat_path.exists() else {}
        if combat.get('status') == 'complete' and combat.get('spatial_combat') is True:
            if int(combat.get('target_seconds', target_seconds)) != target_seconds:
                raise ValueError('Combat and spatial migration rollout durations differ')
            world.append('--spatial-combat')
    return {'codec': codec, 'sr': sr, 'world': world}


def validate_pipeline_codec(codec_path, ready_path, image_size):
    """Honor an explicit frozen-codec acceptance without claiming qualification.

    The ready JSON requests the exceptional path; the checkpoint's real frozen
    weights, integrity checks and acceptance metadata must independently pass
    the normal validator. A boolean in a sidecar alone grants no exception.
    """
    from .vq_world import validate_frozen_codec
    codec_path, ready_path = Path(codec_path).resolve(), Path(ready_path)
    ready = json.loads(ready_path.read_text(encoding='utf-8')) if ready_path.exists() else {}
    if not isinstance(ready, dict):
        raise ValueError('Malformed codec-ready metadata')
    acceptance = ready.get('quality_acceptance')
    requested = ready.get('accepted_by_user') is True
    if acceptance is not None:
        if not isinstance(acceptance, dict):
            raise ValueError('Malformed codec user acceptance')
        requested = requested or acceptance.get('accepted') is True
    options = {}
    if requested:
        if (ready.get('accepted_by_user') is not True or not isinstance(acceptance, dict)
                or acceptance.get('mode') != 'user_requested' or acceptance.get('accepted') is not True
                or not isinstance(ready.get('quality_gate_passed'), bool)
                or acceptance.get('quality_gate_passed') is not ready['quality_gate_passed']):
            raise ValueError('Codec-ready file does not contain consistent explicit user acceptance')
        options['accept_user_requested'] = True
    info = validate_frozen_codec(codec_path, expected_image_size=image_size, **options)
    if ready:
        checkpoint = ready.get('checkpoint')
        if (not isinstance(checkpoint, str) or Path(checkpoint).resolve() != codec_path
                or ready.get('codec_fingerprint') != info['codec_fingerprint']):
            raise ValueError('Codec-ready metadata belongs to a different frozen checkpoint')
    if requested and (info.get('accepted_by_user') is not True
            or info.get('quality_acceptance') != acceptance
            or info['quality_gate'].get('passed') is not ready['quality_gate_passed']):
        raise ValueError('Codec checkpoint does not confirm the recorded user acceptance')
    return info


class PipelineStopped(Exception):
    pass


class FreshPipeline:
    def __init__(self, args):
        self.args = args
        for name in ('run_dir', 'data_dir', 'codec_run_dir', 'sr_run_dir', 'world_run_dir', 'monitor_control_dir'):
            setattr(args, name, getattr(args, name).resolve())
        if len({args.run_dir, args.codec_run_dir, args.sr_run_dir, args.world_run_dir}) != 4:
            raise ValueError('Each worker and the controller require separate run directories')
        self.root = args.run_dir
        self.root.mkdir(parents=True, exist_ok=True)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        previous = self.root/'experiment_config.json'
        if not previous.exists() and any(any((root/'checkpoints').glob('*.pt'))
                for root in (args.codec_run_dir, args.sr_run_dir, args.world_run_dir)):
            raise ValueError('A fresh pipeline requires new codec, SR and world directories; existing weights are preserved')
        if previous.exists() and json.loads(previous.read_text(encoding='utf-8')) != config:
            raise ValueError('Fresh pipeline configuration changed; use a new run directory')
        atomic_json(previous, config)
        self.state = {'run_id': self.root.name, 'status': 'running', 'stage': 'data',
            'stage_label': 'Подготовка нового корпуса и этапов обучения с нуля',
            'current': 0, 'total': 128000, 'started_at': now(),
            'image_size': args.image_size, 'output_size': args.image_size*16,
            'evaluation': {'status': 'pending', 'target_seconds': 30, 'passed': False},
            'pipeline': {'codec_from_scratch': True, 'sr_from_scratch': True,
                'world_from_scratch': True, 'renderer_from_scratch': True, 'discriminator_from_scratch': True,
                'stage_order': ['codec', 'sr', 'world'], 'old_weights_reused': False,
                'native_size': args.image_size, 'sr_size': args.image_size*4,
                'bilinear_output_size': args.image_size*16}, 'completed_stages': []}

    def update(self, **fields):
        self.state.update(updated_at=now(), **fields)
        atomic_json(self.root/'status.json', self.state)

    def stopped(self, stage_root=None):
        return (self.root/'stop_training').exists() or (stage_root is not None and (stage_root/'stop_training').exists())

    def live_worker(self, root):
        path = root/'training.pid'
        return path.exists() and process_alive(int(path.read_text()))

    def publish(self, root, pid):
        record = {'run_dir': str(root), 'pid': pid, 'updated_at': now(),
                  'pipeline_parent': str(self.root), 'native_image_size': self.args.image_size}
        atomic_json(self.root/'active_run.json', record)
        self.args.monitor_control_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.args.monitor_control_dir/'active_run.json', record)

    def worker(self, name, root, command):
        if self.stopped(root):
            raise PipelineStopped()
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root/'pipeline_parent.json', {'run_dir': str(self.root), 'stage': name})
        atomic_json(self.root/f'{name}_command.json', command)
        pid_path = root/'training.pid'
        pid = int(pid_path.read_text()) if pid_path.exists() else 0
        process = None
        if not process_alive(pid):
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            with (root/'training.stdout.log').open('ab') as stdout, (root/'training.stderr.log').open('ab') as stderr:
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parent.parent,
                    stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, close_fds=True, env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            pid = process.pid
            atomic_json(pid_path, pid)
        self.publish(root, pid)
        self.update(stage=name, stage_label={'codec': 'VQGAN: обучение с нуля',
            'sr': 'Нейроапскейлер ×4: обучение на настоящих парах',
            'world': 'Динамика, Base, Flow и GAN: обучение с нуля'}[name], worker_pid=pid, worker_run_dir=str(root))
        while process_alive(pid):
            if self.stopped():
                (root/'stop_training').touch()
            time.sleep(2)
        code = process.wait() if process is not None else 0
        if self.stopped(root):
            raise PipelineStopped()
        if code:
            raise RuntimeError(f'{name} worker exited with code{code}; inspect {root}/training.stderr.log')
        stage_status = json.loads((root/'status.json').read_text(encoding='utf-8'))
        if stage_status.get('status') in ('stopped', 'failed'):
            if stage_status['status'] == 'stopped':
                raise PipelineStopped()
            raise RuntimeError(stage_status.get('error', f'{name} worker failed'))

    def run(self):
        args = self.args
        self.publish(self.root, os.getpid())
        while True:
            if self.stopped():
                raise PipelineStopped()
            path = args.data_dir/'status.json'
            data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            if data.get('status') == 'failed':
                raise RuntimeError('New data collection failed')
            self.update(data_preparation=data, current=data.get('groups', {}).get('rgb/train', {}).get('transitions', 0))
            if data.get('status') == 'complete':
                break
            time.sleep(2)
        subset = self.root/'training_subset_128k.json'
        selected = create_training_subset(args.data_dir/'rgb/train', subset, 128000, 173)
        self.update(training_subset={'manifest_path': str(subset), 'fingerprint': selected['fingerprint'],
            'train_transitions': 128000, 'train_clips': 4000, 'frozen': True})
        masks = self.root/'object_masks'
        commands = build_commands(args, subset, masks)
        codec_path = args.codec_run_dir/'checkpoints/codec_for_renderer.pt'
        if not codec_path.exists() or self.live_worker(args.codec_run_dir):
            self.worker('codec', args.codec_run_dir, commands['codec'])
        codec_info = validate_pipeline_codec(codec_path, args.codec_run_dir/'codec_ready.json', args.image_size)
        commands = build_commands(args, subset, masks, codec_info=codec_info)
        atomic_json(self.root/'codec_handoff.json', codec_info)
        self.update(codec_ready=codec_info, codec_validation=codec_info['quality_gate'],
            pipeline={**self.state['pipeline'], 'codec_fingerprint': codec_info['codec_fingerprint'],
                'codec_qualified': codec_info['qualified'],
                'codec_quality_accepted_by_user': codec_info.get('accepted_by_user', False)})
        self.state['completed_stages'] = ['codec']
        sr_path = args.sr_run_dir/'checkpoints/sr_for_game.pt'
        if not sr_path.exists() or self.live_worker(args.sr_run_dir):
            self.worker('sr', args.sr_run_dir, commands['sr'])
        import torch
        sr = torch.load(sr_path, map_location='cpu', weights_only=False)
        if (sr.get('stage') != 'sr' or sr.get('training_steps', 0) < args.sr_steps
                or sr.get('input_size') != args.image_size or sr.get('output_size') != args.image_size*4):
            raise ValueError('SR handoff is not the completed native x4 training stage')
        del sr
        self.state['completed_stages'] = ['codec', 'sr']
        if self.stopped():
            raise PipelineStopped()
        # Offline annotations run after the earlier GPU workers exit.
        self.publish(self.root, os.getpid())
        self.update(stage='object_masks', stage_label='Подготовка точных масок NPC и фоновых целей')
        from .object_mask_cache import prepare_object_masks
        prepare_object_masks(args.data_dir, masks, subset, workers=4, include_backgrounds=True)
        args.world_run_dir.mkdir(parents=True, exist_ok=True)
        link_data(args.world_run_dir/'data', args.data_dir)
        self.worker('world', args.world_run_dir, commands['world'])
        self.state['completed_stages'] = ['codec', 'sr', 'world']
        self.update(status='awaiting_user', stage='awaiting_user', stage_label='Бюджет нового учителя завершён',
            message='Веса сохранены; качество двухминутной игры ещё требует отдельной проверки')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--codec-run-dir', type=Path, required=True)
    p.add_argument('--sr-run-dir', type=Path, required=True)
    p.add_argument('--world-run-dir', type=Path, required=True)
    p.add_argument('--monitor-control-dir', type=Path, default=Path('training_runs/vqgan_20261109'))
    p.add_argument('--image-size', type=int, choices=(32, 64), default=32)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--gradient-accumulation-steps', type=int, default=21)
    p.add_argument('--codec-batch', type=int, default=32)
    p.add_argument('--sr-batch', type=int, default=8)
    p.add_argument('--sr-steps', type=int, default=2000)
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    if min(args.batch, args.gradient_accumulation_steps, args.codec_batch, args.sr_batch, args.sr_steps) < 1:
        p.error('Batch sizes, accumulation and training steps must be positive')
    existing_pid = args.run_dir/'training.pid'
    if (existing_pid.exists() and int(existing_pid.read_text()) != os.getpid()
            and process_alive(int(existing_pid.read_text()))):
        p.error('This pipeline controller is already running')
    pipeline = FreshPipeline(args)
    atomic_json(pipeline.root/'training.pid', os.getpid())
    try:
        with KeepAwake():
            pipeline.run()
    except PipelineStopped:
        pipeline.update(status='stopped', stage_label='Обучение остановлено с сохранением этапов')
    except Exception as error:
        pipeline.update(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
