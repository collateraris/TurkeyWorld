"""Freeze the latest stopped codec under an explicit user quality acceptance.

This does not train, choose the historical best model, alter the source resume
checkpoint, launch a world worker, or convert a failed quality gate into a pass.
Final reconstruction metrics come from the held-out panel; latent calibration
comes from a separately sampled bounded TRAIN subset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .power_guard import KeepAwake
from .training_pipeline import atomic_json, now, replace_file
from .visual_quality import strict_render_precision
from .vq_world import codec_fingerprint
from .vqgan import VQGAN, VQGANConfig
from .vqgan_data import VQGANData
from .vqgan_pipeline import process_alive
from .vqgan_validation import DEFAULT_CODEC_THRESHOLDS, evaluate_vqgan, save_vqgan_preview


ACCEPTANCE_REASON = 'User accepted the current codec quality and requested the next training stage'
INTEGRITY_CHECKS = ('all_frames_finite', 'all_frames_in_range', 'quantization_finite',
                    'valid_code_indices', 'codebook_broad_usage')


def _json(path, default=None):
    import json
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def finalize_current_vqgan(run_dir, data_dir, *, device='cuda', accept_current_quality=False,
                           validation_pairs=128, validation_batch=4, calibration_pairs=128,
                           train_initial_files=1024):
    """Return ready metadata for the exact latest stopped checkpoint weights."""
    if accept_current_quality is not True:
        raise ValueError('Explicit --accept-current-quality is required to freeze an unqualified codec')
    if min(validation_pairs, validation_batch, calibration_pairs, train_initial_files) < 1:
        raise ValueError('Validation/calibration counts and bounded train subset must be positive')
    root, data_dir = Path(run_dir).resolve(), Path(data_dir).resolve()
    pid_path = root/'training.pid'
    if pid_path.exists() and process_alive(int(pid_path.read_text(encoding='ascii').strip())):
        raise RuntimeError('Codec worker is still running; wait for its graceful checkpoint and exit')
    source_path = root/'checkpoints'/'vqgan_last.pt'
    saved = torch.load(source_path, map_location='cpu', weights_only=False)
    if saved.get('stage') != 'vqgan_codec' or int(saved.get('training_steps', 0)) < 1:
        raise ValueError('A trained latest vqgan_last.pt resume checkpoint is required')
    config = VQGANConfig(**saved['codec_config'])
    model = VQGAN(config).eval().requires_grad_(False)
    model.load_state_dict(saved['model'], strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value, saved['model'][name]):
            raise RuntimeError('Loading changed the source codec tensor: '+name)
    if not bool(model.quantize.initialized):
        raise ValueError('Latest codec codebook has not been initialized from training')
    before_fingerprint = codec_fingerprint(model)
    model = model.to(device)
    step = int(saved['training_steps'])
    original = _json(root/'experiment_config.json', {})
    seed = int(original.get('seed', 273))
    required = max(3, int(original.get('quality_passes', 3)))
    thresholds = {'psnr': original.get('quality_psnr', DEFAULT_CODEC_THRESHOLDS['psnr']),
        'ssim': original.get('quality_ssim', DEFAULT_CODEC_THRESHOLDS['ssim']),
        'mae': original.get('quality_mae', DEFAULT_CODEC_THRESHOLDS['mae']),
        'required_fraction': original.get('quality_fraction', DEFAULT_CODEC_THRESHOLDS['required_fraction'])}
    validation = VQGANData(data_dir/'rgb'/'val', device=device, split='val', image_size=config.image_size,
                          require_native_size=True, cache_bytes=128*1024**2, seed=seed)
    panel = _json(root/'validation_panel.json')
    if panel is None:
        panel = validation.fixed_panel(count=validation_pairs, seed=seed)
    report, preview = evaluate_vqgan(model, validation, panel, batch_size=validation_batch,
        device=torch.device(device), thresholds=thresholds, seed=seed, step=step)
    # This final check does not fabricate a new history of consecutive passes.
    aligned = bool(report['passed'] and saved.get('last_validation_fingerprint') == report['fingerprint'])
    consecutive = int(saved.get('quality_streak', 0)) if aligned else 0
    report.update(consecutive_passes=consecutive, required_passes=required, frozen=True,
                  user_requested_freeze=True, source_checkpoint=str(source_path))
    atomic_json(root/'codec_validation_final.json', report)
    atomic_json(root/'codec_validation.json', report)
    preview_every = int(original.get('validate_every', 500))
    preview.setdefault('metadata', {})['every'] = preview_every
    preview_metadata = save_vqgan_preview(preview, root/'preview.png', step=step)
    if not all(report.get('checks', {}).get(name) is True for name in INTEGRITY_CHECKS):
        raise ValueError('Current codec fails finite/range/quantization integrity; quality acceptance cannot bypass it')
    train = VQGANData(data_dir/'rgb'/'train', device=device, split='train', image_size=config.image_size,
        require_native_size=True, cache_bytes=128*1024**2, seed=seed,
        initial_file_limit=train_initial_files, growth_files=0)
    calibration_seed = seed+991
    calibration_rng = np.random.default_rng(calibration_seed)
    total, squares, count, indices = None, None, 0, []
    with strict_render_precision(), torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
        for first in range(0, calibration_pairs, validation_batch):
            batch = train.batch(min(validation_batch, calibration_pairs-first), rng=calibration_rng)
            indices.extend(batch['indices'].cpu().tolist())
            latent = model.encode(batch['frames'].flatten(0, 1))['quantized'].double()
            one, two = latent.sum((0, 2, 3)), latent.square().sum((0, 2, 3))
            total = one if total is None else total+one
            squares = two if squares is None else squares+two
            count += latent.shape[0]*latent.shape[2]*latent.shape[3]
    mean = total/count
    std = (squares/count-mean.square()).clamp_min(1e-6).sqrt()
    normalization = {'mean': mean.cpu().tolist(), 'std': std.cpu().tolist(),
        'source': 'independent_training_calibration', 'samples': count,
        'seed': calibration_seed, 'pair_indices': indices, 'initial_file_limit': train_initial_files}
    final_fingerprint = codec_fingerprint(model)
    if final_fingerprint != before_fingerprint:
        raise RuntimeError('Validation/calibration changed codec weights or buffers; refusing export')
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError('Training calibration produced nonfinite latent normalization')
    acceptance = {'mode': 'user_requested', 'accepted': True, 'reason': ACCEPTANCE_REASON,
        'accepted_at': now(), 'quality_gate_passed': bool(report['passed'])}
    output = root/'checkpoints'/'codec_for_renderer.pt'
    temporary = output.with_suffix('.pt.tmp')
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({'format_version': 1, 'stage': 'frozen_vqgan_codec', 'codec_config': model.get_config(),
        'model': state, 'training_steps': step, 'codec_fingerprint': final_fingerprint,
        'quality_gate': report, 'quality_acceptance': acceptance, 'latent_normalization': normalization,
        'training_objective': saved.get('training_objective'),
        'world_plan': saved.get('world_plan'),
        'generator_parameters': sum(value.numel() for value in model.parameters()),
        'source_checkpoint': str(source_path), 'pretrained_codec_weights_loaded': False,
        'created_at': now()}, temporary)
    replace_file(temporary, output)
    ready = {'checkpoint': str(output), 'codec_fingerprint': final_fingerprint, 'training_steps': step,
        'codec_config': model.get_config(), 'world_plan': saved.get('world_plan'),
        'quality_gate_passed': bool(report['passed']), 'quality_acceptance': acceptance,
        'accepted_by_user': True, 'consecutive_passes': consecutive, 'required_passes': required,
        'shared_teacher_and_student': True, 'created_at': now()}
    atomic_json(root/'codec_ready.json', ready)
    status = _json(root/'status.json', {})
    status.update(run_id=root.name, status='completed', stage='vqgan_ready_user_accepted',
        stage_label='VQGAN принят пользователем; переходим к следующему этапу обучения',
        current=step, updated_at=now(), codec_ready=ready, codec_validation=report,
        prediction_every=preview_every, prediction_preview={**preview_metadata, 'every': preview_every},
        message='Кодек заморожен по запросу пользователя. Фактическая проверка качества сохранена без подмены результата.')
    status.setdefault('codec_training', {}).update(frozen=True, accepted_by_user=True,
        quality_gate_passed=bool(report['passed']), consecutive_passes=consecutive,
        required_passes=required, current=step)
    atomic_json(root/'status.json', status)
    return ready


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--accept-current-quality', action='store_true', required=True)
    parser.add_argument('--validation-pairs', type=int, default=128)
    parser.add_argument('--validation-batch', type=int, default=4)
    parser.add_argument('--calibration-pairs', type=int, default=128)
    parser.add_argument('--train-initial-files', type=int, default=1024)
    args = parser.parse_args(argv)
    with KeepAwake():
        ready = finalize_current_vqgan(args.run_dir, args.data_dir, device=args.device,
            accept_current_quality=args.accept_current_quality, validation_pairs=args.validation_pairs,
            validation_batch=args.validation_batch, calibration_pairs=args.calibration_pairs,
            train_initial_files=args.train_initial_files)
    import json
    print(json.dumps(ready, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
