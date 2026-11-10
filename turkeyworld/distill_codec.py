"""Distill a <=1M VQGAN codec in the original fixed latent/codebook space.

Teacher Base, Source RGB and teacher-generated RGB supervise the encoder;
actual teacher Flow terminal embeddings additionally supervise the decoder.
The original teacher codec, embedding table and latent normalization do not
learn here. Quality is imitation of the teacher, never gameplay qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .distill_renderer import _batch, _digest, _save, _restore_random_state
from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint
from .object_losses import balanced_edge_l1, decoded_ssim_loss
from .power_guard import KeepAwake
from .training_pipeline import atomic_json, now, replace_file
from .visual_quality import image_quality_metrics, strict_render_precision


DOMAINS = ('source_rgb', 'base_rgb', 'teacher_rgb', 'flow_terminal')


def _freeze_codebook(student):
    student.quantize.eval().requires_grad_(False)
    return student


def _same_codebook(student, teacher):
    first, second = student.quantize.state_dict(), teacher.quantize.state_dict()
    return (first.keys() == second.keys() and all(torch.equal(value.detach().cpu(), second[name].detach().cpu())
                                                 for name, value in first.items()))


def prepare_codec_targets(teacher_flow, renderer_buffer, *, batch_size=16, cache_path=None, progress=None):
    """Build four explicit domain groups without changing original RF targets."""
    from .renderer_student import codec_tensor_fingerprint
    teacher = teacher_flow.codec
    device = next(teacher.parameters()).device
    identity = {'format_version': 1, 'renderer_buffer_identity': renderer_buffer['identity'],
        'teacher_codec_fingerprint': codec_tensor_fingerprint(teacher), 'domains': list(DOMAINS),
        'latent_normalization': {'mean': teacher_flow.latent_mean.flatten().cpu().tolist(),
                                'std': teacher_flow.latent_std.flatten().cpu().tolist()}}
    if cache_path is not None and Path(cache_path).exists():
        saved = torch.load(cache_path, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Codec target identity/teacher/RF source changed')
        return saved
    values = {name: [] for name in ('input_rgb', 'prequant', 'quantized', 'indices', 'teacher_rgb', 'domain', 'pose_index')}
    modes = [(module, module.training) for module in teacher.modules()]
    mean, std = teacher_flow.latent_mean.to(device), teacher_flow.latent_std.to(device)
    try:
        teacher.eval()
        with torch.inference_mode(), strict_render_precision(), torch.autocast(device.type, enabled=False):
            for first in range(0, len(renderer_buffer['state']), batch_size):
                batch = _batch(renderer_buffer, slice(first, first+batch_size), device)
                for domain, key in enumerate(('source_rgb', 'base', 'teacher_rgb', 'teacher_rgb')):
                    inputs = batch[key]
                    if domain < 3:
                        encoded = teacher.encode(inputs)
                        prequant, quantized, indices = encoded['prequant'], encoded['quantized'], encoded['indices']
                    else:
                        # These are the exact coordinates used by the full
                        # frozen Flow teacher, including pre-quantization scale.
                        encoded = teacher.quantize(batch['endpoint']*std+mean)
                        quantized, indices = encoded['quantized'], encoded['indices']
                        prequant = torch.zeros_like(quantized)
                    rgb = teacher.decode(quantized)
                    row = dict(input_rgb=inputs, prequant=prequant, quantized=quantized, indices=indices,
                        teacher_rgb=rgb, domain=torch.full((len(inputs),), domain, device=device, dtype=torch.long),
                        pose_index=torch.arange(first, first+len(inputs), device=device))
                    if not all(bool(torch.isfinite(value).all()) for value in row.values()):
                        raise FloatingPointError('Nonfinite teacher codec targets')
                    for name, value in row.items():
                        values[name].append(value.cpu())
                if progress:
                    progress(min(first+batch_size, len(renderer_buffer['state'])), len(renderer_buffer['state']))
    finally:
        for module, mode in modes:
            module.training = mode
    result = {'identity': identity, **{name: torch.cat(rows) for name, rows in values.items()}}
    if cache_path is not None:
        _save(cache_path, result)
    return result


def codec_distillation_loss(student, batch, latent_std):
    """Train E/D against teacher features and images, with a fixed quantizer."""
    _freeze_codebook(student)
    target_rgb, target_z = batch['teacher_rgb'].detach(), batch['quantized'].detach()
    std = latent_std.detach().to(target_z.device).reshape(1, -1, 1, 1)
    decoded = student.decode(target_z)
    decoder_l1 = F.l1_loss(decoded, target_rgb)
    decoder_ssim = decoded_ssim_loss(decoded, target_rgb)
    decoder_edge = balanced_edge_l1(decoded, target_rgb)
    selected = batch['domain'] < 3
    zero = decoded.sum()*0
    latent_loss = code_loss = encoder_l1 = encoder_ssim = encoder_edge = zero
    if bool(selected.any()):
        encoded = student.encode(batch['input_rgb'][selected].detach())
        latent_loss = F.mse_loss(encoded['prequant']/std, batch['prequant'][selected].detach()/std)
        code_loss = F.smooth_l1_loss(encoded['quantized']/std, target_z[selected]/std)
        reconstructed = student.decode(encoded['quantized'])
        encoder_l1 = F.l1_loss(reconstructed, target_rgb[selected])
        encoder_ssim = decoded_ssim_loss(reconstructed, target_rgb[selected])
        encoder_edge = balanced_edge_l1(reconstructed, target_rgb[selected])
    loss = (latent_loss+.25*code_loss+decoder_l1+.2*decoder_ssim+.1*decoder_edge
            +encoder_l1+.2*encoder_ssim+.1*encoder_edge)
    metrics = {name: float(value.detach()) for name, value in dict(latent=latent_loss, code=code_loss,
        decoder_l1=decoder_l1, decoder_ssim=decoder_ssim, decoder_edge=decoder_edge,
        encoder_l1=encoder_l1, encoder_ssim=encoder_ssim, encoder_edge=encoder_edge).items()}
    return loss, {**metrics, 'train_loss': float(loss.detach()), 'codebook_trainable': False}


def evaluate_codec_student(student, teacher, targets, *, batch_size=32, step=0, ssim=.95, mae=.015):
    if targets['identity']['renderer_buffer_identity']['split'] != 'val':
        raise ValueError('Compact codec qualification requires held-out validation poses')
    device = next(student.parameters()).device
    modes = [(module, module.training) for module in student.modules()]
    rows, previews = [], []
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        student.eval()
        with torch.inference_mode(), strict_render_precision(), torch.random.fork_rng(devices=devices), \
                torch.autocast(device.type, enabled=False):
            for first in range(0, len(targets['domain']), batch_size):
                batch = _batch(targets, slice(first, first+batch_size), device)
                decoded = student.decode(batch['quantized'])
                encoded = student.encode(batch['input_rgb'])
                reconstructed = student.decode(encoded['quantized'])
                for value in (decoded, reconstructed):
                    if not bool(torch.isfinite(value).all()) or bool((value < 0).any() or (value > 1).any()):
                        raise FloatingPointError('Compact codec returned invalid RGB')
                decoder_metrics = image_quality_metrics(decoded, batch['teacher_rgb'])
                encoder_metrics = image_quality_metrics(reconstructed, batch['teacher_rgb'])
                source_student = image_quality_metrics(reconstructed, batch['input_rgb'])
                source_teacher = image_quality_metrics(batch['teacher_rgb'], batch['input_rgb'])
                for index in range(len(decoded)):
                    row = {'domain': int(batch['domain'][index]), 'pose_index': int(batch['pose_index'][index]),
                        'index_agreement': float((encoded['indices'][index] == batch['indices'][index]).float().mean())}
                    for prefix, values in (('decoder', decoder_metrics), ('encoder', encoder_metrics),
                                           ('source_student', source_student), ('source_teacher', source_teacher)):
                        row.update({prefix+'_'+name: float(value[index]) for name, value in values.items()})
                    rows.append(row)
                    if len(previews) < 4:
                        previews.append(torch.stack([batch['input_rgb'][index], batch['teacher_rgb'][index],
                                                      decoded[index], reconstructed[index]]).cpu())
    finally:
        for module, mode in modes:
            module.training = mode
    encoder_rows = [row for row in rows if row['domain'] < 3]
    source_rows = [row for row in rows if row['domain'] == 0]
    summary = {prefix+'_'+name+'_mean': float(np.mean([row[prefix+'_'+name] for row in selected]))
        for prefix, selected in (('decoder', rows), ('encoder', encoder_rows)) for name in ('ssim', 'mae', 'psnr')}
    per_domain = {name: {metric: (float(np.mean([row['decoder_'+metric] for row in rows if row['domain'] == index]))
                                 if any(row['domain'] == index for row in rows) else None)
        for metric in ('ssim', 'mae', 'psnr')} for index, name in enumerate(DOMAINS)}
    for index, name in enumerate(DOMAINS):
        cohort = [row for row in rows if row['domain'] == index]
        per_domain[name].update(frames=len(cohort),
            decoder={metric+'_mean': (float(np.mean([row['decoder_'+metric] for row in cohort])) if cohort else None)
                     for metric in ('ssim', 'mae', 'psnr')},
            encoder={metric+'_mean': (float(np.mean([row['encoder_'+metric] for row in cohort]))
                                     if cohort and index < 3 else None) for metric in ('ssim', 'mae', 'psnr')},
            encoder_index_agreement=float(np.mean([row['index_agreement'] for row in cohort])) if cohort and index < 3 else None,
            encoder_applicable=index < 3)
    summary['encoder_index_agreement'] = float(np.mean([row['index_agreement'] for row in encoder_rows]))
    fixed = _same_codebook(student, teacher)
    passed = fixed and all(summary[prefix+'_ssim_mean'] >= ssim and summary[prefix+'_mae_mean'] <= mae
                           for prefix in ('decoder', 'encoder'))
    return {'step': step, 'passed': passed, 'summary': summary, 'per_domain': per_domain,
        'thresholds': {'ssim_mean': ssim, 'mae_mean': mae}, 'fixed_codebook': fixed,
        'frames': len(rows), 'encoder_frames': len(encoder_rows), 'source_held_out': True,
        'source_reference': {prefix+'_'+metric: float(np.mean([row[prefix+'_'+metric] for row in source_rows]))
            for prefix in ('source_teacher', 'source_student') for metric in ('ssim', 'mae', 'psnr')},
        'scope': 'Compact codec versus original teacher codec; no world or RF qualification',
        'renderer_panel_fingerprint': targets['identity']['renderer_buffer_identity']['fingerprint']}, previews


def export_codec_student(student, teacher_flow, path, *, training_steps, report):
    from .renderer_student import codec_tensor_fingerprint
    if training_steps < 1 or sum(p.numel() for p in student.parameters()) > 1_000_000:
        raise ValueError('Compact codec export requires completed training and <=1M total parameters')
    if not _same_codebook(student, teacher_flow.codec):
        raise ValueError('Compact codec must preserve the full original quantizer state exactly')
    if not all(bool(torch.isfinite(value).all()) for value in student.state_dict().values()):
        raise FloatingPointError('Nonfinite compact codec weights')
    payload = {'stage': 'distilled_vqgan_codec', 'format_version': 1,
        'codec_config': student.get_config(), 'model': student.state_dict(), 'training_steps': training_steps,
        'teacher_codec_fingerprint': codec_tensor_fingerprint(teacher_flow.codec),
        'latent_normalization': {'mean': teacher_flow.latent_mean.flatten().cpu().tolist(),
                                'std': teacher_flow.latent_std.flatten().cpu().tolist()},
        'quality_gate': {**report, 'frozen': True}, 'frozen': True, 'fixed_codebook': True,
        'parameters': sum(p.numel() for p in student.parameters()), 'source_codec_unchanged': True}
    _save(path, payload)
    return {key: value for key, value in payload.items() if key != 'model'}


def _preview(rows, path, step):
    size = 160
    canvas = Image.new('RGB', (len(rows)*size, 4*(size+22)), (12, 18, 25))
    draw = ImageDraw.Draw(canvas)
    for column, images in enumerate(rows):
        for row, (image, label) in enumerate(zip(images, ('INPUT RGB', 'TEACHER CODEC', 'COMPACT D(teacherZ)', 'COMPACT E+Q+D'))):
            pixels = image.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(pixels).resize((size, size), Image.Resampling.NEAREST),
                         (column*size, row*(size+22)+22))
            draw.text((column*size+2, row*(size+22)+4), f'{label} | {step}', fill='white')
    temporary = Path(str(path)+'.tmp.png'); canvas.save(temporary); replace_file(temporary, Path(path))


class CodecDistillation:
    def __init__(self, args):
        self.args, self.root = args, Path(args.run_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.status = {'run_id': self.root.name, 'status': 'running', 'stage': 'codec_distillation',
            'stage_label': 'Distilling a compact fixed-codebook codec', 'current': 0, 'total': args.steps,
            'model': 'VQGAN codec student <=1M', 'image_size': 32, 'prediction_every': args.preview_every,
            'validation_every': args.validate_every, 'loss_protocol': 'fixed_codebook_codec_distillation_v1',
            'world_training': False, 'flow_training': False}

    def publish(self, **values):
        self.status.update(values, updated_at=now(), elapsed_seconds=time.perf_counter()-self.started)
        atomic_json(self.root/'status.json', self.status)

    def run(self):
        from .compact_codec import build_compact_codec
        args, device = self.args, torch.device(self.args.device)
        torch.set_num_threads(int(os.environ.get('TW_TORCH_THREADS', '4')))
        if device.type == 'cuda': torch.cuda.init()
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        world, metadata = load_world_checkpoint(args.teacher_world, 'cpu')
        if int(metadata.get('training_steps', metadata.get('global_training_step', 0))) < 1:
            raise ValueError('Compact codec distillation requires an actually trained teacher world')
        teacher_flow = world.flow_renderer
        if not bool(teacher_flow.codec_loaded):
            raise ValueError('Teacher world has no loaded codec')
        teacher_flow.codec.to(device).requires_grad_(False).eval()
        student = build_compact_codec(teacher_flow.codec, seed=args.seed).to(device)
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
        fingerprint = _digest(args.teacher_world)
        targets = {}
        for split in ('train', 'val'):
            buffer = torch.load(Path(args.renderer_data_buffers)/f'teacher_{split}_buffer.pt', map_location='cpu', weights_only=False)
            if buffer['identity']['split'] != split or buffer['identity']['teacher_fingerprint'] != fingerprint:
                raise ValueError('Renderer teacher target buffer belongs to another teacher or split')
            targets[split] = prepare_codec_targets(teacher_flow, buffer, batch_size=args.batch,
                cache_path=self.root/f'codec_{split}_targets.pt',
                progress=lambda done, total: self.publish(stage='codec_targets', data_preparation={
                    'split': split, 'completed': done, 'total': total}))
        train, validation = targets['train'], targets['val']
        source_seeds = lambda data: {row['seed'] for row in data['identity']['renderer_buffer_identity']['frames']}
        if source_seeds(train) & source_seeds(validation):
            raise ValueError('Compact codec train/validation source seeds overlap')
        configuration = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
        identity = {split: target['identity'] for split, target in targets.items()}
        self.publish(model_parameters={'student_total': sum(p.numel() for p in student.parameters()),
            'student_trainable': sum(p.numel() for p in student.parameters() if p.requires_grad),
            'teacher_total': sum(p.numel() for p in teacher_flow.codec.parameters())},
            fixed_codebook=True, data={'train_examples': len(train['domain']), 'val_examples': len(validation['domain'])})
        checkpoint = self.root/'checkpoints/codec_student_last.pt'
        step, best, streak = 0, None, 0
        initial_budget, budget_history = args.steps, []
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            previous_configuration = saved['configuration']
            previous_budget = previous_configuration['steps']
            if ({key: value for key, value in previous_configuration.items() if key != 'steps'} !=
                    {key: value for key, value in configuration.items() if key != 'steps'}
                    or args.steps < previous_budget or saved['identity'] != identity):
                raise ValueError('Compact codec resume configuration/teacher/targets changed')
            student.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            step, best, streak = saved['step'], saved['best'], saved['streak']
            initial_budget = saved.get('initial_budget', previous_budget)
            budget_history = saved.get('budget_history', [])
            if args.steps > previous_budget:
                budget_history.append({'from_steps': previous_budget, 'to_steps': args.steps,
                    'at_optimizer_step': step, 'requested_at': now()})
            _restore_random_state(saved, device)
        def save():
            if not _same_codebook(student, teacher_flow.codec): raise ValueError('Fixed quantizer state changed')
            _save(checkpoint, {'model': student.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'best': best, 'streak': streak, 'configuration': configuration, 'identity': identity,
                'initial_budget': initial_budget, 'budget_history': budget_history,
                'rng_torch': torch.random.get_rng_state(), 'rng_numpy': np.random.get_state(),
                'rng_python': random.getstate(), 'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})
        def validate(full):
            panel = validation if full else {**validation, **{name: value[:4] for name, value in validation.items() if torch.is_tensor(value)}}
            report, preview = evaluate_codec_student(student, teacher_flow.codec, panel, batch_size=args.batch,
                step=step, ssim=args.target_ssim, mae=args.target_mae)
            _preview(preview, self.root/'preview.png', step)
            atomic_json(self.root/'preview.json', {'step': step, 'every': args.preview_every,
                'scope': report['scope'], 'quality_gate_evaluated': full})
            return report
        self.publish(initial_budget=initial_budget, budget_history=budget_history,
            resumed_from_optimizer_step=step, global_training_step=step, current=step)
        if step == 0: atomic_json(self.root/'validation_initial.json', validate(True))
        started, first_step = time.perf_counter(), step
        std = teacher_flow.latent_std.to(device)
        for step in range(step+1, args.steps+1):
            _freeze_codebook(student.train())
            indices = np.random.randint(len(train['domain']), size=args.batch)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = codec_distillation_loss(student, _batch(train, indices, device), std)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite codec distillation objective')
            loss.backward()
            metrics['gradient_norm'] = float(clip_finite_grad_norm_(student.named_parameters(), context=f'Codec update{step}'))
            optimizer.step()
            report = None
            if step % args.validate_every == 0 or step == args.steps:
                report = validate(True)
                atomic_json(self.root/'validation.json', report)
                score = sum(report['summary'][prefix+'_mae_mean']+.5*(1-report['summary'][prefix+'_ssim_mean'])
                            for prefix in ('encoder', 'decoder'))
                if best is None or (not report['passed'], score) < (not best['report']['passed'], best['score']):
                    best = {'step': step, 'score': score, 'report': report}
                    _save(self.root/'checkpoints/codec_student_best.pt', {'model': student.state_dict(), **best})
                streak = streak+1 if report['passed'] else 0
            elif step % args.preview_every == 0:
                validate(False)
            with (self.root/'metrics.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps({'timestamp': now(), 'step': step, 'global_training_step': step,
                    'stage': 'codec_distillation', **metrics})+'\n')
            self.publish(stage='codec_distillation', current=step, global_training_step=step,
                loss=metrics['train_loss'], metrics=metrics, validation=report if report else self.status.get('validation'),
                best=best, progress=step/args.steps, eta_seconds=(time.perf_counter()-started)/(step-first_step)*(args.steps-step))
            if report is not None or step % args.save_every == 0: save()
            if streak >= args.required_passes: break
        save()
        selected = torch.load(self.root/'checkpoints/codec_student_best.pt', map_location=device, weights_only=False)
        student.load_state_dict(selected['model'])
        report, preview = evaluate_codec_student(student, teacher_flow.codec, validation,
            batch_size=args.batch, step=selected['step'], ssim=args.target_ssim, mae=args.target_mae)
        export = export_codec_student(student, teacher_flow, self.root/'checkpoints/codec_student.pt',
            training_steps=selected['step'], report=report)
        atomic_json(self.root/'validation_selected.json', report); _preview(preview, self.root/'preview.png', selected['step'])
        self.publish(status='completed', stage='codec_distilled', current=step, selected_step=selected['step'],
            quality_gate_passed=report['passed'], validation=report, export=export, eta_seconds=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-world', type=Path, required=True)
    parser.add_argument('--renderer-data-buffers', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    for name, default in [('steps', 2000), ('batch', 32), ('validate-every', 100), ('preview-every', 10),
                          ('save-every', 10), ('required-passes', 2), ('seed', 173)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--target-ssim', type=float, default=.95)
    parser.add_argument('--target-mae', type=float, default=.015)
    args = parser.parse_args(argv)
    if min(args.steps, args.batch, args.validate_every, args.preview_every, args.save_every, args.required_passes) < 1 or args.lr <= 0:
        parser.error('Counts, cadence and learning rate must be positive')
    experiment = CodecDistillation(args)
    try:
        with KeepAwake(): experiment.run()
    except BaseException as error:
        experiment.publish(status='failed', stage='error', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__': main()
