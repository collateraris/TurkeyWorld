"""Source one-enemy E/D adaptation of the existing compact VQGAN (no GAN)."""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from .codec_enemy_objectives import codec_enemy_loss
from .compact_codec import CompactCodec, compact_codec_parameter_counts
from .distill_codec import export_codec_student
from .distill_renderer import _digest, _save
from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_enemy_data import EnemyFrames
from .renderer_enemy_objectives import enemy_conditions
from .renderer_enemy_validation import renderer_metrics
from .renderer_refinement import refinement_forward
from .renderer_student import _hash_tensors, load_renderer_student
from .staged_validation import _isolated_validation
from .training_pipeline import atomic_json, now


def configure_codec_phase(student, decoder_only):
    """Keep a fixed Q in both phases; Adam retains both E/D parameter groups."""
    student.train().requires_grad_(True)
    student.encoder.requires_grad_(not decoder_only)
    student.quant_conv.requires_grad_(not decoder_only)
    student.quantize.eval().requires_grad_(False)
    return {'phase': 'decoder_warmup' if decoder_only else 'encoder_decoder',
            'encoder_trainable': not decoder_only, 'decoder_trainable': True,
            'quantizer_trainable': False, 'gan_active': False}


@torch.no_grad()
def prepare_references(working, original_codec, batch):
    """Targets are labels only; causal dynamic features never see next RGB."""
    conditions = enemy_conditions(working, batch)
    base, old_rgb, endpoint = refinement_forward(working, conditions)
    old = working.flow_renderer.codec
    source_z = original_codec.encode(conditions['target'])
    old_base = old.encode(base)
    rf_z = old.quantize(working.flow_renderer.denormalize_latents(endpoint))['quantized']
    references = {'original_prequant': source_z['prequant'],
        'original_quantized': source_z['quantized'], 'base_rgb': base,
        'old_base_prequant': old_base['prequant'], 'rf_quantized': rf_z,
        'old_rf_rgb': old_rgb, 'original_rf_rgb': original_codec.decode(rf_z)}
    return conditions, {key: value.detach() for key, value in references.items()}


def codec_assessment(report, baseline=None):
    """Source goal and unchanged-renderer regressions are separate evidence."""
    reconstruction = report['source_reconstruction']
    visible = reconstruction['visible_enemy']
    source_passed = (reconstruction['overall']['ssim_mean'] >= .90
        and reconstruction['overall']['mae_mean'] <= .03
        and visible.get('foreground_mae') is not None and visible['foreground_mae'] <= .08
        and visible.get('closer_to_source_than_without_npc_rate', 0.) >= .75)
    failures = []
    if baseline is not None:
        for prediction in ('source_reconstruction', 'full_rf1'):
            for category in ('empty_navigation', 'empty_hidden'):
                current = report[prediction]['categories'].get(category, {})
                old = baseline[prediction]['categories'].get(category, {})
                if (not current.get('frames') or not old.get('frames')
                        or current['mae_mean'] > old['mae_mean']+.005
                        or current['ssim_mean'] < old['ssim_mean']-.02):
                    failures.append(prediction+'_'+category+'_regression')
        if report['full_rf1']['overall']['mae_mean'] > baseline['full_rf1']['overall']['mae_mean']+.01:
            failures.append('full_rf1_source_mae_regression')
    return {'passed': bool(source_passed and not failures), 'source_quality_passed': bool(source_passed),
        'rf_and_empty_guards_passed': not failures, 'failure_reasons': failures,
        'thresholds': {'source_ssim_mean': .90, 'source_mae_mean': .03,
            'npc_foreground_mae': .08, 'npc_presence_rate': .75,
            'empty_mae_regression_tolerance': .005, 'empty_ssim_regression_tolerance': .02,
            'full_rf1_mae_regression_tolerance': .01},
        'scope': 'Held-out Source reconstruction and fixed-parent RF compatibility; not gameplay qualification'}


def evaluate_codec_enemy(student, working, original_codec, data, panel, *, step=0, batch_size=8, baseline=None):
    """Measure Source E/D, original-code D and both old/new-condition RF paths."""
    candidate = copy.deepcopy(student).eval().requires_grad_(False)
    old_codec = working.flow_renderer.codec
    parts = {key: [] for key in ('source', 'mask', 'background', 'dynamic',
        'player_hp0', 'states',
        'source_reconstruction', 'original_codes_decoder', 'fixed_old_rf_endpoint',
        'full_rf1', 'old_full_rf1', 'old_source_reconstruction', 'original_source_reconstruction')}
    agreement, pixels = 0, 0
    try:
        with _isolated_validation(working, data, 173):
            for first in range(0, len(panel), batch_size):
                ids = panel[first:first+batch_size]
                batch = data.batch(len(ids), indices=ids)
                conditions, refs = prepare_references(working, original_codec, batch)
                encoded = candidate.encode(conditions['target'])
                old_base = old_codec.encode(refs['base_rgb'])
                new_base = candidate.encode(refs['base_rgb'])
                agreement += int((old_base['indices'] == new_base['indices']).sum())
                pixels += old_base['indices'].numel()
                values = {'source': conditions['target'], 'mask': conditions['object_mask'],
                    'background': conditions['background'], 'dynamic': conditions['dynamic'],
                    'player_hp0': conditions['state'][:, 4] <= 0,
                    'states': conditions['state'],
                    'source_reconstruction': candidate.decode(encoded['quantized']),
                    'original_codes_decoder': candidate.decode(refs['original_quantized']),
                    'fixed_old_rf_endpoint': candidate.decode(refs['rf_quantized']),
                    'old_full_rf1': refs['old_rf_rgb'],
                    'old_source_reconstruction': old_codec.decode(old_codec.encode(conditions['target'])['quantized']),
                    'original_source_reconstruction': original_codec.decode(refs['original_quantized'])}
                # This private eval clone cannot freeze the actual E/D student.
                working.flow_renderer.codec = candidate
                try:
                    _, values['full_rf1'], _ = refinement_forward(working, conditions)
                finally:
                    working.flow_renderer.codec = old_codec
                if any(not bool(torch.isfinite(value).all()) for value in values.values()):
                    raise FloatingPointError('Nonfinite codec enemy held-out reconstruction')
                for key, value in values.items():
                    parts[key].append(value.detach().cpu())
    finally:
        working.flow_renderer.codec = old_codec
        del candidate
    arrays = {key: torch.cat(value) for key, value in parts.items()}
    categories = [data.rows[int(index)]['category'] for index in panel]
    arrays['categories'] = categories
    report = {'step': step, 'frames': len(panel), 'source_held_out': True,
        'panel_fingerprint': data.panel_fingerprint(panel), 'all_finite': True,
        'static_feature_policy': 'snapshot_zero128_no_target_encoding',
        'dynamic_feature_policy': 'frozen_prior_previous_Source_RGB_state_action_only',
        'base_encoder_index_agreement_with_parent': agreement/max(pixels, 1),
        'quantizer_frozen': True, 'latent_normalization_unchanged': True,
        'gan_active': False, 'automatic_deployment': False, 'gameplay_qualified': False}
    for name in parts:
        if name not in ('source', 'mask', 'background', 'dynamic', 'player_hp0', 'states'):
            report[name] = renderer_metrics(arrays[name], arrays['source'], arrays['mask'],
                arrays['background'], categories, arrays['dynamic'])
    report['player_health_cohorts'] = {}
    for cohort, selected in (('hp0', arrays['player_hp0']), ('alive', ~arrays['player_hp0'])):
        report['player_health_cohorts'][cohort] = {}
        selected_categories = [name for name, flag in zip(categories, selected.tolist()) if flag]
        for name in ('source_reconstruction', 'original_codes_decoder', 'full_rf1', 'fixed_old_rf_endpoint'):
            report['player_health_cohorts'][cohort][name] = renderer_metrics(arrays[name][selected],
                arrays['source'][selected], arrays['mask'][selected], arrays['background'][selected],
                selected_categories, arrays['dynamic'][selected])
    report.update(codec_assessment(report, baseline))
    report['summary'] = report['source_reconstruction']['overall']
    report['selection_score'] = (report['source_reconstruction']['visible_enemy']['foreground_mae']
        +.5*report['original_codes_decoder']['visible_enemy']['foreground_mae']
        +.25*report['source_reconstruction']['overall']['background_mae']
        +2*max(0., report['full_rf1']['overall']['mae_mean']-report['old_full_rf1']['overall']['mae_mean']))
    return report, arrays


def attach_terminal_assessment(report, terminal, *, baseline=None):
    """Keep original main quality thresholds, with a separate real-HP0 goal."""
    report['terminal_supplement'] = terminal
    reconstruction = terminal['source_reconstruction']['overall']
    reasons = []
    if baseline is not None:
        for prediction, tolerance in (('source_reconstruction', .005), ('full_rf1', .01)):
            current = report['player_health_cohorts']['alive'][prediction]['overall']
            old = baseline['player_health_cohorts']['alive'][prediction]['overall']
            if (not current.get('frames') or current['mae_mean'] > old['mae_mean']+tolerance
                    or current['ssim_mean'] < old['ssim_mean']-.02): reasons.append(prediction+'_alive_regression')
        for prediction in ('source_reconstruction', 'full_rf1'):
            current = report[prediction]['visible_enemy']; old = baseline[prediction]['visible_enemy']
            if (current['foreground_mae'] > old['foreground_mae']+.005
                    or current['closer_to_source_than_without_npc_rate']
                    < old['closer_to_source_than_without_npc_rate']-.05):
                reasons.append(prediction+'_npc_regression')
            current_alive = report['player_health_cohorts']['alive'][prediction]['visible_enemy']
            old_alive = baseline['player_health_cohorts']['alive'][prediction]['visible_enemy']
            if (current_alive['foreground_mae'] > old_alive['foreground_mae']+.005
                    or current_alive['closer_to_source_than_without_npc_rate']
                    < old_alive['closer_to_source_than_without_npc_rate']-.05):
                reasons.append(prediction+'_alive_npc_regression')
        old_hp0 = baseline['terminal_supplement']['source_reconstruction']['overall']['mae_mean']
        if reconstruction['mae_mean'] >= old_hp0-.005: reasons.append('hp0_not_improved')
    terminal_passed = reconstruction['frames'] >= 18 and reconstruction['mae_mean'] <= .04
    report['terminal_focus'] = {'passed': terminal_passed and not reasons,
        'frames': reconstruction['frames'], 'mae_mean': reconstruction['mae_mean'],
        'ssim_mean': reconstruction['ssim_mean'], 'failure_reasons': reasons,
        'thresholds': {'hp0_frames_min': 18, 'hp0_mae_max': .04, 'improvement_min': .005,
            'alive_source_mae_regression_max': .005, 'alive_rf_mae_regression_max': .01},
        'scope': 'Additional held-out Source HP0 tint; not full Source/gameplay qualification'}
    report['main_source_quality_passed'] = report['passed']
    report['rf_and_empty_guards_passed'] = report['rf_and_empty_guards_passed'] and not any('regression' in name for name in reasons)
    report['passed'] = bool(report['main_source_quality_passed'] and report['terminal_focus']['passed'])
    report['selection_score'] = (reconstruction['mae_mean']
        +.5*report['source_reconstruction']['overall']['mae_mean']
        +.25*report['source_reconstruction']['visible_enemy']['foreground_mae'])
    return report


def save_preview(arrays, path, step):
    chosen = []
    categories=('tiny','near','corpse','navigation')if'tiny'in arrays['categories']else('large','partial','corpse','empty_hidden')
    for category in categories:
        found = [index for index, name in enumerate(arrays['categories']) if name == category]
        if found: chosen.append(found[0])
    if not chosen: chosen = list(range(min(4, len(arrays['source']))))
    rows = [('source', 'SOURCE NPC (GT pose)'), ('old_source_reconstruction', 'PARENT COMPACT CODEC'),
        ('source_reconstruction', 'TRAINED COMPACT E+Q+D'), ('original_codes_decoder', 'NEW D / ORIGINAL E CODES'),
        ('full_rf1', 'FIXED RF1 + NEW E/D')]
    size = 144
    canvas = Image.new('RGB', (len(chosen)*size, len(rows)*(size+22)), (12, 18, 25))
    draw = ImageDraw.Draw(canvas)
    for row, (key, label) in enumerate(rows):
        for column, index in enumerate(chosen):
            image = arrays[key][index].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(image).resize((size, size), Image.Resampling.NEAREST),
                (column*size, row*(size+22)+22))
            draw.text((column*size+2, row*(size+22)+3), f'{label} g{step}', fill='white')
    path = Path(path); temporary = path.with_name(path.name+'.tmp.png')
    canvas.save(temporary); temporary.replace(path)


class CodecEnemyTraining:
    def __init__(self, args):
        self.args, self.root = args, Path(args.run_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.status = {'run_id': self.root.name, 'status': 'initializing', 'stage': 'codec_enemy_finetune',
            'model': 'Compact VQGAN E/D 997539; frozen 1488-code book; GAN off', 'image_size': 32,
            'current': 0, 'global_training_step': 0, 'total': args.steps,
            'prediction_every': args.preview_every, 'validation_every': args.validate_every,
            'loss_protocol': 'source_enemy_codec_ed_v1', 'started_at': now(),
            'gameplay_qualified': False, 'active_renderer_changed': False}

    def publish(self, **values):
        self.status.update(values, updated_at=now(), elapsed_seconds=time.perf_counter()-self.started)
        atomic_json(self.root/'status.json', self.status)

    def run(self):
        args = self.args; device = torch.device(args.device)
        if device.type == 'cuda': torch.cuda.init()
        torch.set_num_threads(args.threads)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        self.publish(stage_label='Loading immutable world and selected parent RF1')
        source, meta = load_world_checkpoint(args.world, 'cpu')
        source_step = meta['training_steps']; del meta
        source.eval().requires_grad_(False)
        working, meta = load_world_checkpoint(args.world, 'cpu'); del meta
        parent = load_renderer_student(working, args.renderer_checkpoint, allow_unqualified_codec=True)
        working.float().to(device).eval().requires_grad_(False)
        original = source.flow_renderer.codec.float().to(device).eval().requires_grad_(False)
        old = working.flow_renderer.codec
        student = CompactCodec(old.config).to(device)
        student.load_state_dict(old.state_dict(), strict=True)
        configure_codec_phase(student, True)
        if sum(p.numel() for p in student.parameters()) != 997539:
            raise ValueError('Expected the existing 997539-parameter compact codec')
        e_names = ('encoder.', 'quant_conv.')
        encoder = [(name, value) for name, value in student.named_parameters() if name.startswith(e_names)]
        decoder = [(name, value) for name, value in student.named_parameters() if name.startswith(('decoder.', 'post_quant_conv.'))]
        optimizer = torch.optim.AdamW([{'params': [p for _, p in encoder], 'lr': args.encoder_lr},
            {'params': [p for _, p in decoder], 'lr': args.decoder_lr}], weight_decay=.01)
        fixed = {'working': _hash_tensors(working.state_dict()), 'original': _hash_tensors(original.state_dict()),
            'quantizer': _hash_tensors(student.quantize.state_dict()),
            'latent_mean': working.flow_renderer.latent_mean.flatten().cpu().tolist(),
            'latent_std': working.flow_renderer.latent_std.flatten().cpu().tolist()}
        parent_gate = copy.deepcopy(getattr(working.flow_renderer, '_renderer_codec_distillation', {}).get('quality_gate'))
        self.publish(stage_label='Indexing unchanged Source enemy and empty held-out panels', source_world_step=source_step,
            model_parameters=compact_codec_parameter_counts(student), gan_active=False, quantizer_frozen=True,
            world_frozen=True, rf_frozen=True, base_frozen=True, map_frozen=True, sr_frozen=True,
            decoder_warmup_steps=args.decoder_warmup_steps)
        terminal_val = terminal_panel = None
        small_mode=bool(getattr(args,'small_data_root',None))
        if small_mode:
            from .small_enemy_data import SmallEnemyData
            train=SmallEnemyData(args.small_data_root,'train',seed=args.seed)
            val=SmallEnemyData(args.small_data_root,'val',seed=args.seed+1,exclude_scene_keys=train.scene_keys)
            panel=val.fixed_panel(args.validation_frames,seed=args.seed)
            self.publish(loss_protocol='source_tiny_codec_fixedQ_v1',small_enemy_data=train.identity,
                scope='Source native32 NPC1..16pixels;25%navigation/near/corpse/HP0 replay')
        elif getattr(args, 'combat_data_root', None) is not None:
            from .codec_battle_data import CodecBattleFrames
            from .renderer_combat_data import CombatRendererFrames
            train = CodecBattleFrames(args.combat_data_root, args.appearance_data_root, 'train', seed=args.seed)
            val = CombatRendererFrames(args.combat_data_root, args.appearance_data_root, 'val',
                seed=args.seed+1, exclude_scene_keys=train.scene_keys)
            terminal_val = CodecBattleFrames(args.combat_data_root, args.appearance_data_root, 'val',
                seed=args.seed+1, exclude_render_keys=train.render_keys)
            terminal_panel = terminal_val.terminal_indices
            if len(terminal_panel) < 18: raise ValueError('Require all18 distinct genuine held-out HP0 frames')
            parent_identity = json.loads(args.validation_identity.read_text(encoding='utf-8'))
            panel = parent_identity['panel']
            if val.panel_fingerprint(panel) != parent_identity['panel_fingerprint']:
                raise ValueError('Original256-frame Source panel changed')
            self.publish(loss_protocol='source_battle_terminal_codec_ed_v2',
                terminal_training_frames=len(train.terminal_indices), terminal_validation_frames=len(terminal_panel),
                terminal_supplement_fingerprint=terminal_val.panel_fingerprint(terminal_panel))
        else:
            train = EnemyFrames(args.data_root, args.empty_data_root, 'train', seed=args.seed)
            val = EnemyFrames(args.data_root, args.empty_data_root, 'val', seed=args.seed+1, exclude_scene_keys=train.scene_keys)
            panel = val.fixed_panel(args.validation_frames, seed=args.seed)
        if {item['seed'] for item in train.files} & {item['seed'] for item in val.files}:
            raise ValueError('Source train/held-out seeds overlap')
        identity = {'world_sha256': _digest(args.world), 'renderer_sha256': _digest(args.renderer_checkpoint),
            'train': train.identity, 'val': val.identity, 'panel': panel,
            'panel_fingerprint': val.panel_fingerprint(panel), 'parent_codec_sha256': _hash_tensors(old.state_dict())}
        if terminal_val is not None:
            identity.update(terminal_val=terminal_val.identity, terminal_panel=terminal_panel,
                terminal_fingerprint=terminal_val.panel_fingerprint(terminal_panel))
        config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config = {key: value for key,value in config.items() if value is not None}
        for key in ('object_crop_weight',):
            if config.get(key)==0:config.pop(key)
        config_path = self.root/'experiment_config.json'
        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding='utf-8'))
            if ({k: v for k, v in previous.items() if k != 'steps'} != {k: v for k, v in config.items() if k != 'steps'}
                    or args.steps < previous['steps']): raise ValueError('Codec enemy immutable protocol changed')
        atomic_json(config_path, config); atomic_json(self.root/'source_identity.json', identity)
        checkpoint, best_path = self.root/'checkpoints/codec_enemy_last.pt', self.root/'checkpoints/codec_enemy_best.pt'
        step, best, streak = 0, None, 0
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if (saved['identity'] != identity or saved['fixed'] != fixed
                    or {k: v for k, v in saved['config'].items() if k != 'steps'} != {k: v for k, v in config.items() if k != 'steps'}
                    or args.steps < saved['config']['steps']): raise ValueError('Codec enemy resume identity changed')
            student.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
            step, best, streak = saved['step'], saved['best'], saved['streak']
            train.restore_sampler(saved['sampler'])
            torch.set_rng_state(saved['rng_torch'].cpu()); random.setstate(saved['rng_python']); np.random.set_state(saved['rng_numpy'])
            if device.type == 'cuda': torch.cuda.set_rng_state_all([v.cpu() for v in saved['rng_cuda']])
            del saved
        configure_codec_phase(student, step < args.decoder_warmup_steps)
        def save():
            _save(checkpoint, {'model': student.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'best': best, 'streak': streak, 'identity': identity, 'fixed': fixed, 'config': config,
                'sampler': train.sampler_state(), 'rng_torch': torch.get_rng_state(),
                'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})
        def verify():
            if (_hash_tensors(working.state_dict()) != fixed['working']
                    or _hash_tensors(original.state_dict()) != fixed['original']
                    or _hash_tensors(student.quantize.state_dict()) != fixed['quantizer']
                    or student.quantize.training or any(p.requires_grad for p in student.quantize.parameters())):
                raise AssertionError('Frozen world/Base/RF/original codec/quantizer changed')
            self.publish(frozen_verification={'step': step, 'all_quantizer_parameters_buffers_bitexact': True,
                'world_base_rf_map_original_codec_bitexact': True, 'latent_normalization_bitexact': True,
                'quantizer_sha256': fixed['quantizer']})
        def evaluate(step, baseline=None):
            report, arrays = evaluate_codec_enemy(student, working, original, val, panel, step=step,
                batch_size=args.validation_batch, baseline=baseline)
            if small_mode:
                from .small_enemy_validation import small_enemy_cohorts,relative_small_enemy_gate
                names=('source_reconstruction','original_codes_decoder','full_rf1')
                report['small_enemy_cohorts']={name:small_enemy_cohorts(arrays[name],arrays['source'],
                    arrays['mask'],arrays['states'])for name in names}
                report['legacy_missing_cohort_assessment']={'failure_reasons':report['failure_reasons'],
                    'rf_and_empty_guards_passed':report['rf_and_empty_guards_passed']}
                # This pool intentionally replaces hidden-empty examples with
                # explicit navigation,near,corpse,HP0 replay; validate these
                # named guards rather than pretending missing cohorts passed.
                if baseline is not None:
                    gate=relative_small_enemy_gate(report,baseline,predictions=('source_reconstruction',),compatibility=('full_rf1',))
                    report['small_enemy_stage_gate']=gate
                    report['rf_and_empty_guards_passed']=not any('regression'in reason or'missing'in reason
                        or'nonfinite'in reason or'identity'in reason for reason in gate['failure_reasons'])
                    report['failure_reasons']=list(gate['failure_reasons'])
                else:
                    report['small_enemy_stage_gate']={'passed':False,'failure_reasons':['initial_reference_only']}
                    report['rf_and_empty_guards_passed']=True
                    report['failure_reasons']=[]
                report['passed']=bool(report['source_quality_passed']and report['rf_and_empty_guards_passed'])
                report['selection_score']=report['small_enemy_cohorts']['source_reconstruction']['tiny']['foreground_mae']
            if terminal_val is not None:
                terminal, terminal_arrays = evaluate_codec_enemy(student, working, original, terminal_val,
                    terminal_panel, step=step, batch_size=args.validation_batch)
                main_ids = {(val.files[val.rows[i]['file']]['path'], val.rows[i]['frame']) for i in panel}
                supplement_ids = {(terminal_val.files[terminal_val.rows[i]['file']]['path'], terminal_val.rows[i]['frame']) for i in terminal_panel}
                terminal.update(main_panel_overlap_frames=len(main_ids & supplement_ids),
                    combined_distinct_frames=len(main_ids | supplement_ids),
                    cohorts_are_reported_separately_not_pooled=True)
                attach_terminal_assessment(report, terminal, baseline=baseline)
                save_preview(terminal_arrays, self.root/'terminal_preview.png', step)
            return report, arrays
        baseline_path = self.root/'validation_initial.json'
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
            if baseline['panel_fingerprint'] != identity['panel_fingerprint']: raise ValueError('Codec held-out panel changed')
        else:
            if step: raise ValueError('Resume requires the immutable initial validation')
            self.publish(stage_label='Initial held-out Source reconstruction and full RF compatibility',
                validation_panel_fingerprint=identity['panel_fingerprint'])
            baseline, arrays = evaluate(0)
            atomic_json(baseline_path, baseline); save_preview(arrays, self.root/'preview.png', 0)
            self.publish(validation_initial=baseline, validation=baseline); del arrays
        verify(); gc.collect()
        if terminal_val is not None and best_path.exists():
            previous_best = torch.load(best_path, map_location='cpu', weights_only=False)
            reviewed = attach_terminal_assessment(previous_best['report'],
                previous_best['report']['terminal_supplement'], baseline=baseline)
            if reviewed['rf_and_empty_guards_passed']:
                previous_best['report'] = reviewed
                best = {'step': previous_best['step'], 'score': reviewed['selection_score'],
                        'regression_guards_passed': True}
                _save(best_path, previous_best)
            else:
                latest = json.loads((self.root/'validation.json').read_text(encoding='utf-8'))
                latest = attach_terminal_assessment(latest, latest['terminal_supplement'], baseline=baseline)
                if latest['step'] == step and latest['rf_and_empty_guards_passed']:
                    best = {'step': step, 'score': latest['selection_score'], 'regression_guards_passed': True}
                    _save(best_path, {'model': student.state_dict(), 'step': step, 'report': latest, 'identity': identity})
                else: best = None
            atomic_json(self.root/'npc_guard_resume_review.json',
                {'step': step, 'policy': 'global_and_alive_NPC_ROI_0.005_presence_0.05',
                 'best_after_review': best, 'objective_optimizer_rng_unchanged': True})
            del previous_best
        preview_panel = []
        for category in (('tiny','near','corpse','navigation')if small_mode else('large','partial','corpse','empty_hidden')):
            preview_panel.append(next(index for index in panel if val.rows[index]['category'] == category))
        metrics_path = self.root/'metrics.jsonl'
        while step < args.steps:
            if (self.root/'stop_requested').exists():
                save(); self.publish(status='stopped', current=step, global_training_step=step); return
            start = time.perf_counter(); decoder_only = step < args.decoder_warmup_steps
            phase = configure_codec_phase(student, decoder_only)
            batch = train.batch(args.batch)
            conditions, references = prepare_references(working, original, batch)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = codec_enemy_loss(student, conditions, references, decoder_only=decoder_only,
                **({'object_crop_weight':args.object_crop_weight}if getattr(args,'object_crop_weight',0.)>0 else {}))
            loss.backward()
            e_norm = clip_finite_grad_norm_(encoder, context='codec enemy encoder')
            d_norm = clip_finite_grad_norm_(decoder, context='codec enemy decoder')
            optimizer.step(); step += 1
            metrics.update(step=step, global_training_step=step, timestamp=now(), phase=phase['phase'],
                encoder_gradient_norm=float(e_norm), decoder_gradient_norm=float(d_norm),
                seconds_per_step=time.perf_counter()-start, examples_seen=train.draws,
                source_train_frames=len(train.rows), loss_without_flow=metrics['train_loss'])
            with metrics_path.open('a', encoding='utf-8') as handle: handle.write(json.dumps(metrics)+'\n')
            self.publish(status='running', stage_label='D-only Source NPC warmup' if decoder_only else 'Source NPC encoder/decoder refinement',
                current=step, global_training_step=step, metrics=metrics, component_phase=phase,
                model_parameters=compact_codec_parameter_counts(student),
                loss=metrics['train_loss'], latest_metrics=metrics,
                eta_seconds=metrics['seconds_per_step']*(args.steps-step),
                encoder_lr=args.encoder_lr, decoder_lr=args.decoder_lr)
            if step % args.preview_every == 0:
                _, arrays = evaluate_codec_enemy(student, working, original, val, preview_panel, step=step, batch_size=4)
                save_preview(arrays, self.root/'preview.png', step); del arrays
                self.publish(prediction_preview={'step': step, 'every': args.preview_every,
                    'scope': 'Source GT-pose codec reconstruction and fixed RF1; not autonomous dynamics',
                    'row_labels': ['Source', 'Parent codec', 'Trained codec', 'Original E / new D', 'RF1 / new E+D']})
            if step % args.validate_every == 0 or step == args.steps:
                report, arrays = evaluate(step, baseline)
                verify(); streak = streak+1 if report['passed'] else 0
                report['consecutive_passes'] = streak; report['required_passes'] = 2
                eligible = report['rf_and_empty_guards_passed']
                if small_mode:eligible=eligible and report['small_enemy_stage_gate']['passed']
                if (best is None or eligible and not best['regression_guards_passed']
                        or eligible == best['regression_guards_passed'] and report['selection_score'] < best['score']):
                    best = {'step': step, 'score': report['selection_score'], 'regression_guards_passed': eligible}
                    _save(best_path, {'model': student.state_dict(), 'step': step, 'report': report, 'identity': identity})
                atomic_json(self.root/'validation.json', report)
                atomic_json(self.root/f'validation/step_{step:06d}.json', report)
                save_preview(arrays, self.root/'preview.png', step); del arrays
                self.publish(validation=report, best=best, quality_gate_streak=streak)
            if step % args.save_every == 0 or step == args.steps: save()
        save(); verify()
        chosen = torch.load(best_path, map_location='cpu', weights_only=False)
        student.load_state_dict(chosen['model'], strict=True)
        report = chosen['report']
        if terminal_val is not None:
            report, arrays = evaluate(chosen['step'], baseline)
            if not report['rf_and_empty_guards_passed']:
                raise AssertionError('Selected battle codec failed fresh NPC/alive/empty/RF guards')
            save_preview(arrays, self.root/'preview_selected.png', chosen['step']); del arrays
            verify()
        report.update(refinement_protocol='source_enemy_codec_ed_v1', selected_step=chosen['step'],
            original_imitation_quality_gate=parent_gate, parent_codec_sha256=identity['parent_codec_sha256'],
            parent_renderer_sha256=identity['renderer_sha256'], source_world_step=source_step,
            gan_active=False, codebook_trainable=False, normalization_changed=False,
            automatic_active_promotion=False, gameplay_qualified=False)
        exported = export_codec_student(student, source.flow_renderer, self.root/'checkpoints/codec_student.pt',
            training_steps=int(parent['codec_distillation']['training_steps'])+chosen['step'], report=report)
        atomic_json(self.root/'validation_selected.json', report)
        self.publish(status='completed', current=step, global_training_step=step, selected_step=chosen['step'],
            model_parameters=compact_codec_parameter_counts(student),
            validation=report, codec_export=exported, source_quality_passed=report['source_quality_passed'],
            combined_quality_passed=report['passed'], experimental_unqualified_fallback=not best['regression_guards_passed'],
            stage_label='Source codec refinement saved; not deployed; RF adaptation/validation required')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('world', 'renderer-checkpoint', 'data-root', 'empty-data-root', 'run-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('combat-data-root','appearance-data-root','validation-identity'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--small-data-root',type=Path)
    parser.add_argument('--object-crop-weight',type=float,default=0.)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    for name, value in (('steps', 2000), ('decoder-warmup-steps', 200), ('batch', 8),
            ('validation-frames', 256), ('validation-batch', 8), ('validate-every', 100),
            ('preview-every', 10), ('save-every', 10), ('seed', 173), ('threads', 4)):
        parser.add_argument('--'+name, type=int, default=value)
    parser.add_argument('--encoder-lr', type=float, default=2e-5)
    parser.add_argument('--decoder-lr', type=float, default=1e-4)
    args = parser.parse_args(argv)
    if any(getattr(args,key) is not None for key in ('combat_data_root','appearance_data_root','validation_identity')) \
            and not all(getattr(args,key) is not None for key in ('combat_data_root','appearance_data_root','validation_identity')):
        parser.error('Battle codec mode requires data, appearance replay and immutable validation identity together')
    if (min(args.steps, args.batch, args.validation_frames, args.validation_batch,
            args.validate_every, args.preview_every, args.save_every, args.threads) < 1
            or args.decoder_warmup_steps < 0 or min(args.encoder_lr, args.decoder_lr) <= 0):
        parser.error('Positive budgets/rates and nonnegative decoder warmup required')
    args = argparse.Namespace(**{key: value.resolve() if isinstance(value, Path) else value for key, value in vars(args).items()})
    from .vqgan_pipeline import process_alive
    args.run_dir.mkdir(parents=True, exist_ok=True)
    pid = args.run_dir/'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())): parser.error('Codec enemy worker already active')
    atomic_json(pid, __import__('os').getpid())
    experiment = CodecEnemyTraining(args); experiment.publish()
    try:
        with KeepAwake(): experiment.run()
    except BaseException as error:
        experiment.publish(status='failed', error=f'{type(error).__name__}: {error}'); raise


if __name__ == '__main__': main()
