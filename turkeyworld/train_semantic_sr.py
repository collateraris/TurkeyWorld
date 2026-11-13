"""Train object-aware SR on frozen NPR32 / true Source128 pairs.

This worker owns only SR parameters. Dynamics, NPR, codec and Source are absent
from its forward path: the immutable paired cache supplies their actual images.
The optional enemy decoder also receives pinned, past-derived RSSM hypotheses.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .distill_renderer import _save
from .gradient_safety import clip_finite_grad_norm_
from .models import build_upscaler, load_sr_checkpoint
from .power_guard import KeepAwake
from .rssm_preview_assets import preserve_preview_rng
from .semantic_sr_objectives import semantic_sr_loss, LOSS_WEIGHTS
from .sr_semantic_migration import restore_semantic_optimizer, _sha
from .training_pipeline import atomic_json, now


PROTOCOL = 'rgb_only_semantic_npr_sr_v1'


def mix_semantic_inputs(batch, rng):
    """Exactly >=75% actual NPR; guards are clean/codec, never HR downscales."""
    npr = batch['lr_npr'].detach()
    size = len(npr)
    if size < 4:
        raise ValueError('Semantic SR needs at least4 examples per batch')
    main = math.ceil(.75*size)
    order = rng.permutation(size)
    inputs = npr.clone()
    kinds = ['npr']*size
    codec = batch.get('codec_lr')
    for index, row in enumerate(order[main:]):
        kind = 'codec' if codec is not None and index % 2 else 'clean'
        inputs[int(row)] = (codec if kind == 'codec' else batch['lr_source'])[int(row)].detach()
        kinds[int(row)] = kind
    return inputs, kinds


def _restore_rng(saved, rng, device):
    if 'rng_numpy' in saved:
        rng.bit_generator.state = deepcopy(saved['rng_numpy'])
    if 'rng_torch' in saved:
        torch.set_rng_state(saved['rng_torch'].cpu())
    if 'rng_python' in saved:
        random.setstate(saved['rng_python'])
    if device.type == 'cuda' and saved.get('rng_cuda'):
        torch.cuda.set_rng_state_all([value.cpu() for value in saved['rng_cuda']])


def mix_enemy_hypotheses(hypothesis, rng):
    """Train visual rejection of wrong RSSM proposals; no Source labels involved."""
    value = hypothesis.detach().clone()
    available = torch.nonzero(value[:, -1] > .5, as_tuple=False).flatten().cpu().numpy()
    count = min(len(value)//4, len(available))
    if count < 1 or len(available) < 2:
        return value, 0
    order = rng.permutation(available)
    destination, source = order[:count], np.roll(order, 1)[:count]
    value[torch.as_tensor(destination, device=value.device)] = hypothesis[
        torch.as_tensor(source, device=value.device)].detach()
    return value, count


def mix_visual_first_hypotheses(hypothesis, rng):
    """Half the queries have no hint; another quarter gets a wrong prediction."""
    value = hypothesis.detach().clone()
    order = rng.permutation(len(value))
    dropped = order[:len(value)//2]
    value[torch.as_tensor(dropped, device=value.device)] = 0
    wrong = order[len(value)//2:len(value)//2+len(value)//4]
    available = torch.nonzero(hypothesis[:, -1] > .5, as_tuple=False).flatten().cpu().numpy()
    corrupted = 0
    for destination in wrong:
        if not len(available):
            continue
        alternatives = available[available != destination]
        source = int(rng.choice(alternatives if len(alternatives) else available))
        replacement = hypothesis[source].detach().clone()
        if source == destination:
            replacement[8:10] = (2*replacement[:2]-replacement[8:10]).clamp(0, 1)
        value[int(destination)] = replacement
        corrupted += 1
    return value, len(dropped), corrupted


def _choose_best(best, report, step):
    eligible = bool(report['relative_progress']['passed'])
    score = report['selection_score']
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError('Semantic SR validation needs a finite selection score')
    if (best is None or eligible and not best['eligible']
            or eligible == best['eligible'] and score < best['score']):
        return dict(step=step, score=score, eligible=eligible)
    return best


def _light_preview(model, data, panel, *, step, baseline_report, path, every, batch_size):
    """Four fixed held-out NPR images cannot advance any training RNG/sampler."""
    from .semantic_sr_validation import preview_semantic_sr, save_semantic_sr_preview
    modes = [(module, module.training) for module in model.modules()]
    sampler = deepcopy(data.sampler_state())
    device = next(model.parameters()).device
    try:
        with preserve_preview_rng(device, getattr(data, 'rng', None)):
            preview = preview_semantic_sr(model, data, panel, step=step,
                baseline_report=baseline_report, batch_size=batch_size)
            metadata = save_semantic_sr_preview(preview, path, every=every)
    finally:
        data.restore_sampler(sampler)
        for module, training in modes:
            module.training = training
    return metadata


def run(args):
    from .sr_neural_pairs_data import SemanticSRData
    from .semantic_sr_validation import evaluate_semantic_sr, save_semantic_sr_preview
    guided = bool(getattr(args, 'enemy_decoder', False))
    visual_first = bool(getattr(args, 'visual_first', False))
    restoration = bool(getattr(args, 'restoration', False))
    protocol, weights, objective_fn = PROTOCOL, LOSS_WEIGHTS, semantic_sr_loss
    if guided:
        from .enemy_decoder_data import EnemyDecoderData
        from .enemy_decoder_objectives import enemy_decoder_loss, LOSS_WEIGHTS as decoder_weights
        protocol, weights, objective_fn = 'rssm_guided_enemy_decoder_sr_v1', decoder_weights, enemy_decoder_loss
    if visual_first:
        from .visual_first_objectives import visual_first_loss, LOSS_WEIGHTS as vision_weights
        if not guided:
            raise ValueError('Visual-first training requires the learned enemy decoder')
        protocol, weights, objective_fn = 'visual_first_enemy_decoder_sr_v2', vision_weights, visual_first_loss
    if restoration:
        from .decoder_restoration_objectives import decoder_restoration_loss,LOSS_WEIGHTS as restoration_weights
        from .decoder_restoration_protocol import (SETTINGS as RESTORATION_SETTINGS,
            validate_restoration_schedule,next_restoration_blend_scale,commit_restoration_update,
            bind_restoration_baseline,consider_restoration_gate)
        if not visual_first or (args.validation_every,args.prediction_every)!=(100,15):
            raise ValueError('Restoration needs visual-first profile and pinned validation100/preview15')
        protocol,weights,objective_fn='decoder_restoration_sr_v3',restoration_weights,decoder_restoration_loss
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.init()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(args.seed)
    initial = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=False)
    architecture = 'enemy_decoder_edsr' if guided else 'semantic_edsr'
    if initial.get('config', {}).get('architecture') != architecture or 'semantic_migration' not in initial:
        raise ValueError('Use a validated append-only semantic SR initial checkpoint')
    if guided and 'enemy_decoder_migration' not in initial:
        raise ValueError('Enemy decoder requires its full-parent migration proof')
    if visual_first and (not initial['config'].get('enemy_decoder_visual_first')
                         or 'visual_first_migration' not in initial):
        raise ValueError('Visual-first training requires its explicit full-parent continuation')
    if bool(initial['config'].get('enemy_decoder_visual_first')) != visual_first:
        raise ValueError('Selected training profile differs from the SR architecture')
    if bool(initial['config'].get('enemy_decoder_restoration')) != restoration:
        raise ValueError('Restoration training and the explicit SR config disagree')
    if restoration and 'restoration_migration'not in initial:
        raise ValueError('Restoration requires its full-parent continuation proof')
    with torch.random.fork_rng(devices=[]):
        model = build_upscaler(initial['config'])
    model.load_state_dict(initial['model'], strict=True)
    model.to(device).float()
    optimizer = restore_semantic_optimizer(model, initial)
    names = initial['optimizer_parameter_names']
    parent_steps = initial['parent_training_steps']
    reader = (lambda split, seed: EnemyDecoderData(args.data_root, split,
              hypotheses_root=args.hypotheses_root, seed=seed, device=device)) if guided else (
              lambda split, seed: SemanticSRData(args.data_root, split, seed=seed, device=device))
    train = reader('train', args.seed)
    val = reader('val', args.seed+1)
    panel = val.fixed_panel(args.validation_frames or None, seed=args.seed)
    plan = dict(version=1, objective_protocol=protocol,
        initial_checkpoint=str(args.initial_checkpoint.resolve()), initial_sha256=_sha(args.initial_checkpoint),
        train=train.identity, val=val.identity, panel_fingerprint=val.panel_fingerprint(panel),
        panel=panel, batch=args.batch, seed=args.seed, actual_npr_fraction_min=.75,
        guard_domains=['clean_source32', 'codec_source32_if_cached'], loss_weights=weights,
        baseline38100_sha256=_sha(args.baseline38100), baseline40100_sha256=_sha(args.baseline40100),
        validation_every=args.validation_every, prediction_every=args.prediction_every,
        inference_inputs=['RGB32', 'predicted_RSSM_state20', 'hypothesis_available'] if guided else ['RGB32'],
        source_masks_are_loss_only=True,
        frozen_other_models=True, automatic_deployment=False)
    if guided:
        plan.update(hypotheses_root=str(args.hypotheses_root.resolve()),
                    sampler_policy='actual_visible_empty_1_to_1_with_balanced_visible_distances',
                    animation_condition=('predicted_RSSM_7_class_onehot' if initial['config'].get('enemy_decoder_stable')
                                         else 'raw_predicted_RSSM_timers'),
                    animation_sampling='equal_present_Source_phases_per_visible_distance',
                    parent_RGB_frozen=bool(initial['config'].get('enemy_decoder_stable')),
                    mismatched_predicted_proposal_fraction_max=.25,
                    future_Source_state_as_input=False)
    if visual_first:
        plan.update(version=2, hint_dropout_fraction=.5, mismatched_predicted_proposal_fraction_max=.25,
                    visual_route_cannot_be_vetoed_by_RSSM=True,
                    sampler_initialization='preserve_existing_sampler_and_draw_ledger',
                    inference_inputs=['RGB32', 'optional_predicted_RSSM_state20_and_validbit'])
    if restoration:
        plan.update(version=3,restoration_schedule=deepcopy(RESTORATION_SETTINGS),
            restoration_migration=deepcopy(initial['restoration_migration']),
            raw_targets='Source128_in_actual_predicted_bbox_detached_loss_only',
            inference_uses_saved_blend_scale=True,learned_display_gains_unused=True)
    plan_path = root/'training_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Immutable semantic SR training plan changed')
    else:
        atomic_json(plan_path, plan)
    plan_sha256=_sha(plan_path)
    checkpoint = root/'checkpoints/sr_last.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint.exists() else initial
    if checkpoint.exists():
        if saved.get('protocol') != plan or saved['optimizer_parameter_names'] != names:
            raise ValueError('Semantic SR resume identity or optimizer names changed')
        model.load_state_dict(saved['model'], strict=True)
        optimizer = restore_semantic_optimizer(model, saved)
        train.restore_sampler(saved['sampler'])
    elif visual_first:
        train.restore_sampler(initial['sampler'])
    restoration_schedule=None
    if restoration:
        restoration_schedule=deepcopy(saved.get('restoration_schedule'))
        if not isinstance(restoration_schedule,dict):
            raise ValueError('Restoration continuation lost its schedule')
        saved_config=saved['config']
        if ({k:v for k,v in saved_config.items()if k!='restoration_blend_scale'}!=
                {k:v for k,v in initial['config'].items()if k!='restoration_blend_scale'}):
            raise ValueError('Restoration resume changed an architecture field')
        validate_restoration_schedule(restoration_schedule,step=saved.get('semantic_updates',0),
                                      config_scale=saved_config.get('restoration_blend_scale'))
        model.set_restoration_blend_scale(saved_config['restoration_blend_scale'])
        model.restoration_phase_metadata=deepcopy(restoration_schedule)
    _restore_rng(saved, rng, device)
    if device.type=='cuda' and not saved.get('rng_cuda'):
        if not initial['semantic_migration'].get('explicit_fresh_adam'):
            raise ValueError('CUDA continuation lost the saved CUDA RNG state')
        torch.cuda.manual_seed_all(args.seed)
    step, best = saved.get('semantic_updates', 0), saved.get('best')
    if args.updates and args.updates < step:
        raise ValueError('Requested semantic SR budget is below its completed updates')
    state = dict(run_id=root.name, stage='sr', status='initializing', started_at=now(),
        current=step, total=args.updates or None, training_until_quality=not args.updates,
        model=('Raw enemy RGBA restoration with measured gradual blending' if restoration else
               'Visual-first learned enemy decoder with optional RSSM hints' if visual_first else
               'EDSR + compact learned enemy RGBA decoder + causal RSSM proposals' if guided
               else 'EDSR + learned RGB-only enemy recognition/residual'), loss_protocol=protocol,
        image_size=32, output_size=128, validation_every=args.validation_every,
        prediction_every=args.prediction_every, objective_protocol=protocol,
        model_parameters={'total':sum(p.numel() for p in model.parameters()),
                          'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)},
        sr_training={'parent_training_steps':parent_steps, 'input_size':32, 'output_size':128,
            'scale':4, 'actual_npr_fraction_min':.75, 'source_masks_are_loss_only':True,
            'all_other_models_frozen':True},
        gameplay_qualified=False, automatic_deployment=False,
        migration=initial['semantic_migration'])
    if guided:
        state['enemy_decoder_training'] = dict(
            protocol=protocol, migration=initial['enemy_decoder_migration'],
            hypothesis_source=train.identity, visible_empty_sampling='1:1',
            prediction_source='frozen_RSSM_prior_from_actual_previous_context',
            source_future_states_are_inputs=False, atlas_or_sprite_pasting=False,
            compact_learned_RGBA=True, visual_confirmation_required=True)
        state['enemy_decoder_training']['parent_rgb_frozen'] = bool(initial['config'].get('enemy_decoder_stable'))
        if visual_first:
            state['enemy_decoder_training'].update(visual_first=True, RSSM_hint_optional=True,
                high_confidence_visual_has_priority=True, hint_dropout_fraction=.5,
                migration=initial['visual_first_migration'])
    def publish(**values):
        if restoration:
            state['restoration_training']=dict(protocol=protocol,plan_sha256=plan_sha256,
                                               **deepcopy(restoration_schedule))
        state.update(values, updated_at=now())
        atomic_json(root/'status.json', state)
    def save():
        payload = dict(stage='sr', format_version=2, config=model.config,
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            optimizer_parameter_names=names, semantic_migration=initial['semantic_migration'],
            training_steps=parent_steps+step, parent_training_steps=parent_steps,
            semantic_updates=step, completed_updates=step, protocol=plan,
            sampler=train.sampler_state(), parent_sampler=initial.get('parent_sampler'), best=best,
            rng_numpy=deepcopy(rng.bit_generator.state), rng_torch=torch.get_rng_state(),
            rng_python=random.getstate(), rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            gameplay_qualified=False, automatic_deployment=False)
        if guided:
            payload.update(decoder_updates=step, enemy_decoder_migration=initial['enemy_decoder_migration'])
        if visual_first:
            payload.update(visual_first_updates=step, visual_first_migration=initial['visual_first_migration'],
                           parent_decoder_updates=initial.get('parent_decoder_updates'))
        if restoration:
            validate_restoration_schedule(restoration_schedule,step=step,
                                          config_scale=model.config['restoration_blend_scale'])
            payload.update(restoration_updates=step,restoration_migration=initial['restoration_migration'],
                           restoration_schedule=deepcopy(restoration_schedule))
        _save(checkpoint, payload)
    baseline_path = root/'validation_initial.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    else:
        if step:
            raise ValueError('Semantic SR resume lost its fixed initial validation')
        references = {}
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            for key, path in (('sr38100', args.baseline38100), ('sr40100', args.baseline40100)):
                reference, _ = load_sr_checkpoint(path, device)
                references[key] = reference.float().eval().requires_grad_(False)
        baseline, preview = evaluate_semantic_sr(model, val, panel, step=0,
            batch_size=args.batch, baselines=references)
        atomic_json(baseline_path, baseline)
        del references
        metadata = save_semantic_sr_preview(preview, root/'preview.png', every=args.prediction_every)
        publish(prediction_preview=metadata)
    if restoration:
        bind_restoration_baseline(restoration_schedule,baseline,sha256=_sha(baseline_path))
    save()
    state['semantic_sr_training'] = dict(protocol=('decoder_restoration_super_resolution_v3' if restoration else
        'visual_first_enemy_decoder_super_resolution_v2' if visual_first
        else 'enemy_decoder_super_resolution_v1' if guided
        else 'semantic_npr_super_resolution_v1'),
        objective_protocol=protocol, validation_panel_fingerprint=plan['panel_fingerprint'],
        data_identity_sha256=baseline['data_identity_sha256'], plan_sha256=plan_sha256,
        training_parent_cumulative_steps=parent_steps,
        baselines=dict(sr38100=plan['baseline38100_sha256'],sr40100=plan['baseline40100_sha256']))
    publish(status='running', validation_initial=baseline,
        semantic_sr_validation=baseline, sr_validation=baseline['domains']['npr']['models']['current'])
    first = step+1
    while not args.updates or step < args.updates:
        if (root/'stop_training').exists():
            save();publish(status='stopped', current=step);return
        started = time.perf_counter()
        if restoration:
            model.set_restoration_blend_scale(next_restoration_blend_scale(restoration_schedule))
            model.restoration_phase_metadata=deepcopy(restoration_schedule)
        model.train()
        batch = train.batch(args.batch, rng=rng)
        inputs, domains = mix_semantic_inputs(batch, rng)
        if guided:
            if visual_first:
                batch['hypothesis'], dropped, mismatched = mix_visual_first_hypotheses(batch['hypothesis'], rng)
            else:
                batch['hypothesis'], mismatched = mix_enemy_hypotheses(batch['hypothesis'], rng)
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_with_aux(inputs, batch['hypothesis'].detach()) if guided else model.forward_with_aux(inputs)
        loss, metrics = objective_fn(output, batch)
        loss.backward()
        metrics.update(gradient_norm=float(clip_finite_grad_norm_(model.named_parameters())),
            input_domain_counts=dict(Counter(domains)), source_cohorts=dict(Counter(batch['cohort'])),
            objective_protocol=protocol, loss_protocol=protocol, stage='sr',
            actual_npr_fraction=domains.count('npr')/len(domains),
            source_labels_detached=True, inference_rgb_only=not guided,
            source_row_indices=batch['row_index'].detach().cpu().tolist())
        if guided:
            metrics.update(mismatched_predicted_hypotheses=mismatched,
                actual_hypotheses_available=int((batch['hypothesis'][:, -1] > .5).sum()),
                future_Source_states_used_as_input=False)
            if visual_first:
                metrics.update(hypotheses_deliberately_dropped=dropped,
                               RSSM_hint_optional=True, visual_can_render_without_hint=True)
            if 'animation_label' in batch:
                metrics['source_animation_counts'] = dict(Counter(map(int,
                    batch['animation_label'].detach().cpu().reshape(-1).tolist())))
            if 'enemy_animation_index' in output:
                metrics['predicted_animation_counts'] = dict(Counter(map(int,
                    output['enemy_animation_index'].detach().cpu().reshape(-1).tolist())))
        json.dumps(metrics, allow_nan=False)
        optimizer.step()
        step += 1
        if restoration:
            commit_restoration_update(restoration_schedule,step=step)
            model.set_restoration_blend_scale(restoration_schedule['blend_scale'])
            model.restoration_phase_metadata=deepcopy(restoration_schedule)
            metrics.update(restoration_phase=restoration_schedule['phase'],
                restoration_updates=step,restoration_plan_sha256=plan_sha256,
                restoration_blend_scale=restoration_schedule['blend_scale'])
        metrics.update(step=step, cumulative_sr_training_steps=parent_steps+step,
                       timestamp=now(), optimizer_update_seconds=time.perf_counter()-started)
        with (root/'metrics.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(metrics,ensure_ascii=False,allow_nan=False)+'\n')
        publish(current=step, global_training_step=step, cumulative_sr_training_steps=parent_steps+step,
                loss=metrics['train_loss'], latest_metrics=metrics)
        if step == first or step % 10 == 0:
            save()
        if step % args.validation_every == 0 or args.updates and step == args.updates:
            report, preview = evaluate_semantic_sr(model, val, panel, step=step,
                batch_size=args.batch, baseline_report=baseline)
            if restoration:
                decision=(consider_restoration_gate(restoration_schedule,report,baseline,step=step)
                          if step%RESTORATION_SETTINGS['validation_every']==0 else None)
                model.restoration_phase_metadata=deepcopy(restoration_schedule)
                report['restoration_gate_decision']=decision
                report['restoration_schedule_after_validation']=deepcopy(restoration_schedule)
                report['training_goal'].update(
                    raw_restoration_gate_passed=restoration_schedule['blend_started_at']is not None,
                    restoration_full_blend=restoration_schedule['blend_scale']==1.)
                report['training_goal']['passed']=bool(report['training_goal']['passed']
                    and restoration_schedule['phase']=='restoration')
            atomic_json(root/f'validation/step_{step:06d}.json', report)
            atomic_json(root/'validation.json', report)
            metadata = save_semantic_sr_preview(preview, root/'preview.png', every=args.prediction_every)
            selected = _choose_best(best, report, step)
            if selected != best:
                best = selected
                _save(root/'checkpoints/sr_best.pt', dict(model=model.state_dict(),config=model.config,
                    step=step,report=report,eligible=best['eligible']))
                export = dict(stage='sr',format_version=2,
                    config=model.config,model=model.state_dict(),input_size=32,output_size=128,scale=4,
                    training_steps=parent_steps+step,parent_training_steps=parent_steps,
                    semantic_updates=step,validation=report,semantic_migration=initial['semantic_migration'],
                    source_quality_passed=bool(report.get('training_goal',{}).get('passed')),
                    gameplay_qualified=False,automatic_deployment=False)
                if guided:
                    export.update(decoder_updates=step, enemy_decoder_migration=initial['enemy_decoder_migration'])
                if visual_first:
                    export.update(visual_first_updates=step, visual_first_migration=initial['visual_first_migration'])
                if restoration:
                    export.update(restoration_updates=step,restoration_migration=initial['restoration_migration'],
                                  restoration_schedule=deepcopy(restoration_schedule))
                _save(root/'checkpoints/sr_for_game.pt', export)
                atomic_json(root/'validation_selected.json',report)
            objective = report.get('validation_objective')
            chart = (dict(val_loss=objective['loss'],
                          latest_validation_objective={**objective, 'step': step})
                     if objective is not None else {})
            publish(semantic_sr_validation=report, sr_validation=report['domains']['npr']['models']['current'],
                    prediction_preview=metadata, selected_best=best, **chart)
            if guided:
                publish(enemy_decoder_validation=report.get('enemy_decoder', {}))
            save()
            if report.get('training_goal',{}).get('passed'):
                publish(status='completed_quality_target',current=step);return
        elif step % args.prediction_every == 0:
            metadata = _light_preview(model, val, panel, step=step, baseline_report=baseline,
                path=root/'preview.png', every=args.prediction_every, batch_size=args.batch)
            publish(prediction_preview=metadata)
    save();publish(status='completed_budget',current=step)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run-dir','data-root','initial-checkpoint','baseline38100','baseline40100'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--updates',type=int,default=0,help='0 continues until the complete measured quality target')
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--validation-every',type=int,default=100)
    parser.add_argument('--prediction-every',type=int,default=15,
                        help='Light near/mid/far/empty preview cadence; full validation stays separate')
    parser.add_argument('--validation-frames',type=int,default=0)
    parser.add_argument('--seed',type=int,default=173)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--enemy-decoder',action='store_true',help='Train the append-only RSSM-guided learned enemy decoder')
    parser.add_argument('--visual-first',action='store_true',help='Train independent RGB restoration with optional RSSM hints')
    parser.add_argument('--restoration',action='store_true',help='Train raw RGBA before a measured persistent gradual blend')
    parser.add_argument('--hypotheses-root',type=Path,help='Pinned causal RSSM proposal sidecar; required for --enemy-decoder')
    args=parser.parse_args(argv)
    if (args.updates<0 or args.batch<4 or args.validation_every<1
            or args.prediction_every<1 or args.validation_frames<0):
        parser.error('Invalid semantic SR batch, cadence or budget')
    if args.enemy_decoder and args.hypotheses_root is None:
        parser.error('--enemy-decoder requires --hypotheses-root')
    if args.visual_first and not args.enemy_decoder:
        parser.error('--visual-first requires --enemy-decoder')
    if args.restoration and not args.visual_first:
        parser.error('--restoration requires --visual-first')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    (args.run_dir/'training.pid').write_text(str(os.getpid()),encoding='ascii')
    try:
        with KeepAwake():run(args)
    except Exception as error:
        atomic_json(args.run_dir/'error.json',dict(error=str(error),timestamp=now()))
        raise


if __name__=='__main__':main()
