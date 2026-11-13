"""Matched ROI-PatchGAN / reconstruction-only SR continuation.

The two generators inherit the same complete weights and Adam state. Each
completed update uses one shared data draw, domain mixture and optional hint.
Source labels never enter either generator's forward arguments.
"""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .decoder_restoration_objectives import decoder_restoration_loss
from .decoder_restoration_protocol import commit_restoration_update
from .distill_renderer import _save
from .enemy_decoder_data import EnemyDecoderData
from .gradient_safety import clip_finite_grad_norm_
from .models import build_upscaler
from .power_guard import KeepAwake
from .sr_enemy_patchgan import (EnemyPatchDiscriminator, build_enemy_patch_pairs,
    enemy_patch_discriminator_loss, enemy_patch_generator_loss)
from .sr_patchgan_protocol import PROTOCOL, SETTINGS, ledger_update, scheduled_gan_weight
from .sr_semantic_migration import restore_semantic_optimizer, _sha
from .train_semantic_sr import _restore_rng, mix_semantic_inputs, mix_visual_first_hypotheses
from .training_pipeline import atomic_json, now


def patchgan_ab_step(gan,control,discriminator,optimizer,control_optimizer,d_optimizer,
                     batch,inputs,*,scheduled_weight):
    """One matched complete update; the adversary cannot evade raw RGB/alpha loss."""
    gan.train();control.train();discriminator.train()
    optimizer.zero_grad(set_to_none=True);control_optimizer.zero_grad(set_to_none=True)
    d_optimizer.zero_grad(set_to_none=True)
    hint=batch['hypothesis'].detach()
    generated=gan.forward_with_aux(inputs,hint)
    reference=control.forward_with_aux(inputs,hint)
    reconstruction,g_metrics=decoder_restoration_loss(generated,batch)
    control_loss,c_metrics=decoder_restoration_loss(reference,batch)
    pairs=build_enemy_patch_pairs(generated,batch,low_rgb=inputs)
    has_pairs=len(pairs['condition'])>0
    d_norm=0.
    if has_pairs:
        d_loss,d_metrics=enemy_patch_discriminator_loss(discriminator,pairs)
        d_loss.backward()
        d_norm=float(clip_finite_grad_norm_(discriminator.named_parameters(),context='enemy ROI discriminator'))
        d_optimizer.step();d_optimizer.zero_grad(set_to_none=True)
        adversarial,a_metrics=enemy_patch_generator_loss(discriminator,pairs,
            feature_weight=SETTINGS['feature_matching_weight'])
    else:
        d_metrics={'discriminator_loss':0.}
        adversarial=generated['visual_rgba'].sum()*0
        a_metrics={'adv_loss':0.,'feature_matching_loss':0.}
    probe=gan.enemy_decoder_rgba[-1].weight
    base_norm=adv_norm=0.
    effective=float(scheduled_weight) if has_pairs else 0.
    if effective:
        base_gradient=torch.autograd.grad(reconstruction,probe,retain_graph=True,allow_unused=True)[0]
        adv_gradient=torch.autograd.grad(adversarial,probe,retain_graph=True,allow_unused=True)[0]
        base_norm=0. if base_gradient is None else float(base_gradient.float().norm())
        adv_norm=0. if adv_gradient is None else float(adv_gradient.float().norm())
        if adv_norm>0:effective=min(effective,SETTINGS['gradient_ratio_limit']*base_norm/adv_norm)
    total=reconstruction+effective*adversarial if effective else reconstruction
    if not bool(torch.isfinite(total))or not bool(torch.isfinite(control_loss)):
        raise FloatingPointError('Nonfinite matched PatchGAN generator objective')
    # CUDA grid_sample backward uses atomic accumulation. At zero GAN weight
    # identical twins have the SAME mathematical reconstruction gradient; reuse
    # that gradient instead of introducing a second noisy atomic reduction.
    # Both named Adam optimizers still take their own complete update. Once the
    # arms differ, or GAN is scheduled, each arm differentiates its own loss.
    control_values=control.state_dict()
    shared_gradient=(scheduled_weight==0. and torch.equal(reconstruction.detach(),control_loss.detach())
        and all(torch.equal(value,control_values[name])for name,value in gan.state_dict().items()))
    total.backward()
    if shared_gradient:
        for (name,parameter),(other_name,other)in zip(gan.named_parameters(),control.named_parameters()):
            if name!=other_name:raise ValueError('Matched generator parameter order changed')
            other.grad=None if parameter.grad is None else parameter.grad.detach().clone()
    else:control_loss.backward()
    g_norm=float(clip_finite_grad_norm_(gan.named_parameters(),context='PatchGAN SR generator'))
    c_norm=float(clip_finite_grad_norm_(control.named_parameters(),context='matched NoGAN SR generator'))
    optimizer.step();control_optimizer.step()
    metrics=dict(train_loss=float(total.detach()),generator_total_loss=float(total.detach()),
        reconstruction_loss=float(reconstruction.detach()),
        control_reconstruction_loss=float(control_loss.detach()),
        adv_loss=a_metrics['adv_loss'],feature_matching_loss=a_metrics['feature_matching_loss'],
        discriminator_loss=d_metrics['discriminator_loss'],
        gan_weight_scheduled=float(scheduled_weight),gan_weight_effective=effective,
        gan_readout_reconstruction_grad_norm=base_norm,gan_readout_adversarial_grad_norm=adv_norm,
        gan_readout_weighted_gradient_ratio=effective*adv_norm/max(base_norm,1e-30),
        generator_gradient_norm=g_norm,control_gradient_norm=c_norm,discriminator_gradient_norm=d_norm,
        discriminator_updated=has_pairs,patch_pairs=pairs['receipt'],
        warmup_shared_reconstruction_gradient=shared_gradient,
        generator_components=g_metrics,control_components=c_metrics,
        Source_future_inputs=False,matched_input_batches=True)
    json.dumps(metrics,allow_nan=False)
    return metrics


def run(args):
    from .sr_patchgan_validation import (evaluate_patchgan_pair,preview_patchgan_pair,
                                        save_patchgan_preview)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.init()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False)
    if initial.get('patchgan_migration',{}).get('kind')!=PROTOCOL:
        raise ValueError('Use the immutable full-state matched PatchGAN initialization')
    train=EnemyDecoderData(args.data_root,'train',hypotheses_root=args.hypotheses_root,seed=args.seed,device=device)
    val=EnemyDecoderData(args.data_root,'val',hypotheses_root=args.hypotheses_root,seed=args.seed+1,device=device)
    panel=val.fixed_panel(None,seed=args.seed)
    plan=dict(version=1,objective_protocol=PROTOCOL,settings=deepcopy(SETTINGS),
        initial_checkpoint=str(args.initial_checkpoint.resolve()),initial_sha256=_sha(args.initial_checkpoint),
        source_checkpoint_sha256=initial['patchgan_migration']['source_checkpoint_sha256'],
        train=train.identity,val=val.identity,panel=panel,panel_fingerprint=val.panel_fingerprint(panel),
        batch=8,seed=args.seed,inference_inputs=['NPR_RGB32','optional_predicted_RSSM_state20_and_validbit'],
        Source_targets_are_labels_only=True,automatic_deployment=False)
    plan_path=root/'training_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8'))!=plan:raise ValueError('Immutable matched GAN plan changed')
    else:atomic_json(plan_path,plan)
    plan_sha=_sha(plan_path)
    last=root/'checkpoints/sr_last.pt'
    saved=torch.load(last,map_location='cpu',weights_only=False)if last.exists()else initial
    if last.exists()and saved.get('protocol')!=plan:raise ValueError('Matched GAN resume plan changed')
    state=deepcopy(saved['patchgan_state'])
    step=state['updates']
    if (state['settings']!=SETTINGS or saved.get('patchgan_updates')!=step
            or not 0<=state['discriminator_steps']<=step):raise ValueError('Matched GAN update clocks changed')
    with torch.random.fork_rng(devices=[]):
        gan=build_upscaler(saved['config']);control=build_upscaler(saved['config'])
        discriminator=EnemyPatchDiscriminator()
    gan.load_state_dict(saved['model'],strict=True)
    control.load_state_dict(state['control_model'],strict=True)
    discriminator.load_state_dict(state['discriminator'],strict=True)
    gan.to(device).float();control.to(device).float();discriminator.to(device).float()
    for model in (gan,control):
        if any(isinstance(m,torch.nn.Dropout)and m.p for m in model.modules()):
            raise ValueError('Matched-input protocol does not allow independent stochastic generator dropout')
        if model.config.get('restoration_blend_scale')!=1.:raise ValueError('Both arms must retain full source blend')
    optimizer=restore_semantic_optimizer(gan,saved)
    control_saved={**saved,'optimizer':state['control_optimizer']}
    control_optimizer=restore_semantic_optimizer(control,control_saved)
    d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=SETTINGS['discriminator_lr'],
                                  betas=tuple(SETTINGS['discriminator_betas']))
    d_optimizer.load_state_dict(state['discriminator_optimizer'])
    train.restore_sampler(saved['sampler'])
    rng=np.random.default_rng(args.seed);_restore_rng(saved,rng,device)
    parent_steps=initial['training_steps'];parent_semantic=initial['semantic_updates']
    if saved['training_steps']!=parent_steps+step or saved['semantic_updates']!=parent_semantic+step:
        raise ValueError('Matched GAN source or cumulative generator clock changed')
    restoration=deepcopy(saved['restoration_schedule'])
    ledger=state['batch_ledger_sha256'];d_steps=state['discriminator_steps']
    base_payload={k:deepcopy(initial[k])for k in ('semantic_migration','enemy_decoder_migration',
        'visual_first_migration','restoration_migration','patchgan_migration','parent_protocol','parent_best',
        'parent_sampler','parent_training_steps','parent_decoder_updates')if k in initial}
    status=dict(run_id=root.parent.name,stage='sr',status='initializing',current=step,total=args.updates or None,
        loss_protocol=PROTOCOL,objective_protocol=PROTOCOL,started_at=now(),
        model='Matched enemy-ROI conditional PatchGAN / NoGAN',image_size=32,output_size=128,
        validation_every=100,prediction_every=15,automatic_deployment=False,gameplay_qualified=False,
        model_parameters={'total':sum(p.numel()for p in gan.parameters()),
                          'trainable':sum(p.numel()for p in gan.parameters()if p.requires_grad)},
        sr_training=dict(input_size=32,output_size=128,scale=4,all_other_models_frozen=True))
    pin_identity={}
    def publish(**values):
        status.update(values,updated_at=now(),current=step,cumulative_sr_training_steps=parent_steps+step)
        status['patchgan_training']=dict(protocol=PROTOCOL,plan_sha256=plan_sha,
            source_checkpoint_sha256=plan['source_checkpoint_sha256'],source_cumulative_steps=parent_steps,
            updates=step,gan_steps=step,no_gan_steps=step,discriminator_steps=d_steps,
            settings=deepcopy(SETTINGS),batch_ledger_sha256=ledger,matched_input_batches=True,
            blends={'gan':1.,'no_gan':1.},gan_weight_scheduled=scheduled_gan_weight(step),
            gan_weight_effective=status.get('latest_metrics',{}).get('gan_weight_effective',0.),**pin_identity)
        atomic_json(root/'status.json',status)
    def save():
        payload={**base_payload, 'stage':'sr','format_version':2,'config':deepcopy(gan.config),
            'model':gan.state_dict(),'optimizer':optimizer.state_dict(),
            'optimizer_parameter_names':saved['optimizer_parameter_names'],
            'training_steps':parent_steps+step,'semantic_updates':parent_semantic+step,
            'decoder_updates':parent_semantic+step,'visual_first_updates':parent_semantic+step,
            'restoration_updates':parent_semantic+step,'restoration_schedule':deepcopy(restoration),
            'patchgan_updates':step,'completed_updates':step,'protocol':plan,
            'sampler':train.sampler_state(),'rng_numpy':deepcopy(rng.bit_generator.state),
            'rng_torch':torch.get_rng_state(),'rng_python':random.getstate(),
            'rng_cuda':torch.cuda.get_rng_state_all()if device.type=='cuda'else[],
            'patchgan_state':dict(version=1,updates=step,discriminator_steps=d_steps,
                settings=deepcopy(SETTINGS),batch_ledger_sha256=ledger,
                control_model=control.state_dict(),control_optimizer=control_optimizer.state_dict(),
                discriminator=discriminator.state_dict(),discriminator_optimizer=d_optimizer.state_dict()),
            'automatic_deployment':False,'gameplay_qualified':False}
        _save(last,payload)
    def attach(report,preview):
        from .renderer_student import _hash_tensors
        common=dict(plan_sha256=plan_sha,parent_source_sha256=plan['source_checkpoint_sha256'],
                    arm_steps={'gan':step,'no_gan':step},batch_ledger_sha256=ledger,
                    objective_protocol=PROTOCOL,
                    arm_model_sha256={'gan':_hash_tensors(gan.state_dict()),
                                      'no_gan':_hash_tensors(control.state_dict())},
                    matched_training_verified_by_trainer=True)
        report.update(common);preview['metadata'].update(common)
        return report,preview
    baseline_path=root/'validation_initial.json'
    if baseline_path.exists():baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    else:
        if step:raise ValueError('Matched GAN resume lost its fixed paired baseline')
        baseline,preview=evaluate_patchgan_pair(gan,control,val,panel,step=step,batch_size=8)
        attach(baseline,preview);atomic_json(baseline_path,baseline)
        metadata=save_patchgan_preview(preview,root/'preview.png',every=15)
        status['prediction_preview']=metadata
    if (baseline['plan_sha256']!=plan_sha or baseline['panel_fingerprint']!=plan['panel_fingerprint']
            or baseline.get('parent_source_sha256')!=plan['source_checkpoint_sha256']):
        raise ValueError('Matched GAN validation baseline identity changed')
    pin_identity.update(validation_panel_fingerprint=baseline['panel_fingerprint'],
        data_identity_sha256=baseline['data_identity_sha256'],
        temporal_panel_fingerprint=baseline['temporal']['panel_fingerprint'])
    publish(status='running',patchgan_validation=baseline)
    save()
    while not args.updates or step<args.updates:
        if(root/'stop_training').exists():save();publish(status='stopped');return
        started=time.perf_counter()
        batch=train.batch(8,rng=rng);inputs,domains=mix_semantic_inputs(batch,rng)
        batch['hypothesis'],dropped,wrong=mix_visual_first_hypotheses(batch['hypothesis'],rng)
        metrics=patchgan_ab_step(gan,control,discriminator,optimizer,control_optimizer,d_optimizer,
            batch,inputs,scheduled_weight=scheduled_gan_weight(step+1))
        step+=1;d_steps+=int(metrics['discriminator_updated'])
        commit_restoration_update(restoration,step=parent_semantic+step)
        ledger=ledger_update(ledger,step,batch['row_index'].cpu().tolist(),domains,inputs,batch['hypothesis'])
        metrics.update(step=step,stage='sr',objective_protocol=PROTOCOL,loss_protocol=PROTOCOL,
            timestamp=now(),input_domain_counts=dict(Counter(domains)),source_row_indices=batch['row_index'].cpu().tolist(),
            hypotheses_deliberately_dropped=dropped,mismatched_predicted_hypotheses=wrong,
            batch_ledger_sha256=ledger,matched_input_batches=True,gan_steps=step,no_gan_steps=step,
            discriminator_steps=d_steps,cumulative_sr_training_steps=parent_steps+step,
            optimizer_update_seconds=time.perf_counter()-started)
        with(root/'metrics.jsonl').open('a',encoding='utf-8')as h:h.write(json.dumps(metrics,allow_nan=False)+'\n')
        publish(status='running',loss=metrics['generator_total_loss'],latest_metrics=metrics)
        if step<=5 or step%10==0:save()
        if step%100==0:
            report,preview=evaluate_patchgan_pair(gan,control,val,panel,step=step,batch_size=8)
            attach(report,preview);atomic_json(root/'validation.json',report)
            atomic_json(root/'validation'/f'step_{step:06d}.json',report)
            metadata=save_patchgan_preview(preview,root/'preview.png',every=15)
            publish(patchgan_validation=report,prediction_preview=metadata);save()
        elif step%15==0:
            preview=preview_patchgan_pair(gan,control,val,panel,step=step,baseline_report=baseline,batch_size=8)
            attach({},preview)
            metadata=save_patchgan_preview(preview,root/'preview.png',every=15)
            publish(prediction_preview=metadata)
    save();publish(status='completed')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','initial-checkpoint','data-root','hypotheses-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--updates',type=int,default=0)
    parser.add_argument('--seed',type=int,default=173)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args(argv)
    if args.updates<0:parser.error('Updates must be nonnegative')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    (args.run_dir/'training.pid').write_text(str(__import__('os').getpid()))
    try:
        with KeepAwake():run(args)
    except Exception as error:
        path=args.run_dir/'status.json'
        previous=json.loads(path.read_text(encoding='utf-8'))if path.exists()else{}
        atomic_json(path,{**previous,'status':'failed','error':str(error),
                         'message':'Matched SR training failed before the next complete saved boundary',
                         'updated_at':now()})
        raise


if __name__=='__main__':main()
