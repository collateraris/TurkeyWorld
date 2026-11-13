"""Train only NN sprite placement, animation, visibility and player-death tint."""
from collections import Counter
from copy import deepcopy
from functools import partial
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .atlas_enemy_objectives import atlas_enemy_loss,LOSS_WEIGHTS,PROTOCOL
from .distill_renderer import _save
from .gradient_safety import clip_finite_grad_norm_
from .models import build_upscaler
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .sr_semantic_migration import restore_semantic_optimizer,_sha
from .train_semantic_sr import _restore_rng,mix_semantic_inputs,mix_visual_first_hypotheses
from .training_pipeline import atomic_json,now


def run(args):
    from .enemy_decoder_data import EnemyDecoderData
    from .atlas_enemy_validation import evaluate_atlas_enemy_sr,preview_atlas_enemy_sr,save_atlas_enemy_preview
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.init()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False)
    if initial.get('config',{}).get('architecture')!='atlas_enemy_edsr'or 'atlas_migration'not in initial:
        raise ValueError('Use an explicit full-state fixed-atlas initialization')
    train=EnemyDecoderData(args.data_root,'train',hypotheses_root=args.hypotheses_root,seed=args.seed,device=device)
    val=EnemyDecoderData(args.data_root,'val',hypotheses_root=args.hypotheses_root,seed=args.seed+1,device=device)
    panel=val.fixed_panel(None,seed=args.seed)
    plan=dict(version=1,objective_protocol=PROTOCOL,
        initial_checkpoint=str(args.initial_checkpoint.resolve()),initial_sha256=_sha(args.initial_checkpoint),
        source_checkpoint_sha256=initial['atlas_migration']['source_checkpoint_sha256'],
        reference_checkpoint=str(args.reference_checkpoint.resolve()),reference_sha256=_sha(args.reference_checkpoint),
        train=train.identity,val=val.identity,panel=panel,panel_fingerprint=val.panel_fingerprint(panel),
        batch=8,seed=args.seed,validation_every=100,prediction_every=15,loss_weights=LOSS_WEIGHTS,
        atlas_asset_sha256=initial['config']['atlas_asset_sha256'],
        atlas_pixels_sha256=initial['config']['atlas_pixels_sha256'],
        geometry='NN_full_unclipped_sprite_rectangle',phase='RGB_NN_hard_argmax',
        visibility='NN_binary_mask_preserves_intrinsic_PNG_alpha',adversarial_loss=False,
        reference_is_frozen=True,source_labels_are_inference_inputs=False,automatic_deployment=False)
    path=root/'training_plan.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=plan:raise ValueError('Immutable atlas SR plan changed')
    else:atomic_json(path,plan)
    plan_sha=_sha(path);last=root/'checkpoints/sr_last.pt'
    saved=torch.load(last,map_location='cpu',weights_only=False)if last.exists()else initial
    if last.exists()and saved.get('protocol')!=plan:raise ValueError('Atlas SR resume belongs to another plan')
    step=saved['atlas_updates'];parent_steps=initial['parent_training_steps']
    if (saved['training_steps']!=parent_steps+step or saved['semantic_updates']!=step
            or saved['config']!=initial['config']):raise ValueError('Atlas SR architecture or completed clock changed')
    with torch.random.fork_rng(devices=[]):model=build_upscaler(saved['config'])
    model.load_state_dict(saved['model'],strict=True);model.to(device).float()
    optimizer=restore_semantic_optimizer(model,saved)
    source=torch.load(args.reference_checkpoint,map_location='cpu',weights_only=False)
    if _sha(args.reference_checkpoint)!=plan['source_checkpoint_sha256']:
        raise ValueError('Atlas reference must be its exact pre-migration source model')
    with torch.random.fork_rng(devices=[]):reference=build_upscaler(source['config'])
    reference.load_state_dict(source['model'],strict=True)
    reference.to(device).float().eval().requires_grad_(False)
    del source
    train.restore_sampler(saved['sampler'])
    rng=np.random.default_rng(args.seed);_restore_rng(saved,rng,device)
    objective=partial(atlas_enemy_loss,label_renderer=model.render_label_atlas)
    frozen_names=initial['atlas_migration']['frozen_parameter_names']
    fixed_names=(*frozen_names,'atlas_frames_rgba','atlas_native_sizes')
    fixed_hash=_hash_tensors({k:v for k,v in model.state_dict().items()if k in fixed_names})
    status=dict(run_id=root.parent.name,stage='sr',status='initializing',current=step,
        total=args.updates or None,started_at=now(),loss_protocol=PROTOCOL,objective_protocol=PROTOCOL,
        model='NN full sprite geometry / animation / visibility + embedded enemy atlas',
        image_size=32,output_size=128,validation_every=100,prediction_every=15,
        model_parameters={'total':sum(p.numel()for p in model.parameters()),
                          'trainable':sum(p.numel()for p in model.parameters()if p.requires_grad)},
        sr_training=dict(input_size=32,output_size=128,scale=4,all_other_models_frozen=True),
        gameplay_qualified=False,automatic_deployment=False)
    pins={}
    def publish(**values):
        status.update(values,current=step,cumulative_sr_training_steps=parent_steps+step,updated_at=now())
        status['atlas_training']=dict(protocol=PROTOCOL,plan_sha256=plan_sha,
            source_checkpoint_sha256=plan['source_checkpoint_sha256'],source_cumulative_steps=parent_steps,
            updates=step,phase='bbox_animation_visibility',atlas_asset_sha256=model.config['atlas_asset_sha256'],
            atlas_pixels_sha256=model.config['atlas_pixels_sha256'],trainable_parameters=status['model_parameters']['trainable'],
            full_bbox_unclipped=True,binary_visibility=True,embedded_atlas=True,
            fixed_enemy_pixels=True,adversarial_training=False,reference_is_frozen=True,**pins)
        atomic_json(root/'status.json',status)
    metadata_keys=('semantic_migration','enemy_decoder_migration','visual_first_migration',
        'restoration_migration','patchgan_migration','atlas_migration','parent_protocol','parent_best',
        'parent_sampler','parent_semantic_updates','parent_patchgan_updates','parent_restoration_schedule')
    def save():
        payload={k:deepcopy(initial[k])for k in metadata_keys if k in initial}
        payload.update(stage='sr',format_version=2,config=deepcopy(model.config),model=model.state_dict(),
            optimizer=optimizer.state_dict(),optimizer_parameter_names=saved['optimizer_parameter_names'],
            training_steps=parent_steps+step,parent_training_steps=parent_steps,
            semantic_updates=step,atlas_updates=step,completed_updates=step,protocol=plan,
            sampler=train.sampler_state(),rng_numpy=deepcopy(rng.bit_generator.state),
            rng_torch=torch.get_rng_state(),rng_python=random.getstate(),
            rng_cuda=torch.cuda.get_rng_state_all()if device.type=='cuda'else[],
            automatic_deployment=False,gameplay_qualified=False)
        _save(last,payload)
    def attach(report,preview):
        fields=dict(plan_sha256=plan_sha,parent_source_sha256=plan['source_checkpoint_sha256'],
            atlas_asset_sha256=model.config['atlas_asset_sha256'],atlas_pixels_sha256=model.config['atlas_pixels_sha256'],
            reference_is_frozen=True,source_future_as_input=False,
            model_sha256=_hash_tensors(model.state_dict()))
        report.update(fields);preview['metadata'].update(fields)
    baseline_path=root/'validation_initial.json'
    if baseline_path.exists():baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    else:
        if step:raise ValueError('Atlas continuation lost its complete initial evaluation')
        baseline,preview=evaluate_atlas_enemy_sr(model,reference,val,panel,step=0,batch_size=8)
        attach(baseline,preview);atomic_json(baseline_path,baseline)
        status['prediction_preview']=save_atlas_enemy_preview(preview,root/'preview.png',every=15)
    if (baseline.get('plan_sha256')!=plan_sha or baseline.get('parent_source_sha256')!=plan['source_checkpoint_sha256']
            or baseline.get('panel_fingerprint')!=plan['panel_fingerprint']):
        raise ValueError('Atlas initial validation source or panel changed')
    pins.update(validation_panel_fingerprint=baseline['panel_fingerprint'],data_identity_sha256=baseline['data_identity_sha256'],
                temporal_panel_fingerprint=baseline['temporal']['panel_fingerprint'])
    publish(status='running',atlas_validation=baseline);save()
    while not args.updates or step<args.updates:
        if(root/'stop_training').exists():save();publish(status='stopped');return
        started=time.perf_counter();model.train()
        batch=train.batch(8,rng=rng);inputs,domains=mix_semantic_inputs(batch,rng)
        batch['hypothesis'],dropped,wrong=mix_visual_first_hypotheses(batch['hypothesis'],rng)
        optimizer.zero_grad(set_to_none=True)
        output=model.forward_with_aux(inputs,batch['hypothesis'].detach())
        loss,metrics=objective(output,batch);loss.backward()
        metrics['gradient_norm']=float(clip_finite_grad_norm_(model.named_parameters(),context='fixed atlas SR'))
        optimizer.step();step+=1
        metrics.update(step=step,stage='sr',objective_protocol=PROTOCOL,loss_protocol=PROTOCOL,
            timestamp=now(),input_domain_counts=dict(Counter(domains)),
            source_row_indices=batch['row_index'].cpu().tolist(),
            hypotheses_deliberately_dropped=dropped,mismatched_predicted_hypotheses=wrong,
            cumulative_sr_training_steps=parent_steps+step,optimizer_update_seconds=time.perf_counter()-started)
        with(root/'metrics.jsonl').open('a',encoding='utf-8')as h:h.write(json.dumps(metrics,allow_nan=False)+'\n')
        publish(status='running',loss=float(loss.detach()),latest_metrics=metrics)
        if step<=5 or step%10==0:save()
        if step%100==0:
            report,preview=evaluate_atlas_enemy_sr(model,reference,val,panel,step=step,batch_size=8,baseline_report=baseline)
            attach(report,preview);atomic_json(root/'validation.json',report)
            atomic_json(root/'validation'/f'step_{step:06d}.json',report)
            current=_hash_tensors({k:v for k,v in model.state_dict().items()if k in fixed_names})
            if current!=fixed_hash:raise AssertionError('Fixed atlas, inherited RGB or unused pixel decoder changed')
            metadata=save_atlas_enemy_preview(preview,root/'preview.png',every=15)
            publish(atlas_validation=report,prediction_preview=metadata,frozen_verification=dict(step=step,passed=True,sha256=fixed_hash));save()
        elif step%15==0:
            preview=preview_atlas_enemy_sr(model,reference,val,panel,step=step,baseline_report=baseline,batch_size=8)
            attach({},preview);metadata=save_atlas_enemy_preview(preview,root/'preview.png',every=15)
            publish(prediction_preview=metadata)
    save();publish(status='completed')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('run-dir','initial-checkpoint','reference-checkpoint','data-root','hypotheses-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--updates',type=int,default=0);parser.add_argument('--seed',type=int,default=173)
    args=parser.parse_args(argv)
    if args.updates<0:parser.error('Updates must be nonnegative')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    (args.run_dir/'training.pid').write_text(str(__import__('os').getpid()))
    try:
        with KeepAwake():run(args)
    except Exception as error:
        p=args.run_dir/'status.json';old=json.loads(p.read_text(encoding='utf-8'))if p.exists()else{}
        atomic_json(p,{**old,'status':'failed','error':str(error),'updated_at':now()});raise


if __name__=='__main__':main()
