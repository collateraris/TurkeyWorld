"""Train only five appended SR Transformer experts on refreshed causal NPR/Source pairs."""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .distill_renderer import _save
from .enemy_decoder_data import EnemyDecoderData
from .gradient_safety import clip_finite_grad_norm_
from .models import build_upscaler
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .sr_semantic_migration import restore_semantic_optimizer, _sha
from .sr_moe_phase import PROTOCOL, sr_moe_protocol
from .sr_moe_objectives import sr_moe_loss, LOSS_WEIGHTS
from .sr_moe_validation import evaluate_sr_moe, preview_sr_moe, save_sr_moe_preview
from .sr_transformer_moe import configure_sr_transformer_moe, sr_moe_metadata
from .train_semantic_sr import _restore_rng, mix_semantic_inputs, mix_visual_first_hypotheses
from .training_pipeline import atomic_json, now


def run(args):
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.init()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False)
    if initial.get('config',{}).get('sr_moe_enabled') is not True or 'sr_moe_migration' not in initial:
        raise ValueError('Use an explicit complete five-expert frozen-Atlas initialization')
    train=EnemyDecoderData(args.data_root,'train',hypotheses_root=args.hypotheses_root,seed=args.seed,device=device)
    val=EnemyDecoderData(args.data_root,'val',hypotheses_root=args.hypotheses_root,seed=args.seed+1,device=device)
    panel=val.fixed_panel(256,seed=args.seed)
    cache=train.identity['base']['cache_identity']
    if (cache.get('world_step')!=5947 or cache.get('renderer_step')!=29609
            or cache.get('future_rgb_features') is not False or cache.get('paired_same_pose_map_tick') is not True):
        raise ValueError('SR five-expert phase needs the explicitly refreshed5947/29609 causal NPR cache')
    settings=sr_moe_protocol()
    plan=dict(version=1,objective_protocol=PROTOCOL,settings=settings,
        initial_checkpoint=str(args.initial_checkpoint.resolve()),initial_sha256=_sha(args.initial_checkpoint),
        source_checkpoint_sha256=initial['sr_moe_migration']['source_checkpoint_sha256'],
        reference_checkpoint=str(args.reference_checkpoint.resolve()),reference_sha256=_sha(args.reference_checkpoint),
        train=train.identity,val=val.identity,panel=panel,panel_fingerprint=val.panel_fingerprint(panel),
        batch=8,steps=args.steps,seed=args.seed,validation_every=100,prediction_every=15,loss_weights=LOSS_WEIGHTS,
        parent_training_steps=initial['parent_training_steps'],fresh_private_sampler_draws=0,
        sampler_rng='fresh_identity_bound_PCG173',augmentation_rng='parent_mix_rng_preserved',
        parent_sampler_retained_as_history=True,all_old_SR_parameters_frozen=True,
        source_labels_are_inference_inputs=False,automatic_deployment=False,
        quality_goal=dict(input_kind='actual_NPR_RGB32',ssim_strictly_greater_than=.91,mae255_strictly_less_than=2.5))
    plan_path=root/'training_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8'))!=plan:raise ValueError('Immutable SR five-expert plan changed')
    else:atomic_json(plan_path,plan)
    plan_sha=_sha(plan_path);last=root/'checkpoints/sr_last.pt'
    saved=torch.load(last,map_location='cpu',weights_only=False) if last.exists() else initial
    if last.exists() and saved.get('protocol')!=plan:raise ValueError('SR expert resume belongs to another plan')
    step=saved['sr_moe_updates'];parent_steps=initial['parent_training_steps']
    if (type(step) is not int or not 0<=step<=args.steps or saved['training_steps']!=parent_steps+step
            or saved['config']!=initial['config']):raise ValueError('SR expert architecture or cumulative clock changed')
    with torch.random.fork_rng(devices=[]):model=build_upscaler(saved['config'])
    model.load_state_dict(saved['model'],strict=True);model.sr_moe_migration=deepcopy(saved['sr_moe_migration'])
    model.to(device).float();configure_sr_transformer_moe(model)
    if saved.get('sr_moe_binding')!=sr_moe_metadata(model):raise ValueError('SR expert model/binding changed')
    optimizer=restore_semantic_optimizer(model,saved)
    reference_payload=torch.load(args.reference_checkpoint,map_location='cpu',weights_only=False)
    if _sha(args.reference_checkpoint)!=initial['sr_moe_migration']['selected_checkpoint_sha256']:
        raise ValueError('SR frozen reference must be the exact selected106105 Atlas parent')
    with torch.random.fork_rng(devices=[]):reference=build_upscaler(reference_payload['config'])
    reference.load_state_dict(reference_payload['model'],strict=True)
    reference.to(device).float().eval().requires_grad_(False);del reference_payload
    if last.exists():train.restore_sampler(saved['sampler'])
    elif train.draws!=0:raise AssertionError('Fresh SR paired-data sampler must begin at zero')
    rng=np.random.default_rng(args.seed);_restore_rng(saved,rng,device)
    frozen_names=[n for n in model.state_dict() if not n.startswith('sr_moe.')]
    fixed_hash=_hash_tensors({n:model.state_dict()[n] for n in frozen_names})
    started=time.perf_counter()
    status=dict(run_id=root.parent.name,stage='sr',status='initializing',current=step,total=args.steps,
        started_at=now(),objective_protocol=PROTOCOL,loss_protocol=PROTOCOL,
        model='Frozen AtlasSR + five RGB token Transformer experts',image_size=32,output_size=128,
        validation_every=100,prediction_every=15,gameplay_qualified=False,automatic_deployment=False,
        model_parameters=dict(total=sum(p.numel() for p in model.parameters()),
                              trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)),
        sr_training=dict(input_size=32,output_size=128,scale=4,all_other_models_frozen=True))
    training=dict(protocol=PROTOCOL,plan_sha256=plan_sha,parent_training_steps=parent_steps,
        source_parent_sha256=plan['source_checkpoint_sha256'],local_step=step,local_budget=args.steps,
        world_sha256=cache['world_sha256'],world_step=cache['world_step'],
        renderer_sha256=cache['renderer_sha256'],renderer_training_steps=cache['renderer_step'],
        roles=settings['roles'],experts=5,private_Transformer_blocks_per_expert=1,
        width=64,heads=4,expert_hidden=128,grid=8,rgb_resolution=128,visibility_resolution=128,
        heat_resolution=32,rgb_amplitude=1.,frozen_parent=True,Source_targets_are_labels_only=True,
        source_future_feedback=False,parameters=status['model_parameters'],quality_goal=plan['quality_goal'])
    def publish(**values):
        training['local_step']=step
        status.update(values,current=step,sr_moe_training=training,cumulative_sr_training_steps=parent_steps+step,
                      updated_at=now(),elapsed_seconds=time.perf_counter()-started)
        atomic_json(root/'status.json',status)
    def save():
        payload={k:deepcopy(initial[k]) for k in ('semantic_migration','enemy_decoder_migration',
            'visual_first_migration','restoration_migration','atlas_migration','sr_moe_migration') if k in initial}
        payload.update(stage='sr',config=deepcopy(model.config),model=model.state_dict(),
            optimizer=optimizer.state_dict(),optimizer_parameter_names=deepcopy(initial['optimizer_parameter_names']),
            sr_moe_binding=sr_moe_metadata(model),training_steps=parent_steps+step,parent_training_steps=parent_steps,
            sr_moe_updates=step,completed_updates=step,protocol=plan,
            parent_atlas_updates=initial.get('parent_atlas_updates'),parent_sampler=deepcopy(initial.get('parent_sampler')),
            sampler=train.sampler_state(),rng_numpy=deepcopy(rng.bit_generator.state),
            rng_torch=torch.get_rng_state(),rng_python=random.getstate(),
            rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            automatic_deployment=False,gameplay_qualified=False)
        _save(last,payload)
    def attach(report,preview):
        pins=dict(plan_sha256=plan_sha,parent_source_sha256=plan['source_checkpoint_sha256'],
            world_sha256=cache['world_sha256'],renderer_sha256=cache['renderer_sha256'],
            model_sha256=_hash_tensors(model.state_dict()),old_SR_frozen=True)
        report.update(pins);preview['metadata'].update(pins)
    baseline_path=root/'validation_initial.json'
    if baseline_path.exists():baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    else:
        if step:raise ValueError('SR expert continuation lost the initial held-out report')
        publish(stage_label='Initial held-out current-NPR validation before SGD')
        baseline,preview=evaluate_sr_moe(model,reference,val,panel,step=0,batch_size=8)
        attach(baseline,preview);atomic_json(baseline_path,baseline)
        status['prediction_preview']=save_sr_moe_preview(preview,root/'preview.png',every=15)
    if (baseline.get('plan_sha256')!=plan_sha or baseline.get('panel_fingerprint')!=plan['panel_fingerprint']):
        raise ValueError('SR expert initial validation belongs to another immutable plan/panel')
    training.update(validation_panel_fingerprint=baseline['panel_fingerprint'],
                    data_identity_sha256=baseline['data_identity_sha256'])
    publish(status='running',sr_moe_validation=baseline);save()
    while step<args.steps:
        if (root/'stop_training').exists():save();publish(status='stopped');return
        update_started=time.perf_counter();configure_sr_transformer_moe(model)
        batch=train.batch(8)  # Independent fresh private data PCG, never the inherited augmentation PCG.
        inputs,domains=mix_semantic_inputs(batch,rng)
        batch['hypothesis'],dropped,wrong=mix_visual_first_hypotheses(batch['hypothesis'],rng)
        optimizer.zero_grad(set_to_none=True)
        output=model.forward_with_aux(inputs,batch['hypothesis'].detach())
        loss,metrics=sr_moe_loss(output,batch,label_renderer=model.render_label_atlas);loss.backward()
        metrics['gradient_norm']=float(clip_finite_grad_norm_(model.named_parameters(),context='SR five experts'))
        if step<3:
            metrics['expert_gradient_norms']={str(i):sum(float(p.grad.detach().float().square().sum())
                for p in expert.parameters() if p.grad is not None)**.5 for i,expert in enumerate(model.sr_moe.experts)}
        optimizer.step();step+=1
        if _hash_tensors({n:model.state_dict()[n] for n in frozen_names})!=fixed_hash:
            raise AssertionError('SR expert training changed a frozen parent weight or atlas buffer')
        metrics.update(step=step,stage='sr',objective_protocol=PROTOCOL,loss_protocol=PROTOCOL,timestamp=now(),
            sr_moe_plan_sha256=plan_sha,input_domain_counts=dict(Counter(domains)),
            source_row_indices=batch['row_index'].detach().cpu().tolist(),
            hypotheses_deliberately_dropped=dropped,mismatched_predicted_hypotheses=wrong,
            cumulative_sr_training_steps=parent_steps+step,optimizer_update_seconds=time.perf_counter()-update_started)
        with (root/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
        publish(status='running',loss=float(loss.detach()),latest_metrics=metrics,
            frozen_verification=dict(step=step,passed=True,sha256=fixed_hash,all_old_SR_parameters_buffers_bitexact=True,
                                     sr_moe_plan_sha256=plan_sha))
        if step<=5 or step%10==0:save()
        if step%100==0:
            report,preview=evaluate_sr_moe(model,reference,val,panel,step=step,batch_size=8,baseline_report=baseline)
            attach(report,preview);atomic_json(root/'validation.json',report)
            atomic_json(root/'validation'/f'step_{step:06d}.json',report)
            publish(sr_moe_validation=report,prediction_preview=save_sr_moe_preview(preview,root/'preview.png',every=15));save()
        elif step%15==0:
            preview=preview_sr_moe(model,reference,val,panel,step=step,baseline_report=baseline,batch_size=8)
            attach({},preview);publish(prediction_preview=save_sr_moe_preview(preview,root/'preview.png',every=15))
    save();publish(status='completed',message='Experimental SR budget completed; quality goal and gameplay qualification remain separate')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','initial-checkpoint','reference-checkpoint','data-root','hypotheses-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2000);parser.add_argument('--seed',type=int,default=173)
    for name,default in (('batch',8),('validate-every',100),('prediction-every',15),
                         ('validation-frames',256),('validation-batch',8)):
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args(argv)
    if args.steps<1:parser.error('Positive initial experimental budget required')
    if (args.batch,args.validate_every,args.prediction_every,args.validation_frames,args.validation_batch)!=(8,100,15,256,8):
        parser.error('SR experts use B8, preview15, full VAL100 on fixed256 held-out frames with B8')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    from .vqgan_pipeline import process_alive
    pid=args.run_dir/'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())):parser.error('SR expert worker is already active')
    import os
    pid.write_text(str(os.getpid()),encoding='ascii')
    try:
        with KeepAwake():run(args)
    except BaseException as error:
        path=args.run_dir/'status.json';old=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        atomic_json(path,{**old,'status':'failed','error':f'{type(error).__name__}: {error}','updated_at':now()});raise


if __name__=='__main__':main()
