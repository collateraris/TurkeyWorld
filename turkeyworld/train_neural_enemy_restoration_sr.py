"""Isolated RGB-only enemy restoration experiment on frozen final-renderer pairs.

The selected full Atlas checkpoint is a tensor-copy source and validation
baseline only. The trained standalone model contains no atlas assets or state
hints. Its new decoder has a fresh optimizer; background parameters stay exact.
"""
from copy import deepcopy
from functools import partial
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .distill_renderer import _save
from .gradient_safety import clip_finite_grad_norm_
from .models import build_upscaler
from .neural_enemy_restoration_data import NeuralEnemyRestorationData
from .neural_enemy_restoration_objectives import (PROTOCOL,FOREGROUND_PROTOCOL,FOREGROUND_TOPK_PIXELS,LOSS_WEIGHTS,
    DETAIL_PROTOCOL,DETAIL_SETTINGS,loss_weights,neural_enemy_restoration_loss)
from .neural_enemy_restoration_continuation import continuation_source,validate_continuation_checkpoint
from .neural_enemy_detail_selection import POLICY as DETAIL_SELECTION_POLICY,detail_visual_candidate
from .neural_enemy_restoration_sr import(initialize_neural_enemy_restoration,frozen_background_state,
    neural_restoration_metadata)
from .neural_enemy_restoration_validation import(evaluate_neural_enemy_restoration,
    preview_neural_enemy_restoration,save_neural_enemy_restoration_preview,build_detail_animation_panel)
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .sr_neural_pairs_data import fingerprint
from .training_pipeline import atomic_json,now


def file_sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError('Neural restoration immutable plan changed')
    else:atomic_json(path,value)


def initialize_runtime(device):
    """Initialize CUDA and its CPU thread budget before any guard/worker work."""
    device=torch.device(device)
    if device.type=='cuda':torch.cuda.init()
    torch.set_num_threads(2)
    return device


def prepare_experiment(args):
    """Complete parent copy and immutable paired-data/panel selection.

    Windows must create the CUDA context before deserializing a CUDA-authored
    checkpoint, even though the payload and migration are mapped to CPU.
    """
    if torch.device(getattr(args,'device','cpu')).type=='cuda':torch.cuda.init()
    parent_path=args.parent_checkpoint.resolve();parent_sha=file_sha(parent_path)
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    if file_sha(parent_path)!=parent_sha:raise ValueError('Parent checkpoint changed while reading')
    model,migration=initialize_neural_enemy_restoration(parent,seed=args.seed)
    train=NeuralEnemyRestorationData(args.data_root,'train',seed=args.seed,device='cpu')
    val=NeuralEnemyRestorationData(args.data_root,'val',seed=args.seed+1,device='cpu')
    if train.scene_keys&val.scene_keys:raise ValueError('TRAIN/VAL render conditions overlap')
    panel=val.fixed_panel(args.validation_frames,seed=args.seed)
    # Identical256 indices are retained when the selected parent has that same
    # panel; new model/data fingerprints remain separate and explicit.
    parent_plan=parent.get('protocol')or{}
    for split,data in(('train',train),('val',val)):
        old_base=(parent_plan.get(split)or{}).get('base')
        if old_base is not None and old_base!=data.base.identity:
            raise ValueError('Selected parent and new experiment paired '+split+' corpus differ')
    old_panel=parent_plan.get('panel')
    if args.validation_frames==256 and isinstance(old_panel,list)and len(old_panel)==256:
        if panel!=old_panel:raise ValueError('Selected parent and new experiment256 held-out indices differ')
    cache=train.base.manifest.get('cache_identity',{})
    if(cache.get('future_rgb_features')is not False or cache.get('paired_same_pose_map_tick')is not True
            or cache.get('restorer_present')is not False or cache.get('solver_steps')!=3):
        raise ValueError('Need the final RF3/Q/decoder causal paired cache without a restorer')
    plan=dict(kind=PROTOCOL,version=1,parent_checkpoint=str(parent_path),parent_checkpoint_sha256=parent_sha,
        parent_training_steps=parent['training_steps'],parent_model_sha256=migration['source_model_sha256'],
        config=deepcopy(model.config),migration=deepcopy(migration),data_root=str(args.data_root.resolve()),
        train=deepcopy(train.identity),val=deepcopy(val.identity),panel=panel,panel_fingerprint=val.panel_fingerprint(panel),
        parent_panel_indices_equal=panel==old_panel,
        parent_paired_data_equal=all((parent_plan.get(split)or{}).get('base')is not None
            and(parent_plan.get(split)or{}).get('base')==data.base.identity
            for split,data in(('train',train),('val',val))),
        batch=args.batch,steps=args.steps,seed=args.seed,
        preview_every=args.prediction_every,validation_every=args.validate_every,validation_frames=len(panel),
        validation_batch=args.validation_batch,loss_weights=deepcopy(LOSS_WEIGHTS),
        frozen_background=True,new_decoder_only=True,optimizer='fresh_Adam_for_new_architecture',learning_rate=1e-4,
        hypothesis_sidecar_required=False,inference_inputs='actual_NPR_RGB32_only',
        Source_targets_are_labels_only=True,atlas_in_new_model=False,GAN=False,
        automatic_deployment=False,gameplay_qualified=False,budget_completion_is_quality=False,
        disposable_probe=bool(getattr(args,'probe',False)))
    protocol=getattr(args,'loss_protocol',PROTOCOL)
    weights=loss_weights(protocol)
    resume_path=getattr(args,'resume_checkpoint',None)
    if protocol!=PROTOCOL:
        if resume_path is None:raise ValueError('Corrected foreground phase requires its complete source checkpoint')
        resume_path=resume_path.resolve();resume_sha=file_sha(resume_path)
        source=torch.load(resume_path,map_location='cpu',weights_only=False)
        if file_sha(resume_path)!=resume_sha:raise ValueError('Restoration source changed while reading')
        binding=continuation_source(source,plan,model)
        restore_decoder_optimizer(model,source,step=binding['training_steps'])
        binding.update(checkpoint_path=str(resume_path),checkpoint_sha256=resume_sha)
        plan.update(kind=protocol,version=2,loss_weights=weights,resume_source=binding,
            phase_start_step=binding['training_steps'],phase_budget=args.steps,
            target_step=binding['training_steps']+args.steps,
            optimizer='continued_Adam_same_parameters',learning_rate=1e-4,
            old_metrics_preserved_in_source_run=True,new_loss_curve_separate=True,
            background_Source_reconstruction_gradient=False,hard_negative_alpha=True,
            foreground_topk_pixels=FOREGROUND_TOPK_PIXELS,
            edge_ssim_background_context='same_frozen_display_background_both_sides',
            preservation_target='detached_frozen_display_background')
        if protocol==DETAIL_PROTOCOL:
            plan.update(version=3,detail_loss_settings=deepcopy(DETAIL_SETTINGS),
                detail_animation_panel=build_detail_animation_panel(val,seed=args.seed),
                detail_selection_policy=deepcopy(DETAIL_SELECTION_POLICY),
                background_ablation='matched_current_frozen_display_background_no_extra_forward')
    elif resume_path is not None:
        raise ValueError('A loss-only continuation must select the corrected foreground objective')
    root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    immutable_json(root/'training_plan.json',plan);immutable_json(root/'migration.json',migration)
    return plan,parent,model


def restore_decoder_optimizer(model,payload,*,step):
    names=[name for name,p in model.named_parameters()if p.requires_grad]
    optimizer=torch.optim.Adam([p for p in model.parameters()if p.requires_grad],lr=1e-4)
    if payload is None:return optimizer,names
    if payload.get('optimizer_parameter_names')!=names:raise ValueError('New decoder optimizer names changed')
    expected=[{key:value for key,value in group.items()if key!='params'}for group in optimizer.param_groups]
    optimizer.load_state_dict(payload['optimizer'])
    actual=[{key:value for key,value in group.items()if key!='params'}for group in optimizer.param_groups]
    if actual!=expected:raise ValueError('New decoder Adam hyperparameters changed')
    if any(float(state.get('step',-1))!=step for state in optimizer.state.values()):
        raise ValueError('New decoder Adam step does not match its completed local update')
    if len(optimizer.state)!=(len(names)if step else 0):raise ValueError('New decoder lost its exact fresh/completed Adam states')
    for parameter,state in optimizer.state.items():
        if(set(state)!=set(('step','exp_avg','exp_avg_sq'))or any(
                not torch.is_tensor(state[name])or state[name].shape!=parameter.shape
                or not bool(torch.isfinite(state[name]).all())for name in('exp_avg','exp_avg_sq'))
                or bool((state['exp_avg_sq']<0).any())):
            raise ValueError('New decoder Adam moment shape or values changed')
    return optimizer,names


def validate_resume_sampler(train,*,step,batch):
    if(type(step)is not int or step<0 or type(batch)is not int or batch<1
            or type(getattr(train,'draws',None))is not int or train.draws!=step*batch):
        raise ValueError('Restoration private sampler draws differ from completed update exposure')


def _rng(payload=None,*,seed,device):
    if payload is None:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    else:
        random.setstate(payload['rng_python']);np.random.set_state(payload['rng_numpy'])
        torch.set_rng_state(payload['rng_torch'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([value.cpu()for value in payload['rng_cuda']])


def run(args):
    device=initialize_runtime(args.device)
    plan,parent,model=prepare_experiment(args);root=args.run_dir.resolve();plan_sha=file_sha(root/'training_plan.json')
    if args.prepare_only:return dict(plan_sha256=plan_sha,prepared=True,model=neural_restoration_metadata(model))
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train=NeuralEnemyRestorationData(args.data_root,'train',seed=args.seed,device=device)
    val=NeuralEnemyRestorationData(args.data_root,'val',seed=args.seed+1,device=device)
    if train.identity!=plan['train']or val.identity!=plan['val']:raise ValueError('Paired cache changed after preparation')
    protocol=plan['kind'];weights=plan['loss_weights']
    objective=partial(neural_enemy_restoration_loss,protocol=protocol)
    source_step=plan.get('phase_start_step',0);target_step=plan.get('target_step',args.steps)
    panel=plan['panel'];last=root/'checkpoints/sr_last.pt';saved=None;step=source_step;best=None;visual_best=None
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if(saved.get('training_plan')!=plan or saved.get('plan_sha256')!=plan_sha
                or saved.get('config')!=model.config or type(saved.get('training_steps'))is not int
                or not source_step<=saved['training_steps']<=target_step):raise ValueError('Standalone restoration resume identity changed')
        step=saved['training_steps'];model.load_state_dict(saved['model'],strict=True)
        train.restore_sampler(saved['sampler']);best=deepcopy(saved.get('best'))
        visual_best=deepcopy(saved.get('visual_best'))
    elif 'resume_source'in plan:
        pin=plan['resume_source'];source_path=Path(pin['checkpoint_path'])
        if file_sha(source_path)!=pin['checkpoint_sha256']:raise ValueError('Pinned restoration source bytes changed')
        saved=torch.load(source_path,map_location='cpu',weights_only=False)
        validate_continuation_checkpoint(saved,plan)
        model.load_state_dict(saved['model'],strict=True);train.restore_sampler(saved['sampler'])
    validate_resume_sampler(train,step=step,batch=args.batch)
    model.to(device).float().configure_training()
    fixed=plan['migration']['copied_background_sha256']
    if _hash_tensors(frozen_background_state(model))!=fixed:raise ValueError('Standalone model frozen background changed')
    optimizer,names=restore_decoder_optimizer(model,saved,step=step)
    # Atlas is constructed only for this independent reference arm. It is never
    # part of the new model, its optimizer, saved standalone weights or inputs.
    if(not parent['config'].get('atlas_buffer_shape')or 'atlas_frames_rgba'not in parent['model']):
        raise ValueError('Validation baseline needs its embedded full atlas checkpoint')
    with torch.random.fork_rng(devices=[]):reference=build_upscaler(parent['config'])
    reference.load_state_dict(parent['model'],strict=True);reference.to(device).float().eval().requires_grad_(False)
    del parent
    _rng(saved,seed=args.seed,device=device);started=time.perf_counter()
    training=dict(protocol=protocol,plan_sha256=plan_sha,parent_checkpoint_sha256=plan['parent_checkpoint_sha256'],
        parent_training_steps=plan['parent_training_steps'],local_budget=args.steps,new_decoder_only=True,
        frozen_background=True,new_optimizer='resume_source'not in plan,atlas_in_new_model=False,hypothesis_used=False,
        input_size=32,output_size=128,parameters=neural_restoration_metadata(model),
        data_identity_sha256=fingerprint(val.identity),validation_panel_fingerprint=plan['panel_fingerprint'])
    status=dict(run_id=root.name,stage='sr',status='initializing',model='Frozen RGB background + learned full-frame enemy RGB/alpha',
        objective_protocol=protocol,loss_protocol=protocol,current=step,total=target_step,started_at=now(),
        sr_training=dict(input_size=32,output_size=128,scale=4,all_other_models_frozen=True),
        prediction_every=args.prediction_every,validation_every=args.validate_every,gameplay_qualified=False,
        automatic_deployment=False,disposable_probe=plan['disposable_probe'],checkpoint=str(last))
    if 'resume_source'in plan:
        training.update(source_step=source_step,phase_updates=step-source_step,phase_budget=args.steps,
            total_target_step=target_step,source_checkpoint_sha256=plan['resume_source']['checkpoint_sha256'],
            source_plan_sha256=plan['resume_source']['plan_sha256'],optimizer_reset=False,
            optimizer=plan['optimizer'],old_loss_curve_separate=True)
    detail_kwargs={}
    if protocol==DETAIL_PROTOCOL:
        detail=plan['detail_animation_panel'];detail_kwargs=dict(detail_panel=detail)
        training.update(detail_animation_panel_fingerprint=detail['fingerprint'],
            detail_animation_panel_frames=detail['frames'],detail_loss_settings=plan['detail_loss_settings'],
            visual_selection_policy=plan['detail_selection_policy'])
    def publish(**values):
        training['local_step']=step
        if 'resume_source'in plan:training['phase_updates']=step-source_step
        status.update(values,current=step,neural_enemy_restoration_training=training,
            updated_at=now(),elapsed_seconds=time.perf_counter()-started);atomic_json(root/'status.json',status)
    def save():
        _save(last,dict(stage='sr',config=deepcopy(model.config),model=model.state_dict(),
            training_steps=step,neural_enemy_restoration_updates=step,training_plan=plan,plan_sha256=plan_sha,
            loss_protocol=protocol,loss_revision_updates=step-source_step,
            parent_training_steps=plan['parent_training_steps'],migration=plan['migration'],
            optimizer=optimizer.state_dict(),optimizer_parameter_names=names,sampler=train.sampler_state(),
            best=deepcopy(best),visual_best=deepcopy(visual_best),rng_python=random.getstate(),rng_numpy=np.random.get_state(),
            rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()if device.type=='cuda'else[],
            standalone_RGB_only=True,atlas_in_new_model=False,automatic_deployment=False,gameplay_qualified=False,
            disposable_probe=plan['disposable_probe']))
    def attach(report,preview):
        pins=dict(plan_sha256=plan_sha,parent_checkpoint_sha256=plan['parent_checkpoint_sha256'],
            frozen_background_sha256=fixed,model_sha256=_hash_tensors(model.state_dict()))
        report.update(pins);preview['metadata'].update(pins)
    def validate(baseline_report=None):
        report,preview=evaluate_neural_enemy_restoration(model,reference,val,panel,step=step,
            batch_size=args.validation_batch,baseline_report=baseline_report,objective_protocol=protocol,
            objective_fn=objective,objective_weights=weights,**detail_kwargs)
        attach(report,preview);atomic_json(root/'validation'/f'step_{step:06d}.json',report)
        atomic_json(root/'validation.json',report)
        publish(neural_enemy_restoration_validation=report,
            prediction_preview=save_neural_enemy_restoration_preview(preview,root/'preview.png',every=args.prediction_every))
        return report
    def select_visual(report,baseline_report):
        nonlocal visual_best
        if protocol!=DETAIL_PROTOCOL:return
        candidate=detail_visual_candidate(report,baseline_report)
        publish(detail_visual_candidate=candidate)
        if candidate['eligible']and(visual_best is None or candidate['score']>visual_best['score']):
            candidate.update(plan_sha256=plan_sha,model_sha256=report['model_sha256'])
            visual_best=candidate;save()
            _save(root/'checkpoints/sr_visual_best.pt',torch.load(last,map_location='cpu',weights_only=False))
            atomic_json(root/'visual_best.json',visual_best)
    baseline_path=root/'validation_initial.json'
    initial=root/'checkpoints/sr_initial.pt'
    if 'resume_source'in plan and not initial.exists():
        if step!=source_step:raise ValueError('Corrected restoration lost its exact initial source state')
        save();_save(initial,torch.load(last,map_location='cpu',weights_only=False))
    if baseline_path.exists():
        baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
        if(baseline.get('plan_sha256')!=plan_sha or baseline.get('panel_fingerprint')!=plan['panel_fingerprint']
                or baseline.get('objective_protocol')!=protocol):
            raise ValueError('Initial restoration validation pin changed')
    else:
        if step!=source_step:raise ValueError('Restoration resume lost its initial held-out baseline')
        publish(stage_label='Initial same-panel RGB-only restoration versus frozen Atlas baseline')
        baseline=validate();atomic_json(baseline_path,baseline)
    if 'resume_source'in plan and best is None:
        if step!=source_step:raise ValueError('Corrected restoration lost its same-objective best ledger')
        best=dict(step=step,loss=baseline['validation_objective']['loss'],objective_protocol=protocol,
            panel_fingerprint=plan['panel_fingerprint'],automatic_deployment=False,initial_baseline=True)
        save();_save(root/'checkpoints/sr_best.pt',torch.load(last,map_location='cpu',weights_only=False))
        atomic_json(root/'best.json',best)
    if protocol==DETAIL_PROTOCOL and visual_best is None:
        if step!=source_step:raise ValueError('Detail phase lost its physical selection baseline')
        select_visual(baseline,baseline)
    publish(status='running',stage_label='RGB-only enemy restoration: training learned RGB/alpha decoder');save()
    while step<target_step:
        if(root/'stop_training').exists():save();publish(status='stopped');return
        begin=time.perf_counter();model.configure_training();batch=train.batch(args.batch)
        optimizer.zero_grad(set_to_none=True);output=model.forward_with_aux(batch['lr_npr'].detach())
        loss,metrics=objective(output,batch);loss.backward()
        metrics['grad_norm']=float(clip_finite_grad_norm_(model.named_parameters(),context='Neural enemy restoration'))
        optimizer.step();step+=1
        if _hash_tensors(frozen_background_state(model))!=fixed:raise AssertionError('Restoration changed the frozen background')
        metrics.update(step=step,stage='sr',objective_protocol=protocol,loss_protocol=protocol,
            neural_enemy_restoration_plan_sha256=plan_sha,timestamp=now(),
            source_row_indices=batch['row_index'].detach().cpu().tolist(),optimizer_update_seconds=time.perf_counter()-begin,
            frozen_background_bitexact=True,atlas_used=False,hypothesis_used=False)
        if 'resume_source'in plan:metrics.update(phase_updates=step-source_step,source_step=source_step)
        save()  # Full atomic model/Adam/sampler/RNG checkpoint at every completed update.
        if 'resume_source'in plan and step==source_step+1:
            _save(root/'checkpoints/sr_first.pt',torch.load(last,map_location='cpu',weights_only=False))
        with(root/'metrics.jsonl').open('a',encoding='utf-8')as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
        publish(loss=float(loss.detach()),latest_metrics=metrics,
            frozen_verification=dict(passed=True,step=step,frozen_background_sha256=fixed))
        if(step-source_step)%args.validate_every==0 or step==target_step:
            report=validate(baseline);score=report['validation_objective']['loss']
            select_visual(report,baseline)
            if best is None or score<best['loss']:
                best=dict(step=step,loss=score,panel_fingerprint=plan['panel_fingerprint'],automatic_deployment=False,
                    objective_protocol=protocol)
                save();payload=torch.load(last,map_location='cpu',weights_only=False);_save(root/'checkpoints/sr_best.pt',payload)
                atomic_json(root/'best.json',best)
            save()
        elif(step-source_step)%args.prediction_every==0:
            preview=preview_neural_enemy_restoration(model,reference,val,panel,step=step,
                baseline_report=baseline,batch_size=args.validation_batch,objective_protocol=protocol,
                **detail_kwargs);attach({},preview)
            publish(prediction_preview=save_neural_enemy_restoration_preview(preview,root/'preview.png',every=args.prediction_every))
    save();publish(status='completed',stage_label='RGB-only enemy restoration budget completed; quality assessed separately')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('run-dir','parent-checkpoint','data-root'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=3000);parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--seed',type=int,default=173);parser.add_argument('--validation-frames',type=int,default=256)
    parser.add_argument('--validation-batch',type=int,default=8);parser.add_argument('--validate-every',type=int,default=100)
    parser.add_argument('--prediction-every',type=int,default=15);parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume-checkpoint',type=Path,help='Complete RGB restoration source for a separate loss-only continuation')
    parser.add_argument('--loss-protocol',choices=(PROTOCOL,FOREGROUND_PROTOCOL,DETAIL_PROTOCOL),default=PROTOCOL)
    parser.add_argument('--probe',action='store_true',help='Three discarded steps in an isolated run; never promote these weights')
    args=parser.parse_args(argv)
    if args.probe:args.steps=3
    if min(args.steps,args.batch,args.validation_frames,args.validation_batch,args.validate_every,args.prediction_every)<1:
        parser.error('Positive budget, batch and reporting intervals required')
    initialize_runtime(args.device)  # Windows: must precede KeepAwake, run and checkpoint preparation.
    if args.prepare_only:
        print(json.dumps(run(args),allow_nan=False));return
    args.run_dir.mkdir(parents=True,exist_ok=True)
    from .vqgan_pipeline import process_alive
    pid=args.run_dir/'training.pid'
    if pid.exists()and process_alive(int(pid.read_text())):parser.error('Restoration worker is already active')
    pid.write_text(str(os.getpid()),encoding='ascii')
    try:
        with KeepAwake():run(args)
    except BaseException as error:
        path=args.run_dir/'status.json';old=json.loads(path.read_text(encoding='utf-8'))if path.exists()else{}
        atomic_json(path,{**old,'status':'failed','error':f'{type(error).__name__}: {error}','updated_at':now()});raise


if __name__=='__main__':main()
