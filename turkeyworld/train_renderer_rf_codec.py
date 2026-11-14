"""RF3/Q/frozen-VQGAN training without a restorer, until held-out SSIM>.85."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import os
import random
import time

import numpy as np
import torch

from .distill_renderer import _digest
from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_combat_data import CombatRendererFrames
from .renderer_small_enemy_data import SmallEnemyBalancedFrames
from .renderer_far_wall_data import FarWallFrames
from .renderer_geometry_data import GeometryWallFrames
from .renderer_moe_five_objectives import FiveProfiledFrames
from .renderer_moe_five_phase import original_renderer_configuration
from .renderer_native_phase import _same,restore_renderer_native_rng
from .renderer_student import load_renderer_student,_hash_tensors
from .renderer_rf_codec_phase import (PROTOCOL,configure_rf_codec_training,adopt_rf_codec_optimizer,
    rf_codec_frozen_fingerprint,rf_codec_parameter_counts)
from .renderer_rf_codec_objectives import rf_codec_refinement_loss,renderer_rf_codec_protocol
from .renderer_rf_codec_validation import evaluate_rf_codec,save_rf_codec_preview
from .renderer_rf_codec_bundle import export_rf_codec_renderer
from .renderer_rf_codec_goal import (rf_codec_goal_spec,evaluate_rf_codec_goal,
    update_rf_codec_best,validate_saved_rf_codec_goal)
from .train_renderer_rf5 import durable_save
from .training_pipeline import atomic_json,now


class RFCodecTraining:
    def __init__(self,args):
        self.args,self.root=args,args.run_dir
        self.root.mkdir(parents=True,exist_ok=True)
        self.started=time.perf_counter()
        self.state=dict(run_id=self.root.name,status='initializing',stage='renderer_rf_codec_refinement',
            loss_protocol=PROTOCOL,current=0,global_training_step=0,total=None,stop_on_ssim=True,
            image_size=32,prediction_every=10,validation_every=100,gameplay_qualified=False,
            model='RF3 → terminal Q → frozen VQGAN; no restorer')
        self.step,self.best,self.goal_history,self.goal_met=0,None,[],False

    def publish(self,**values):
        self.state.update(values,updated_at=now(),elapsed_seconds=time.perf_counter()-self.started)
        atomic_json(self.root/'status.json',self.state)

    def prepare(self):
        a=self.args
        if a.device=='cuda':torch.cuda.init()
        torch.set_num_threads(a.threads)
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.publish(stage_label='Loading RF30717 and frozen decoder1725; no restorer')
        self.parent=torch.load(a.parent_checkpoint,map_location='cpu',weights_only=False)
        recipe=torch.load(a.source_recipe_parent,map_location='cpu',weights_only=False)
        config=original_renderer_configuration(recipe['configuration'])
        self.source_world,meta=load_world_checkpoint(a.world,'cpu')
        if meta['training_steps']!=5906 or self.parent['renderer_training_steps']!=30717:
            raise ValueError('RF-codec requires the exact frozen feature-world/RF clocks')
        self.source_world.eval().requires_grad_(False)
        self.model=deepcopy(self.source_world)
        parent_info=load_renderer_student(self.model,a.renderer_checkpoint,allow_unqualified_codec=True)
        if parent_info.get('format_version')!=3 or parent_info.get('decoder_refinement_updates')!=1725:
            raise ValueError('RF-codec requires its exact current format3 decoder1725 parent')
        self.model.float().to(a.device).eval().requires_grad_(False)
        self.reference=deepcopy(self.model).eval().requires_grad_(False)
        configure_rf_codec_training(self.model)
        self.optimizer,self.adoption=adopt_rf_codec_optimizer(self.model,self.parent)
        self.frozen=rf_codec_frozen_fingerprint(self.model)
        self.reference_hash=_hash_tensors(self.reference.state_dict())
        self.codec_hash=_hash_tensors(self.model.flow_renderer.codec.state_dict())
        identity=self.parent['phase']['identity']['parent_identity']
        self.publish(stage_label='Restoring the unchanged146386 Source frames and wall samplers')
        raw=CombatRendererFrames(config['combat_data_root'],config['appearance_data_root'],'train',seed=a.seed)
        val=CombatRendererFrames(config['combat_data_root'],config['appearance_data_root'],'val',
                                seed=a.seed+1,exclude_scene_keys=raw.scene_keys)
        self.train=SmallEnemyBalancedFrames(raw)
        self.far=FarWallFrames(self.train,candidate_count=2048,seed=2718)
        self.geometry=GeometryWallFrames(self.train,candidate_count=8192,seed=31415)
        self.panel,self.geometry_panel=list(identity['panel']),list(identity['geometry_wall_panel'])
        observed=dict(train=raw.identity,val=val.identity,distance_sampling=self.train.identity['distance_sampling'],
            far_wall_aux_pool=self.far.identity,geometry_wall_aux_pool=self.geometry.identity,
            panel_fingerprint=val.panel_fingerprint(self.panel),geometry_wall_fingerprint=val.panel_fingerprint(self.geometry_panel))
        for key,value in observed.items():
            if not _same(identity[key],value):raise ValueError('RF-codec changed its immutable Source data: '+key)
        self.profiled=dict(main=FiveProfiledFrames(self.train),far=FiveProfiledFrames(self.far,self.train.rows),
            geometry=FiveProfiledFrames(self.geometry,self.train.rows),val=FiveProfiledFrames(val))
        self.preview_ids=[]
        for category in ('large','partial','distant','empty_navigation'):
            index=next((i for i in self.panel if val.rows[i]['category']==category),None)
            if index is not None and index not in self.preview_ids:self.preview_ids.append(index)
        self.preview_ids += [i for i in self.panel if i not in self.preview_ids][:4-len(self.preview_ids)]
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        self.phase=dict(protocol=renderer_rf_codec_protocol(),configuration=configuration,
            inputs={key:_digest(getattr(a,key)) for key in ('world','parent_checkpoint','renderer_checkpoint','source_recipe_parent')},
            adoption=self.adoption,source_train_identity=raw.identity,source_val_identity=val.identity,
            source_sampling_identity={k:observed[k] for k in ('distance_sampling','far_wall_aux_pool','geometry_wall_aux_pool')},
            panel_fingerprint=observed['panel_fingerprint'],geometry_panel_fingerprint=observed['geometry_wall_fingerprint'],
            preview_panel_fingerprint=val.panel_fingerprint(self.preview_ids),frozen_fingerprint=self.frozen,
            reference_model_sha256=self.reference_hash,codec_sha256=self.codec_hash,
            parent_RF_training_steps=30717,parent_RF_local_step=1108,base_training_steps=29209,
            frozen_decoder_updates=1725,goal=rf_codec_goal_spec(),hard_update_budget=None,
            conditional_restorer_parameters=0,automatic_active_promotion=False)
        phase_path=self.root/'phase_plan.json'
        if phase_path.exists():
            if json.loads(phase_path.read_text(encoding='utf-8'))!=self.phase:
                raise ValueError('Immutable RF-codec phase, objective, source or goal changed')
        else:atomic_json(phase_path,self.phase)
        self.plan_sha=_digest(phase_path)
        self.checkpoint=self.root/'checkpoints/renderer_last.pt'
        saved=self.parent
        if self.checkpoint.exists() and not a.probe:
            saved=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
            if (saved['phase']!=self.phase or saved['plan_sha256']!=self.plan_sha
                    or type(saved.get('step')) is not int or saved['step']<0
                    or saved.get('renderer_training_steps')!=30717+saved['step']
                    or saved['optimizer_parameter_names']!=self.adoption['optimizer_parameter_names']):
                raise ValueError('RF-codec resume changed its pinned phase, clock or Adam names')
            self.model.flow_renderer.load_state_dict(saved['flow'],strict=True)
            if not _same(saved['base'],self.model.base_student.state_dict()):raise ValueError('Resume changed frozen Base')
            self.optimizer.load_state_dict(deepcopy(saved['optimizer']))
            self.step=saved['step']
            self.best,self.goal_history,self.goal_met=validate_saved_rf_codec_goal(saved,
                panel_fingerprint=self.phase['panel_fingerprint'],plan_sha256=self.plan_sha,
                feature_world_sha256=self.phase['inputs']['world'],run_dir=self.root)
            if rf_codec_frozen_fingerprint(self.model)!=self.frozen:raise ValueError('Resume changed a frozen asset')
        for stream,key in ((self.train,'sampler'),(self.far,'far_wall_aux_sampler'),(self.geometry,'geometry_wall_aux_sampler')):
            stream.restore_sampler(saved[key])
            if not _same(stream.sampler_state(),saved[key]):raise AssertionError('RF-codec sampler restoration was not exact')
        restore_renderer_native_rng(saved,device=a.device)
        counts=rf_codec_parameter_counts(self.model)
        self.training=dict(protocol=PROTOCOL,plan_sha256=self.plan_sha,
            feature_world_sha256=self.phase['inputs']['world'],feature_world_step=5906,
            parent_decoder_sha256=self.phase['inputs']['renderer_checkpoint'],parent_RF_training_steps=30717,
            parent_RF_local_step=1108,base_training_steps=29209,frozen_decoder_updates=1725,solver_steps=3,
            local_step=self.step,local_budget=None,parameters=counts,codec_frozen=True,base_frozen=True,
            private_opacity_frozen=True,restorer_active=False,main_batch=16,far_aux_batch=4,geometry_aux_batch=8,
            main_stream_weight=1.,far_stream_weight=.25,geometry_stream_weight=1.,
            preview_panel_fingerprint=self.phase['preview_panel_fingerprint'])
        self.publish(rf_codec_training=self.training,current=self.step,global_training_step=self.step,
            renderer_training_steps=30717+self.step,base_training_steps=29209,
            validation_panel_fingerprint=self.phase['panel_fingerprint'],ssim_goal=self.goal_status(),
            renderer_training=dict(active=True,rf_active=True,base_active=False,codec_frozen=True,
                decoder_active=False,restorer_active=False,rssm_frozen=True,gan_active=False))

    def goal_status(self):
        latest=self.goal_history[-1] if self.goal_history else {}
        return dict(protocol='renderer_rf_codec_ssim_goal_v1',target=.85,comparison='strict_greater_than',
            metric='models.rf_new.summary.ssim_mean',frames=256,panel_fingerprint=self.phase['panel_fingerprint'],
            current_ssim=latest.get('ssim'),best_ssim=self.best.get('ssim') if self.best else None,
            best_step=self.best.get('step') if self.best else None,validation_step=latest.get('step'),goal_met=self.goal_met)

    def payload(self):
        return dict(base=self.model.base_student.state_dict(),flow=self.model.flow_renderer.state_dict(),
            optimizer=self.optimizer.state_dict(),optimizer_parameter_names=self.adoption['optimizer_parameter_names'],
            step=self.step,renderer_training_steps=30717+self.step,base_training_steps=29209,
            phase=self.phase,plan_sha256=self.plan_sha,frozen_fingerprint=self.frozen,
            sampler=self.train.sampler_state(),far_wall_aux_sampler=self.far.sampler_state(),
            geometry_wall_aux_sampler=self.geometry.sampler_state(),rng_torch=torch.get_rng_state(),
            rng_numpy=np.random.get_state(),rng_python=random.getstate(),
            rng_cuda=torch.cuda.get_rng_state_all() if self.args.device=='cuda' else [],
            best=self.best,goal_history=self.goal_history,goal_met=self.goal_met)

    def save(self):durable_save(self.checkpoint,self.payload())

    def update(self):
        configure_rf_codec_training(self.model)
        self.optimizer.zero_grad(set_to_none=True)
        records,total={},0.
        started=time.perf_counter()
        for kind,size,weight in (('main',16,1.),('far',4,.25),('geometry',8,1.)):
            value,metrics=rf_codec_refinement_loss(self.model,self.profiled[kind].batch(size),kind=kind,checkpoint_velocity=True)
            (weight*value).backward();total+=weight*float(value.detach());records[kind]=metrics
            del value
        deep={str(i):float(sum(p.grad.detach().float().abs().sum() for p in expert.private_transformer.parameters()
                              if p.grad is not None)) for i,expert in enumerate(self.model.flow_renderer.output_moe.experts)}
        norm=float(clip_finite_grad_norm_(self.model.named_parameters(),context='RF-codec'))
        self.optimizer.step();self.step+=1
        if self.step<=5 or self.step%100==0:
            if (rf_codec_frozen_fingerprint(self.model)!=self.frozen
                    or _hash_tensors(self.reference.state_dict())!=self.reference_hash):
                raise AssertionError('RF-codec update changed a frozen codec/Base/world/reference')
        return dict(protocol=PROTOCOL,loss_protocol=PROTOCOL,step=self.step,timestamp=now(),
            plan_sha256=self.plan_sha,feature_world_sha256=self.phase['inputs']['world'],solver_steps=3,
            train_loss=total,main_loss=records['main']['train_loss'],main=records['main'],
            far=records['far'],geometry=records['geometry'],gradient_norm=norm,
            private_transformer_gradient_abs=deep,optimizer_update_seconds=time.perf_counter()-started,
            renderer_training_steps=30717+self.step,base_training_steps=29209,restorer_active=False)

    def preview(self):
        _,arrays=evaluate_rf_codec(self.model,self.reference,self.profiled['val'],self.preview_ids,
                                   step=self.step,batch_size=4)
        meta=save_rf_codec_preview(arrays,self.root/'preview.png',self.step)
        meta.update(kind='renderer_rf_codec_GT_query',every=10,loss_protocol=PROTOCOL,
            plan_sha256=self.plan_sha,feature_world_sha256=self.phase['inputs']['world'],
            validation_panel_fingerprint=self.phase['panel_fingerprint'])
        atomic_json(self.root/'preview.json',meta)
        self.publish(prediction_preview=meta,preview_url='/preview.png')

    def export(self,report=None,path=None):
        return export_rf_codec_renderer(self.model,self.args.renderer_checkpoint,
            path or self.root/'checkpoints/renderer_for_play.pt',local_updates=self.step,
            source_model=self.source_world,metrics=dict(validation=report or {}),provenance=dict(plan_sha256=self.plan_sha))

    def validate(self,baseline):
        report,_=evaluate_rf_codec(self.model,self.reference,self.profiled['val'],self.panel,self.geometry_panel,
            step=self.step,batch_size=8,baseline=baseline)
        report.update(plan_sha256=self.plan_sha,feature_world_sha256=self.phase['inputs']['world'])
        pins=dict(panel_fingerprint=self.phase['panel_fingerprint'],plan_sha256=self.plan_sha,
                  feature_world_sha256=self.phase['inputs']['world'])
        decision=evaluate_rf_codec_goal(report,**pins)
        old_best=deepcopy(self.best)
        self.best=update_rf_codec_best(self.best,report,**pins)
        self.goal_met=decision['goal_met']
        self.goal_history.append(dict(step=self.step,ssim=decision['measured_ssim'],goal_met=self.goal_met))
        report['ssim_goal']=self.goal_status()
        path=self.root/'validation'/f'step_{self.step:06d}.json'
        atomic_json(path,report);atomic_json(self.root/'validation.json',report)
        digest=_digest(path);self.goal_history[-1]['report_sha256']=digest
        if self.best['improved']:
            self.best=update_rf_codec_best(old_best,report,report_sha256=digest,**pins)
            atomic_json(self.root/'best.json',self.best)
            durable_save(self.root/'checkpoints/renderer_best.pt',self.payload())
            self.export(report,self.root/'checkpoints/renderer_best_for_play.pt')
        self.save();self.export(report)
        row=dict(step=self.step,protocol=PROTOCOL,loss_protocol=PROTOCOL,plan_sha256=self.plan_sha,
            feature_world_sha256=self.phase['inputs']['world'],solver_steps=3,
            panel_fingerprint=self.phase['panel_fingerprint'],val_loss=report['val_loss'],
            validation_objective=report['validation_objective'],source_held_out=True,renderer_only=True,
            world_rollout=False,validation_objective_matches_main_training=True,restorer_active=False)
        with (self.root/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        self.publish(validation=report,rf_codec_validation=report,ssim_goal=self.goal_status())
        self.preview()
        if self.goal_met:atomic_json(self.root/'goal_reached.json',dict(step=self.step,decision=decision,best=self.best,timestamp=now()))
        return report

    def run(self):
        self.prepare()
        if self.args.probe:
            rows=[self.update() for _ in range(3)]
            if not all(row['main']['router_velocity_calls']==3 for row in rows):
                raise AssertionError('Missing actual Euler gradient calls')
            self.save();self.export()
            proof=dict(passed=True,discarded_optimizer_updates=3,production_training_updates=0,rows=rows,
                frozen_assets_exact=True,parameters=rf_codec_parameter_counts(self.model),restorer_active=False,
                seconds_per_update=sum(r['optimizer_update_seconds'] for r in rows)/3,
                peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30 if self.args.device=='cuda' else 0.)
            atomic_json(self.root/'gpu_warmstart.json',proof)
            self.publish(status='completed',message='Three discarded RF/Q/VQGAN updates passed')
            print(json.dumps({k:v for k,v in proof.items() if k!='rows'}),flush=True);return
        initial=self.root/'validation_initial.json'
        baseline=json.loads(initial.read_text(encoding='utf-8')) if initial.exists() else None
        if self.step and baseline is None:raise ValueError('Resume requires the original full-panel baseline')
        if baseline is None:
            self.publish(stage_label='Measuring RF3/Q/VQGAN before the first optimizer update')
            baseline=self.validate(None);atomic_json(initial,baseline)
        self.save()
        self.publish(status='running',stage_label='RF only: whole-frame SSIM + wall geometry; frozen VQGAN')
        while not self.goal_met:
            if (self.root/'stop_training').exists():self.save();self.publish(status='stopped');return
            row=self.update()
            with (self.root/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            self.training['local_step']=self.step
            self.save()
            self.publish(current=self.step,global_training_step=self.step,loss=row['train_loss'],latest_metrics=row,
                rf_codec_training=self.training,renderer_training_steps=30717+self.step)
            if self.step%100==0:self.validate(baseline)
            elif self.step%10==0:self.preview()
        self.save();self.publish(status='completed',message='RF-Q/VQGAN held-out SSIM is strictly above0.85',ssim_goal=self.goal_status())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','world','parent-checkpoint','renderer-checkpoint','source-recipe-parent'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=173)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda');p.add_argument('--probe',action='store_true')
    a=p.parse_args(argv)
    if a.threads<1:p.error('Positive thread count required')
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    from .vqgan_pipeline import process_alive
    a.run_dir.mkdir(parents=True,exist_ok=True);pid=a.run_dir/'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())):p.error('RF-codec worker is already active')
    pid.write_text(str(os.getpid()),encoding='ascii')
    trainer=RFCodecTraining(a)
    try:
        with KeepAwake():trainer.run()
    except BaseException as error:
        trainer.publish(status='failed',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
