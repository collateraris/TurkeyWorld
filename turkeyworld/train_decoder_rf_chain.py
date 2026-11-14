"""Exactly 3000 decoder/post-quant updates after the selected Base/RF3 stages."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .decoder_rf_adaptation_objectives import PROTOCOL,decoder_rf_adaptation_protocol
from .decoder_rf_adaptation_validation import evaluate_decoder_adaptation
from .decoder_rf_chain_best import decoder_chain_measurement,validate_saved_decoder_chain_best,METRIC
from .decoder_rf_chain_bundle import export_decoder_rf_chain
from .decoder_rf_chain_phase import (configure_decoder_rf_chain,adopt_decoder_rf_chain_optimizer,
    validate_decoder_rf_chain_optimizer)
from .distill_renderer import _digest
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
from .train_decoder_rf_adaptation import DecoderTraining,decoder_state,load_decoder_state,frozen_hash
from .train_renderer_rf5 import durable_save
from .training_pipeline import atomic_json,now


class DecoderRFChainTraining(DecoderTraining):
    def __init__(self,args):
        super().__init__(args)
        self.step,self.best,self.validation_history=0,None,[]
        self.state.update(stage='decoder_rf_chain_adaptation',
            model='D + post_quant_conv after selected expanded Base / RF3; no restorer')

    def prepare(self):
        a=self.args
        if a.device=='cuda':torch.cuda.init()
        torch.set_num_threads(a.threads)
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.publish(stage_label='Loading the selected complete Base/RF3 boundary and matching D Adam')
        self.generator=torch.load(a.generator_checkpoint,map_location='cpu',weights_only=False)
        self.generator_full=torch.load(a.generator_full_checkpoint,map_location='cpu',weights_only=False)
        decoder_parent=torch.load(a.decoder_parent,map_location='cpu',weights_only=False)
        recipe=torch.load(a.source_recipe_parent,map_location='cpu',weights_only=False)
        config=original_renderer_configuration(recipe['configuration'])
        self.source,meta=load_world_checkpoint(a.world,'cpu')
        if meta['training_steps']!=5906:raise ValueError('Decoder chain retains its frozen feature world5906')
        self.source.eval().requires_grad_(False);self.model=deepcopy(self.source)
        info=load_renderer_student(self.model,a.generator_checkpoint,allow_unqualified_codec=True)
        if (info.get('format_version')!=7 or self.generator['training_steps']!=self.generator_full.get('renderer_training_steps')
                or self.generator['rf_chain_local_updates']!=self.generator_full.get('step')
                or self.generator['base_training_steps']!=self.generator_full.get('base_training_steps')
                or self.generator['decoder_refinement_updates']!=decoder_parent.get('step')
                or not _same(self.model.flow_renderer.state_dict(),self.generator_full.get('flow'))
                or not _same(self.model.base_student.state_dict(),self.generator_full.get('base'))):
            raise ValueError('Decoder chain requires the exact selected RF bundle, full checkpoint and D Adam parent')
        self.model.float().to(a.device).eval().requires_grad_(False)
        self.reference=deepcopy(self.model.flow_renderer.codec).eval().requires_grad_(False)
        policy=configure_decoder_rf_chain(self.model)
        if (policy['trainable_parameters'],policy['trainable_parameter_tensors'])!=(528403,62):
            raise ValueError('Decoder chain may train only its62 D/post_quant tensors')
        self.optimizer,self.adoption=adopt_decoder_rf_chain_optimizer(self.model,self.generator_full,decoder_parent,lr=a.lr)
        self.names=self.adoption['optimizer_parameter_names']
        self.frozen=frozen_hash(self.model);self.reference_hash=_hash_tensors(self.reference.state_dict())
        self.publish(stage_label='Restoring the three unchanged Source sampler streams')
        parent_phase=self.generator_full['phase'];identity=parent_phase['source_parent_identity']
        raw=CombatRendererFrames(config['combat_data_root'],config['appearance_data_root'],'train',seed=a.seed)
        val=CombatRendererFrames(config['combat_data_root'],config['appearance_data_root'],'val',seed=a.seed+1,exclude_scene_keys=raw.scene_keys)
        self.train=SmallEnemyBalancedFrames(raw)
        self.far=FarWallFrames(self.train,candidate_count=2048,seed=2718)
        self.geometry=GeometryWallFrames(self.train,candidate_count=8192,seed=31415)
        self.panel=list(parent_phase['panel']);self.geometry_panel=list(parent_phase['geometry_panel'])
        observed=dict(distance_sampling=self.train.identity['distance_sampling'],far_wall_aux_pool=self.far.identity,
            geometry_wall_aux_pool=self.geometry.identity)
        checks=dict(source_train_identity=raw.identity,source_val_identity=val.identity,source_sampling_identity=observed,
            panel_fingerprint=val.panel_fingerprint(self.panel),geometry_panel_fingerprint=val.panel_fingerprint(self.geometry_panel))
        if len(self.panel)!=256 or len(self.geometry_panel)!=128 or any(not _same(parent_phase.get(k),v)for k,v in checks.items()):
            raise ValueError('Decoder chain changed its Source identity, samplers or fixed256/geometry128 panels')
        if (not _same(identity['train'],raw.identity)or not _same(identity['val'],val.identity)
                or any(not _same(identity[k],v)for k,v in observed.items())):
            raise ValueError('Decoder chain differs from the original Source recipe identity')
        self.data,self.validation_data=FiveProfiledFrames(self.train),FiveProfiledFrames(val)
        self.preview_ids=[]
        for category in('large','partial','distant','empty_navigation'):
            index=next((i for i in self.panel if val.rows[i]['category']==category),None)
            if index is not None and index not in self.preview_ids:self.preview_ids.append(index)
        self.preview_ids += [i for i in self.panel if i not in self.preview_ids][:4-len(self.preview_ids)]
        configuration={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()}
        self.phase=dict(kind='decoder_after_selected_Base_RF3_v1',protocol=decoder_rf_adaptation_protocol(),
            configuration=configuration,inputs={k:_digest(getattr(a,k))for k in('world','generator_checkpoint',
                'generator_full_checkpoint','decoder_parent','source_recipe_parent')},adoption=self.adoption,
            generator_training_steps=self.generator['training_steps'],generator_local_step=self.generator['rf_chain_local_updates'],
            base_training_steps=self.generator['base_training_steps'],decoder_parent_updates=decoder_parent['step'],
            feature_world_step=5906,source_parent_identity=identity,**checks,panel=self.panel,geometry_panel=self.geometry_panel,
            preview_panel_fingerprint=val.panel_fingerprint(self.preview_ids),frozen_fingerprint=self.frozen,
            reference_codec_sha256=self.reference_hash,decoder_parameter_names=self.names[-1],restorer_present=False,
            fixed_new_optimizer_updates=a.steps,selection_metric=METRIC,automatic_promotion=False)
        phase_path=self.root/'phase_plan.json'
        if phase_path.exists():
            if json.loads(phase_path.read_text(encoding='utf-8'))!=self.phase:raise ValueError('Immutable decoder chain inputs changed')
        else:atomic_json(phase_path,self.phase)
        self.plan_sha=_digest(phase_path);self.checkpoint=self.root/'checkpoints/decoder_last.pt'
        saved=self.generator_full
        if self.checkpoint.exists()and not a.probe:
            saved=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
            if (saved.get('phase')!=self.phase or saved.get('plan_sha256')!=self.plan_sha
                    or type(saved.get('step'))is not int or not 0<=saved['step']<=a.steps
                    or saved.get('optimizer_parameter_names')!=self.names
                    or saved.get('renderer_training_steps')!=self.generator['training_steps']
                    or saved.get('base_training_steps')!=self.generator['base_training_steps']
                    or saved.get('decoder_refinement_updates')!=decoder_parent['step']+saved['step']
                    or not _same(saved.get('base'),self.model.base_student.state_dict())):
                raise ValueError('Decoder continuation changed its frozen parent, phase, optimizer or clocks')
            load_decoder_state(self.model,saved['decoder'])
            if not _same(saved.get('flow'),self.model.flow_renderer.state_dict()):raise ValueError('Decoder resume changed frozen RF/E/Q/private weights')
            validate_decoder_rf_chain_optimizer(self.model,saved['optimizer'],self.generator_full,decoder_parent,local_updates=saved['step'])
            self.optimizer.load_state_dict(deepcopy(saved['optimizer']));self.step=saved['step']
            if not _same(self.optimizer.state_dict(),saved['optimizer']):raise AssertionError('Decoder resume changed an Adam state or option')
            self.best,self.validation_history=validate_saved_decoder_chain_best(saved,panel_fingerprint=self.phase['panel_fingerprint'],
                plan_sha256=self.plan_sha,feature_world_sha256=self.phase['inputs']['world'],run_dir=self.root)
            if frozen_hash(self.model)!=self.frozen:raise ValueError('Decoder continuation changed a frozen asset')
        for stream,key in((self.train,'sampler'),(self.far,'far_wall_aux_sampler'),(self.geometry,'geometry_wall_aux_sampler')):
            stream.restore_sampler(saved[key])
            if not _same(stream.sampler_state(),saved[key]):raise AssertionError('Decoder chain sampler restoration was not exact')
        restore_renderer_native_rng(saved,device=a.device)
        self.training=dict(protocol=PROTOCOL,plan_sha256=self.plan_sha,feature_world_sha256=self.phase['inputs']['world'],
            feature_world_step=5906,frozen_RF_training_steps=self.generator['training_steps'],
            frozen_RF_local_step=self.generator['rf_chain_local_updates'],base_training_steps=self.generator['base_training_steps'],
            decoder_parent_updates=decoder_parent['step'],solver_steps=3,decoder_parameters=528403,decoder_parameter_tensors=62,
            rf_weight=.5,source_weight=.5,source_queries_per_update=a.batch,decoder_examples_per_update=2*a.batch,
            local_step=self.step,local_budget=a.steps,encoder_frozen=True,codebook_frozen=True,RF_frozen=True,Base_frozen=True,
            private_opacity_frozen=True,restorer_present=False,post_quant_conv_trainable=True,
            preview_panel_fingerprint=self.phase['preview_panel_fingerprint'],selection_metric=METRIC,
            inherited_RF_Adam_states=self.adoption['old_RF_Adam_states'],inherited_decoder_Adam_states=62,optimizer_groups=8)
        self.publish(decoder_adaptation_training=self.training,decoder_rf_chain_training=self.training,current=self.step,
            global_training_step=self.step,renderer_training_steps=self.generator['training_steps'],
            base_training_steps=self.generator['base_training_steps'],decoder_refinement_updates=decoder_parent['step']+self.step,
            validation_panel_fingerprint=self.phase['panel_fingerprint'],
            renderer_training=dict(active=True,rf_active=False,base_active=False,decoder_active=True,
                encoder_frozen=True,codebook_frozen=True,restorer_active=False,gan_active=False))

    def payload(self):
        return dict(base=self.model.base_student.state_dict(),base_config=asdict(self.model.base_student.config),
            flow=self.model.flow_renderer.state_dict(),decoder=decoder_state(self.model),optimizer=self.optimizer.state_dict(),
            optimizer_parameter_names=self.names,step=self.step,phase=self.phase,plan_sha256=self.plan_sha,
            renderer_training_steps=self.generator['training_steps'],base_training_steps=self.generator['base_training_steps'],
            decoder_refinement_updates=self.phase['decoder_parent_updates']+self.step,frozen_fingerprint=self.frozen,
            best=self.best,validation_history=deepcopy(self.validation_history),sampler=self.train.sampler_state(),
            far_wall_aux_sampler=self.far.sampler_state(),geometry_wall_aux_sampler=self.geometry.sampler_state(),
            rng_torch=torch.get_rng_state(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),
            rng_cuda=torch.cuda.get_rng_state_all()if self.args.device=='cuda'else [])

    def save(self):durable_save(self.checkpoint,self.payload())

    def update(self):
        row=super().update()
        row.update(decoder_local_updates=self.step,decoder_refinement_updates=self.phase['decoder_parent_updates']+self.step,
            renderer_training_steps=self.generator['training_steps'],base_training_steps=self.generator['base_training_steps'],solver_steps=3)
        return row

    def preview(self,arrays=None):
        super().preview(arrays)
        meta=self.state['prediction_preview'];meta.update(kind='decoder_rf_chain_GT_query',restorer_present=False,
            renderer_training_steps=self.generator['training_steps'],base_training_steps=self.generator['base_training_steps'],
            decoder_refinement_updates=self.phase['decoder_parent_updates']+self.step)
        atomic_json(self.root/'preview.json',meta);self.publish(prediction_preview=meta)

    def export(self,report=None,path=None):
        return export_decoder_rf_chain(self.model,self.args.generator_checkpoint,
            path or self.root/'checkpoints/renderer_for_play.pt',local_updates=self.step,source_model=self.source,
            metrics=dict(validation=report or {}),provenance=dict(plan_sha256=self.plan_sha))

    def validate(self,baseline):
        report,_=evaluate_decoder_adaptation(self.model,self.reference,self.validation_data,self.panel,
            step=self.step,batch_size=8,geometry_panel=self.geometry_panel,baseline=baseline)
        report.update(plan_sha256=self.plan_sha,feature_world_sha256=self.phase['inputs']['world'],
            restorer_present=False,renderer_training_steps=self.generator['training_steps'],
            base_training_steps=self.generator['base_training_steps'],decoder_refinement_updates=self.phase['decoder_parent_updates']+self.step)
        measured=decoder_chain_measurement(report,panel_fingerprint=self.phase['panel_fingerprint'],plan_sha256=self.plan_sha,
            feature_world_sha256=self.phase['inputs']['world'])
        path=self.root/'validation'/f'step_{self.step:06d}.json';atomic_json(path,report);atomic_json(self.root/'validation.json',report)
        sha=_digest(path);score=measured['ssim'];improved=self.best is None or score>self.best['ssim']
        self.validation_history.append(dict(step=self.step,ssim=score,report_sha256=sha))
        if improved:
            self.best={k:v for k,v in measured.items()if k not in('data_identity_sha256','reference_codec_sha256')}
            self.best['report_sha256']=sha;atomic_json(self.root/'best.json',self.best)
            durable_save(self.root/'checkpoints/decoder_best.pt',self.payload())
            self.export(report,self.root/'checkpoints/decoder_best_for_preview.pt')
        self.save();self.export(report)
        row=dict(step=self.step,protocol=PROTOCOL,loss_protocol=PROTOCOL,plan_sha256=self.plan_sha,
            feature_world_sha256=self.phase['inputs']['world'],solver_steps=3,panel_fingerprint=self.phase['panel_fingerprint'],
            val_loss=report['val_loss'],validation_objective=report['validation_objective'],source_held_out=True,
            renderer_only=True,world_rollout=False,restorer_present=False)
        with(self.root/'metrics.jsonl').open('a',encoding='utf-8')as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        self.publish(validation=report,decoder_adaptation_validation=report,decoder_rf_chain_validation=report,best_decoder=self.best)
        self.preview();return report

    def run(self):
        self.prepare()
        if self.args.probe:
            rows=[self.update()for _ in range(3)];self.save();self.export()
            proof=dict(passed=True,discarded_optimizer_updates=3,production_training_updates=0,rows=rows,
                frozen_assets_exact=True,trainable_parameters=528403,trainable_tensors=62,
                optimizer_adoption=self.adoption,seconds_per_update=sum(r['optimizer_update_seconds']for r in rows)/3,
                peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30 if self.args.device=='cuda'else 0.)
            atomic_json(self.root/'gpu_warmstart.json',proof)
            self.publish(status='completed',message='Three discarded decoder-chain updates passed')
            print(json.dumps(dict(probe_passed=True,report=str(self.root/'gpu_warmstart.json'))),flush=True);return
        initial=self.root/'validation_initial.json'
        baseline=None
        if self.validation_history:
            first=self.validation_history[0]
            source=self.root/'validation'/f"step_{first['step']:06d}.json"
            if first['step']!=0 or _digest(source)!=first['report_sha256']:
                raise ValueError('Decoder chain lost its SHA-pinned initial measurement')
            if initial.exists()and _digest(initial)!=first['report_sha256']:
                raise ValueError('Decoder initial baseline differs from its preserved validation0')
            baseline=json.loads(source.read_text(encoding='utf-8'))
            if not initial.exists():atomic_json(initial,baseline)
        elif initial.exists():raise ValueError('Decoder chain cannot reuse an untracked initial baseline')
        if self.step and baseline is None:raise ValueError('Decoder chain resume requires its original baseline')
        if baseline is None:
            self.publish(stage_label='Measuring both unchanged decoders before the first new update')
            baseline=self.validate(None);atomic_json(initial,baseline)
        self.save();self.publish(status='running',stage_label='D/post-quant only: paired RF3/Source-Q 50/50; no restorer')
        while self.step<self.args.steps:
            if(self.root/'stop_training').exists():self.save();self.publish(status='stopped');return
            row=self.update()
            with(self.root/'metrics.jsonl').open('a',encoding='utf-8')as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            self.training['local_step']=self.step;self.save()
            self.publish(current=self.step,global_training_step=self.step,loss=row['train_loss'],latest_metrics=row,
                decoder_adaptation_training=self.training,decoder_rf_chain_training=self.training,
                decoder_refinement_updates=self.phase['decoder_parent_updates']+self.step)
            if self.step%100==0:self.validate(baseline)
            elif self.step%10==0:self.preview()
        if not self.validation_history or self.validation_history[-1]['step']!=self.step:self.validate(baseline)
        self.save();self.publish(status='completed',message='3000 decoder updates completed; quality and Source preservation are reported separately')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in('run-dir','world','generator-checkpoint','generator-full-checkpoint','decoder-parent','source-recipe-parent'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=173);p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--probe',action='store_true');a=p.parse_args(argv)
    if a.steps!=3000 or a.batch!=8 or not np.isfinite(a.lr)or a.lr<=0 or a.threads<1:
        p.error('The chain requires exactly3000 new updates, B8 paired queries and positive LR/threads')
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    from .vqgan_pipeline import process_alive
    a.run_dir.mkdir(parents=True,exist_ok=True);pid=a.run_dir/'training.pid'
    if pid.exists()and process_alive(int(pid.read_text())):p.error('Decoder-chain worker is already active')
    pid.write_text(str(os.getpid()),encoding='ascii');experiment=DecoderRFChainTraining(a)
    try:
        with KeepAwake():experiment.run()
    except BaseException as error:
        experiment.publish(status='failed',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
