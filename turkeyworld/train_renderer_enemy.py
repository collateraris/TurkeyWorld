"""Independent Source-supervised one-enemy compact Base/RF1 training."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .distill_renderer import _digest,_save
from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_enemy_data import EnemyFrames
from .renderer_enemy_objectives import enemy_refinement_loss
from .renderer_enemy_validation import evaluate_enemy_renderer,save_enemy_preview
from .renderer_refinement import configure_refinement,frozen_fingerprint,export_refined_bundle
from .renderer_student import load_renderer_student,_hash_tensors
from .training_pipeline import atomic_json,now


class EnemyRendererTraining:
    def __init__(self,args):
        self.args=args;self.root=Path(args.run_dir).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.started=time.perf_counter()
        self.state={'run_id':self.root.name,'stage':'renderer_enemy_refinement','status':'initializing',
            'model':'One-NPC Source → compact Base + RF1; VQGAN/RSSM/map frozen',
            'current':0,'global_training_step':0,'total':args.steps,'started_at':now(),'image_size':32,
            'prediction_every':args.preview_every,'validation_every':args.validate_every,
            'loss_protocol':'one_enemy_source_balanced_renderer_v1','scope':'Fixed Source poses; not autonomous game qualification',
            'world_training':False,'gameplay_qualified':False}
    def publish(self,**values):
        self.state.update(values,updated_at=now(),elapsed_seconds=time.perf_counter()-self.started)
        atomic_json(self.root/'status.json',self.state)

    def run(self):
        args=self.args;device=torch.device(args.device)
        if device.type=='cuda':torch.cuda.init()
        torch.set_num_threads(args.threads)
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        source,meta=load_world_checkpoint(args.world,'cpu');source_step=meta['training_steps'];del meta
        source.eval().requires_grad_(False)
        working,meta=load_world_checkpoint(args.world,'cpu');del meta;gc.collect()
        initial=load_renderer_student(working,args.renderer_checkpoint,
            allow_unqualified_codec=args.allow_unqualified_codec)
        initial_renderer_steps=initial['training_steps']
        initial_base_steps=working.base_student_bundle['training_steps']
        working.float().to(device)
        policy=configure_refinement(working)
        if policy['base_trainable']>1000000 or policy['flow_trainable']>7000000 or policy['codec_parameters']>1000000:
            raise ValueError('User compact renderer parameter budgets changed')
        optimizer=torch.optim.AdamW([
            {'params':working.base_student.parameters(),'lr':args.base_lr},
            {'params':[p for p in working.flow_renderer.parameters() if p.requires_grad],'lr':args.lr}],weight_decay=.01)
        frozen=frozen_fingerprint(working)
        codec_identity=_hash_tensors(working.flow_renderer.codec.state_dict())
        codec_ids={id(p) for p in working.flow_renderer.codec.parameters()}
        if any(id(p) in codec_ids for group in optimizer.param_groups for p in group['params']):
            raise AssertionError('Codec entered enemy renderer optimizer')
        self.publish(stage_label='Indexing genuine one-enemy Source frames',
            source_world_step=source_step,model_parameters=policy,solver_steps=1,
            renderer_training={'active':True,'base_active':True,'codec_frozen':True,'codec_all_buffers_frozen':True,
                'rssm_frozen':True,'map_frozen':True,'geometry_frozen':True,'sr_frozen':True,'gan_active':False})
        train=EnemyFrames(args.data_root,args.empty_data_root,'train',seed=args.seed)
        validation=EnemyFrames(args.data_root,args.empty_data_root,'val',seed=args.seed+1,exclude_scene_keys=train.scene_keys)
        if ({r['seed'] for r in train.files}&{r['seed'] for r in validation.files}
                or {r['sha256'] for r in train.files}&{r['sha256'] for r in validation.files}):
            raise ValueError('Enemy Source train/val identities overlap')
        panel=validation.fixed_panel(args.validation_frames,seed=args.seed)
        identity={'world_sha256':_digest(args.world),'renderer_sha256':_digest(args.renderer_checkpoint),
            'train':train.identity,'val':validation.identity,'panel':panel,
            'panel_fingerprint':validation.panel_fingerprint(panel),'codec_sha256':codec_identity}
        configuration={key:str(value.resolve()) if isinstance(value,Path) else value for key,value in vars(args).items()}
        config_path=self.root/'experiment_config.json'
        if config_path.exists():
            old=json.loads(config_path.read_text(encoding='utf-8'))
            if ({k:v for k,v in old.items() if k!='steps'}!={k:v for k,v in configuration.items() if k!='steps'}
                    or args.steps<old['steps']):raise ValueError('Enemy refinement settings changed or budget shrank')
        atomic_json(config_path,configuration)
        atomic_json(self.root/'source_identity.json',identity)
        self.publish(data_preparation={'status':'complete','train_frames':len(train.rows),'val_frames':len(validation.rows),
            'train_categories':train.identity['category_counts'],'val_categories':validation.identity['category_counts'],
            'excluded_val_frames':validation.excluded_frames,'excluded_val_scene_keys':validation.excluded_scene_keys,
            'train_fingerprint':train.identity['fingerprint'],'validation_fingerprint':validation.identity['fingerprint']},
            validation_panel_fingerprint=identity['panel_fingerprint'])
        checkpoint=self.root/'checkpoints/renderer_enemy_last.pt'
        best_path=self.root/'checkpoints/renderer_enemy_best.pt'
        baseline_path=self.root/'parent_renderer_baseline.pt'
        step,best,streak=0,None,0
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if (saved['identity']!=identity or saved['frozen_fingerprint']!=frozen
                    or {k:v for k,v in saved['configuration'].items() if k!='steps'}!={k:v for k,v in configuration.items() if k!='steps'}
                    or args.steps<saved['configuration']['steps']):
                raise ValueError('Enemy refinement checkpoint/data/protocol identity changed')
            working.base_student.load_state_dict(saved['base'],strict=True)
            working.flow_renderer.load_state_dict(saved['flow'],strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            step,best,streak=saved['step'],saved['best'],saved['streak']
            train.restore_sampler(saved['sampler'])
            torch.set_rng_state(saved['rng_torch'].cpu());np.random.set_state(saved['rng_numpy']);random.setstate(saved['rng_python'])
            if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in saved['rng_cuda']])
            del saved;gc.collect()
        def save():
            _save(checkpoint,{'base':working.base_student.state_dict(),'flow':working.flow_renderer.state_dict(),
                'optimizer':optimizer.state_dict(),'step':step,'best':best,'streak':streak,
                'configuration':configuration,'identity':identity,'frozen_fingerprint':frozen,
                'sampler':train.sampler_state(),'rng_torch':torch.get_rng_state(),
                'rng_numpy':np.random.get_state(),'rng_python':random.getstate(),
                'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []})
        def verify_frozen():
            if (frozen_fingerprint(working)!=frozen or _hash_tensors(working.flow_renderer.codec.state_dict())!=codec_identity
                    or working.flow_renderer.codec.training or working.flow_renderer.codec.quantize.training):
                raise AssertionError('Frozen codec/dynamics/map/geometry parameters or buffers changed')
            self.publish(frozen_verification={'step':step,'all_frozen_parameters_buffers_bitexact':True,
                'codec_all_parameters_buffers_bitexact':True,'codec_optimizer_excluded':True,'codec_sha256':codec_identity})
        if baseline_path.exists():
            baseline_saved=torch.load(baseline_path,map_location='cpu',weights_only=False)
            if baseline_saved['identity']!=identity:raise ValueError('Immutable parent baseline changed')
            baseline=baseline_saved['report'];del baseline_saved
        else:
            if step:raise ValueError('Resume requires the original parent renderer baseline')
            self.publish(stage_label='Initial held-out enemy/empty renderer validation')
            baseline,arrays=evaluate_enemy_renderer(working,validation,panel,step=0,batch_size=args.validation_batch)
            verify_frozen()
            _save(baseline_path,{'identity':identity,'report':baseline,'arrays':arrays})
            atomic_json(self.root/'validation_initial.json',baseline);save_enemy_preview(arrays,self.root/'preview.png',0)
            self.publish(validation_initial=baseline,validation=baseline)
        preview_ids=[]
        for category in ('large','partial','corpse','empty_navigation'):
            matches=[i for i in panel if validation.rows[i]['category']==category]
            if matches:preview_ids.append(matches[0])
        first_step=step+1
        while step<args.steps and streak<2:
            if (self.root/'stop_training').exists():
                save();self.publish(status='stopped',current=step,global_training_step=step);return
            step+=1;configure_refinement(working);optimizer.zero_grad(set_to_none=True)
            batch=train.batch(args.batch)
            loss,metrics=enemy_refinement_loss(working,batch,
                ssim_weight=args.ssim_weight,edge_weight=args.edge_weight,contrast_weight=args.contrast_weight)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Invalid enemy renderer loss')
            loss.backward()
            metrics['gradient_norm']=float(clip_finite_grad_norm_(working.named_parameters(),context=f'Enemy renderer update{step}'))
            optimizer.step()
            if step==first_step:verify_frozen()
            report=None
            if step%args.validate_every==0 or step==args.steps:
                report,arrays=evaluate_enemy_renderer(working,validation,panel,step=step,batch_size=args.validation_batch,baseline=baseline)
                verify_frozen();report['frozen_components_bitexact']=True
                eligible=report['experimental_selection']['validated_improvement']
                if eligible and (best is None or report['selection_score']<best['score']):
                    best={'step':step,'score':report['selection_score']}
                    _save(best_path,{'base':working.base_student.state_dict(),'flow':working.flow_renderer.state_dict(),**best})
                streak=streak+1 if report['passed'] else 0
                atomic_json(self.root/'validation.json',report)
                atomic_json(self.root/'validation'/f'step_{step:06d}.json',report)
                save_enemy_preview(arrays,self.root/'preview.png',step)
            elif step%args.preview_every==0:
                _,arrays=evaluate_enemy_renderer(working,validation,preview_ids,step=step,batch_size=args.validation_batch)
                save_enemy_preview(arrays,self.root/'preview.png',step)
            if step%args.preview_every==0 or report is not None:
                atomic_json(self.root/'preview.json',{'step':step,'every':args.preview_every,'kind':'Source_enemy_GT_query',
                    'source_held_out':True,'renderer_only':True,'world_rollout':False,'codec_frozen':True,
                    'feature_policy':'static_zero128_or_dynamic_frozen_past_prior','source_quality_gate_evaluated':report is not None})
            record={'step':step,'global_training_step':step,'stage':'renderer_enemy_refinement','timestamp':now(),**metrics}
            if report is not None:record['source_validation']=report
            with (self.root/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
            self.publish(status='running',stage_label='Training compact Base/RF1 on genuine one-enemy Source',
                current=step,global_training_step=step,progress=step/args.steps,loss=metrics['train_loss'],
                latest_metrics=metrics,validation=report or self.state.get('validation'),selected_best=best,
                sampler_draws=train.draws,sampler_case_counts=train.case_counts,
                eta_seconds=(time.perf_counter()-self.started)/max(step,1)*(args.steps-step))
            if step%args.save_every==0 or report is not None or step==first_step:save()
        save()
        if best is None:
            self.publish(status='completed',current=step,experimental_candidate_validated=False,
                source_quality_passed=False,message='Training completed; no candidate passed NPC-improvement and empty-regression guards',
                active_renderer_changed=False);return
        selected=torch.load(best_path,map_location='cpu',weights_only=False)
        working.base_student.load_state_dict(selected['base'],strict=True);working.flow_renderer.load_state_dict(selected['flow'],strict=True)
        report,arrays=evaluate_enemy_renderer(working,validation,panel,step=selected['step'],batch_size=args.validation_batch,baseline=baseline)
        verify_frozen()
        if not report['experimental_selection']['validated_improvement']:
            raise AssertionError('Selected enemy candidate no longer passes its held-out guards')
        exported=export_refined_bundle(source,working,self.root/'checkpoints',updates=selected['step'],report=report,
            initial_renderer_steps=initial_renderer_steps,initial_base_steps=initial_base_steps,
            provenance={'kind':'one_enemy_Source_GT_renderer_refinement','source_identity':identity,
                'feature_policies':['static_zero128','dynamic_frozen_past_prior'],'codec_frozen':True,
                'dynamics_frozen':True,'automatic_active_promotion':False})
        atomic_json(self.root/'validation_selected.json',report);save_enemy_preview(arrays,self.root/'preview.png',selected['step'])
        self.publish(status='completed',current=step,selected_step=selected['step'],validation=report,
            renderer_export=exported,source_quality_passed=report['passed'],experimental_candidate_validated=True,
            active_renderer_changed=False,gameplay_qualified=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('world','renderer-checkpoint','data-root','empty-data-root','run-dir'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--allow-unqualified-codec',action='store_true')
    for name,default in (('steps',2000),('batch',8),('validation-frames',256),('validation-batch',8),
                         ('validate-every',100),('preview-every',10),('save-every',10),('seed',173),('threads',4)):
        parser.add_argument('--'+name,type=int,default=default)
    for name,default in (('lr',.0001),('base-lr',.0001),('ssim-weight',.2),('edge-weight',.1),('contrast-weight',.1)):
        parser.add_argument('--'+name,type=float,default=default)
    args=parser.parse_args(argv)
    if min(args.steps,args.batch,args.validation_frames,args.validation_batch,args.validate_every,args.preview_every,args.save_every,args.threads)<=0:
        parser.error('Positive renderer budgets and batch sizes required')
    if min(args.lr,args.base_lr)<=0 or min(args.ssim_weight,args.edge_weight,args.contrast_weight)<0:
        parser.error('Positive rates and nonnegative loss weights required')
    args=argparse.Namespace(**{k:v.resolve() if isinstance(v,Path) else v for k,v in vars(args).items()})
    from .vqgan_pipeline import process_alive
    args.run_dir.mkdir(parents=True,exist_ok=True);pid=args.run_dir/'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())):parser.error('Enemy renderer worker is already active')
    atomic_json(pid,__import__('os').getpid())
    experiment=EnemyRendererTraining(args);experiment.publish()
    try:
        with KeepAwake():experiment.run()
    except BaseException as error:
        experiment.publish(status='failed',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()

