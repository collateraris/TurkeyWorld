"""Continue the trained EDSR on true paired Source enemy/combat RGB only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .compact_codec import CompactCodec
from .gradient_safety import clip_finite_grad_norm_
from .models import load_sr_checkpoint
from .object_losses import balanced_image_l1, balanced_edge_l1, decoded_ssim_loss
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .sr_combat_data import SRCombatPairData, digest
from .train_sr import evaluate_sr, save_sr_preview
from .training_pipeline import atomic_json, now, replace_file
from .vqgan import VQGANConfig


QUALITY_SSIM = .91
QUALITY_MAE = 2.5/255


def clean_quality_gate(report, *, expected_frames=384):
    """The user's strict clean-Source mean criterion; other domains are separate."""
    summary=report.get('summary',{}) if isinstance(report,dict) else {}
    if not isinstance(summary,dict):summary={}
    def number(name):
        value=summary.get(name)
        return (float(value) if isinstance(value,(int,float)) and not isinstance(value,bool)
                and np.isfinite(value) else None)
    ssim,mae=number('ssim_mean'),number('mae_mean')
    valid=(isinstance(report,dict) and report.get('source_held_out') is True
        and report.get('world_rollout') is False and report.get('input_kind')=='clean_source_lr'
        and report.get('frames')==expected_frames and ssim is not None and mae is not None
        and 0<=ssim<=1 and 0<=mae<=1)
    passed=bool(valid and ssim>QUALITY_SSIM and 0<=mae<QUALITY_MAE)
    return dict(mode='until_clean_quality',passed=passed,target_met=passed,
        current_mean_ssim=ssim,current_mean_mae=mae,current_mean_mae_255=None if mae is None else mae*255,
        last_validation_step=report.get('step') if isinstance(report,dict) else None,
        validation_frames=report.get('frames') if isinstance(report,dict) else None,
        panel_fingerprint=report.get('panel_fingerprint') if isinstance(report,dict) else None,
        thresholds=dict(ssim_mean_min=QUALITY_SSIM,ssim_operator='>',mae_mean_max=QUALITY_MAE,
            mae_operator='<',mae_255_max=2.5),required_consecutive_passes=1,consecutive_passes=int(passed),
        decision='qualified' if passed else 'training',metrics_valid=bool(valid),
        scope='clean Source LR fixed384 held-out mean; not codec or gameplay qualification')


def load_augmentation_codec(path, device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved.get('stage')!='distilled_vqgan_codec' or not saved.get('frozen')
            or not saved.get('fixed_codebook') or saved.get('training_steps',0)<=0):
        raise ValueError('SR augmentation requires an explicitly trained fixed-book compact codec')
    codec=CompactCodec(VQGANConfig(**saved['codec_config']))
    codec.load_state_dict(saved['model'],strict=True)
    if (codec.config.image_size!=32 or sum(p.numel() for p in codec.parameters())>1000000
            or not bool(codec.quantize.initialized)):
        raise ValueError('Expected initialized native32 compact codec<=1M')
    codec.to(device).eval().requires_grad_(False)
    return codec,dict(checkpoint=str(Path(path).resolve()),sha256=digest(path),
        tensor_sha256=_hash_tensors(codec.state_dict()),training_steps=saved['training_steps'],
        quality_gate_passed=bool(saved.get('quality_gate',{}).get('passed',False)),
        augmentation_only=True,world_generated_rgb=False,all_parameters_and_buffers_frozen=True)


def source_sr_loss(prediction,target,mask):
    pixel=balanced_image_l1(prediction,target,mask)
    ssim=decoded_ssim_loss(prediction,target,mask)
    edge=balanced_edge_l1(prediction,target,mask)
    return pixel+.2*ssim+.1*edge,dict(pixel=float(pixel.detach()),ssim_loss=float(ssim.detach()),edge=float(edge.detach()))


def _save(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp.pt')
    torch.save(payload,temporary);replace_file(temporary,path)


def validate_continuation(path,saved,target_steps,training_plan_sha256,resume_sha256):
    plan=json.loads(Path(path).read_text())
    start,end=plan['from_completed_updates'],plan['to_budget_updates']
    if (plan.get('reason')!='explicit_user_request' or not isinstance(start,int) or not isinstance(end,int)
            or not 0<=start<end or end!=target_steps or saved['refinement_updates']<start
            or saved['run_state']['total'] not in (plan['from_budget_updates'],end)
            or plan['training_plan_sha256']!=training_plan_sha256
            or digest(plan['source_checkpoint'])!=plan['source_checkpoint_sha256']):
        raise ValueError('SR continuation budget/source identity mismatch')
    if saved['refinement_updates']==start and resume_sha256!=plan['source_checkpoint_sha256']:
        raise ValueError('First continuation must restore the exact archived source Adam checkpoint')
    return {**plan,'plan':str(Path(path).resolve()),'plan_sha256':digest(path)}


def validate_quality_revision(path,saved,training_plan_sha256,panel_fingerprint,resume_sha256):
    plan=json.loads(Path(path).read_text())
    expected=dict(input_kind='clean_source_lr',frames=384,ssim_gt=QUALITY_SSIM,mae_lt=QUALITY_MAE,
        mae_255_lt=2.5,required_consecutive_passes=1)
    if (plan.get('reason')!='explicit_user_request' or plan.get('criteria')!=expected
            or plan.get('fixed_max_updates') is not None or plan.get('validation_every')!=100
            or plan.get('preview_every')!=100 or plan['training_plan_sha256']!=training_plan_sha256
            or plan['panel_fingerprint']!=panel_fingerprint
            or digest(plan['source_checkpoint'])!=plan['source_checkpoint_sha256']
            or digest(plan['previous_budget_plan'])!=plan['previous_budget_plan_sha256']
            or saved['refinement_updates']<plan['from_completed_updates']):
        raise ValueError('SR quality revision identity/strict criteria mismatch')
    if saved['refinement_updates']==plan['from_completed_updates'] and resume_sha256!=plan['source_checkpoint_sha256']:
        raise ValueError('Quality continuation must restore the archived exact Adam boundary')
    previous=saved['run_state'].get('quality_revision')
    if previous is not None and previous['plan_sha256']!=digest(path):
        raise ValueError('Immutable SR quality revision changed')
    return {**plan,'plan':str(Path(path).resolve()),'plan_sha256':digest(path)}


def resume_revisions(saved,*,steps,continuation_plan,quality_revision_plan,
                     training_plan_sha256,panel_fingerprint,resume_sha256):
    continuation=None
    if saved['run_state'].get('training_until_quality',False):
        if not quality_revision_plan:raise ValueError('Quality continuation requires its immutable revision plan')
        continuation=saved['run_state'].get('training_continuation')
    elif steps!=saved['run_state']['total'] or saved['run_state'].get('training_continuation'):
        if not continuation_plan:raise ValueError('Budget changes require an explicit continuation plan')
        continuation=validate_continuation(continuation_plan,saved,steps,training_plan_sha256,resume_sha256)
    quality=(validate_quality_revision(quality_revision_plan,saved,training_plan_sha256,panel_fingerprint,resume_sha256)
             if quality_revision_plan else None)
    return continuation,quality


class SRCombatTraining:
    def __init__(self,args):
        self.args=args;self.root=args.run_dir.resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.device=torch.device(args.device);self.step=0;self.rng=np.random.default_rng(args.seed);self.started=time.perf_counter()
        self.quality_revision=None
        self.state=dict(run_id=self.root.name,status='initializing',stage='sr',model='Trained EDSR x4: Source one-enemy/combat continuation',
            current=0,global_training_step=0,total=args.steps,started_at=now(),image_size=32,
            prediction_every=100,validation_every=100,loss_protocol='sr_source_combat_v1',
            gameplay_qualified=False,active_renderer_changed=False,automatic_deployment=False,
            sr_training=dict(from_scratch=False,pretrained_weights_loaded=True,input_size=32,output_size=128,
                scale=4,target_ssim=.9,world_rollout_qualified=False,requested_updates=args.steps))
        self.publish()

    def publish(self,**values):
        self.state.update(current=self.step,global_training_step=self.step,elapsed_seconds=time.perf_counter()-self.started,
            updated_at=now(),progress=None if self.quality_revision else self.step/max(1,self.args.steps),**values)
        if self.quality_revision:self.state.update(total=None,eta_seconds=None,training_until_quality=True)
        atomic_json(self.root/'status.json',self.state)

    def checkpoint(self):
        payload=dict(stage='sr',format_version=1,config=self.model.config,model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),training_steps=self.parent_steps+self.step,refinement_updates=self.step,
            parent_checkpoint=str(self.args.parent),parent_training_steps=self.parent_steps,training_protocol=self.protocol,
            codec_augmentation=self.codec_info,rng_numpy=self.rng.bit_generator.state,rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],run_state=self.state)
        _save(self.root/'checkpoints/sr_last.pt',payload)

    def evaluate(self):
        reports={};preview=None
        for name,input_kind in (('clean','clean_source_lr'),('codec','codec_reconstruction')):
            measured,picture=evaluate_sr(self.model,self.val,self.panel,batch_size=8,step=self.step,
                codec=self.codec,input_kind=input_kind)
            groups=defaultdict(list)
            by_name={episode['path'].name:cohort for episode,cohort in zip(self.val.episodes,self.val.cohorts)}
            for row in measured['per_frame']:groups[by_name[row['episode']]].append(row)
            measured['cohorts']={cohort:dict(frames=len(rows),ssim_mean=float(np.mean([r['ssim']for r in rows])),
                mae_mean=float(np.mean([r['mae']for r in rows])))for cohort,rows in groups.items()}
            reports[name]=measured
            if name=='clean':preview=picture
        if _hash_tensors(self.codec.state_dict())!=self.codec_info['tensor_sha256']:
            raise AssertionError('Frozen augmentation codec parameters/buffers changed')
        baseline=self.state.get('baseline_validation')
        guards=[]
        if baseline:
            for name in reports:
                current=reports[name]['cohorts']['navigation'];old=baseline[name]['cohorts']['navigation']
                if current['mae_mean']>old['mae_mean']+.005 or current['ssim_mean']<old['ssim_mean']-.02:
                    guards.append(name+'_navigation_regression')
        legacy_passed=not guards and all(v['summary']['ssim_mean']>=.9 and v['summary']['mae_mean']<=.03 for v in reports.values())
        quality=clean_quality_gate(reports['clean']) if self.quality_revision else None
        passed=quality['passed'] if quality is not None else legacy_passed
        report=dict(step=self.step,cumulative_sr_step=self.parent_steps+self.step,**reports,passed=passed,
            source_held_out=True,world_rollout=False,codec_bitexact=True,navigation_regression_failures=guards,
            thresholds=dict(ssim_mean_min=.9,mae_mean_max=.03,navigation_mae_tolerance=.005,navigation_ssim_tolerance=.02),
            gameplay_qualified=False,automatically_deployed=False)
        if quality is not None:
            report['legacy_thresholds']=report['thresholds']
            report.update(primary_clean_quality=quality,legacy_combined_passed=legacy_passed,
                thresholds=quality['thresholds'],
                codec_secondary_passed=(reports['codec']['summary']['ssim_mean']>=.9
                    and reports['codec']['summary']['mae_mean']<=.03),
                primary_scope='STRICT clean384 only; degraded-codec/nav checks remain separate')
            self.state['sr_quality']=quality
            if quality['passed']:
                _save(self.root/'checkpoints/sr_quality_passed.pt',dict(model=self.model.state_dict(),report=report,step=self.step))
        report['selection_score']=1-reports['clean']['summary']['ssim_mean']+reports['clean']['summary']['mae_mean']
        atomic_json(self.root/'validation'/f'step_{self.step:06d}.json',report)
        atomic_json(self.root/'sr_validation.json',reports['clean']);atomic_json(self.root/'sr_validation_codec.json',reports['codec'])
        self.publish(sr_validation=reports['clean'],sr_validation_codec=reports['codec'],
            codec_frozen_verification=dict(all_tensors_bitexact=True,tensor_sha256=self.codec_info['tensor_sha256']),
            sr_validation_combined=report)
        self.state['prediction_preview']=save_sr_preview(preview,self.root/'preview.png',every=100)
        if not guards and (self.state.get('best_score') is None or report['selection_score']<self.state['best_score']):
            self.state.update(best_score=report['selection_score'],best_step=self.step)
            _save(self.root/'checkpoints/sr_best.pt',dict(model=self.model.state_dict(),report=report,step=self.step))
        if baseline is None:self.state['baseline_validation']=report;atomic_json(self.root/'validation_initial.json',report)
        self.publish();return report

    def run(self):
        args=self.args;torch.set_num_threads(2)
        if self.device.type=='cuda':torch.cuda.init()
        torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.model,parent=load_sr_checkpoint(args.parent,'cpu');self.parent_steps=int(parent['training_steps'])
        if self.model.config.get('architecture')!='edsr' or sum(p.numel()for p in self.model.parameters())!=1517571:
            raise ValueError('Continue the existing1517571-parameter EDSR, without replacing its architecture')
        self.model.to(self.device).float();self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=args.lr,weight_decay=.01)
        self.codec,self.codec_info=load_augmentation_codec(args.codec_checkpoint,self.device)
        self.train=SRCombatPairData(args.data_dir/'rgb/train',self.device,cache_bytes=64*1024**2,seed=args.seed)
        self.val=SRCombatPairData(args.data_dir/'rgb/val',self.device,cache_bytes=64*1024**2)
        self.panel=self.val.fixed_panel()
        self.protocol=dict(parent_sha256=digest(args.parent),data_manifest_sha256=digest(args.data_dir/'manifest.json'),
            codec_sha256=self.codec_info['sha256'],codec_fraction=.25,batch=args.batch,lr=args.lr,
            pixel_weight=1.,ssim_weight=.2,edge_weight=.1,loss_regions='equal_actual_HR_foreground_background',
            target_source='exact_replayed_reference_engine128',interpolated_HR=False,training_target_frame=1)
        path=self.root/'training_plan.json'
        if path.exists() and json.loads(path.read_text())!=self.protocol:raise ValueError('Immutable SR continuation plan changed')
        if not path.exists():atomic_json(path,self.protocol)
        resume=self.root/'checkpoints/sr_last.pt'
        if resume.exists():
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            if saved['training_protocol']!=self.protocol:raise ValueError('SR checkpoint protocol mismatch')
            continuation,self.quality_revision=resume_revisions(saved,steps=args.steps,
                continuation_plan=getattr(args,'continuation_plan',None),quality_revision_plan=getattr(args,'quality_revision_plan',None),
                training_plan_sha256=digest(path),panel_fingerprint=self.val.panel_fingerprint(self.panel),resume_sha256=digest(resume))
            self.model.load_state_dict(saved['model']);self.optimizer.load_state_dict(saved['optimizer'])
            self.step=saved['refinement_updates'];self.state=saved['run_state'];self.rng.bit_generator.state=saved['rng_numpy']
            torch.set_rng_state(saved['rng_torch'].cpu())
            if self.device.type=='cuda':torch.cuda.set_rng_state_all([value.cpu()for value in saved['rng_cuda']])
            if continuation is not None:
                self.state['training_continuation']=continuation
                unchanged=all(torch.equal(value,self.model.state_dict()[name].detach().cpu())for name,value in saved['model'].items())
                restored=self.optimizer.state_dict()
                adam_equal=all(torch.equal(value,restored['state'][key][name].detach().cpu())
                    for key,entry in saved['optimizer']['state'].items()for name,value in entry.items()if torch.is_tensor(value))
                proof=dict(step=self.step,all_model_tensors_bitexact=unchanged,all_adam_tensors_bitexact=adam_equal,
                    torch_rng_bitexact=torch.equal(saved['rng_torch'].cpu(),torch.get_rng_state()),
                    numpy_rng_bitexact=saved['rng_numpy']==self.rng.bit_generator.state,
                    cuda_rng_bitexact=(self.device.type!='cuda' or all(torch.equal(a.cpu(),b.cpu())for a,b in zip(saved['rng_cuda'],torch.cuda.get_rng_state_all()))),
                    continuation_plan_sha256=continuation['plan_sha256'],optimizer_reset=False)
                if self.quality_revision:proof['quality_revision_sha256']=self.quality_revision['plan_sha256']
                if not all(proof[k]for k in ('all_model_tensors_bitexact','all_adam_tensors_bitexact','torch_rng_bitexact','numpy_rng_bitexact','cuda_rng_bitexact')):
                    raise AssertionError('SR continuation restoration changed learned state')
                name='quality_resume_verification' if self.quality_revision else 'resume_verification'
                atomic_json(self.root/f'{name}_g{self.step}.json',proof)
        self.state['total']=args.steps
        self.state['sr_training'].update(parent_training_steps=self.parent_steps,codec_augmentation=self.codec_info,
            dataset=self.train.stats,validation_frames=len(self.panel),requested_updates=args.steps)
        if self.quality_revision:
            self.state.update(training_until_quality=True,total=None,eta_seconds=None,quality_revision=self.quality_revision)
            self.state['sr_training'].update(training_until_quality=True,requested_updates=None,target_ssim=QUALITY_SSIM,
                target_mae=QUALITY_MAE,target_mae_255=2.5,target_scope='strict clean384 Source mean; SSIM> and MAE<')
        self.publish(status='running',stage_label='SR Source combat: existing EDSR20k + frozen codec25%',
            model_parameters=dict(total=1517571,trainable=1517571),codec_frozen=True)
        if self.step==0 or self.quality_revision:self.evaluate();self.checkpoint()
        while ((not self.state.get('sr_quality',{}).get('target_met',False)) if self.quality_revision else self.step<args.steps):
            if (self.root/'stop_training').exists():self.publish(status='stopped');self.checkpoint();return
            self.model.train();self.codec.eval();batch=self.train.batch(args.batch,rng=self.rng)
            selected=np.flatnonzero(self.rng.random(args.batch)<.25);inputs=batch['lr'].clone()
            if len(selected):
                ids=torch.as_tensor(selected,device=self.device)
                with torch.no_grad():inputs[ids]=self.codec(inputs[ids])['reconstruction']
            self.optimizer.zero_grad(set_to_none=True);prediction=self.model(inputs)
            loss,parts=source_sr_loss(prediction,batch['hr'],batch['hr_object_mask']);loss.backward()
            norm=float(clip_finite_grad_norm_(self.model.named_parameters(),context='SR combat'))
            self.optimizer.step();self.step+=1
            metrics=dict(step=self.step,train_loss=float(loss.detach()),**parts,world_grad_norm=norm,
                codec_fraction=len(selected)/args.batch,loss_protocol='sr_source_combat_v1',timestamp=now())
            with (self.root/'metrics.jsonl').open('a',encoding='utf8')as handle:handle.write(json.dumps(metrics)+'\n')
            self.publish(loss=metrics['train_loss'],latest_metrics=metrics,cumulative_sr_training_steps=self.parent_steps+self.step)
            first_continuation=self.step==self.state.get('training_continuation',{}).get('from_completed_updates',-2)+1
            first_quality=self.quality_revision and self.step==self.quality_revision['from_completed_updates']+1
            if self.step%100==0 or (self.step==args.steps and not self.quality_revision):self.evaluate();self.checkpoint()
            elif first_continuation or first_quality:
                codec_bitexact=_hash_tensors(self.codec.state_dict())==self.codec_info['tensor_sha256']
                if not codec_bitexact:raise AssertionError('Frozen codec changed at the first continued update')
                self.checkpoint()
                if first_continuation:
                    atomic_json(self.root/'first_continuation_update.json',dict(step=self.step,
                        cumulative_sr_training_steps=self.parent_steps+self.step,optimizer_updated=True,
                        codec_bitexact=codec_bitexact))
                if first_quality:
                    atomic_json(self.root/'first_quality_revision_update.json',dict(step=self.step,
                        cumulative_sr_training_steps=self.parent_steps+self.step,optimizer_updated=True,
                        quality_revision_sha256=self.quality_revision['plan_sha256'],codec_bitexact=codec_bitexact))
        selected_path=self.root/'checkpoints'/('sr_quality_passed.pt' if self.quality_revision else 'sr_best.pt')
        best=torch.load(selected_path,map_location='cpu',weights_only=False)
        payload=dict(stage='sr',format_version=1,config=self.model.config,model=best['model'],
            input_size=32,output_size=128,scale=4,training_steps=self.parent_steps+best['step'],
            parent_training_steps=self.parent_steps,refinement_updates=best['step'],completed_refinement_updates=self.step,
            training_protocol=self.protocol,codec_augmentation=self.codec_info,validation=best['report']['clean'],
            validation_codec=best['report']['codec'],source_quality_passed=best['report']['passed'],
            quality_status='Source combat SR validation; not deployed; no gameplay qualification')
        if self.quality_revision:
            payload.update(quality_revision=self.quality_revision,primary_clean_quality=best['report']['primary_clean_quality'])
        _save(self.root/'checkpoints/sr_for_game.pt',payload);atomic_json(self.root/'validation_selected.json',best['report'])
        self.publish(status='qualified' if self.quality_revision else 'completed',selected_step=best['step'],source_quality_passed=best['report']['passed'],
            ready_checkpoint=str(self.root/'checkpoints/sr_for_game.pt'),automatic_deployment=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','data-dir','parent','codec-checkpoint'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2000);parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--lr',type=float,default=5e-5);parser.add_argument('--seed',type=int,default=173)
    parser.add_argument('--continuation-plan',type=Path)
    parser.add_argument('--quality-revision-plan',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');args=parser.parse_args()
    experiment=SRCombatTraining(args)
    try:
        with KeepAwake():experiment.run()
    except BaseException as error:
        experiment.publish(status='failed',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
