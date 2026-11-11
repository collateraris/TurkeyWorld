"""Paired true-Source tiny-enemy EDSR continuation with a frozen selected codec."""
import argparse
from pathlib import Path
import random
import time
import numpy as np
import torch

from .distill_renderer import _digest,_save
from .gradient_safety import clip_finite_grad_norm_
from .models import load_sr_checkpoint
from .object_losses import balanced_image_l1,balanced_edge_l1,decoded_ssim_loss
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .small_enemy_validation import small_enemy_cohorts,relative_small_enemy_gate
from .train_sr import _pixels,save_sr_preview
from .train_sr_combat import load_augmentation_codec
from .training_pipeline import atomic_json,now
from .visual_quality import strict_render_precision


@torch.no_grad()
def evaluate_small_sr(model,codec,data,panel,*,step,batch_size=8,baseline=None):
    from .staged_validation import _isolated_validation
    reports={};preview=None
    for name in ('clean','codec'):
        parts={key:[]for key in ('prediction','target','mask','native_mask','state','input')}
        with _isolated_validation(model,data,173),_isolated_validation(codec,data,173):
            for start in range(0,len(panel),batch_size):
                batch=data.batch(len(panel[start:start+batch_size]),indices=panel[start:start+batch_size])
                low=batch['lr'];high=batch['hr'];network_input=codec(low)['reconstruction']if name=='codec'else low
                prediction=model(network_input).float()
                if prediction.shape!=high.shape or not torch.isfinite(prediction).all() or ((prediction<0)|(prediction>1)).any():
                    raise FloatingPointError('Invalid native32-to128 SR prediction')
                for key,value in {'prediction':prediction,'target':high,'mask':batch['hr_object_mask'],
                    'native_mask':batch['object_mask'],'state':batch['state'],'input':network_input}.items():
                    parts[key].append(value.detach().cpu())
        arrays={key:torch.cat(values)for key,values in parts.items()}
        reports[name]=small_enemy_cohorts(arrays['prediction'],arrays['target'],arrays['mask'],arrays['state'],native_mask=arrays['native_mask'])
        if name=='clean':
            preview={key:_pixels(arrays[value][:4])for key,value in
                (('source_lr','input'),('sr','prediction'),('source_hr','target'))}
            preview['difference']=np.abs(preview['sr'].astype(np.int16)-preview['source_hr'].astype(np.int16)).astype(np.uint8)
    report={'step':step,'source_held_out':True,'frames':len(panel),'world_rollout':False,
        'panel_fingerprint':data.panel_fingerprint(panel),'small_enemy_cohorts':reports,
        'clean':{'summary':reports['clean']['overall']},'codec':{'summary':reports['codec']['overall']},
        'source_quality_passed':all(rows['overall']['ssim_mean']>=.9 and rows['overall']['mae_mean']<=.03 for rows in reports.values()),
        'source_hr_target':'Exact reference-engine128 at the same physical tick; never interpolated',
        'gameplay_qualified':False,'automatic_deployment':False}
    report['small_enemy_stage_gate']=(relative_small_enemy_gate(report,baseline,predictions=('clean','codec'))
        if baseline is not None else {'passed':False,'failure_reasons':['initial_reference_only']})
    report['selection_score']=sum(rows['tiny']['foreground_mae']+.25*rows['overall']['mae_mean']for rows in reports.values())
    preview['metadata']={'kind':'single_frame_super_resolution','step':step,'source_held_out':True,
        'world_rollout':False,'source_input_allowed_for_sr':True,'image_size':32,'output_size':128,
        'panel_fingerprint':report['panel_fingerprint'],'input_kind':'clean_source_lr'}
    return report,preview


def run(args):
    from .small_enemy_data import SmallEnemyData
    root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.init()
    torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state={'run_id':root.name,'stage':'sr','status':'initializing','current':0,'total':args.steps,
        'started_at':now(),'image_size':32,'validation_every':100,'prediction_every':100,
        'model':'Existing EDSR x4; true Source tiny enemies; frozen selected compact codec',
        'loss_protocol':'paired_small_enemy_sr_v1','gameplay_qualified':False,'automatic_deployment':False}
    def publish(**kwargs):state.update(kwargs,updated_at=now());atomic_json(root/'status.json',state)
    publish()
    model,parent=load_sr_checkpoint(args.parent,'cpu');parent_steps=parent['training_steps'];del parent
    if sum(p.numel()for p in model.parameters())!=1517571:raise ValueError('Keep the existing1517571-parameter SR')
    model.to(device).float();optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    codec,codec_info=load_augmentation_codec(args.codec_checkpoint,device)
    train=SmallEnemyData(args.data_root,'train',seed=args.seed,device=device)
    val=SmallEnemyData(args.data_root,'val',seed=args.seed+1,device=device,exclude_scene_keys=train.scene_keys)
    panel=val.fixed_panel(args.validation_frames,seed=args.seed)
    protocol={'parent_sha256':_digest(args.parent),'codec_sha256':_digest(args.codec_checkpoint),
        'train':train.identity,'val':val.identity,'panel':panel,'codec_fraction':.25,'lr':args.lr,
        'batch':args.batch,'pixel':1.,'ssim':.2,'edge':.1,'hr_source':'reference_engine128','hr_interpolated':False}
    plan=root/'training_plan.json'
    if plan.exists():
        import json
        if json.loads(plan.read_text(encoding='utf-8'))!=protocol:raise ValueError('SR small-enemy immutable plan changed')
    else:atomic_json(plan,protocol)
    step=0;best=None;rng=np.random.default_rng(args.seed)
    checkpoint=root/'checkpoints/sr_last.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['protocol']!=protocol or args.steps<saved['completed_updates']:raise ValueError('SR small-enemy resume changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        step=saved['completed_updates'];best=saved['best'];train.restore_sampler(saved['sampler'])
        rng.bit_generator.state=saved['rng_numpy'];torch.set_rng_state(saved['rng_torch'].cpu())
        random.setstate(saved['rng_python'])
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu()for v in saved['rng_cuda']])
        del saved
    def save():_save(checkpoint,{'stage':'sr','config':model.config,'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),'completed_updates':step,'training_steps':parent_steps+step,
        'protocol':protocol,'sampler':train.sampler_state(),'best':best,'rng_numpy':rng.bit_generator.state,
        'rng_torch':torch.get_rng_state(),'rng_python':random.getstate(),
        'rng_cuda':torch.cuda.get_rng_state_all()if device.type=='cuda'else[]})
    import json
    baseline_path=root/'validation_initial.json'
    if baseline_path.exists():baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    else:
        if step:raise ValueError('SR resume lost fixed initial validation')
        baseline,preview=evaluate_small_sr(model,codec,val,panel,step=0)
        atomic_json(baseline_path,baseline);save_sr_preview(preview,root/'preview.png',every=100)
    publish(status='running',current=step,validation_initial=baseline,
        model_parameters={'total':1517571,'trainable':1517571},codec_frozen=True,
        optimizer_initialization='New AdamW for selected-best SR export; no mismatched latest18788 moments reused',
        sr_training={'parent_training_steps':parent_steps,'input_size':32,'output_size':128,'scale':4})
    first=step+1
    while step<args.steps:
        if (root/'stop_training').exists():save();publish(status='stopped',current=step);return
        model.train();codec.eval();batch=train.batch(args.batch,rng=rng)
        inputs=batch['lr'].clone();selected=np.flatnonzero(rng.random(args.batch)<.25)
        if len(selected):
            ids=torch.as_tensor(selected,device=device)
            with torch.no_grad():inputs[ids]=codec(inputs[ids])['reconstruction']
        optimizer.zero_grad(set_to_none=True);prediction=model(inputs)
        loss=balanced_image_l1(prediction,batch['hr'],batch['hr_object_mask'])
        loss=loss+.2*decoded_ssim_loss(prediction,batch['hr'],batch['hr_object_mask'])+.1*balanced_edge_l1(prediction,batch['hr'],batch['hr_object_mask'])
        loss.backward();norm=float(clip_finite_grad_norm_(model.named_parameters()));optimizer.step();step+=1
        metrics={'step':step,'train_loss':float(loss.detach()),'gradient_norm':norm,'codec_fraction':len(selected)/args.batch}
        with(root/'metrics.jsonl').open('a',encoding='utf-8')as stream:stream.write(json.dumps(metrics)+'\n')
        publish(current=step,global_training_step=step,loss=metrics['train_loss'],latest_metrics=metrics)
        if step%100==0 or step==args.steps:
            report,preview=evaluate_small_sr(model,codec,val,panel,step=step,baseline=baseline)
            if _hash_tensors(codec.state_dict())!=codec_info['tensor_sha256']:raise AssertionError('Frozen codec changed in SR stage')
            report['codec_all_tensors_bitexact']=True
            atomic_json(root/f'validation/step_{step:06d}.json',report);atomic_json(root/'validation.json',report)
            save_sr_preview(preview,root/'preview.png',every=100)
            eligible=report['small_enemy_stage_gate']['passed']
            if best is None or eligible and not best['eligible'] or eligible==best['eligible']and report['selection_score']<best['score']:
                best={'step':step,'score':report['selection_score'],'eligible':eligible}
                _save(root/'checkpoints/sr_best.pt',{'model':model.state_dict(),'step':step,'report':report})
            publish(validation=report,selected_best=best);save()
        elif step==first:save()
    chosen=torch.load(root/'checkpoints/sr_best.pt',map_location='cpu',weights_only=False)
    _save(root/'checkpoints/sr_for_game.pt',{'stage':'sr','format_version':1,'config':model.config,
        'model':chosen['model'],'input_size':32,'output_size':128,'scale':4,
        'training_steps':parent_steps+chosen['step'],'refinement_updates':chosen['step'],
        'validation':chosen['report'],'source_quality_passed':chosen['report']['source_quality_passed'],
        'small_enemy_stage_gate':chosen['report']['small_enemy_stage_gate'],'gameplay_qualified':False,'automatic_deployment':False})
    atomic_json(root/'validation_selected.json',chosen['report'])
    publish(status='completed',current=step,selected_step=chosen['step'],validation=chosen['report'],
        small_enemy_stage_gate=chosen['report']['small_enemy_stage_gate'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','data-root','parent','codec-checkpoint'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2000);parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--validation-frames',type=int,default=512);parser.add_argument('--seed',type=int,default=173)
    parser.add_argument('--lr',type=float,default=5e-5);parser.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    args=parser.parse_args(argv)
    try:
        with KeepAwake():run(args)
    except BaseException as error:
        atomic_json(args.run_dir/'failure.json',{'error':f'{type(error).__name__}: {error}','timestamp':now()});raise


if __name__=='__main__':main()
