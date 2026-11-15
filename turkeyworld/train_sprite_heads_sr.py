"""Train sprite heads on final RF3 NPR pairs; freeze all SR9000 wall pixels."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from .distill_renderer import _save
from .enemy_animation import enemy_animation_index
from .gradient_safety import clip_finite_grad_norm_
from .models import build_upscaler
from .neural_enemy_restoration_data import NeuralEnemyRestorationData
from .patrol_sprite_sr import patrol_alpha
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .sr_sprite_heads import PROTOCOL,WEIGHTS,GROUND_PROTOCOL,GROUND_WEIGHTS,configure,objective,sample_indices
from .atlas_enemy_objectives import source_full_sprite_geometry
from .train_background_detail_sr import file_sha
from .train_background_detail_sr import restore_rng
from .training_pipeline import atomic_json,now
from .workspace_paths import ROOT


def run(args):
    root=args.run_dir;root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=args.source_checkpoint or ROOT/'training_runs/sr_wall_patchgan_20261115/checkpoints/sr_last.pt'
    protocol=GROUND_PROTOCOL if args.ground_align else PROTOCOL
    weights=GROUND_WEIGHTS if args.ground_align else WEIGHTS
    parent=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    base_step=parent['training_steps'];head_source_updates=parent.get('sprite_updates',0)
    if base_step<9000:raise ValueError('Expected SR9000 or its complete sprite-head descendant')
    train=NeuralEnemyRestorationData(ROOT/'training_runs/image_chain_20261114/data/paired','train',seed=6177,device='cuda')
    val=NeuralEnemyRestorationData(ROOT/'training_runs/image_chain_20261114/data/paired','val',seed=6178,device='cuda')
    if train.scene_keys&val.scene_keys:raise ValueError('TRAIN/VAL overlap')
    original=json.loads((ROOT/'training_runs/sr_background_detail_20261114/training_plan.json').read_text());panel=original['panel']
    plan=dict(protocol=protocol,source_sha256=file_sha(source),source_step=base_step,updates=args.updates,
        lr=1e-5,weights=weights,ground_alignment=args.ground_align,threshold_unchanged=.9,patrol_visible=4,combat_negative=2,empty=2,
        train_identity=train.identity,val_identity=val.identity,panel=panel,panel_fingerprint=val.panel_fingerprint(panel),
        ground_projection=args.ground_project,frozen_RGB_SR9000=True,combat_animation_sprites=False,navigation_sprites=False,Source_inputs=False)
    if (root/'training_plan.json').exists() and json.loads((root/'training_plan.json').read_text())!=plan:raise ValueError('Sprite plan changed')
    atomic_json(root/'training_plan.json',plan);last=root/'checkpoints/sr_last.pt'
    saved=torch.load(last,map_location='cpu',weights_only=False,mmap=True) if last.exists() else None
    if saved is not None and saved['training_plan']!=plan:raise ValueError('Sprite checkpoint plan changed')
    config=deepcopy(parent['config'])
    if args.ground_project:config['sprite_config']['atlas_ground_anchor']=True
    plan['ground_projection']=args.ground_project
    atomic_json(root/'training_plan.json',plan)
    model=build_upscaler(config);model.load_state_dict(parent['model'] if saved is None else saved['model'],strict=True)
    model.to('cuda').float();names=configure(model);parameters=dict(model.named_parameters())
    optimizer=torch.optim.Adam([parameters[n] for n in names],lr=plan['lr'])
    if saved is not None:optimizer.load_state_dict(saved['sprite_optimizer'])
    elif 'sprite_optimizer' in parent:
        if parent['sprite_optimizer_names']!=names:raise ValueError('Sprite optimizer names changed')
        optimizer.load_state_dict(parent['sprite_optimizer'])
    count=0 if saved is None else saved['training_steps']-base_step;rng=np.random.default_rng(6177)
    if saved is not None:rng.bit_generator.state=saved['sprite_sampler_rng']
    frozen=lambda:{k:v for k,v in model.state_dict().items() if k not in names}
    frozen_sha=_hash_tensors(frozen())
    if saved is not None and saved['frozen_sha256']!=frozen_sha:raise ValueError('Frozen wall/SR/atlas tensors changed')
    reference=build_upscaler(parent['config']);reference.load_state_dict(parent['model'],strict=True);reference.to('cuda').float().eval().requires_grad_(False)
    restore_rng(saved,seed=6177,device=torch.device('cuda'))
    status=dict(stage='sr',run_id=root.name,model='SR9000 + trained patrol sprite heads',current=base_step+count,
        total=base_step+args.updates,objective_protocol=protocol,loss_protocol=protocol,
        sprite_head_training=plan,prediction_every=100,started_at=now(),
        sr_training=dict(input_size=32,output_size=128,scale=4,all_other_models_frozen=True))
    def publish(**values):
        status.update(current=base_step+count,sprite_updates=head_source_updates+count,status='running',updated_at=now(),**values)
        atomic_json(root/'status.json',status)
    def save():
        _save(last,dict(stage='sr',config=model.config,model=model.state_dict(),training_steps=base_step+count,
            sprite_updates=head_source_updates+count,sprite_optimizer=optimizer.state_dict(),sprite_optimizer_names=names,
            source_detail_optimizer=parent.get('source_detail_optimizer') or parent.get('optimizer'),
            source_detail_optimizer_names=parent.get('source_detail_optimizer_names') or parent.get('optimizer_parameter_names'),
            source_detail_step=9000,source_checkpoint_sha256=plan['source_sha256'],training_plan=plan,
            sprite_sampler_rng=deepcopy(rng.bit_generator.state),frozen_sha256=frozen_sha,
            rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all()))
    @torch.no_grad()
    def evaluate():
        model.eval();bottom_errors=[];center_errors=[];numbers=dict(patrol=0,detected=0,empty=0,false_positive=0,visible=0,phase_correct=0);total=0.
        for first in range(0,len(panel),8):
            batch=val.batch(len(panel[first:first+8]),indices=panel[first:first+8])
            raw=model.sprite_renderer.forward_with_aux(batch['lr_npr']);loss,_=objective(raw,batch,ground=args.ground_align);total+=float(loss)*len(batch['state'])
            target_box=source_full_sprite_geometry(batch['state'])['full_bbox'];pred_box=raw['visual_full_bbox']
            phase=enemy_animation_index(batch['state']);visible=batch['mask_hr'].flatten(1).any(-1)
            alpha=patrol_alpha(raw['visual_composite_alpha_hr'],raw['visual_animation_index'])
            active=(alpha>.01).flatten(1).any(-1);overlap=(alpha*batch['mask_hr']).flatten(1).sum(-1)>0
            patrol=visible&(phase<3)
            bottom_errors.extend(((pred_box[:,3]-target_box[:,3]).abs()*128)[patrol].cpu().tolist())
            center_errors.extend((((pred_box[:,1]+pred_box[:,3]-target_box[:,1]-target_box[:,3])/2).abs()*128)[patrol].cpu().tolist())
            numbers['patrol']+=int(patrol.sum());numbers['detected']+=int((patrol&active&overlap).sum())
            numbers['empty']+=int((~visible).sum());numbers['false_positive']+=int((~visible&active).sum())
            numbers['visible']+=int(visible.sum());numbers['phase_correct']+=int((visible&(phase==raw['visual_animation_index'])).sum())
        report=dict(step=base_step+count,protocol=protocol,panel_fingerprint=plan['panel_fingerprint'],
            model_sha256=_hash_tensors(model.state_dict()),source_held_out=True,Source_inputs=False,
            val_loss=total/len(panel),bottom_mae_px=float(np.mean(bottom_errors)),center_y_mae_px=float(np.mean(center_errors)),**numbers,
            patrol_recall=numbers['detected']/max(1,numbers['patrol']),
            empty_false_positive_rate=numbers['false_positive']/max(1,numbers['empty']),
            phase_accuracy=numbers['phase_correct']/max(1,numbers['visible']))
        atomic_json(root/'validation.json',report);atomic_json(root/'validation'/f'step_{base_step+count:06d}.json',report)
        if count==0:atomic_json(root/'validation_initial.json',report)
        ids=[]
        for distance in ('near','mid','far'):
            ids.append(next(i for i in panel if val.visible[i] and val.animation_labels[i]<3 and val.distance[i]==distance))
        ids.append(next(i for i in panel if not val.visible[i]));batch=val.batch(4,indices=ids)
        previous=reference(batch['lr_npr']);current=model(batch['lr_npr'])
        raw_near=model.sprite_renderer.forward_with_aux(batch['lr_npr'][:1]);target_near=source_full_sprite_geometry(batch['state'][:1])
        atomic_json(root/'near_geometry.json',dict(step=base_step+count,row_index=ids[0],target_bbox=target_near['full_bbox'][0].tolist(),predicted_bbox=raw_near['visual_full_bbox'][0].tolist()))
        labels=['Оригинал Source128','Нейрорендер32',f'SR{base_step} до коррекции','SR с дообученными головами спрайта']
        canvas=Image.new('RGB',(1024,1120),(14,22,31));draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
        for row,images in enumerate((batch['hr_source'],batch['lr_npr'],previous,current)):
            draw.text((8,row*280+3),labels[row],font=font,fill=(230,240,235))
            for column,image in enumerate(images):
                pixels=image.clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
                canvas.paste(Image.fromarray(pixels).resize((256,256),Image.Resampling.NEAREST),(column*256,row*280+24))
        temp=root/'preview.tmp.png';canvas.save(temp);temp.replace(root/'preview.png')
        preview=dict(kind='sprite_heads_sr_preview_v1',step=base_step+count,every=100,row_count=4,row_labels=labels,
            model_sha256=report['model_sha256'],panel_fingerprint=plan['panel_fingerprint'],
            source_held_out=True,Source_inputs=False,world_rollout=False,generated_at=now())
        atomic_json(root/'preview.json',preview);publish(validation=report,prediction_preview=preview)
        return report
    atomic_json(root/'preparation.json',dict(source_step=base_step,source_sha256=plan['source_sha256'],
        trainable_names=names,trainable_parameters=sum(parameters[n].numel() for n in names),
        frozen_sha256=frozen_sha,detail_optimizer_archived=True,head_optimizer='continued_Adam_1e-5' if 'sprite_optimizer' in parent else 'fresh_Adam_1e-5'))
    publish();evaluate();save()
    while count<args.updates and not (root/'stop_training').exists():
        model.eval();model.sprite_renderer.train()
        batch=train.batch(8,indices=sample_indices(train,rng,count))
        optimizer.zero_grad(set_to_none=True);raw=model.sprite_renderer.forward_with_aux(batch['lr_npr'])
        loss,metrics=objective(raw,batch,ground=args.ground_align);loss.backward()
        clip_finite_grad_norm_(model.named_parameters(),context='sprite-head calibration');optimizer.step();count+=1
        metrics.update(step=base_step+count,stage='sr',objective_protocol=protocol,loss_protocol=protocol,lr=plan['lr'])
        with (root/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
        publish(loss=metrics['train_loss'],latest_metrics=metrics);save()
        if count%100==0:
            if _hash_tensors(frozen())!=frozen_sha:raise AssertionError('Sprite refinement changed walls/atlas pixels')
            evaluate()
    status.update(status='completed' if count==args.updates else 'stopped');atomic_json(root/'status.json',status)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--updates',type=int,default=3000);p.add_argument('--source-checkpoint',type=Path);p.add_argument('--ground-align',action='store_true');p.add_argument('--ground-project',action='store_true');a=p.parse_args()
    a.run_dir.mkdir(parents=True,exist_ok=True)
    from .process_runtime import pid_file_alive,write_pid_file
    if pid_file_alive(a.run_dir/'training.pid'):raise ValueError('Sprite head worker already running')
    write_pid_file(a.run_dir/'training.pid')
    try:
        with KeepAwake():run(a)
    except BaseException as error:
        atomic_json(a.run_dir/'failure.json',dict(error=f'{type(error).__name__}: {error}'));raise


if __name__=='__main__':main()
