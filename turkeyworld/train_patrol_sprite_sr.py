"""Continue SR3000 detail Adam with frozen patrol sprites and held-out reports."""
import argparse
from copy import deepcopy
import json
import random
from pathlib import Path

import numpy as np
import torch

from .background_detail_objectives import background_detail_loss, LOSS_WEIGHTS
from .background_detail_sr_validation import evaluate_background_detail_sr, save_background_detail_sr_preview
from .distill_renderer import _save
from .gradient_safety import clip_finite_grad_norm_
from .models import build_upscaler
from .neural_enemy_restoration_data import NeuralEnemyRestorationData
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .train_background_detail_sr import restore_detail_optimizer, restore_rng, file_sha
from .training_pipeline import atomic_json, now
from .workspace_paths import ROOT


def run(args):
    root=args.run_dir;root.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'training_runs/sr_background_detail_20261114/checkpoints/sr_last.pt'
    atlas=ROOT/'training_runs/sr_atlas_enemy_20261113/training/checkpoints/sr_last.pt'
    parent=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    sprite=torch.load(atlas,map_location='cpu',weights_only=False,mmap=True)
    last=root/'checkpoints/sr_last.pt'
    config={**parent['config'],'architecture':'patrol_sprite_detail_sr','sprite_config':sprite['config']}
    plan=dict(protocol='patrol_sprite_detail_sr_v1',parent_sha256=file_sha(source),
        sprite_sha256=file_sha(atlas),source_step=3000,target_step=args.steps,
        sprite_frames=[0,1,2],excluded_sprite_frames=[3,4,5,6],
        navigation_sprites=False,Source_inputs=False,
        frozen_sprite_placement=True,detail_Adam_sampler_RNG_continued=True)
    path=root/'training_plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('SR continuation plan changed')
    atomic_json(path,plan)
    saved=torch.load(last,map_location='cpu',weights_only=False,mmap=True) if last.exists() else parent
    model=build_upscaler(config)
    if last.exists():model.load_state_dict(saved['model'],strict=True)
    else:
        values={**parent['model'],**{'sprite_renderer.'+k:v for k,v in sprite['model'].items()}}
        model.load_state_dict(values,strict=True)
    model.to(device).float().configure_training()
    step=saved['training_steps']
    optimizer,names=restore_detail_optimizer(model,saved,step=step)
    data=ROOT/'training_runs/image_chain_20261114/data/paired'
    train=NeuralEnemyRestorationData(data,'train',seed=173,device=device)
    val=NeuralEnemyRestorationData(data,'val',seed=174,device=device)
    train.restore_sampler(saved['sampler'])
    if train.draws!=step*8:raise ValueError('Inherited sampler clock changed')
    restore_rng(saved,seed=173,device=device)
    frozen=lambda:{k:v for k,v in model.state_dict().items() if not k.startswith('detail_residual.')}
    frozen_sha=_hash_tensors(frozen())
    panel=parent['training_plan']['panel'];detail=parent['training_plan']['detail_animation_panel']
    status=dict(run_id=root.name,stage='sr',model='SR3000 + patrol sprites only; RGB combat details',
        total=args.steps,started_at=now(),prediction_every=100,validation_every=100,
        sr_training=dict(input_size=32,output_size=128,scale=4,all_other_models_frozen=True),
        sprite_frames=[0,1,2],combat_animation_sprites=False,navigation_sprites=False)
    def publish(**extra):
        status.update(current=step,status='running',updated_at=now(),**extra)
        atomic_json(root/'status.json',status)
    def save():
        _save(last,dict(stage='sr',config=model.config,model=model.state_dict(),training_steps=step,
            optimizer=optimizer.state_dict(),optimizer_parameter_names=names,sampler=train.sampler_state(),
            rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            training_plan=plan,frozen_sha256=frozen_sha,source_training_steps=3000))
    def validate():
        report,preview=evaluate_background_detail_sr(model,val,panel,step=step,batch_size=8,detail_panel=detail,
            objective_fn=objective,objective_weights=LOSS_WEIGHTS)
        report['sprite_policy']=plan
        atomic_json(root/'validation.json',report)
        atomic_json(root/'validation'/f'step_{step:06d}.json',report)
        metadata=save_background_detail_sr_preview(preview,root/'preview.png',every=100)
        publish(validation=report,prediction_preview=metadata)
    def objective(output,batch):
        loss,metrics=background_detail_loss(output,batch)
        metrics.update(alpha_used=True,atlas_pixels_used=True,patrol_sprites_only=True,
            combat_animation_sprites=False,navigation_sprites=False)
        return loss,metrics
    atomic_json(root/'resume_verification.json',dict(passed=True,step=step,Adam_states=len(optimizer.state),
        sampler_draws=train.draws,original_model_bitexact=all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in parent['model'].items()) if not last.exists() else None,
        frozen_sha256=frozen_sha))
    publish();validate();save()
    while step<args.steps and not (root/'stop_training').exists():
        model.train();batch=train.batch(8)
        optimizer.zero_grad(set_to_none=True)
        result=model.forward_with_aux(batch['lr_npr'])
        loss,metrics=objective(result,batch)
        loss.backward();clip_finite_grad_norm_(model.named_parameters(),context='patrol sprite SR')
        optimizer.step();step+=1
        with (root/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=step,**metrics))+'\n')
        publish(loss=float(loss.detach()),latest_metrics=metrics);save()
        if step%100==0:
            if _hash_tensors(frozen())!=frozen_sha:raise AssertionError('Frozen SR or sprites changed')
            validate()
    status.update(status='completed' if step==args.steps else 'stopped',current=step)
    atomic_json(root/'status.json',status)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--device',default='cuda')
    args=p.parse_args()
    with KeepAwake():run(args)


if __name__=='__main__':main()
