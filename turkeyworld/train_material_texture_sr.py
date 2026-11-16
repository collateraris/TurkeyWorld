"""Low-LR wall PatchGAN continuation of the complete patrol SR6000 checkpoint."""
import argparse
from copy import deepcopy
import json
import random
from pathlib import Path
import time

import numpy as np
import torch

from .sr_material_texture import material_labels,texture_loss
from .background_detail_objectives import background_detail_loss,LOSS_WEIGHTS
from .background_detail_sr_validation import evaluate_background_detail_sr,save_background_detail_sr_preview
from .distill_renderer import _save
from .gradient_safety import clip_finite_grad_norm_
from .models import build_upscaler
from .neural_enemy_restoration_data import NeuralEnemyRestorationData
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .sr_wall_patchgan import (SETTINGS,WallPatchDiscriminator,wall_labels,wall_reconstruction,
    wall_patch_pairs,discriminator_loss,generator_loss,gan_weight)
from .train_background_detail_sr import restore_detail_optimizer,restore_rng,file_sha
from .training_pipeline import atomic_json,now
from .workspace_paths import ROOT

SETTINGS={**SETTINGS,'protocol':'material_texture_patchgan_v2','texture_strata':['brick','mortar','earth','grass','grass_edge'],'texture_weight':1.}
PROTOCOL=SETTINGS['protocol']


def reconstruction(output,batch):
    loss,metrics=background_detail_loss(output,batch)
    texture,texture_metrics=texture_loss(output['image'],batch['hr_source'],material_labels(batch))
    loss=loss+texture
    metrics.update(texture_metrics,texture_loss=float(texture.detach()))
    wall,details=wall_reconstruction(output['image'],batch['hr_source'],wall_labels(batch))
    metrics.update(details,alpha_used=True,atlas_pixels_used=True,GAN_active=False,
        reconstruction_loss=float((loss+wall).detach()))
    return loss+wall,metrics


def run(args):
    root=args.run_dir;root.mkdir(parents=True,exist_ok=True);device=torch.device(args.device)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=root/'assets/source.pt'
    parent=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    base_step=parent['training_steps']
    if base_step!=12613:raise ValueError('Expected corrected patrol SR12613 parent')
    plan=dict(protocol=PROTOCOL,source=str(source),source_sha256=file_sha(source),source_step=base_step,
        target_step=args.steps,settings=SETTINGS,Source_labels_only=True,
        frozen_sprites=True,combat_animation_sprites=False,navigation_sprites=False)
    planpath=root/'training_plan.json'
    if planpath.exists() and json.loads(planpath.read_text())!=plan:raise ValueError('Wall GAN plan changed')
    atomic_json(planpath,plan);last=root/'checkpoints/sr_last.pt'
    saved=torch.load(last,map_location='cpu',weights_only=False,mmap=True) if last.exists() else parent
    if last.exists() and saved['training_plan']!=plan:raise ValueError('GAN checkpoint plan changed')
    model=build_upscaler(parent['config']);model.load_state_dict(saved['model'],strict=True)
    model.to(device).float().configure_training();step=saved['training_steps']
    detail_saved={**saved,'optimizer':saved.get('source_detail_optimizer') or saved.get('optimizer'),'optimizer_parameter_names':saved.get('source_detail_optimizer_names') or saved.get('optimizer_parameter_names')}
    optimizer,names=restore_detail_optimizer(model,detail_saved,step=9000 if not last.exists() else step-base_step+9000,learning_rate=SETTINGS['generator_lr'])
    for group in optimizer.param_groups:group['lr']=SETTINGS['generator_lr']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6151);discriminator=WallPatchDiscriminator()
    discriminator.to(device).float()
    d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=SETTINGS['discriminator_lr'],betas=tuple(SETTINGS['discriminator_betas']))
    if last.exists():
        discriminator.load_state_dict(saved['discriminator']);d_optimizer.load_state_dict(saved['discriminator_optimizer'])
    if any(group['lr']!=SETTINGS['discriminator_lr'] for group in d_optimizer.param_groups):
        raise ValueError('Discriminator low learning rate changed on resume')
    d_steps=saved.get('discriminator_steps',0) if last.exists() else 0;crop_rng=np.random.default_rng(6151)
    if last.exists():crop_rng.bit_generator.state=saved['crop_rng']
    data=ROOT/'training_runs/image_chain_20261114/data/paired'
    train=NeuralEnemyRestorationData(data,'train',seed=173,device=device)
    val=NeuralEnemyRestorationData(data,'val',seed=174,device=device)
    sampler=saved['sampler'] if last.exists() else torch.load(ROOT/'training_runs/sr_wall_patchgan_20261115/checkpoints/sr_last.pt',map_location='cpu',weights_only=False,mmap=True)['sampler']
    train.restore_sampler(sampler);assert train.draws==(step-base_step+9000)*8
    restore_rng(saved,seed=173,device=device)
    frozen=lambda:{k:v for k,v in model.state_dict().items() if not k.startswith('detail_residual.')}
    frozen_sha=_hash_tensors(frozen())
    if last.exists() and frozen_sha!=saved['frozen_sha256']:raise ValueError('Frozen SR/sprite weights changed')
    old_plan=json.loads((ROOT/'training_runs/sr_background_detail_20261114/training_plan.json').read_text())
    panel=old_plan['panel'];detail=old_plan['detail_animation_panel']
    status=dict(run_id=root.name,stage='sr',model='SR12613 + brick, earth and grass texture PatchGAN',total=args.steps,
        started_at=now(),objective_protocol=PROTOCOL,loss_protocol=PROTOCOL,
        sr_training=dict(input_size=32,output_size=128,scale=4,all_other_models_frozen=True),
        wall_patchgan_training=deepcopy(plan),validation_every=100,prediction_every=100)
    def publish(**extra):
        status.update(status='running',current=step,updated_at=now(),**extra);atomic_json(root/'status.json',status)
    def save():
        _save(last,dict(stage='sr',config=model.config,model=model.state_dict(),training_steps=step,
            optimizer=optimizer.state_dict(),optimizer_parameter_names=names,sampler=train.sampler_state(),
            rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],training_plan=plan,
            frozen_sha256=frozen_sha,source_training_steps=base_step,sprite_optimizer=parent.get('sprite_optimizer'),sprite_optimizer_names=parent.get('sprite_optimizer_names'),discriminator=discriminator.state_dict(),
            discriminator_optimizer=d_optimizer.state_dict(),discriminator_steps=d_steps,
            crop_rng=deepcopy(crop_rng.bit_generator.state)))
    @torch.no_grad()
    def validate():
        report,preview=evaluate_background_detail_sr(model,val,panel,step=step,batch_size=8,detail_panel=detail)
        report['wall_patchgan_protocol']=PROTOCOL;measured=[];model.eval()
        for first in range(0,len(panel),8):
            ids=panel[first:first+8];batch=val.batch(len(ids),indices=ids)
            output=model.forward_with_aux(batch['lr_npr'])
            _,metrics=wall_reconstruction(output['image'],batch['hr_source'],wall_labels(batch))
            _,texture_metrics=texture_loss(output['image'],batch['hr_source'],material_labels(batch));metrics.update(texture_metrics);measured.append((len(ids),metrics))
        wall={key:sum(n*m[key] for n,m in measured)/len(panel) for key in measured[0][1] if key!='wall_images' and key!='wall_reconstruction_loss'}
        baselinepath=root/'wall_validation_initial.json'
        if not baselinepath.exists():
            if step!=base_step:raise ValueError('Missing original wall validation')
            atomic_json(baselinepath,dict(step=step,panel_fingerprint=val.panel_fingerprint(panel),**wall))
        baseline=json.loads(baselinepath.read_text())
        assert baseline['panel_fingerprint']==val.panel_fingerprint(panel)
        report['wall_details']=dict(current=wall,baseline=baseline)
        atomic_json(root/'validation.json',report);atomic_json(root/'validation'/f'step_{step:06d}.json',report)
        metadata=save_background_detail_sr_preview(preview,root/'preview.png',every=100)
        publish(validation=report,prediction_preview=metadata,wall_details=report['wall_details'])
    atomic_json(root/'resume_verification.json',dict(passed=True,step=step,Adam_states=len(optimizer.state),
        sampler_draws=train.draws,model_bitexact=all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in saved['model'].items()),
        generator_lr=SETTINGS['generator_lr'],discriminator_lr=SETTINGS['discriminator_lr'],frozen_sha256=frozen_sha))
    publish();validate();save()
    while step<args.steps and not (root/'stop_training').exists():
        began=time.perf_counter();model.train();discriminator.train();batch=train.batch(8)
        optimizer.zero_grad(set_to_none=True);d_optimizer.zero_grad(set_to_none=True)
        output=model.forward_with_aux(batch['lr_npr']);base,metrics=reconstruction(output,batch)
        pairs=wall_patch_pairs(batch['lr_npr'],output['image'],batch['hr_source'],wall_labels(batch),crop_rng)
        weight=gan_weight(step-base_step+1);effective=0.;d_loss=base.detach()*0;adversarial=base*0
        extra=dict(adversarial_loss=0.,feature_matching_loss=0.);base_norm=adv_norm=0.
        if pairs is not None:
            d_loss=discriminator_loss(discriminator,pairs);d_loss.backward()
            clip_finite_grad_norm_(discriminator.named_parameters(),context='wall discriminator');d_optimizer.step();d_steps+=1
            d_optimizer.zero_grad(set_to_none=True)
            if weight:
                adversarial,extra=generator_loss(discriminator,pairs)
                probe=model.detail_residual.output.weight
                base_norm=float(torch.autograd.grad(base,probe,retain_graph=True)[0].norm())
                adv_norm=float(torch.autograd.grad(adversarial,probe,retain_graph=True)[0].norm())
                effective=min(weight,SETTINGS['gradient_ratio_limit']*base_norm/max(adv_norm,1e-30))
        total=base+effective*adversarial
        if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite wall GAN loss')
        total.backward();clip_finite_grad_norm_(model.named_parameters(),context='wall PatchGAN SR');optimizer.step();step+=1
        metrics.update(extra,step=step,stage='sr',train_loss=float(total.detach()),objective_protocol=PROTOCOL,loss_protocol=PROTOCOL,
            GAN_active=effective>0,gan_weight_scheduled=weight,gan_weight_effective=effective,
            adversarial_gradient_ratio=effective*adv_norm/max(base_norm,1e-30),discriminator_loss=float(d_loss.detach()),
            discriminator_steps=d_steps,wall_patches=0 if pairs is None else len(pairs['coordinates']),
            generator_lr=SETTINGS['generator_lr'],discriminator_lr=SETTINGS['discriminator_lr'],update_seconds=time.perf_counter()-began)
        with (root/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
        publish(loss=metrics['train_loss'],latest_metrics=metrics);save()
        if step%100==0:
            if _hash_tensors(frozen())!=frozen_sha:raise AssertionError('Frozen assets changed')
            validate();save()
    status.update(status='completed' if step==args.steps else 'stopped');atomic_json(root/'status.json',status)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=15613);p.add_argument('--device',default='cuda');args=p.parse_args()
    from .process_runtime import pid_file_alive,write_pid_file
    args.run_dir.mkdir(parents=True,exist_ok=True)
    if pid_file_alive(args.run_dir/'training.pid'):raise ValueError('Wall PatchGAN worker already active')
    write_pid_file(args.run_dir/'training.pid')
    try:
        with KeepAwake():run(args)
    except BaseException as error:
        path=args.run_dir/'status.json';state=json.loads(path.read_text()) if path.exists() else {}
        atomic_json(path,{**state,'status':'failed','error':f'{type(error).__name__}: {error}','updated_at':now()})
        raise


if __name__=='__main__':main()
