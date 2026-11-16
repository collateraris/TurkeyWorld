"""Independent RRDB architecture migration; old checkpoints remain untouched."""
import argparse
from copy import deepcopy
import json,random,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from .models import build_upscaler
from .rrdb_material_sr import ARCHITECTURE
from .sr_material_texture import material_labels,texture_loss
from .sr_wall_patchgan import (WallPatchDiscriminator,wall_labels,wall_reconstruction,wall_patch_pairs,discriminator_loss,generator_loss)
from .neural_enemy_restoration_data import NeuralEnemyRestorationData
from .gradient_safety import clip_finite_grad_norm_
from .renderer_student import _hash_tensors
from .distill_renderer import _save
from .train_background_detail_sr import file_sha,restore_rng
from .training_pipeline import atomic_json,now
from .workspace_paths import ROOT
from .power_guard import KeepAwake

PROTOCOL='rrdb_material_sprite_sr_v1'

def objective(output,batch):
    image=output['image'];target=batch['hr_source'];labels=material_labels(batch)
    texture,metrics=texture_loss(image,target,labels)
    wall,wm=wall_reconstruction(image,target,wall_labels(batch));metrics.update(wm)
    l1=(image-target).abs().mean()
    edge=sum((image.diff(dim=d)-target.diff(dim=d)).abs().mean() for d in (-1,-2))/2
    classes=torch.zeros_like(labels['brick'][:,0],dtype=torch.long)
    for i,key in enumerate(('brick','earth','grass'),1):classes[labels[key][:,0]>.5]=i
    # Every material is normalized separately so small grass regions matter.
    ce=F.cross_entropy(output['material_logits'],classes,reduction='none')
    material=torch.stack([ce[classes==i].mean() for i in range(4) if bool((classes==i).any())]).mean()
    seam_target=torch.cat((labels['mortar'],labels['grass_edge']),1)
    bce=F.binary_cross_entropy_with_logits(output['seam_logits'],seam_target,reduction='none')
    seam_terms=[];exposed=wall_labels(batch).expand_as(seam_target)
    for positive in (True,False):
        support=(seam_target>.5 if positive else seam_target<=.5)&(exposed>.5)
        if bool(support.any()):seam_terms.append(bce[support].mean())
    seam=torch.stack(seam_terms).mean() if seam_terms else bce.sum()*0
    loss=l1+.2*edge+texture+wall+.2*material+.1*seam
    metrics.update(image_l1=float(l1.detach()),image_edge=float(edge.detach()),texture_loss=float(texture.detach()),material_ce=float(material.detach()),seam_bce=float(seam.detach()))
    if output.get('weapon_memory') is not None:
        from .weapon_memory_sr import weapon_objective
        weapon,wm=weapon_objective(output,batch);loss=loss+weapon;metrics.update(wm)
    if output.get('wall_memory') is not None:
        from .wall_texture_memory_sr import wall_memory_objective
        memory,mm=wall_memory_objective(output,batch);loss=loss+memory;metrics.update(mm)
        if 'column_logits' in output['wall_memory']:
            from .wall_texture_memory_sr import column_objective
            column,cm=column_objective(output['wall_memory'],batch);loss=loss+column;metrics.update(cm)
    return loss,metrics

def run(args):
    root=args.run_dir;root.mkdir(parents=True,exist_ok=True);device=torch.device(args.device)
    torch.set_num_threads(2);torch.manual_seed(6211)
    source=root/'assets/source.pt';parent=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    config=dict(architecture=ARCHITECTURE,channels=48,growth=24,blocks=6,scale=4,sprite_config=parent['config']['sprite_config'])
    if args.weapon_memory:
        if parent['config'].get('architecture')!=ARCHITECTURE:raise ValueError('Weapon continuation requires RRDB parent')
        config={**parent['config'],'weapon_memory':True}
    if args.wall_texture_memory:
        if parent['config'].get('architecture')!=ARCHITECTURE:raise ValueError('Wall memory needs RRDB parent')
        config={**parent['config'],'wall_texture_memory':True,'wall_geometry_supervised':args.wall_geometry_supervised,'wall_coherent_uv':args.wall_coherent_uv,'wall_texture_v2':args.wall_texture_v2}
    if not args.wall_texture_v2:config.pop("wall_texture_v2",None)
    base_only=getattr(args,'base_only',False)
    if base_only:config=deepcopy(parent['config'])
    continuation=(args.weapon_memory or args.wall_texture_memory) and not base_only
    plan=dict(protocol=PROTOCOL,source_sha256=file_sha(source),source_step=parent['training_steps'],config=config,target_step=args.steps,
        generator_lr=2e-5,discriminator_lr=5e-6,max_gan_weight=.002,gradient_ratio_limit=.05,gan_warmup=500,gan_ramp=500,
        wall_texture_v2=args.wall_texture_v2,wall_coherent_uv=args.wall_coherent_uv,wall_geometry_supervised=args.wall_geometry_supervised,wall_texture_memory=args.wall_texture_memory,wall_memory_lr=1e-4 if args.wall_texture_memory else None,fresh_RGB_optimizer=not continuation,weapon_memory=args.weapon_memory,weapon_lr=1e-4 if args.weapon_memory else None,bitexact_sprite_transfer=True,Source_inputs=False,Source_labels_only=True,combat_animation_sprites=False,navigation_sprites=False,
        material_classes=['other','brick','earth','grass'],seam_classes=['mortar','grass_edge'],distillation_initial_weight=.25,distillation_updates=500)
    data=getattr(args,'data_root',None) or ROOT/'training_runs/image_chain_20261114/data/paired'
    if base_only:
        manifest=json.loads((data/'manifest.json').read_text())
        if manifest['cache_identity'].get('renderer_input')!='base_only' or manifest['cache_identity'].get('flow_matching') is not False:
            raise ValueError('Base-only adaptation refuses RF-generated input data')
        plan.update(renderer_input='base_only',flow_matching=False,codec=False,data_root=str(data.resolve()),
            data_manifest_sha256=file_sha(data/'manifest.json'),sprite_head_lr=1e-5,
            frozen_sprite_pixels=True,sprite_heads_adapted=True,
            weapon_memory=bool(config.get('weapon_memory')),wall_texture_memory=bool(config.get('wall_texture_memory')),
            wall_geometry_supervised=bool(config.get('wall_geometry_supervised')),wall_coherent_uv=bool(config.get('wall_coherent_uv')),
            wall_texture_v2=bool(config.get('wall_texture_v2')))
    path=root/'training_plan.json'
    if path.exists():
        existing=json.loads(path.read_text())
        # Preserve old immutable plans when newly added optional features are off.
        for key in list(plan):
            if key not in existing and (plan[key] is False or plan[key] is None):plan.pop(key)
        if existing!=plan:raise ValueError('Immutable RRDB plan changed')
    atomic_json(path,plan);last=root/'checkpoints/sr_last.pt';saved=torch.load(last,map_location='cpu',weights_only=False,mmap=True) if last.exists() else None
    if saved is not None and saved['training_plan']!=plan:raise ValueError('Checkpoint plan mismatch')
    model=build_upscaler(config)
    sprite_weights={k[len('sprite_renderer.'):]:v for k,v in parent['model'].items() if k.startswith('sprite_renderer.')}
    model.sprite_renderer.load_state_dict(sprite_weights,strict=True)
    if saved:model.load_state_dict(saved['model'],strict=True)
    elif base_only:model.load_state_dict(parent['model'],strict=True)
    elif args.weapon_memory:
        incompatible=model.load_state_dict(parent['model'],strict=False)
        if incompatible.unexpected_keys or any(not n.startswith('weapon_memory.') for n in incompatible.missing_keys):raise ValueError('Unexpected RRDB migration differences')
        from .weapon_memory_sr import initialize_weapon_memory
        initialize_weapon_memory(model.weapon_memory)
    if args.wall_texture_memory and saved is None:
        incompatible=model.load_state_dict(parent['model'],strict=False)
        if incompatible.unexpected_keys or any(not n.startswith('wall_memory.') for n in incompatible.missing_keys):raise ValueError('Wall memory migration changed old weights')
        from .wall_texture_memory_sr import initialize_wall_memory
        initialize_wall_memory(model.wall_memory)
    model.to(device).configure_training();reference=build_upscaler(parent['config']);reference.load_state_dict(parent['model'],strict=True);reference.to(device).eval().requires_grad_(False)
    from .sr_sprite_heads import PREFIXES as sprite_prefixes
    def frozen_sprites():
        return {k:v for k,v in model.sprite_renderer.state_dict().items() if not base_only or not k.startswith(sprite_prefixes)}
    sprite_sha=_hash_tensors(frozen_sprites())
    if sprite_sha!=_hash_tensors({k:v for k,v in sprite_weights.items() if not base_only or not k.startswith(sprite_prefixes)}):raise ValueError('Sprite transfer mismatch')
    if config.get('wall_geometry_supervised'):model.wall_memory.rgb_logits.requires_grad_(False)
    names=[n for n,p in model.named_parameters() if p.requires_grad];parameters=dict(model.named_parameters())
    if args.wall_texture_memory and not base_only:
        prior_names=parent['optimizer_parameter_names'];groups=parent['optimizer']['param_groups'];offset=0
        first=groups[0];optimizer=torch.optim.Adam([parameters[n] for n in prior_names[:len(first['params'])]],lr=first['lr']);offset=len(first['params'])
        for group in groups[1:]:
            optimizer.add_param_group(dict(params=[parameters[n] for n in prior_names[offset:offset+len(group['params'])]],lr=group['lr']));offset+=len(group['params'])
        optimizer.load_state_dict(parent['optimizer'])
        if [n for n in names if n in set(prior_names)]!=prior_names:raise ValueError('Prior RGB/weapon/wall Adam order changed')
        added=[n for n in names if n not in set(prior_names)]
        if added:optimizer.add_param_group(dict(params=[parameters[n] for n in added],lr=plan['wall_memory_lr']))
    else:
        original=[n for n in names if not n.startswith('weapon_memory.')]
        optimizer=torch.optim.Adam([parameters[n] for n in original],lr=plan['generator_lr'])
        if args.weapon_memory:
            if parent['optimizer_parameter_names']!=original:raise ValueError('Parent RRDB Adam names mismatch')
            optimizer.load_state_dict(parent['optimizer'])
            optimizer.add_param_group(dict(params=[parameters[n] for n in names if n.startswith('weapon_memory.')],lr=plan['weapon_lr']))
    discriminator=WallPatchDiscriminator().to(device);d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=plan['discriminator_lr'],betas=(0.,.99))
    step=parent['training_steps'] if continuation else 0;crop_rng=np.random.default_rng(6211)
    train=NeuralEnemyRestorationData(data,'train',seed=6211,device=device);val=NeuralEnemyRestorationData(data,'val',seed=6212,device=device)
    if continuation and not saved:
        discriminator.load_state_dict(parent['discriminator']);d_optimizer.load_state_dict(parent['discriminator_optimizer']);train.restore_sampler(parent['sampler']);crop_rng.bit_generator.state=parent['crop_rng'];restore_rng(parent,seed=6211,device=device)
    if saved:
        optimizer.load_state_dict(saved['optimizer']);discriminator.load_state_dict(saved['discriminator']);d_optimizer.load_state_dict(saved['discriminator_optimizer']);step=saved['training_steps'];train.restore_sampler(saved['sampler']);crop_rng.bit_generator.state=saved['crop_rng'];restore_rng(saved,seed=6211,device=device)
    sprite_optimizer=None
    if base_only:
        for name,param in model.sprite_renderer.named_parameters():param.requires_grad_(name.startswith(sprite_prefixes))
        sprite_optimizer=torch.optim.Adam([p for p in model.sprite_renderer.parameters() if p.requires_grad],lr=1e-5)
        if saved:
            sprite_optimizer.load_state_dict(saved['sprite_head_optimizer'])
            # Persist actual continuation evidence before any new update.
            atomic_json(root/'base_resume_verification.json',dict(step=step,training_plan_unchanged=saved['training_plan']==plan,
                RGB_optimizer_states=len(optimizer.state),sprite_optimizer_states=len(sprite_optimizer.state),
                sampler_restored=train.sampler_state()==saved['sampler'],
                local_clock_reset=False,RGB_optimizer_reset=False,sprite_optimizer_reset=False))
    panel=json.loads((ROOT/'training_runs/sr_background_detail_20261114/training_plan.json').read_text())['panel']
    status=dict(run_id=root.name,stage='sr',model='RRDB x4 + wall materials and texture + separate patrol sprites',total=args.steps,started_at=now(),loss_protocol=PROTOCOL,objective_protocol=PROTOCOL,rrdb_training=plan,validation_every=100,prediction_every=100,sr_training=dict(input_size=32,output_size=128,scale=4))
    if base_only:status['model']='Base32 -> RRDB materials -> sprites / weapon (no RF, no VQGAN)'
    def publish(**kw):status.update(status='running',current=step,updated_at=now(),**kw);atomic_json(root/'status.json',status)
    def save():
        _save(last,dict(stage='sr',config=model.config,model=model.state_dict(),training_steps=step,training_plan=plan,optimizer=optimizer.state_dict(),optimizer_parameter_names=names,discriminator=discriminator.state_dict(),discriminator_optimizer=d_optimizer.state_dict(),sampler=train.sampler_state(),crop_rng=deepcopy(crop_rng.bit_generator.state),rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],sprite_sha256=sprite_sha,sprite_head_optimizer=sprite_optimizer.state_dict() if sprite_optimizer else None))
    @torch.no_grad()
    def validate():
        model.eval();values=[]
        for first in range(0,len(panel),8):
            ids=panel[first:first+8];batch=val.batch(len(ids),indices=ids);loss,metrics=objective(model.forward_with_aux(batch['lr_npr']),batch)
            if base_only:
                from .sr_sprite_heads import objective as sprite_objective
                _,sprite_metrics=sprite_objective(model.sprite_renderer.forward_with_aux(batch['lr_npr']),batch,ground=True)
                metrics.update({'sprite_'+k:v for k,v in sprite_metrics.items() if type(v) in (int,float)})
            values.append((len(ids),dict(metrics,val_loss=float(loss))))
        means={k:sum(n*m[k] for n,m in values)/len(panel) for k in values[0][1]}
        report=dict(protocol=PROTOCOL,step=step,source_held_out=True,Source_inputs=False,panel_fingerprint=val.panel_fingerprint(panel),model_sha256=_hash_tensors(model.state_dict()),metrics=means)
        atomic_json(root/'validation.json',report);atomic_json(root/'validation'/f'step_{step:06d}.json',report)
        batch=val.batch(4,indices=[774,32,519,38]);low=batch['lr_npr'];output=model.forward_with_aux(low)
        images=[batch['hr_source'],F.interpolate(low,scale_factor=4,mode='nearest'),reference(low),output['image']]
        canvas=Image.new('RGB',(1024,1120));draw=ImageDraw.Draw(canvas)
        for row,(label,imageset) in enumerate(zip(('Source128','Base32 (no RF)' if base_only else 'Neural renderer32','Previous SR on Base' if base_only else 'Previous SR','Base + RRDB materials + sprites' if base_only else 'RRDB material SR'),images)):
            draw.text((4,row*280+3),label,fill='white')
            for col,pixels in enumerate(imageset):canvas.paste(Image.fromarray((pixels.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')).resize((256,256),Image.Resampling.NEAREST),(col*256,row*280+24))
        temp=root/'preview.tmp.png';canvas.save(temp);temp.replace(root/'preview.png')
        preview=dict(kind='rrdb_material_sr_preview_v1',step=step,row_count=4,generated_at=now(),Source_inputs=False,world_rollout=False,source_held_out=True,model_sha256=report['model_sha256'],panel_fingerprint=report['panel_fingerprint'])
        atomic_json(root/'preview.json',preview);publish(validation=report,prediction_preview=preview)
        if _hash_tensors(frozen_sprites())!=sprite_sha:raise ValueError('Frozen sprites changed')
    atomic_json(root/'migration.json',dict(architecture=ARCHITECTURE,sprite_sha256=sprite_sha,sprites_bitexact=True,old_RGB_weights_reused=continuation,new_optimizer=not continuation,wall_memory_initialized=args.wall_texture_memory,weapon_memory_initialized=args.weapon_memory,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    save();publish();validate()
    while step<args.steps and not (root/'stop_training').exists():
        began=time.perf_counter();model.train();discriminator.train();batch=train.batch(8);optimizer.zero_grad(set_to_none=True);d_optimizer.zero_grad(set_to_none=True)
        sprite_metrics={}
        if sprite_optimizer is not None:
            from .sr_sprite_heads import objective as sprite_objective
            sprite_optimizer.zero_grad(set_to_none=True)
            sprite_output=model.sprite_renderer.forward_with_aux(batch['lr_npr'])
            sprite_loss,sprite_metrics=sprite_objective(sprite_output,batch,ground=True)
            sprite_loss.backward();clip_finite_grad_norm_(model.sprite_renderer.named_parameters(),context='Base sprite heads')
            sprite_optimizer.step();sprite_optimizer.zero_grad(set_to_none=True)
        output=model.forward_with_aux(batch['lr_npr']);base,metrics=objective(output,batch)
        metrics.update({'sprite_'+k:v for k,v in sprite_metrics.items()})
        distill=max(0.,1-step/500)*.25
        if distill:
            with torch.no_grad():teacher=reference(batch['lr_npr'])
            base=base+distill*(output['image']-teacher).abs().mean()
        pairs=wall_patch_pairs(batch['lr_npr'],output['image'],batch['hr_source'],wall_labels(batch),crop_rng)
        weight=.002*max(0.,min(1.,(step-500)/500));effective=0.;adversarial=base*0;ratio=0.
        if pairs is not None:
            dl=discriminator_loss(discriminator,pairs);dl.backward();clip_finite_grad_norm_(discriminator.named_parameters(),context='RRDB discriminator');d_optimizer.step();d_optimizer.zero_grad(set_to_none=True)
            if weight:
                adversarial,_=generator_loss(discriminator,pairs);probe=model.color_head.weight
                bn=float(torch.autograd.grad(base,probe,retain_graph=True)[0].norm());an=float(torch.autograd.grad(adversarial,probe,retain_graph=True)[0].norm());effective=min(weight,.05*bn/max(an,1e-30));ratio=effective*an/max(bn,1e-30)
        loss=base+effective*adversarial
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite RRDB loss')
        loss.backward();clip_finite_grad_norm_(model.named_parameters(),context='RRDB SR');optimizer.step();step+=1
        metrics.update(step=step,train_loss=float(loss.detach()),gan_weight_effective=effective,adversarial_gradient_ratio=ratio,generator_lr=plan['generator_lr'],discriminator_lr=plan['discriminator_lr'],update_seconds=time.perf_counter()-began,loss_protocol=PROTOCOL)
        with (root/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
        publish(loss=metrics['train_loss'],latest_metrics=metrics)
        if step%25==0:save()
        if step%100==0:validate();save()
    validate();save();status.update(status='completed' if step==args.steps else 'stopped');atomic_json(root/'status.json',status)

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-only',action='store_true');p.add_argument('--data-root',type=Path);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=5000);p.add_argument('--device',default='cuda');p.add_argument('--weapon-memory',action='store_true');p.add_argument('--wall-texture-memory',action='store_true');p.add_argument('--wall-geometry-supervised',action='store_true');p.add_argument('--wall-coherent-uv',action='store_true');p.add_argument('--wall-texture-v2',action='store_true');args=p.parse_args()
    from .process_runtime import pid_file_alive,write_pid_file
    args.run_dir.mkdir(parents=True,exist_ok=True)
    if pid_file_alive(args.run_dir/'training.pid'):raise ValueError('RRDB worker already active')
    write_pid_file(args.run_dir/'training.pid')
    try:
        with KeepAwake():run(args)
    except BaseException as error:
        path=args.run_dir/'status.json';state=json.loads(path.read_text()) if path.exists() else {};atomic_json(path,{**state,'status':'failed','error':str(error),'updated_at':now()});raise
if __name__=='__main__':main()
