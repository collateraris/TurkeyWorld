"""Append fixed sprite buffers and a tint classifier without resetting old Adam."""
from copy import deepcopy
from pathlib import Path

import torch

from .atlas_enemy_sr import AtlasEnemyEDSRUpscaler
from .distill_renderer import _save
from .models import build_upscaler
from .renderer_student import _hash_tensors
from .sr_semantic_migration import restore_semantic_optimizer,_sha
from .training_pipeline import atomic_json

PROTOCOL='fixed_atlas_enemy_sr_v1'


def prepare_atlas_enemy_continuation(source,destination,*,atlas_path=None,seed=173):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source==destination or destination.exists():raise ValueError('Atlas continuation needs a new immutable destination')
    parent=torch.load(source,map_location='cpu',weights_only=False)
    if (parent.get('config',{}).get('architecture')!='enemy_decoder_edsr'
            or parent['config'].get('enemy_decoder_visual_first')is not True
            or parent['config'].get('enemy_decoder_restoration')is not True
            or any(k not in parent for k in ('optimizer','optimizer_parameter_names','sampler',
                                             'rng_numpy','rng_torch','rng_python','rng_cuda'))):
        raise ValueError('Use a complete visual-first restoration/PatchGAN SR source')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        previous=build_upscaler(parent['config']);previous.load_state_dict(parent['model'],strict=True)
        config={k:v for k,v in parent['config'].items()if k!='architecture'}
        config['restoration_blend_scale']=1.
        model=AtlasEnemyEDSRUpscaler(**config,atlas_asset_path=atlas_path)
    restore_semantic_optimizer(previous,parent)
    values=model.state_dict()
    for name,value in parent['model'].items():
        if name not in values or values[name].shape!=value.shape:
            raise ValueError('Atlas migration changed an inherited model tensor')
        values[name]=value
    model.load_state_dict(values,strict=True)
    old_names=list(dict(previous.named_parameters()));new_names=list(dict(model.named_parameters()))
    if new_names[:len(old_names)]!=old_names:raise ValueError('Atlas changed the canonical old optimizer prefix')
    added=new_names[len(old_names):]
    if added!=['atlas_player_dead_head.weight','atlas_player_dead_head.bias']:
        raise ValueError('Unexpected new atlas parameters')
    names=deepcopy(parent['optimizer_parameter_names'])+[added]
    optimizer=deepcopy(parent['optimizer'])
    last=max(index for group in optimizer['param_groups']for index in group['params'])+1
    group=deepcopy(optimizer['param_groups'][-1]);group['params']=list(range(last,last+len(added)))
    optimizer['param_groups'].append(group)
    migrated={**parent,'optimizer':optimizer,'optimizer_parameter_names':names}
    restore_semantic_optimizer(model,migrated)
    receipt=dict(kind=PROTOCOL,source_checkpoint=str(source),source_checkpoint_sha256=_sha(source),
        source_training_steps=parent['training_steps'],source_semantic_updates=parent.get('semantic_updates'),
        source_patchgan_updates=parent.get('patchgan_updates'),source_sampler_draws=parent['sampler']['draws'],
        inherited_model_tensors=len(parent['model']),inherited_Adam_states=len(parent['optimizer']['state']),
        inherited_weights_Adam_RNG_sampler_preserved=True,added_parameter_names=added,
        added_parameters=sum(dict(model.named_parameters())[name].numel()for name in added),
        parameter_count=sum(p.numel()for p in model.parameters()),
        trainable_parameter_count=sum(p.numel()for p in model.parameters()if p.requires_grad),
        frozen_parameter_names=[name for name,p in model.named_parameters()if not p.requires_grad],
        inherited_weights_sha256=_hash_tensors(parent['model']),
        atlas_asset_sha256=model.config['atlas_asset_sha256'],atlas_pixels_sha256=model.config['atlas_pixels_sha256'],
        atlas_frame_sizes=model.config['atlas_frame_sizes'],atlas_embedded=True,
        source_frames_used_as_inference_inputs=False,unused_pixel_generator_frozen=True,
        geometry='NN_full_unclipped_sprite_rectangle',phase='RGB_NN_hard_argmax',
        occlusion='NN_binary_visibility',adversarial_training=False,automatic_deployment=False)
    payload={k:deepcopy(v)for k,v in parent.items()if k not in
        ('model','optimizer','protocol','patchgan_state','patchgan_updates')}
    payload.update(stage='sr',format_version=2,config=deepcopy(model.config),model=model.state_dict(),
        optimizer=optimizer,optimizer_parameter_names=names,atlas_migration=receipt,
        parent_protocol=deepcopy(parent.get('protocol')),parent_training_steps=parent['training_steps'],
        parent_semantic_updates=parent.get('semantic_updates'),parent_patchgan_updates=parent.get('patchgan_updates'),
        parent_sampler=deepcopy(parent['sampler']),parent_best=deepcopy(parent.get('best')),
        semantic_updates=0,decoder_updates=0,visual_first_updates=0,restoration_updates=0,
        atlas_updates=0,completed_updates=0,best=None,gameplay_qualified=False,automatic_deployment=False)
    payload['parent_restoration_schedule']=payload.pop('restoration_schedule',None)
    _save(destination,payload)
    atomic_json(destination.with_suffix(destination.suffix+'.atlas.json'),receipt)
    return receipt
