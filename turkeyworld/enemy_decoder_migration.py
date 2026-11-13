"""Append learned enemy decoder parameters to a full semantic-SR checkpoint.

All parent parameters, named Adam groups/mixed ages and RNG are retained. New
data sampling starts explicitly in the new trainer; the old sampler/protocol
and completed semantic counter are retained as history, never reused as new
balanced-dataset clocks.
"""
from copy import deepcopy
from pathlib import Path

import torch

from .models import build_upscaler
from .sr_semantic_migration import _sha,restore_semantic_optimizer
from .renderer_student import _hash_tensors
from .distill_renderer import _save
from .training_pipeline import atomic_json


def restore_enemy_decoder_optimizer(model,saved):
    return restore_semantic_optimizer(model,saved)


def prepare_enemy_decoder_continuation(parent_checkpoint,destination,*,seed=173,stable=False):
    source=Path(parent_checkpoint).resolve();destination=Path(destination).resolve()
    if source==destination or destination.exists():
        raise ValueError('Enemy decoder migration needs a new immutable destination')
    parent=torch.load(source,map_location='cpu',weights_only=False)
    if (parent.get('stage')!='sr' or parent.get('config',{}).get('architecture')!='semantic_edsr'
            or 'optimizer'not in parent or 'optimizer_parameter_names'not in parent
            or 'semantic_migration'not in parent or parent.get('training_steps',0)<=0):
        raise ValueError('A full trained semantic SR checkpoint with named Adam is required')
    if any(key not in parent for key in ('rng_numpy','rng_torch','rng_python','rng_cuda','sampler')):
        raise ValueError('Parent semantic SR checkpoint lost sampler or recorded RNG')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        old=build_upscaler(parent['config'])
        config={**parent['config'],'architecture':'enemy_decoder_edsr','enemy_decoder_channels':24}
        if stable:config['enemy_decoder_stable']=True
        model=build_upscaler(config)
    old.load_state_dict(parent['model'],strict=True)
    target=model.state_dict();target.update(parent['model']);model.load_state_dict(target,strict=True)
    old_names=list(dict(old.named_parameters()))
    all_names=list(dict(model.named_parameters()))
    added=all_names[len(old_names):]
    if all_names[:len(old_names)]!=old_names or not added or any(not n.startswith('enemy_decoder_')for n in added):
        raise AssertionError('Enemy decoder changed the parent canonical parameter prefix')
    old_groups=parent['optimizer_parameter_names']
    if [name for group in old_groups for name in group]!=old_names:
        raise ValueError('Parent Adam names do not match canonical semantic SR parameters')
    # Validate moment tensors before binding a new architecture to them.
    restore_semantic_optimizer(old,parent)
    parameters=dict(model.named_parameters())
    specifications=[{**{k:deepcopy(v)for k,v in group.items()if k!='params'},
                     'params':[parameters[name]for name in names]}
        for group,names in zip(parent['optimizer']['param_groups'],old_groups)]
    specifications.append({**{k:deepcopy(v)for k,v in parent['optimizer']['param_groups'][0].items()if k!='params'},
                           'params':[parameters[name]for name in added]})
    optimizer=torch.optim.AdamW(specifications)
    migrated=optimizer.state_dict();new_groups=deepcopy(old_groups)+[added]
    current_ids={name:index for names,group in zip(new_groups,migrated['param_groups'])
                 for name,index in zip(names,group['params'])}
    source_names={index:name for names,group in zip(old_groups,parent['optimizer']['param_groups'])
                  for name,index in zip(names,group['params'])}
    migrated['state']={current_ids[source_names[index]]:deepcopy(row)
                      for index,row in parent['optimizer']['state'].items()}
    optimizer.load_state_dict(migrated)
    if any(not torch.equal(value,model.state_dict()[name])for name,value in parent['model'].items()):
        raise AssertionError('Enemy decoder migration changed parent model tensors')
    report=dict(kind='append_predicted_rssm_enemy_decoder_v1',seed=seed,
        source_checkpoint=str(source),source_checkpoint_sha256=_sha(source),
        source_training_steps=parent['training_steps'],source_semantic_updates=parent['semantic_updates'],
        old_model_tensors=len(parent['model']),old_parameter_names=old_names,
        added_parameter_names=added,old_Adam_states=len(parent['optimizer']['state']),
        old_Adam_groups=len(parent['optimizer']['param_groups']),old_weights_bitexact=True,
        old_Adam_and_RNG_preserved=True,new_Adam_states_fresh=True,
        old_weights_sha256=_hash_tensors(parent['model']),
        added_parameter_count=sum(parameters[name].numel()for name in added),
        parameter_count=sum(p.numel()for p in model.parameters()),
        zero_added_composite_gain=True,inference_inputs=['RGB32','predicted_RSSM_state20+validbit'],
        stable_RGB_parent_frozen=bool(stable),
        frozen_parent_parameter_names=old_names if stable else [],
        trainable_parameter_names=added if stable else all_names,
        no_atlas_or_Source_inference=True,sampler_initialization='explicit_new_balanced_pool',
        automatic_deployment=False)
    payload=deepcopy(parent)
    payload.update(config=model.config,model=model.state_dict(),optimizer=optimizer.state_dict(),
        optimizer_parameter_names=new_groups,enemy_decoder_migration=report,
        parent_training_steps=parent['training_steps'],parent_semantic_updates=parent['semantic_updates'],
        parent_sampler=deepcopy(parent['sampler']),parent_protocol=deepcopy(parent.get('protocol')),
        parent_best=deepcopy(parent.get('best')),semantic_updates=0,decoder_updates=0,
        completed_updates=0,best=None,gameplay_qualified=False,automatic_deployment=False)
    payload.pop('sampler',None);payload.pop('protocol',None)
    _save(destination,payload)
    atomic_json(destination.with_suffix(destination.suffix+'.enemy-decoder.json'),report)
    return report


def prepare_visual_first_continuation(parent_checkpoint,destination,*,seed=173):
    """Append the RGB-first route to its exact full stable decoder parent.

    Current parent weights/mixed-age Adam/RNG and sampler are retained. Only
    phase-local update counters start at zero; old counters and objective are
    history, not a rollback. This is an explicit routing change rather than a
    claim that every initial guided image is unchanged.
    """
    source=Path(parent_checkpoint).resolve();destination=Path(destination).resolve()
    if source==destination or destination.exists():
        raise ValueError('Visual-first migration requires a new immutable destination')
    parent=torch.load(source,map_location='cpu',weights_only=False)
    config=parent.get('config',{})
    if (parent.get('stage')!='sr' or config.get('architecture')!='enemy_decoder_edsr'
            or config.get('enemy_decoder_stable') is not True
            or config.get('enemy_decoder_visual_first',False) is not False
            or 'optimizer'not in parent or 'optimizer_parameter_names'not in parent
            or 'enemy_decoder_migration'not in parent
            or type(parent.get('training_steps')) is not int or parent['training_steps']<=0
            or type(parent.get('decoder_updates')) is not int or parent['decoder_updates']<0):
        raise ValueError('Visual-first decoding requires its exact full trained stable decoder parent')
    if any(key not in parent for key in ('rng_numpy','rng_torch','rng_python','rng_cuda','sampler')):
        raise ValueError('Stable decoder parent lost its recorded sampler or RNG')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        old=build_upscaler(config)
        model=build_upscaler({**config,'enemy_decoder_visual_first':True,
                             'enemy_decoder_visual_threshold':.9})
    old.load_state_dict(parent['model'],strict=True)
    target=model.state_dict();target.update(parent['model']);model.load_state_dict(target,strict=True)
    # A fresh optimizer group may start from trained recognition values. This
    # copies the visual ROI classifier, never its old moments or a Source label.
    model.enemy_decoder_visual_confirmation.load_state_dict(old.enemy_decoder_confirmation.state_dict())
    old_names=list(dict(old.named_parameters()))
    parameters=dict(model.named_parameters());all_names=list(parameters)
    added=all_names[len(old_names):]
    if (all_names[:len(old_names)]!=old_names or not added
            or any(not name.startswith(('enemy_decoder_visual_condition.',
                'enemy_decoder_visual_confirmation.','enemy_decoder_visual_gain.'))for name in added)):
        raise AssertionError('Visual-first migration changed the inherited canonical parameter prefix')
    old_groups=parent['optimizer_parameter_names']
    restore_semantic_optimizer(old,parent)
    specifications=[{**{k:deepcopy(v)for k,v in group.items()if k!='params'},
                     'params':[parameters[name]for name in names]}
        for group,names in zip(parent['optimizer']['param_groups'],old_groups)]
    specifications.append({**{k:deepcopy(v)for k,v in parent['optimizer']['param_groups'][-1].items()if k!='params'},
                           'params':[parameters[name]for name in added]})
    optimizer=torch.optim.AdamW(specifications)
    migrated=optimizer.state_dict();new_groups=deepcopy(old_groups)+[added]
    current_ids={name:index for names,group in zip(new_groups,migrated['param_groups'])
                 for name,index in zip(names,group['params'])}
    source_names={index:name for names,group in zip(old_groups,parent['optimizer']['param_groups'])
                  for name,index in zip(names,group['params'])}
    migrated['state']={current_ids[source_names[index]]:deepcopy(row)
                      for index,row in parent['optimizer']['state'].items()}
    optimizer.load_state_dict(migrated)
    if any(not torch.equal(value,model.state_dict()[name])for name,value in parent['model'].items()):
        raise AssertionError('Visual-first migration changed an inherited model tensor')
    frozen=[name for name,p in model.named_parameters()if not p.requires_grad]
    if frozen != [name for name,p in old.named_parameters()if not p.requires_grad]:
        raise AssertionError('Visual-first profile changed the frozen RGB parent policy')
    report=dict(kind='append_rgb_visual_first_enemy_decoder_v2',seed=seed,
        source_checkpoint=str(source),source_checkpoint_sha256=_sha(source),
        source_training_steps=parent['training_steps'],source_decoder_updates=parent['decoder_updates'],
        old_model_tensors=len(parent['model']),old_parameter_names=old_names,
        added_parameter_names=added,old_Adam_states=len(parent['optimizer']['state']),
        old_Adam_groups=len(parent['optimizer']['param_groups']),old_weights_bitexact=True,
        old_Adam_and_RNG_preserved=True,new_Adam_states_fresh=True,sampler_preserved=True,
        old_weights_sha256=_hash_tensors(parent['model']),
        added_parameter_count=sum(parameters[name].numel()for name in added),
        parameter_count=sum(p.numel()for p in model.parameters()),
        trainable_parameter_count=sum(p.numel()for p in model.parameters()if p.requires_grad),
        frozen_parent_parameter_names=frozen,
        trainable_parameter_names=[name for name,p in model.named_parameters()if p.requires_grad],
        zero_added_visual_gain=True,zero_added_visual_condition_output=True,
        visual_confirmation_initialization='copy_trained_guided_RGB_ROI_classifier_fresh_Adam',
        visual_confidence_threshold=.9,visual_size_upper_range=1.5,
        visual_geometry_and_appearance_inputs=['RGB32'],RSSM_cannot_veto_confident_visual_route=True,
        visual_animation_input='RGB_only',
        routing_change_explicit=True,initial_guided_image_bitexact_not_claimed=True,
        inference_inputs=['RGB32','optional_predicted_RSSM_state20+validbit'],
        no_atlas_or_Source_inference=True,automatic_deployment=False)
    payload=deepcopy(parent)
    payload.update(config=model.config,model=model.state_dict(),optimizer=optimizer.state_dict(),
        optimizer_parameter_names=new_groups,visual_first_migration=report,
        parent_training_steps=parent['training_steps'],parent_decoder_updates=parent['decoder_updates'],
        parent_semantic_updates=parent.get('semantic_updates'),
        parent_sampler=deepcopy(parent['sampler']),parent_protocol=deepcopy(parent.get('protocol')),
        parent_best=deepcopy(parent.get('best')),semantic_updates=0,decoder_updates=0,
        visual_first_updates=0,completed_updates=0,best=None,
        gameplay_qualified=False,automatic_deployment=False)
    payload.pop('protocol',None)
    _save(destination,payload)
    atomic_json(destination.with_suffix(destination.suffix+'.visual-first.json'),report)
    return report


def prepare_restoration_continuation(parent_checkpoint,destination,*,seed=173):
    """Same-parameter continuation; preserve moments/RNG/sampler, start raw warmup."""
    from .decoder_restoration_protocol import initial_restoration_schedule
    source=Path(parent_checkpoint).resolve();destination=Path(destination).resolve()
    if source==destination or destination.exists():
        raise ValueError('Restoration continuation requires a new immutable destination')
    parent=torch.load(source,map_location='cpu',weights_only=False)
    config=parent.get('config',{})
    if (parent.get('stage')!='sr' or config.get('architecture')!='enemy_decoder_edsr'
            or config.get('enemy_decoder_stable')is not True
            or config.get('enemy_decoder_visual_first')is not True
            or config.get('enemy_decoder_restoration',False)is not False
            or 'visual_first_migration'not in parent or 'optimizer'not in parent
            or type(parent.get('training_steps'))is not int or parent['training_steps']<=0
            or any(key not in parent for key in ('optimizer_parameter_names','sampler',
                'rng_torch','rng_numpy','rng_python','rng_cuda'))):
        raise ValueError('Restoration requires the exact full visual-first parent and named optimizer/RNG')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        old=build_upscaler(config)
        model=build_upscaler({**config,'enemy_decoder_restoration':True,'restoration_blend_scale':0.})
    old.load_state_dict(parent['model'],strict=True);model.load_state_dict(parent['model'],strict=True)
    old_optimizer=restore_semantic_optimizer(old,parent)
    optimizer=restore_semantic_optimizer(model,parent)
    if (list(dict(old.named_parameters()))!=list(dict(model.named_parameters()))
            or [(n,p.requires_grad)for n,p in old.named_parameters()]!=
                [(n,p.requires_grad)for n,p in model.named_parameters()]
            or any(not torch.equal(v,model.state_dict()[n])for n,v in parent['model'].items())):
        raise AssertionError('Restoration changed inherited weights, optimizer layout or frozen policy')
    if len(old_optimizer.state)!=len(optimizer.state):
        raise AssertionError('Restoration dropped an inherited Adam state')
    names=[n for n,_ in model.named_parameters()]
    report=dict(kind='coordinate_consistent_raw_decoder_restoration_v3',seed=seed,
        source_checkpoint=str(source),source_checkpoint_sha256=_sha(source),
        source_training_steps=parent['training_steps'],source_visual_first_updates=parent.get('visual_first_updates'),
        old_model_tensors=len(parent['model']),old_parameter_names=names,
        old_Adam_states=len(parent['optimizer']['state']),old_Adam_groups=len(parent['optimizer']['param_groups']),
        added_parameter_names=[],added_parameter_count=0,
        parameter_count=sum(p.numel()for p in model.parameters()),
        trainable_parameter_count=sum(p.numel()for p in model.parameters()if p.requires_grad),
        old_weights_bitexact=True,old_Adam_and_RNG_preserved=True,sampler_preserved=True,
        frozen_parent_parameter_names=[n for n,p in model.named_parameters()if not p.requires_grad],
        old_weights_sha256=_hash_tensors(parent['model']),
        initial_blend_scale=0.,learned_gains_preserved_but_not_used_for_restoration=True,
        inference_uses_explicit_saved_blend_scale=True,
        raw_loss_coordinates='actual_predicted_bbox_detached',Source_background_is_loss_labels_only=True,
        automatic_deployment=False)
    payload=deepcopy(parent)
    payload.update(config=deepcopy(model.config),model=model.state_dict(),optimizer=optimizer.state_dict(),
        restoration_migration=report,restoration_schedule=initial_restoration_schedule(),
        parent_training_steps=parent['training_steps'],parent_visual_first_updates=parent.get('visual_first_updates'),
        parent_decoder_updates=parent.get('decoder_updates'),parent_semantic_updates=parent.get('semantic_updates'),
        parent_sampler=deepcopy(parent['sampler']),parent_protocol=deepcopy(parent.get('protocol')),
        parent_best=deepcopy(parent.get('best')),semantic_updates=0,decoder_updates=0,visual_first_updates=0,
        restoration_updates=0,completed_updates=0,best=None,gameplay_qualified=False,automatic_deployment=False)
    payload.pop('protocol',None)
    _save(destination,payload)
    atomic_json(destination.with_suffix(destination.suffix+'.restoration.json'),report)
    return report


def main(argv=None):
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=173)
    parser.add_argument('--stable',action='store_true')
    args=parser.parse_args(argv)
    print(json.dumps(prepare_enemy_decoder_continuation(args.parent,args.output,seed=args.seed,stable=args.stable),allow_nan=False))


if __name__=='__main__':main()
