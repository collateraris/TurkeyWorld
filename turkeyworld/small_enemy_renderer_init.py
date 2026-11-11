"""Explicit new-codec RF continuation with the parent's Base/RF Adam retained."""
from __future__ import annotations

from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil

import torch

from .codec_enemy_swap import prepare_codec_enemy_swap
from .distill_renderer import _digest
from .models import load_world_checkpoint
from .renderer_refinement import export_refined_bundle
from .renderer_student import load_renderer_student, _hash_tensors
from .training_pipeline import atomic_json


def _same(left,right):
    if torch.is_tensor(left):
        return torch.is_tensor(right)and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left.detach().cpu(),right.detach().cpu())
    if isinstance(left,dict):return isinstance(right,dict)and left.keys()==right.keys()and all(_same(v,right[k])for k,v in left.items())
    if isinstance(left,(list,tuple)):return type(left)is type(right)and len(left)==len(right)and all(_same(a,b)for a,b in zip(left,right))
    return left==right


def _parent_weights(saved,model):
    if not isinstance(saved,dict)or not all(k in saved for k in ('base','flow','optimizer','step','sampler')):
        raise ValueError('Small-enemy RF continuation requires full parent Base/Flow/Adam/sampler state')
    if saved['step']!=8000 or isinstance(saved['step'],bool):raise ValueError('Expected the pinned completed8000 parent boundary')
    base,flow=model.base_student.state_dict(),model.flow_renderer.state_dict()
    if not _same(saved['base'],base):raise ValueError('Compact Base differs from the pinned Adam parent')
    if saved['flow'].keys()!=flow.keys():raise ValueError('Flow architecture differs from the parent Adam state')
    noncodec={k:v for k,v in saved['flow'].items()if not k.startswith('codec.')}
    if not _same(noncodec,{k:flow[k]for k in noncodec}):
        raise ValueError('Non-codec Flow/noise/normalization differs from the pinned Adam parent')
    quantizer={k:v for k,v in saved['flow'].items()if k.startswith('codec.quantize.')}
    if not quantizer or not _same(quantizer,{k:flow[k]for k in quantizer}):
        raise ValueError('New codec must preserve every original quantizer parameter and buffer')
    if any(saved['flow'][k].shape!=v.shape or saved['flow'][k].dtype!=v.dtype for k,v in flow.items()):
        raise ValueError('Only codec E/D values may change; shapes and dtypes remain fixed')
    return {'base_tensor_count':len(base),'noncodec_flow_tensor_count':len(noncodec),
        'quantizer_tensor_count':len(quantizer),
        'changed_codec_ed_tensors':sum(not torch.equal(saved['flow'][k].cpu(),v.detach().cpu())
            for k,v in flow.items()if k.startswith('codec.')and not k.startswith('codec.quantize.'))}


def restore_small_renderer_parent(saved,model,opt):
    """Load Adam only; loading old Flow here would silently undo the new codec."""
    details=_parent_weights(saved,model)
    expected=[list(model.base_student.parameters()),
        [p for name,p in model.flow_renderer.named_parameters()if not name.startswith('codec.')and p.requires_grad]]
    if (len(opt.param_groups)!=2 or len(saved['optimizer']['param_groups'])!=2
            or any([id(p)for p in group['params']]!=[id(p)for p in wanted]
                   for group,wanted in zip(opt.param_groups,expected))):
        raise ValueError('Parent Adam requires unchanged ordered Base and non-codec Flow parameter groups')
    if any(len(group['params'])!=len(wanted)for group,wanted in zip(saved['optimizer']['param_groups'],expected)):
        raise ValueError('Parent Adam parameter identities/counts differ')
    codec_before=_hash_tensors(model.flow_renderer.codec.state_dict())
    rng_before=torch.get_rng_state().clone()
    opt.load_state_dict(saved['optimizer'])
    if not _same(saved['optimizer'],opt.state_dict()):raise AssertionError('Parent Adam moments/options changed during adoption')
    if not torch.equal(torch.get_rng_state(),rng_before):raise AssertionError('Adam adoption consumed CPU RNG')
    if _hash_tensors(model.flow_renderer.codec.state_dict())!=codec_before:
        raise AssertionError('Adam adoption overwrote the selected new codec')
    _parent_weights(saved,model)
    sampler=saved['sampler'];parent_sampler=sampler.get('parent',sampler)
    if 'rng'not in parent_sampler:raise ValueError('Parent sampler must retain its private PCG random state')
    return {'start_step':saved['step'],'adam_all_moments_options_bitexact':True,
        'adam_state_count':len(opt.state),'adam_steps':sorted({int(v['step'])for v in opt.state.values()if'step'in v}),
        'base_and_noncodec_flow_bitexact':True,'quantizer_all_state_bitexact':True,
        'new_codec_retained':True,'new_codec_sha256':codec_before,**details,
        'parent_sampler_archived':deepcopy(sampler),'parent_sampler_rng':deepcopy(parent_sampler['rng']),
        'new_small_sampler_counts_start_zero':True,'rng_restoration_required':True,
        'automatic_deployment':False,'gameplay_qualified':False}


def prepare_small_renderer_initial(assets_dir,selected_new_codec_path,output_dir,old8000_fullAdam_path):
    """Write a new strict bundle and immutable full-Adam copy, never deploy it."""
    assets,codec,output,parent=map(lambda p:Path(p).resolve(),
        (assets_dir,selected_new_codec_path,output_dir,old8000_fullAdam_path))
    world=assets/'world_navigation.pt';bundle=assets/'renderer_parent_navigation.pt'
    identity={'version':1,'world_navigation_sha256':_digest(world),'parent_renderer_sha256':_digest(bundle),
        'selected_codec_sha256':_digest(codec),'parent_8000_Adam_sha256':_digest(parent)}
    proof_path=output/'small_enemy_renderer_initial.json';destination=output/'renderer_student.pt'
    adam_copy=output/'renderer_adam_parent.pt'
    if proof_path.exists():
        proof=json.loads(proof_path.read_text(encoding='utf-8'))
        if (proof['identity']!=identity or _digest(destination)!=proof['renderer_sha256']
                or _digest(adam_copy)!=identity['parent_8000_Adam_sha256']):
            raise ValueError('Immutable small-enemy renderer initialization changed')
        return destination
    if destination.exists()or adam_copy.exists():raise FileExistsError('Incomplete small-enemy renderer initialization requires review')
    saved=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    with torch.random.fork_rng(devices=[]):
        source,metadata=load_world_checkpoint(world,'cpu');del metadata
        working,metadata=load_world_checkpoint(world,'cpu');del metadata
        source.eval().requires_grad_(False);working.eval().requires_grad_(False)
        approved=load_renderer_student(working,bundle,allow_unqualified_codec=True)
        initial_renderer=int(approved['training_steps']);initial_base=int(working.base_student_bundle['training_steps'])
        _parent_weights(saved,working)
        candidate,swap=prepare_codec_enemy_swap(source,working,codec,allow_unqualified_codec=True)
        working.flow_renderer=candidate
        weights=_parent_weights(saved,working)
        output.mkdir(parents=True,exist_ok=True)
        report={'passed':False,'scope':'Untrained Base/RF adaptation to selected tiny-enemy codec; revalidation required',
            'codec_source_quality_gate':deepcopy(candidate._renderer_codec_distillation['quality_gate']),
            'initialization_only':True,'automatic_deployment':False,'gameplay_qualified':False}
        export_refined_bundle(source,working,output,updates=0,report=report,
            initial_renderer_steps=initial_renderer,initial_base_steps=initial_base,
            provenance={'protocol':'small_enemy_new_codec_old_Adam_v1','identity':identity,
                'parent_optimizer_step':8000,'old_source_world_hash_not_required':True,
                'source_binding':'Strict original renderer assets/config validation; dynamics weights are not renderer inputs'})
        # The exporter attaches Base only to the private source. Reload into
        # another original world view to verify the actual standalone bundle.
        del source;gc.collect()
        fresh,metadata=load_world_checkpoint(world,'cpu');del metadata
        load_renderer_student(fresh,destination,allow_unqualified_codec=True)
        _parent_weights(saved,fresh)
        if _hash_tensors(fresh.flow_renderer.codec.state_dict())!=_hash_tensors(working.flow_renderer.codec.state_dict()):
            raise AssertionError('Strict reload changed the selected new codec')
        proof={'identity':identity,'parent_optimizer_step':8000,'initial_renderer_training_steps':initial_renderer,
            'initial_base_training_steps':initial_base,'strict_bundle_reload_passed':True,
            'new_codec_sha256':_hash_tensors(working.flow_renderer.codec.state_dict()),**weights,
            'swap':swap,'automatic_deployment':False,'gameplay_qualified':False}
        del fresh,working;gc.collect()
    shutil.copy2(parent,adam_copy)
    if _digest(adam_copy)!=identity['parent_8000_Adam_sha256']:raise AssertionError('Archived parent Adam copy differs')
    proof.update(renderer_sha256=_digest(destination),parent_Adam_copy=str(adam_copy))
    atomic_json(proof_path,proof)
    return destination
