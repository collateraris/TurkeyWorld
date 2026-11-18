"""Merge a previously evaluated local player head into a NEW file at a pinned world boundary."""
import argparse
import os
import tempfile
from pathlib import Path

import torch

from .learned_axis_collision import LocalAxisCollisionHead, PROTOCOL, PREFIX, MODE, MODE_NAV, head_digest, with_local_axis_head
from .training_pipeline import atomic_json


def same(a,b):
    if torch.is_tensor(a):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def merge(parent, trained, validated, nav_only=False, expect_step=None):
    step=parent['training_steps'];vstep=validated['training_steps']
    if expect_step is not None and step!=expect_step:raise ValueError('Wrong parent training step')
    if (parent.get('stage')!='world' or validated.get('stage')!='world' or step<vstep or
        parent.get('global_training_step')!=step or parent.get('training_counters',{}).get('pending_microbatches',0)):
        raise ValueError('Expected a complete, current V-JEPA optimizer boundary')
    newer=parent['config'];older=validated['config']
    if newer!={**older,'player_motion_mode':'delta'} or older.get('player_motion_mode') not in (MODE,MODE_NAV):
        raise ValueError('Evaluated head and parent architectures differ')
    if parent['vjepa_sr_feedback']!=validated['vjepa_sr_feedback'] or parent['state_schema']!=validated['state_schema']:
        raise ValueError('Parent and candidate schema or renderer bindings differ')
    if (not parent.get('optimizer',{}).get('state') or not parent.get('optimizer_parameter_names')
            or not parent.get('vjepa_sr_training',{}).get('frozen_parameters_sha256')
            or not parent['vjepa_sr_training'].get('samplers')):
        raise ValueError('Incomplete parent Adam / frozen hash / sampler metadata')
    a=trained['plan'];ledger=validated['local_axis_training']
    if (a['protocol']!='local_axis_train_only_v1' or a['training_uses_heldout'] is not False or a['source_future_feedback'] is not False
            or a['lr']!=1e-5 or trained['steps']<=0 or trained['steps']!=ledger['head_steps']
            or ledger['protocol']!=a['protocol'] or ledger['lr']!=a['lr'] or ledger['seed']!=a['seed']
            or ledger['source_world_step']!=vstep or not same(ledger['optimizer'],trained['optimizer'])):
        raise ValueError('Standalone training ledger differs from evaluated candidate')
    model=LocalAxisCollisionHead()
    model.load_state_dict(trained['model'],strict=True)
    reference={k[len(PREFIX):]:v for k,v in validated['model'].items() if k.startswith(PREFIX)}
    if not same(model.state_dict(),reference): raise ValueError('Standalone head weights differ from evaluated candidate')
    vp=validated['local_axis_head_migration']
    if (vp['head_sha256']!=head_digest(model) or vp['protocol']!=PROTOCOL or vp['source_step']!=vstep or
            vp['source_mode']!='delta' or vp['target_mode']!=older['player_motion_mode'] or not vp['no_reference_future']):
        raise ValueError('Candidate does not identify this learned head')
    new=with_local_axis_head(parent,model,nav_only=nav_only)
    new['local_axis_training']={**ledger,'attached_world_step':step,'trained_head_sha256':head_digest(model)}
    # The head is stored separately from the regular trainer's Adam parameter groups.
    # Keep *exactly* the old frozen hash; verify_frozen_resume checks all old weights
    # plus verifies the extra head on this first optimizer boundary.
    for key in ('optimizer','optimizer_parameter_names','vjepa_sr_training','rng_torch','rng_cuda','ema_target_encoder'):
        if new[key] is not parent[key]:raise AssertionError('Unexpected mutation of '+key)
    return new


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('base','trained','validated','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--nav-only',action='store_true');p.add_argument('--expect-step',type=int)
    a=p.parse_args(argv);out=a.output.absolute();sources=[v.resolve(strict=True) for v in [a.base,a.trained,a.validated]]
    if out.exists() or out.is_symlink() or out.resolve() in sources:raise FileExistsError('Output must be a fresh file, not a source or symlink')
    stamp=lambda f:(f.stat().st_dev,f.stat().st_ino,f.stat().st_size,f.stat().st_mtime_ns)
    identity=list(map(stamp,sources))
    torch.set_num_threads(2)
    parent,head,validated=[torch.load(v,map_location='cpu',mmap=True,weights_only=False) for v in sources]
    new=merge(parent,head,validated,nav_only=a.nav_only,expect_step=a.expect_step)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent,suffix='.pt',delete=False) as f:tmp=Path(f.name)
    try:
        torch.save(new,tmp)
        if list(map(stamp,sources))!=identity:raise OSError('Pinned inputs changed during merge')
        os.link(tmp,out) # atomic no-clobber
    finally:tmp.unlink(missing_ok=True)
    r=dict(base=str(sources[0]),validated=str(sources[2]),output=str(out),
           step=new['training_steps'],updates=new['vjepa_sr_training']['updates'],validation_source_step=validated['training_steps'],
           head_updates=head['steps'],retained_Adam_slots=len(new['optimizer']['state']),**new['local_axis_head_migration'])
    atomic_json(out.with_suffix(out.suffix+'.attachment.json'),r);print(r,flush=True)


if __name__=='__main__':main()
