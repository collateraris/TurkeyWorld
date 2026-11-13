"""Frozen one-step RSSM hypotheses from actual Source past, never future labels."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch

from .enemy_decoder_data import HYPOTHESIS_PROTOCOL
from .prepare_training_data import _atomic_json
from .renderer_student import _hash_tensors
from .small_enemy_data import digest
from .sr_neural_pairs_data import SemanticSRData, fingerprint


@contextmanager
def preserve_rng(device):
    python = random.getstate(); numpy = np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python); np.random.set_state(numpy)


def rng_fingerprint(device):
    values = {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state().numpy().tobytes()}
    if device.type=='cuda':
        values['cuda'] = torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
    return hashlib.sha256(pickle.dumps(values,protocol=4)).hexdigest()


def initialize_cache_device(device):
    device = torch.device(device)
    if device.type=='cuda':
        index = device.index if device.index is not None else 0
        torch.cuda.set_device(index)
        torch.cuda.init()
        return torch.device('cuda',index)
    return device


@torch.inference_mode()
def predict_hypotheses(model,batch):
    """Only dynamic flags, previous RGB/state, action and map are read here."""
    device = next(model.parameters()).device
    dynamic = batch['dynamic'].to(device=device)
    if dynamic.ndim != 1 or dynamic.dtype != torch.bool or not len(dynamic):
        raise ValueError('Hypothesis inputs need explicit per-row genuine-past flags')
    count = len(dynamic); predicted = torch.zeros((count,20),device=device,dtype=torch.float32)
    indices = torch.nonzero(dynamic,as_tuple=False).flatten()
    with preserve_rng(device):
        if len(indices):
            previous = batch['previous_state'].to(device=device,dtype=torch.float32)[indices]
            image = batch['previous_rgb'].to(device=device,dtype=torch.float32)[indices]
            action = batch['action'].to(device=device,dtype=torch.float32)[indices]
            grid = batch['map'].to(device=device,dtype=torch.long)[indices]
            if (previous.shape != (len(indices),20) or image.shape != (len(indices),3,32,32)
                    or action.shape != (len(indices),4) or grid.shape != (len(indices),16,16)
                    or not all(torch.isfinite(v).all() for v in (previous,image,action))
                    or ((image < 0) | (image > 1)).any()):
                raise ValueError('Dynamic hypotheses require genuine finite previous RGB/state/action/map')
            features = model.encode_map(grid)
            memory = model.initialize_from_image(image,previous,features)
            result = model.transition(previous,action,features,memory)
            state = result['state'].detach().float()
            if state.shape != (len(indices),20) or not torch.isfinite(state).all():
                raise FloatingPointError('Frozen RSSM returned invalid predicted state20')
            predicted[indices] = state
    return {'pred_state':predicted.cpu().numpy().astype(np.float32,copy=False),
        'valid':dynamic.detach().cpu().numpy().astype(np.bool_,copy=False)}


def build_hypothesis_cache(data_root,output,model,*,world_sha256,world_step=None,batch_size=8,seed=173):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Positive hypothesis cache batch size required')
    if not isinstance(world_sha256,str) or not world_sha256:
        raise ValueError('Frozen world identity is required')
    data_root = Path(data_root).resolve(); root = Path(output).resolve()
    if data_root == root or root in data_root.parents:
        raise ValueError('Hypothesis sidecar must not replace its immutable base corpus')
    base_path = data_root/'manifest.json'; base_bytes = base_path.read_bytes(); base = json.loads(base_bytes)
    readers = {split:SemanticSRData(data_root,split,seed=seed) for split in ('train','val')}
    if readers['train'].scene_keys & readers['val'].scene_keys:
        raise ValueError('Base TRAIN/VAL Source conditions overlap')
    lookup = {split:{(r['shard'],r['offset']):i for i,r in enumerate(reader.rows)} for split,reader in readers.items()}
    identity = {'protocol':HYPOTHESIS_PROTOCOL,'base_manifest_sha256':digest(base_path),
        'base_data_sha256':base['data_sha256'],'world_sha256':world_sha256,'world_step':world_step,'seed':seed,
        'bootstrap_horizon':1,'past_fields':['previous_rgb','previous_state','action','map'],
        'future_state_or_rgb_conditioning':False,'static_hypothesis_policy':'zero_state_invalid',
        'observation_policy':'one_real_previous_Source_observation_then_frozen_prior',
        'autonomous_450_step_claim':False,'optimizer_steps':0}
    plan_sha = fingerprint(identity); root.mkdir(parents=True,exist_ok=True)
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('fingerprint') != plan_sha:
            raise ValueError('Immutable RSSM hypothesis identity changed')
        if old.get('status') == 'complete':
            if any(digest(root/s['path']) != s['sha256'] for s in old['shards']):
                raise ValueError('Immutable RSSM hypothesis sidecar changed')
            return old
    modes = {module:module.training for module in model.modules()}
    flags = {parameter:parameter.requires_grad for parameter in model.parameters()}
    model_before = _hash_tensors(model.state_dict()); device = next(model.parameters()).device
    rng_before = rng_fingerprint(device); model.eval().requires_grad_(False)
    shards = []; completed = 0; valid_count = 0; started = time.perf_counter()
    def publish(status,verified=False):
        report = {**identity,'fingerprint':plan_sha,'status':status,'stage':'data','current':completed,
            'total':len(base['records']),'shards':shards,'valid_dynamic_rows':valid_count,
            'invalid_static_rows':completed-valid_count,'elapsed_seconds':time.perf_counter()-started,
            'base_arrays_unchanged':True,'frozen_model_tensors_bitexact':bool(verified),
            'rng_state_unchanged':bool(verified),'gameplay_qualified':False,'automatic_deployment':False}
        report['data_sha256'] = fingerprint({'identity':identity,'shards':shards})
        _atomic_json(manifest_path,report)
        _atomic_json(root/'status.json',{k:v for k,v in report.items() if k!='shards'})
        return report
    publish('caching')
    try:
        with preserve_rng(device):
            for entry in base['shards']:
                name = entry['path']; split = entry['split']; count = entry['rows']; source = data_root/name
                if digest(source) != entry['sha256']:
                    raise ValueError('Immutable base corpus changed before prediction')
                states = []; validity = []
                for start in range(0,count,batch_size):
                    indices = [lookup[split][(name,j)] for j in range(start,min(count,start+batch_size))]
                    batch = readers[split].batch(len(indices),indices=indices)
                    # The predictor accepts a deliberately narrow mapping; it
                    # cannot accidentally condition on the available labels.
                    past = {key:batch[key] for key in ('dynamic','previous_rgb','previous_state','action','map')}
                    result = predict_hypotheses(model,past)
                    states.append(result['pred_state']); validity.append(result['valid'])
                arrays = {'pred_state':np.concatenate(states),'valid':np.concatenate(validity)}
                path = root/name; path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():
                    with np.load(path,allow_pickle=False) as old:
                        if set(old.files) != set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                            raise ValueError('Existing aligned RSSM hypothesis shard differs')
                else:
                    temporary = path.with_suffix('.tmp.npz'); np.savez_compressed(temporary,**arrays); temporary.replace(path)
                shards.append({'path':name,'base_shard':name,'base_sha256':entry['sha256'],
                    'sha256':digest(path),'rows':count,'split':split})
                completed += count; valid_count += int(arrays['valid'].sum()); publish('caching')
        if base_bytes != base_path.read_bytes() or any(digest(data_root/s['path']) != s['sha256'] for s in base['shards']):
            raise AssertionError('Frozen hypothesis caching changed or outlived immutable base inputs')
        if model_before != _hash_tensors(model.state_dict()) or rng_before != rng_fingerprint(device):
            raise AssertionError('Frozen hypothesis caching changed model tensors or global RNG')
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError('Frozen hypothesis caching produced unexpected gradients')
        return publish('complete',verified=True)
    except BaseException:
        publish('failed'); raise
    finally:
        for module,training in modes.items():
            module.training = training
        for parameter,enabled in flags.items():
            parameter.requires_grad_(enabled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--world',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--seed',type=int,default=173)
    args = parser.parse_args(); torch.set_num_threads(2)
    device = initialize_cache_device(args.device)
    # Windows trusted checkpoint loading can restore pinned CUDA states. Bring
    # CUDA up before deserializing the full world checkpoint, not afterward.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    from .models import load_world_checkpoint
    with preserve_rng(device):
        torch.manual_seed(args.seed)
        model,saved = load_world_checkpoint(args.world,'cpu'); step = saved['training_steps']; del saved
        model.to(device).float().eval().requires_grad_(False)
        report = build_hypothesis_cache(args.data_root,args.output,model,world_sha256=digest(args.world),
            world_step=step,batch_size=args.batch,seed=args.seed)
    print(json.dumps({'status':report['status'],'current':report['current'],'total':report['total'],
        'valid_dynamic_rows':report['valid_dynamic_rows'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
