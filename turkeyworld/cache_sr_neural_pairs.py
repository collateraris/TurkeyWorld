"""Cache frozen NPR32 inputs paired with untouched same-tick Source128 labels.

No training runs here. Dynamic appearance sees only a genuine preceding
observation and control; independent snapshots use the renderer's trained
zero-appearance policy. All target RGB, masks and HR remain labels only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .prepare_training_data import _atomic_json
from .renderer_student import _hash_tensors, codec_tensor_fingerprint
from .small_enemy_data import digest
from .sr_neural_pairs_data import CACHE_PROTOCOL, SourcePairData, fingerprint
from .visual_quality import strict_render_precision


def _pixels(rgb, size):
    if (rgb.ndim != 4 or rgb.shape[1:] != (3,size,size) or not torch.isfinite(rgb).all()
            or ((rgb < 0) | (rgb > 1)).any()):
        raise FloatingPointError('Frozen renderer/codec produced invalid RGB')
    return (rgb.detach().float().cpu().permute(0,2,3,1).numpy()*255).round().astype(np.uint8)


@torch.inference_mode()
def render_cache_batch(model, batch, *, include_codec=True):
    """The NPR path deliberately never indexes target/HR/mask fields."""
    device = next(model.parameters()).device
    state = batch['state'].to(device=device,dtype=torch.float32)
    grid = batch['map'].to(device=device,dtype=torch.long)
    dynamic = batch['dynamic'].to(device=device,dtype=torch.bool)
    count = len(state)
    if (state.shape != (count,20) or grid.shape != (count,16,16) or dynamic.shape != (count,)
            or count < 1 or not torch.isfinite(state).all()):
        raise ValueError('Cache queries require finite state20/map16 and dynamic flags')
    features = state.new_zeros(count,128)
    indices = torch.nonzero(dynamic,as_tuple=False).flatten()
    if len(indices):
        # Only actual past fields are accepted by the initialization/prior.
        previous = batch['previous_state'].to(device=device,dtype=torch.float32)[indices]
        image = batch['previous_rgb'].to(device=device,dtype=torch.float32)[indices]
        controls = batch['action'].to(device=device,dtype=torch.float32)[indices]
        if (previous.shape != (len(indices),20) or image.shape != (len(indices),3,32,32)
                or controls.shape != (len(indices),4) or not all(torch.isfinite(x).all() for x in (previous,image,controls))
                or ((image < 0) | (image > 1)).any()):
            raise ValueError('Dynamic cache queries need genuine preceding RGB/state/action')
        previous_map = model.encode_map(grid[indices])
        memory = model.initialize_from_image(image,previous,previous_map)
        prior = model.transition(previous,controls,previous_map,memory)
        value = prior['features']
        if value.shape != (len(indices),128) or not torch.isfinite(value).all():
            raise FloatingPointError('Frozen causal prior produced invalid appearance')
        features[indices] = value.detach().float()
    # Subset priors can change ephemeral map/spawn caches. Re-encode the full
    # map batch before rendering the recorded target pose, without a reset.
    target_map = model.encode_map(grid)
    with strict_render_precision():
        generated = _pixels(model.render(state,features,target_map),32)
        codec = None
        if include_codec:
            # A separate guard domain: Source32 is never reused as NPR input.
            low = batch['lr_source'].to(device=device,dtype=torch.float32)
            codec = _pixels(model.flow_renderer.codec(low)['reconstruction'],32)
    return {'lr_npr':generated, **({'codec_lr':codec} if codec is not None else {})}


def build_cache(source_root, output, model, *, cache_identity, batch_size=8, seed=173,
                include_codec=True):
    """Pure frozen-model worker, separately callable with a CPU test backend."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Positive cache batch size required')
    if (not isinstance(cache_identity,dict) or cache_identity.get('future_rgb_features') is not False
            or cache_identity.get('paired_same_pose_map_tick') is not True):
        raise ValueError('Cache identity must explicitly prohibit future RGB features and pin same-tick Source poses')
    source_root = Path(source_root).resolve(); root = Path(output).resolve()
    if source_root == root or source_root in root.parents and root.name == 'source':
        raise ValueError('Neural cache must not overwrite its immutable Source data')
    source_path = source_root/'manifest.json'
    source_bytes = source_path.read_bytes(); source_manifest = json.loads(source_bytes)
    if source_manifest['status'] != 'complete':
        raise ValueError('Source pairs must be complete before neural caching')
    source_sha = digest(source_path)
    identity = {**cache_identity,'source_manifest_sha256':source_sha,'seed':seed,
        'input_precision':'uint8_round_to_nearest','static_feature_policy':'snapshot_zero128_no_target_encoding',
        'dynamic_feature_policy':'frozen_prior_actual_previous_Source_RGB_state_action',
        'source_pose_corrected_with_engine':False,'world_rollout':False,'codec_guard_cached':bool(include_codec)}
    plan_sha = fingerprint(identity)
    root.mkdir(parents=True,exist_ok=True)
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('cache_identity') != identity:
            raise ValueError('Immutable frozen-renderer cache identity changed')
        if old['status'] == 'complete':
            if any(digest(root/s['path']) != s['sha256'] for s in old['shards']):
                raise ValueError('Completed neural SR input cache changed')
            return old
    readers = {split:SourcePairData(source_root,split,seed=seed) for split in ('train','val')}
    train_keys = readers['train'].scene_keys
    if train_keys & readers['val'].scene_keys:
        raise ValueError('Neural SR cache TRAIN/VAL scene overlap')
    lookup = {split:{(r['shard'],r['offset']):i for i,r in enumerate(reader.rows)} for split,reader in readers.items()}
    modes = {module:module.training for module in model.modules()}
    flags = {p:p.requires_grad for p in model.parameters()}
    before = _hash_tensors(model.state_dict())
    model.eval().requires_grad_(False)
    started = time.perf_counter(); shards = []; completed = 0
    def publish(status, *, frozen_verified=False):
        report = {**source_manifest,'protocol':CACHE_PROTOCOL,'status':status,'stage':'data',
            'current':completed,'total':len(source_manifest['records']),'fingerprint':plan_sha,
            'cache_identity':identity,'shards':shards,'source_data_root':str(source_root),
            'source_manifest_sha256':source_sha,'source_data_sha256':source_manifest.get('data_sha256'),
            'elapsed_seconds':time.perf_counter()-started,'source_arrays_bitexact':True,
            'frozen_model_tensors_bitexact':bool(frozen_verified),'optimizer_steps':0,
            'neural_input_cache_complete':status=='complete','gameplay_qualified':False,
            'automatic_deployment':False}
        # Source record ordering, split, provenance and shard offsets are exact.
        report['data_sha256'] = fingerprint({'identity':identity,'shards':shards})
        _atomic_json(manifest_path,report)
        _atomic_json(root/'status.json',{k:v for k,v in report.items() if k not in ('records','shards')})
        return report
    publish('caching')
    try:
        for source_shard in source_manifest['shards']:
            name = source_shard['path']; split = source_shard['split']; source_file = source_root/name
            if digest(source_file) != source_shard['sha256']:
                raise ValueError('Immutable Source shard changed before neural caching')
            with np.load(source_file,allow_pickle=False) as z:
                arrays = {key:z[key] for key in z.files}
            count = source_shard['rows']; generated = []; codecs = []
            for start in range(0,count,batch_size):
                panel = [lookup[split][(name,j)] for j in range(start,min(count,start+batch_size))]
                batch = readers[split].batch(len(panel),indices=panel)
                result = render_cache_batch(model,batch,include_codec=include_codec)
                generated.append(result['lr_npr'])
                if include_codec:
                    codecs.append(result['codec_lr'])
            output_arrays = {**arrays,'lr_npr':np.concatenate(generated)}
            if include_codec:
                output_arrays['codec_lr'] = np.concatenate(codecs)
            target = root/name; target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                with np.load(target,allow_pickle=False) as old:
                    if set(old.files) != set(output_arrays) or any(not np.array_equal(old[k],v) for k,v in output_arrays.items()):
                        raise ValueError('Existing frozen-neural-renderer cache shard differs')
            else:
                temporary = target.with_suffix('.tmp.npz')
                np.savez_compressed(temporary,**output_arrays); temporary.replace(target)
            # Explicitly prove every original array survived serialization.
            with np.load(target,allow_pickle=False) as saved:
                if any(not np.array_equal(saved[k],v) for k,v in arrays.items()):
                    raise AssertionError('Neural cache changed immutable Source labels or causal context')
            shards.append({**source_shard,'sha256':digest(target),'source_sha256':source_shard['sha256']})
            completed += count; publish('caching')
        if source_bytes != source_path.read_bytes():
            raise ValueError('Immutable Source manifest changed during neural caching')
        if before != _hash_tensors(model.state_dict()):
            raise AssertionError('Frozen caching changed model weights or buffers')
        if any(p.grad is not None for p in model.parameters()):
            raise AssertionError('Frozen caching unexpectedly produced parameter gradients')
        return publish('complete',frozen_verified=True)
    except BaseException:
        publish('failed'); raise
    finally:
        for module,training in modes.items():
            module.training = training
        for parameter,enabled in flags.items():
            parameter.requires_grad_(enabled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--world',type=Path,required=True)
    parser.add_argument('--renderer',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--seed',type=int,default=173)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--allow-unqualified-codec',action='store_true')
    args = parser.parse_args()
    from .models import load_world_checkpoint
    from .renderer_student import load_renderer_student
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        torch.manual_seed(args.seed)
        model, saved = load_world_checkpoint(args.world,'cpu')
        world_step = saved['training_steps']; del saved
        renderer_metadata = load_renderer_student(model,args.renderer,
            allow_unqualified_codec=args.allow_unqualified_codec)
        identity = {'world_checkpoint':str(args.world.resolve()),'world_sha256':digest(args.world),
            'world_step':world_step,'renderer_checkpoint':str(args.renderer.resolve()),
            'renderer_sha256':digest(args.renderer),'renderer_step':renderer_metadata.get('training_steps'),
            'codec_tensor_sha256':codec_tensor_fingerprint(model.flow_renderer.codec),
            'world_config':asdict(model.config),'future_rgb_features':False,'paired_same_pose_map_tick':True,
            'allow_unqualified_codec':args.allow_unqualified_codec,'renderer_quality_qualified':False}
        model.to(device).float().eval().requires_grad_(False)
        report = build_cache(args.source_root,args.output,model,cache_identity=identity,
            batch_size=args.batch,seed=args.seed)
        print(json.dumps({'status':report['status'],'current':report['current'],'total':report['total'],
                          'data_sha256':report['data_sha256'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
