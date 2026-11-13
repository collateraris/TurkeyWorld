"""Append a zero-residual RGB recognition branch without losing parent Adam.

Only an explicit ``fresh_adam=True`` accepts a model-only selected checkpoint.
The unrelated latest checkpoint's moments are never attached to selected-best
weights. No file/model belonging to the parent run is modified.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import random

import numpy as np

import torch

from .models import build_upscaler
from .renderer_student import _hash_tensors
from .training_pipeline import atomic_json
from .distill_renderer import _save


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def restore_semantic_optimizer(model, saved):
    """Restore all named old/new groups, their options, and mixed-age states."""
    names = saved['optimizer_parameter_names']
    groups = saved['optimizer']['param_groups']
    params = dict(model.named_parameters())
    flat = [name for row in names for name in row]
    if (len(names) != len(groups) or len(flat) != len(set(flat))
            or flat != list(params)
            or any(len(row) != len(group['params']) for row, group in zip(names, groups))):
        raise ValueError('Semantic SR optimizer canonical names/groups changed')
    optimizer = torch.optim.AdamW([{**{k: deepcopy(v) for k, v in group.items() if k != 'params'},
        'params': [params[name] for name in row]} for group, row in zip(groups, names)])
    optimizer.load_state_dict(saved['optimizer'])
    for parameter, row in optimizer.state.items():
        for name, value in row.items():
            if torch.is_tensor(value) and (not bool(torch.isfinite(value).all())
                    or name in ('exp_avg','exp_avg_sq','max_exp_avg_sq') and value.shape != parameter.shape):
                raise ValueError('Semantic SR optimizer has invalid moment tensors')
    return optimizer


def prepare_semantic_sr_continuation(parent_checkpoint, destination, *, selected_checkpoint=None,
                                     seed=173, fresh_adam=False, fresh_adam_lr=5e-5):
    parent_path, destination = Path(parent_checkpoint).resolve(), Path(destination).resolve()
    if destination.exists() or destination == parent_path:
        raise ValueError('Semantic SR migration needs a new immutable output path')
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    if parent.get('stage') != 'sr' or parent.get('config', {}).get('architecture') != 'edsr':
        raise ValueError('Semantic SR appends to an explicit trained EDSR parent')
    if not isinstance(parent.get('training_steps'), int) or parent['training_steps'] <= 0:
        raise ValueError('Semantic SR requires actual trained parent weights')
    if not fresh_adam and 'optimizer' not in parent:
        raise ValueError('Selected model-only parent needs explicit fresh_adam=True')
    selected = Path(selected_checkpoint).resolve() if selected_checkpoint else (
        parent_path if fresh_adam else parent_path.with_name('sr_for_game.pt'))
    if not selected.exists():
        raise ValueError('Provide the selected SR export to prove parent moments match its weights')
    selected_payload = torch.load(selected, map_location='cpu', weights_only=False)
    if (selected_payload.get('training_steps') != parent['training_steps']
            or selected_payload.get('config') != parent['config']
            or set(selected_payload['model']) != set(parent['model'])
            or any(not torch.equal(value, selected_payload['model'][name])
                   for name, value in parent['model'].items())):
        raise ValueError('Full optimizer checkpoint does not match the selected SR export')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        legacy = build_upscaler(parent['config'])
        legacy.load_state_dict(parent['model'], strict=True)
        config = {**parent['config'], 'architecture': 'semantic_edsr',
                  'semantic_channels': 32, 'semantic_blocks': 4}
        model = build_upscaler(config)
    target = model.state_dict()
    target.update(parent['model'])
    model.load_state_dict(target, strict=True)
    old_names = [name for name, _ in legacy.named_parameters()]
    new_names = [name for name, _ in model.named_parameters() if name not in old_names]
    parameters = dict(model.named_parameters())
    if any(not name.startswith('semantic_') for name in new_names):
        raise AssertionError('Migration appended an unexpected nonsemantic parameter')
    if fresh_adam:
        old_optimizer = torch.optim.AdamW(legacy.parameters(), lr=fresh_adam_lr, weight_decay=.01).state_dict()
    else:
        old_optimizer = parent['optimizer']
    if len(old_optimizer['param_groups']) != 1:
        raise ValueError('The known parent SR trainer uses one canonical ordered Adam group')
    ids = [index for group in old_optimizer['param_groups'] for index in group['params']]
    if len(ids) != len(old_names) or len(set(ids)) != len(ids):
        raise ValueError('Parent SR optimizer does not cover canonical EDSR parameters once')
    parent_id_by_name = dict(zip(old_names, ids))
    names_by_id = {value: key for key, value in parent_id_by_name.items()}
    old_groups = [[names_by_id[index] for index in group['params']]
                  for group in old_optimizer['param_groups']]
    specifications = [{**{k: deepcopy(v) for k, v in group.items() if k != 'params'},
                       'params': [parameters[name] for name in names]}
                      for group, names in zip(old_optimizer['param_groups'], old_groups)]
    specifications.append({**{k: deepcopy(v) for k, v in old_optimizer['param_groups'][0].items()
                              if k != 'params'}, 'params': [parameters[name] for name in new_names]})
    optimizer = torch.optim.AdamW(specifications)
    migrated = optimizer.state_dict()
    all_names = old_groups + [new_names]
    current_ids = {name: index for names, group in zip(all_names, migrated['param_groups'])
                   for name, index in zip(names, group['params'])}
    migrated['state'] = {current_ids[names_by_id[index]]: deepcopy(state)
                         for index, state in old_optimizer['state'].items()}
    optimizer.load_state_dict(migrated)
    if any(not torch.equal(model.state_dict()[name], value) for name, value in parent['model'].items()):
        raise AssertionError('Semantic SR migration changed a parent tensor')
    report = dict(kind='append_rgb_only_semantic_sr_v1', seed=seed,
        source_checkpoint=str(parent_path), source_checkpoint_sha256=_sha(parent_path),
        selected_checkpoint=str(selected), selected_checkpoint_sha256=_sha(selected),
        parent_training_steps=parent['training_steps'], selected_parent_weights_bitexact=True,
        old_optimizer_preserved=not fresh_adam, explicit_fresh_adam=bool(fresh_adam),
        old_optimizer_state_count=len(old_optimizer['state']),
        old_parameter_names=old_names, added_parameter_names=new_names,
        old_parameter_count=sum(p.numel() for p in legacy.parameters()),
        added_parameter_count=sum(parameters[name].numel() for name in new_names),
        parameter_count=sum(p.numel() for p in model.parameters()),
        parent_weights_sha256=_hash_tensors(parent['model']),
        model_weights_sha256=_hash_tensors(model.state_dict()),
        zero_initial_object_rgb=True, inference_inputs=['rgb32'],
        source_masks_are_loss_only=True, automatic_deployment=False,
        old_sampler_retained_as_history=True, new_sampler_initialization='new_paired_NPR_TRAIN_pool')
    payload = {key: deepcopy(parent[key]) for key in
               ('rng_numpy', 'rng_torch', 'rng_python', 'rng_cuda') if key in parent}
    if fresh_adam:
        payload.setdefault('rng_numpy', np.random.default_rng(seed).bit_generator.state)
        payload.setdefault('rng_torch', torch.Generator().manual_seed(seed).get_state())
        payload.setdefault('rng_python', random.Random(seed).getstate())
        payload.setdefault('rng_cuda', [])
    elif any(key not in payload for key in ('rng_numpy','rng_torch','rng_python','rng_cuda')):
        raise ValueError('A full parent continuation must retain all saved RNG domains')
    report['rng_initialization'] = ('parent_recorded_domains_preserved' if not fresh_adam
                                  else 'available_parent_domains_else_explicit_seed')
    payload.update(stage='sr', format_version=2, config=model.config, model=model.state_dict(),
        optimizer=optimizer.state_dict(), optimizer_parameter_names=all_names,
        training_steps=parent['training_steps'], parent_training_steps=parent['training_steps'],
        completed_updates=0, semantic_updates=0, best=None,
        parent_sampler=deepcopy(parent.get('sampler')), semantic_migration=report,
        gameplay_qualified=False, automatic_deployment=False)
    _save(destination, payload)
    atomic_json(destination.with_suffix(destination.suffix+'.semantic.json'), report)
    return report


def main(argv=None):
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--selected',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=173)
    parser.add_argument('--fresh-adam',action='store_true')
    args=parser.parse_args(argv)
    report=prepare_semantic_sr_continuation(args.parent,args.output,
        selected_checkpoint=args.selected,seed=args.seed,fresh_adam=args.fresh_adam)
    print(json.dumps(report,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
