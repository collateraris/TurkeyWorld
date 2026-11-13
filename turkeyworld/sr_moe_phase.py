"""Frozen-Atlas warm start with a separately trained five-expert SR append."""
from copy import deepcopy
from pathlib import Path

import torch

from .distill_renderer import _save
from .models import build_upscaler
from .renderer_native_phase import _same
from .renderer_student import _hash_tensors
from .sr_semantic_migration import restore_semantic_optimizer, _sha
from .training_pipeline import atomic_json


PROTOCOL = 'atlas_sr_five_transformer_training_v1'
ROLES = ('enemy_detection', 'sprite_selection_placement', 'far_walls', 'near_walls', 'player_weapon_effects')


def sr_moe_protocol():
    return dict(version=1, protocol=PROTOCOL, roles=list(ROLES),
        inference_inputs='NPR_RGB32_and_optional_causal_predicted_RSSM_hint',
        Source_targets_are_labels_only=True, source_future_feedback=False,
        only_new_sr_moe_parameters_train=True, all_parent_Atlas_parameters_and_buffers_frozen=True,
        world_renderer_codec_frozen=True, adversarial_loss=False,
        microbatch_size=8, prediction_every=15, validation_every=100, validation_frames=256,
        near_far_boundary_tiles=5., near_comparison='strict_less_than',
        enemy_routing_label_split=[.5, .5],
        new_sampler='fresh_identity_bound_actual_HR_visible_empty_balanced_PCG173',
        old_sampler='retained_as_immutable_parent_history', automatic_deployment=False, gameplay_qualified=False)


def prepare_sr_moe_continuation(parent_checkpoint, destination, *, selected_checkpoint, seed=173, new_lr=1e-4):
    from .sr_transformer_moe import enable_sr_transformer_moe, configure_sr_transformer_moe, sr_moe_metadata
    source, selected, destination = Path(parent_checkpoint).resolve(), Path(selected_checkpoint).resolve(), Path(destination).resolve()
    if source == destination or selected == destination or destination.exists():
        raise FileExistsError('SR five-expert continuation requires a new immutable destination')
    if type(seed) is not int or seed < 0 or type(new_lr) not in (int, float) or not 0 < new_lr <= .001:
        raise ValueError('Explicit valid SR expert seed/learning rate required')
    source_sha, selected_sha = _sha(source), _sha(selected)
    parent = torch.load(source, map_location='cpu', weights_only=False)
    export = torch.load(selected, map_location='cpu', weights_only=False)
    if (parent.get('stage') != 'sr' or parent.get('config', {}).get('architecture') != 'atlas_enemy_edsr'
            or parent['config'].get('sr_moe_enabled', False)
            or any(k not in parent for k in ('optimizer', 'optimizer_parameter_names', 'sampler',
                'rng_numpy', 'rng_torch', 'rng_python', 'rng_cuda'))
            or parent.get('training_steps') != export.get('training_steps')
            or parent['config'] != export.get('config') or not _same(parent['model'], export.get('model'))):
        raise ValueError('Full Atlas Adam checkpoint must exactly match the selected frozen Atlas export')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = build_upscaler(parent['config'])
        model.load_state_dict(parent['model'], strict=True)
        old_names = list(dict(model.named_parameters()))
        restore_semantic_optimizer(model, parent)
        migration = enable_sr_transformer_moe(model)
    configure_sr_transformer_moe(model)
    names = list(dict(model.named_parameters())); added = names[len(old_names):]
    if names[:len(old_names)] != old_names or not added or any(not n.startswith('sr_moe.') for n in added):
        raise ValueError('Only append-only sr_moe parameters may join the SR architecture')
    if (any(p.requires_grad for n, p in model.named_parameters() if not n.startswith('sr_moe.'))
            or any(not p.requires_grad for n, p in model.named_parameters() if n.startswith('sr_moe.'))
            or any(not _same(v, model.state_dict()[n]) for n, v in parent['model'].items())):
        raise AssertionError('SR expert activation changed an old tensor or its required frozen policy')
    optimizer = deepcopy(parent['optimizer']); name_groups = deepcopy(parent['optimizer_parameter_names'])
    old_ids = [i for group in optimizer['param_groups'] for i in group['params']]
    if old_ids != list(range(len(old_names))) or set(optimizer['state']) - set(old_ids):
        raise ValueError('Parent SR Adam IDs must follow the exact canonical old name groups')
    group = deepcopy({k: v for k, v in optimizer['param_groups'][-1].items() if k != 'params'})
    group.update(params=list(range(len(old_names), len(names))), lr=float(new_lr))
    optimizer['param_groups'].append(group); name_groups.append(added)
    report = {**migration, 'source_checkpoint':str(source), 'source_checkpoint_sha256':source_sha,
        'selected_checkpoint':str(selected), 'selected_checkpoint_sha256':selected_sha,
        'parent_training_steps':parent['training_steps'], 'parent_atlas_updates':parent.get('atlas_updates'),
        'old_Adam_states':len(parent['optimizer']['state']), 'old_Adam_groups':len(parent['optimizer']['param_groups']),
        'all_old_Adam_states_options_preserved':True, 'old_parameter_names':old_names,
        'added_parameter_names':added, 'added_parameter_count':sum(dict(model.named_parameters())[n].numel() for n in added),
        'fresh_Adam_state_entries':0, 'new_optimizer_lr':float(new_lr),
        'old_sampler_retained_as_history':True, 'new_sampler_draws':0,
        'parent_global_RNG_payloads_preserved':True, 'parent_weights_sha256':_hash_tensors(parent['model']),
        'all_old_SR_parameters_frozen':True, 'atlas_buffers_frozen':True,
        'source_model_export_match_bitexact':True, 'automatic_deployment':False, 'gameplay_qualified':False}
    model.sr_moe_migration = report
    payload = {k: deepcopy(v) for k, v in parent.items() if k not in ('model', 'optimizer', 'protocol', 'sampler', 'best')}
    payload.update(stage='sr', config=deepcopy(model.config), model=model.state_dict(), optimizer=optimizer,
        optimizer_parameter_names=name_groups, sr_moe_migration=report, sr_moe_binding=sr_moe_metadata(model),
        parent_protocol=deepcopy(parent.get('protocol')), parent_training_steps=parent['training_steps'],
        parent_atlas_updates=parent.get('atlas_updates'), parent_sampler=deepcopy(parent['sampler']),
        parent_best=deepcopy(parent.get('best')), sr_moe_updates=0, completed_updates=0, best=None,
        training_steps=parent['training_steps'], automatic_deployment=False, gameplay_qualified=False)
    restore_semantic_optimizer(model, payload)
    if _sha(source) != source_sha or _sha(selected) != selected_sha:
        raise ValueError('Immutable SR parent/export changed during preparation')
    _save(destination, payload)
    atomic_json(destination.with_suffix(destination.suffix + '.sr-moe.json'), report)
    return report
