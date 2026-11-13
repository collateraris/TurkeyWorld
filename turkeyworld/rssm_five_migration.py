"""Append five zero-residual private fusion experts to a complete three-RSSM boundary."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import torch

from .models import NeuralWorldModel, WorldModelConfig, STATE_SCHEMA, parameter_counts
from .rssm_migration import configure_rssm_training
from .rssm_five_role_fusion import RSSMFiveRoleFusion, RESIDUAL_KEYS, five_role_metadata
from .training_pipeline import atomic_json, replace_file


FIVE_PREFIX = 'vjepa_backend.five_role_fusion.'
MIGRATION_KIND = 'append_five_role_transformer_fusion'


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def report_sha(report):
    return hashlib.sha256(json.dumps({k: v for k, v in report.items() if k != 'revision_sha256'},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def enable_rssm_five_checkpoint(saved, *, seed=173, source_checkpoint_sha256=None):
    """Return an independent append-only payload; every old training ledger stays exact."""
    from .renderer_student import renderer_condition_identity, _hash_tensors
    from .rssm_moe_protocol import moe_training_horizon
    if type(seed) is not int or seed < 0:
        raise ValueError('Five-role initialization requires a nonnegative integer seed')
    if (not isinstance(saved, dict) or saved.get('stage') != 'world'
            or saved.get('state_schema') != STATE_SCHEMA):
        raise ValueError('Five-role migration requires a complete one-enemy world checkpoint')
    old_config = WorldModelConfig(**saved['config'])
    if (old_config.kind != 'teacher' or old_config.architecture != 'dreamer_rssm'
            or not old_config.enemy_rssm_enabled or not old_config.rssm_fusion_moe_enabled
            or not old_config.combat_rssm_enabled or old_config.rssm_five_role_enabled):
        raise ValueError('Expected three existing RSSMs and pretrained shared fusion before the five-role append')
    step = saved.get('training_steps'); run = saved.get('run_state')
    if (type(step) is not int or step < 0 or saved.get('global_training_step') != step
            or not isinstance(run, dict) or run.get('global_training_step') != step
            or saved.get('training_counters', {}).get('pending_microbatches', 0)):
        raise ValueError('Five-role migration requires a complete optimizer boundary')
    moe_training_horizon(run)
    if (not isinstance(saved.get('rssm_fusion_migration'), dict)
            or not isinstance(saved.get('combat_rssm_migration'), dict)):
        raise ValueError('Five-role append requires the full existing architecture lineage')
    state = saved.get('model')
    if (not isinstance(state, dict) or not state or any(not torch.is_tensor(value)
            or value.device.type != 'cpu' or value.is_floating_point() and not bool(torch.isfinite(value).all())
            for value in state.values()) or any(n.startswith(FIVE_PREFIX) for n in state)):
        raise ValueError('Expected finite CPU model tensors without a partial five-role module')
    config = {**asdict(old_config), 'rssm_five_role_enabled': True}
    new_config = WorldModelConfig(**config)
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        old_model = NeuralWorldModel(old_config)
        target = NeuralWorldModel(new_config)
    expected = old_model.state_dict()
    if (set(expected) != set(state) or any(expected[n].shape != v.shape or expected[n].dtype != v.dtype
                                          for n, v in state.items())):
        raise ValueError('Source model namespace/shape/dtype does not match its complete configuration')
    old_model.load_state_dict(dict(state), strict=True, assign=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        fusion = RSSMFiveRoleFusion(width=old_config.rssm_fusion_moe_width,
            heads=old_config.rssm_fusion_moe_heads, expert_hidden=old_config.rssm_fusion_moe_expert_hidden,
            role_bias=old_config.rssm_fusion_moe_role_bias)
    target.vjepa_backend.five_role_fusion = fusion
    added_tensors = set(target.state_dict()) - set(state)
    if not added_tensors or any(not n.startswith(FIVE_PREFIX) for n in added_tensors):
        raise ValueError('Only appended five-role fusion tensors may be new')
    missing, unexpected = target.load_state_dict(dict(state), strict=False, assign=True)
    if unexpected or set(missing) != added_tensors:
        raise ValueError('Five-role append cannot omit any existing model tensor')
    model_state = target.state_dict()
    if any(not torch.equal(v, model_state[n]) for n, v in state.items()):
        raise AssertionError('Five-role append changed an existing model tensor')
    model_state = {n: state[n] if n in state else v.detach().clone() for n, v in model_state.items()}
    configure_rssm_training(old_model); policy = configure_rssm_training(target)
    canonical_old = [n for n, p in old_model.named_parameters() if p.requires_grad]
    canonical_new = [n for n, p in target.named_parameters() if p.requires_grad]
    names, optimizer = saved.get('optimizer_parameter_names'), saved.get('optimizer')
    if (not isinstance(names, list) or names != canonical_old
            or not isinstance(optimizer, dict) or not isinstance(optimizer.get('state'), dict)
            or not optimizer.get('param_groups')):
        raise ValueError('Five-role append requires all old named optimizer groups and states')
    added = [FIVE_PREFIX + n for n, _ in fusion.named_parameters()]
    if canonical_new != names + added:
        raise ValueError('Five-role append must retain the exact canonical old name prefix/order')
    groups = optimizer['param_groups']; ids = [i for g in groups for i in g.get('params', [])]
    if (ids != list(range(len(names))) or any(type(i) is not int for i in ids)
            or set(optimizer['state']) - set(ids)):
        raise ValueError('Old Adam IDs must map exactly to the saved named group order')
    for i, name in enumerate(names):
        for label, value in optimizer['state'].get(i, {}).items():
            if torch.is_tensor(value) and (value.device.type != 'cpu'
                    or label == 'step' and (value.numel() != 1 or not 0 <= float(value) <= step)
                    or label != 'step' and value.shape != state[name].shape
                    or value.is_floating_point() and not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid retained named Adam moment: ' + name + '/' + label)
    for group in groups:
        if 'param_names' in group and group['param_names'] != [names[i] for i in group['params']]:
            raise ValueError('Old Adam group parameter-name order changed')
    new_group = copy.deepcopy({k: v for k, v in groups[0].items() if k not in {'params', 'param_names'}})
    new_group['params'] = list(range(len(names), len(names) + len(added)))
    if 'param_names' in groups[0]: new_group['param_names'] = list(added)
    with torch.no_grad():
        fusion.eval()
        before = torch.get_rng_state().clone()
        result = fusion(torch.zeros(2, 3, fusion.width), torch.zeros(2, 20), torch.zeros(2, 4))
        if any(bool(result[k].count_nonzero()) for k in RESIDUAL_KEYS):
            raise AssertionError('Five-role task additions must initialize to exact zero')
        if not torch.equal(before, torch.get_rng_state()):
            raise AssertionError('Five-role forward changed the established RNG stream')
    old_memory = old_model.vjepa_backend.memory_size
    if target.vjepa_backend.memory_size != old_memory:
        raise AssertionError('Five-role fusion cannot change packed three-RSSM memory')
    identity = renderer_condition_identity(old_model)
    if renderer_condition_identity(target) != identity:
        raise AssertionError('Five-role append changed frozen rendering conditioning')
    counts = parameter_counts(target)
    report = dict(format_version=1, kind=MIGRATION_KIND, source_step=step, seed=seed,
        source_checkpoint_sha256=source_checkpoint_sha256, source_config=asdict(old_config), target_config=config,
        allowed_config_changes=['rssm_five_role_enabled'], old_model_tensors_bitexact=True,
        old_tensor_count=len(state), source_model_sha256=_hash_tensors(state),
        added_parameter_names=added, added_parameter_count=sum(p.numel() for p in fusion.parameters()),
        added_tensor_count=len(added_tensors), memory_size=old_memory, combined_memory_size=old_memory,
        frozen_render_assets_bitexact=True, zero_residual_initialization=True,
        source_renderer_conditioning_identity=identity, old_optimizer_parameter_names_preserved=True,
        old_parameter_order_is_prefix=canonical_new[:len(names)] == names,
        old_optimizer_groups_preserved=len(groups), old_adam_states_preserved=len(optimizer['state']),
        new_optimizer_groups=1, new_optimizer_state_entries=0,
        old_mixed_Adam_ages_preserved_not_reset_to_global=True,
        rng_payloads_preserved=True, all_old_curriculum_sampler_EMA_history_retained=True,
        current_moe_gate_streak_and_all_history_preserved=True, no_objective_or_horizon_restart=True,
        model_parameter_counts=counts, architecture=five_role_metadata(fusion),
        gameplay_qualified=False, automatic_active_promotion=False)
    report['revision_sha256'] = report_sha(report)
    new_run = copy.deepcopy(run)
    # The complete rssm_training subtree (gate, batch, NAV/quiet/recovery PCGs,
    # historical clocks) is intentionally byte-for-byte unchanged at activation.
    new_run.update(rssm_five_migration=report,
        rssm_five_training={**five_role_metadata(fusion), 'enabled': True,
            'migration_sha256': report['revision_sha256'], 'source_step': step, 'current_step': step},
        model_parameters={**run.get('model_parameters', {}), **counts})
    new_run.setdefault('training_protocol', {})['rssm_five_role'] = dict(
        protocol=report['architecture']['protocol'], migration_sha256=report['revision_sha256'], source_step=step)
    payload = {**saved, 'config': config, 'model': model_state,
        'optimizer_parameter_names': names + added,
        'optimizer': {**optimizer, 'state': dict(optimizer['state']),
                      'param_groups': copy.deepcopy(groups) + [new_group]},
        'rssm_five_migration': report, 'run_state': new_run}
    return payload, report


def prepare_rssm_five_continuation(source, destination, *, seed=173):
    """Write a separate checkpoint and immutable proof; never activate a live worker."""
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError('Five-role migration requires a separate new checkpoint destination')
    original_sha = file_sha(source)
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    payload, report = enable_rssm_five_checkpoint(saved, seed=seed, source_checkpoint_sha256=original_sha)
    if file_sha(source) != original_sha:
        raise ValueError('Source changed during migration; use an immutable stopped boundary')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary); replace_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {**report, 'source_checkpoint': str(source), 'checkpoint': str(destination),
               'checkpoint_sha256': file_sha(destination)}
    atomic_json(destination.with_suffix(destination.suffix + '.five.json'), receipt)
    return receipt


class FiveRoleDiagnostics:
    """Detached actual-forward means; this never changes a loss or routing decision."""
    def __init__(self):
        self.active = False

    def begin(self):
        self.active = True; self.calls = self.queries = self.actor_ticks = 0
        self.importance = self.modes = self.role = self.balance = None

    def __call__(self, module, inputs, output):
        if not self.active: return
        probabilities = output['router_probabilities'].detach().float()
        modes = output['player_mode_weights'].detach().float()
        count = len(modes)
        values = (probabilities.sum((0, 1)), modes.sum(0),
                  output['router_role_loss'].detach().float() * count,
                  output['router_balance_loss'].detach().float() * count)
        for key, value in zip(('importance', 'modes', 'role', 'balance'), values):
            old = getattr(self, key); setattr(self, key, value if old is None else old + value)
        self.calls += 1; self.queries += probabilities.shape[0] * probabilities.shape[1]
        self.actor_ticks += count

    def summary(self):
        self.active = False
        if not self.calls: raise ValueError('Enabled five-role architecture had no actual forward calls')
        return dict(router_importance=(self.importance / self.queries).cpu().tolist(),
            player_mode_weights=(self.modes / self.actor_ticks).cpu().tolist(),
            router_role_loss_mean=float(self.role / self.actor_ticks),
            router_balance_loss_mean=float(self.balance / self.actor_ticks),
            forward_calls=self.calls, role_query_count=self.queries, player_mode_actor_ticks=self.actor_ticks,
            scope='all_own_model_forward_calls_during_optimizer_update',
            normalization='actual actor/query means; not weighted optimizer objective normalization',
            source_future_feedback=False)
