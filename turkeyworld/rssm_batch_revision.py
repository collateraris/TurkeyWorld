"""Explicit effective128 continuation; existing tensors, Adam and RNG stay intact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


BATCH_PROTOCOL = 'rssm_one_enemy_combat_effective128_v4'


def bind_batch_revision(state, plan, *, sha256, path, parent_recovery_sha,
                        source_checkpoint_sha, cli_microbatch, cli_accum):
    """Bind32x4 at one preserved Adam boundary, or verify an exact later resume.

    The parent recovery plan remains immutable. Only batch metadata/protocol
    and a single objective-history entry are added; no previous counter,
    quality outcome, sampler ledger, tensor or random state is reset.
    """
    expected = {'version': 4, 'microbatch_size': 32, 'gradient_accumulation_steps': 4,
        'effective_batch_size': 128, 'gradient_reduction': 'mean_over_microbatches',
        'same_environment_per_optimizer_step': True, 'same_context_per_optimizer_step': True,
        'same_latent_mode_per_optimizer_step': True, 'source_future_feedback': False,
        'optimizer_reset': False, 'rollback': False}
    if not isinstance(plan, dict) or any(type(plan.get(k)) is not type(v) or plan.get(k) != v for k, v in expected.items()):
        raise ValueError('Batch revision requires explicit32x4 mean-gradient semantics')
    if (type(cli_microbatch) is not int or cli_microbatch != 32
            or type(cli_accum) is not int or cli_accum != 4):
        raise ValueError('CLI microbatch/accumulation differs from the immutable batch plan')
    if not isinstance(sha256, str) or not sha256:
        raise ValueError('Batch revision requires its immutable plan fingerprint')
    training = state['rssm_training']
    recovery = training.get('combat_recovery_revision')
    feedback = training.get('feedback_revision')
    step = state['global_training_step']
    if (not recovery or recovery.get('sha256') != parent_recovery_sha
            or plan.get('parent_recovery_sha256') != parent_recovery_sha
            or recovery.get('last_optimizer_step') != step
            or not feedback or feedback.get('numeric_state_feedback') != 'detached'
            or feedback.get('last_optimizer_step') != step
            or plan.get('feedback_revision_sha256') != feedback.get('sha256')
            or plan.get('source_checkpoint_sha256') != source_checkpoint_sha):
        raise ValueError('Batch revision parent/source or numeric-feedback identity changed')
    parent_path = Path(recovery['plan'])
    parent_bytes = parent_path.read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != parent_recovery_sha:
        raise ValueError('Pinned recovery parent plan content changed')
    parent = json.loads(parent_bytes.decode('utf-8'))
    if (parent.get('version') != 3
            or parent.get('base_plan_sha256') != plan.get('base_plan_sha256')
            or parent.get('feedback_revision_sha256') != plan.get('feedback_revision_sha256')):
        raise ValueError('Batch revision must retain the exact base and feedback plan pins')
    start = plan.get('start_optimizer_step')
    if type(start) is not int or start < 0:
        raise ValueError('Batch revision requires an exact nonnegative optimizer boundary')
    previous = training.get('batch_revision')
    if previous is not None:
        if (previous.get('sha256') != sha256 or previous.get('start_optimizer_step') != start
                or previous.get('last_optimizer_step') != step
                or previous.get('updates') != step-start or step < start
                or previous.get('parent_recovery_sha256') != parent_recovery_sha
                or any(previous.get(k) != expected[k] for k in
                       ('version', 'microbatch_size', 'gradient_accumulation_steps', 'effective_batch_size'))):
            raise ValueError('Saved batch revision identity/counters changed')
    else:
        if start != step:
            raise ValueError('Batch revision must begin at the exact saved Adam boundary')
        previous = {'version': 4, 'sha256': sha256, 'plan': str(path),
            'start_optimizer_step': start, 'last_optimizer_step': start, 'updates': 0,
            'microbatch_size': 32, 'gradient_accumulation_steps': 4, 'effective_batch_size': 128,
            'parent_recovery_sha256': parent_recovery_sha,
            'source_checkpoint_sha256': source_checkpoint_sha}
        state.setdefault('objective_history', []).append({
            'loss_protocol': state['loss_protocol'], 'last_step': step,
            'reason': 'user_requested_microbatch32_accumulation4_effective128',
            'previous_microbatch_size': state['training_protocol'].get('microbatch_size', 16),
            'previous_gradient_accumulation_steps': state['training_protocol'].get('gradient_accumulation_steps', 1),
            'last_validation_report': training.get('last_validation_report'),
            'last_validation_passed': training.get('last_validation_passed', False)})
        training['batch_revision'] = previous
    state['loss_protocol'] = BATCH_PROTOCOL
    state['training_protocol'].update(microbatch_size=32, gradient_accumulation_steps=4,
        effective_batch_size=128, auxiliary_fragments_per_update=128,
        auxiliary_combat_quotas={'batch': 32, 'spawn': 8, 'attack_negative': 8, 'ordinary': 16},
        gradient_reduction='mean_over_microbatches', same_environment_per_optimizer_step=True,
        same_context_per_optimizer_step=True, same_latent_mode_per_optimizer_step=True)
    return previous


def batch_commit_size(state):
    """Return16 for legacy or128 only for a validated, current v4 ledger."""
    revision = state['rssm_training'].get('batch_revision')
    if revision is None:
        return 16
    step = state['global_training_step']
    from .rssm_loss_balance import balanced_batch_boundary_valid
    if (revision.get('version') != 4 or not revision.get('sha256')
            or (state.get('loss_protocol') != BATCH_PROTOCOL and not balanced_batch_boundary_valid(state))
            or revision.get('microbatch_size') != 32
            or revision.get('gradient_accumulation_steps') != 4
            or revision.get('effective_batch_size') != 128
            or revision.get('last_optimizer_step') != step
            or revision.get('updates') != step-revision.get('start_optimizer_step', step+1)):
        raise ValueError('Effective128 commits require the exact bound32x4 batch revision')
    if state['rssm_training'].get('batch256_revision') is not None:
        from .rssm_batch256 import batch256_boundary_valid
        if not batch256_boundary_valid(state):
            raise ValueError('Effective256 requires the exact bound64x4 revision')
        return 256
    protocol = state.get('training_protocol', {})
    if (state.get('batch_protocol') == 'rssm_effective256_v7'
            or protocol.get('microbatch_size') == 64 or protocol.get('effective_batch_size') == 256):
        raise ValueError('Saved effective256 metadata cannot silently lose its revision ledger')
    return 128
