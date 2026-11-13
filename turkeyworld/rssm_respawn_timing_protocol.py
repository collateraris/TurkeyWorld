"""Record a death-clock correctness revision at a full optimizer boundary."""
from copy import deepcopy
import json
from pathlib import Path

from .rssm_combat_memory_protocol import digest_json, file_sha
from .rssm_moe_protocol import moe_training_horizon
from .simulation_clock import NORMALIZED_TIMER_VERSION


RESPAWN_TIMING_PROTOCOL = 'respawn_phase_alignment_v1'
CODE_FILES = ('simulation_clock.py', 'structured_state.py', 'engine.py',
    'spawn.py', 'anchor_spawn.py', 'spawn_render_training.py', 'runtime_checks.py')


def code_fingerprints():
    package = Path(__file__).parent
    return {name: file_sha(package/name) for name in CODE_FILES}


def respawn_timing_plan(state, *, source_checkpoint):
    moe_training_horizon(state)
    return dict(version=1, kind=RESPAWN_TIMING_PROTOCOL,
        timer_protocol=NORMALIZED_TIMER_VERSION, scope='death_and_respawn_only',
        source_optimizer_step=state['global_training_step'],
        source_checkpoint=str(Path(source_checkpoint).resolve()),
        source_checkpoint_sha256=file_sha(source_checkpoint),
        source_moe_gate_sha256=digest_json(state['rssm_training']['moe_training']),
        moe_training_plan_sha256=state['rssm_training']['moe_training']['sha256'],
        code_sha256=code_fingerprints(), batch_size=64, accumulation_steps=4,
        effective_batch_size=256, gate_reset=False, optimizer_reset=False,
        source_state_feedback=False, canonical_adapter_changed=False)


def bind_respawn_timing(state, plan, *, path, sha256, loaded_checkpoint_sha256):
    if (file_sha(path) != sha256 or json.loads(Path(path).read_text(encoding='utf-8')) != plan
            or plan.get('kind') != RESPAWN_TIMING_PROTOCOL or plan.get('version') != 1
            or plan.get('timer_protocol') != NORMALIZED_TIMER_VERSION
            or plan.get('scope') != 'death_and_respawn_only'
            or plan.get('code_sha256') != code_fingerprints()
            or any(plan.get(k) is not False for k in ('gate_reset', 'optimizer_reset',
                                                    'source_state_feedback', 'canonical_adapter_changed'))
            or (plan.get('batch_size'), plan.get('accumulation_steps'), plan.get('effective_batch_size')) != (64, 4, 256)
            or file_sha(plan['source_checkpoint']) != plan.get('source_checkpoint_sha256')):
        raise ValueError('Pinned respawn-clock correctness revision changed')
    moe_training_horizon(state)
    q = state['rssm_training']
    step, start = state['global_training_step'], plan['source_optimizer_step']
    if type(start) is not int or start > step or q['moe_training']['sha256'] != plan['moe_training_plan_sha256']:
        raise ValueError('Respawn-clock revision cannot replace the current MoE clock')
    revision = q.get('respawn_timing_revision')
    if revision is None:
        if (step != start or loaded_checkpoint_sha256 != plan['source_checkpoint_sha256']
                or digest_json(q['moe_training']) != plan['source_moe_gate_sha256']):
            raise ValueError('Respawn fix must start from its exact complete saved optimizer/gate boundary')
        revision = dict(version=1, kind=RESPAWN_TIMING_PROTOCOL, plan=str(Path(path).resolve()),
            sha256=sha256, timer_protocol=NORMALIZED_TIMER_VERSION,
            source_optimizer_step=start, last_optimizer_step=step, updates=0,
            source_checkpoint_sha256=plan['source_checkpoint_sha256'],
            source_gate=deepcopy(q['moe_training']))
        q['respawn_timing_revision'] = revision
    elif (revision.get('sha256') != sha256 or revision.get('source_optimizer_step') != start
            or revision.get('last_optimizer_step') != step or revision.get('updates') != step-start
            or revision.get('source_checkpoint_sha256') != plan['source_checkpoint_sha256']):
        raise ValueError('Respawn-clock saved update counter or source identity changed')
    state['respawn_clock_protocol'] = NORMALIZED_TIMER_VERSION
    state.setdefault('training_protocol', {}).update(respawn_clock_protocol=NORMALIZED_TIMER_VERSION,
        respawn_timing_plan_sha256=sha256, respawn_timer_scope='death_only',
        source_state_feedback=False, canonical_state_tolerance=2e-7)
    return revision


def commit_respawn_timing(state, *, step):
    revision = state['rssm_training']['respawn_timing_revision']
    if step != state['global_training_step'] or step != revision['last_optimizer_step']+1:
        raise ValueError('Respawn-clock revision requires exactly one complete optimizer update')
    revision['updates'] += 1
    revision['last_optimizer_step'] = step
    return revision
