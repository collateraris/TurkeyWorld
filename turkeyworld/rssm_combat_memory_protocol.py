"""Pin the additive combat-memory architecture without restarting the MoE gate."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .rssm_moe_protocol import moe_training_horizon


COMBAT_MEMORY_PROTOCOL = 'append_combat_rssm20m_moe_v1'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def combat_memory_plan(state, *, initial_checkpoint, migration_proof, source_checkpoint,
                       added_parameters, old_memory_size, new_memory_size):
    """Construct a separate architecture pin; the old objective plan stays intact."""
    gate = state['rssm_training']['moe_training']
    moe_training_horizon(state)
    if (type(added_parameters) is not int or not 19_990_000 <= added_parameters <= 20_010_000
            or old_memory_size != 3632 or type(new_memory_size) is not int
            or new_memory_size <= old_memory_size):
        raise ValueError('Combat continuation requires an actual measured20M additive memory')
    return dict(version=1, kind=COMBAT_MEMORY_PROTOCOL,
        source_optimizer_step=state['global_training_step'],
        initial_checkpoint=str(Path(initial_checkpoint).resolve()),
        initial_checkpoint_sha256=file_sha(initial_checkpoint),
        migration_proof=str(Path(migration_proof).resolve()),
        migration_proof_sha256=file_sha(migration_proof),
        source_checkpoint=str(Path(source_checkpoint).resolve()),
        source_checkpoint_sha256=file_sha(source_checkpoint),
        moe_training_plan_sha256=gate['sha256'], source_moe_gate_sha256=digest_json(gate),
        added_parameters=added_parameters, old_memory_size=old_memory_size,
        new_memory_size=new_memory_size, gate_reset=False, optimizer_reset=False,
        batch_size=64, accumulation_steps=4, effective_batch_size=256)


def bind_combat_memory(state, plan, *, path, sha256, loaded_checkpoint_sha256):
    """Bind once at the migrated boundary, or verify an exact continuation."""
    raw = Path(path).read_bytes()
    if (hashlib.sha256(raw).hexdigest() != sha256 or json.loads(raw.decode('utf-8')) != plan
            or plan.get('kind') != COMBAT_MEMORY_PROTOCOL or plan.get('version') != 1
            or plan.get('gate_reset') is not False or plan.get('optimizer_reset') is not False
            or (plan.get('batch_size'), plan.get('accumulation_steps'), plan.get('effective_batch_size')) != (64, 4, 256)):
        raise ValueError('Combat-memory immutable architecture or batch plan changed')
    if (file_sha(plan['migration_proof']) != plan['migration_proof_sha256']
            or file_sha(plan['initial_checkpoint']) != plan['initial_checkpoint_sha256']):
        raise ValueError('Pinned combat-memory checkpoint or migration proof changed')
    gate = state['rssm_training']['moe_training']
    moe_training_horizon(state)
    step, start = state['global_training_step'], plan['source_optimizer_step']
    if (type(start) is not int or start > step or gate['sha256'] != plan['moe_training_plan_sha256']):
        raise ValueError('Combat memory cannot replace the existing objective/curriculum clock')
    revision = state['rssm_training'].get('combat_rssm_revision')
    if revision is None:
        if (step != start or loaded_checkpoint_sha256 != plan['initial_checkpoint_sha256']
                or digest_json(gate) != plan['source_moe_gate_sha256']):
            raise ValueError('Combat-memory activation must preserve the complete source MoE gate')
        revision = dict(version=1, kind=COMBAT_MEMORY_PROTOCOL, plan=str(Path(path).resolve()),
            sha256=sha256, source_optimizer_step=start, last_optimizer_step=step,
            updates=0, initial_checkpoint_sha256=plan['initial_checkpoint_sha256'],
            migration_proof_sha256=plan['migration_proof_sha256'],
            added_parameters=plan['added_parameters'], old_memory_size=plan['old_memory_size'],
            new_memory_size=plan['new_memory_size'], source_gate=deepcopy(gate))
        state['rssm_training']['combat_rssm_revision'] = revision
    elif (revision.get('sha256') != sha256 or revision.get('source_optimizer_step') != start
            or revision.get('last_optimizer_step') != step
            or revision.get('updates') != step-start
            or revision.get('initial_checkpoint_sha256') != plan['initial_checkpoint_sha256']
            or revision.get('migration_proof_sha256') != plan['migration_proof_sha256']):
        raise ValueError('Combat-memory saved optimizer clock or architecture identity changed')
    state['combat_memory_protocol'] = COMBAT_MEMORY_PROTOCOL
    state.setdefault('training_protocol', {}).update(combat_rssm_enabled=True,
        combat_rssm_updates_every_tick=True, combat_memory_plan_sha256=sha256,
        combat_memory_is_source_corrected=False, separate_combat_xy_or_HP_truth=False)
    return revision


def commit_combat_memory(state, *, step):
    revision = state['rssm_training']['combat_rssm_revision']
    if step != state['global_training_step'] or step != revision['last_optimizer_step']+1:
        raise ValueError('Combat memory requires one complete optimizer commit')
    revision['updates'] += 1
    revision['last_optimizer_step'] = step
    return revision
