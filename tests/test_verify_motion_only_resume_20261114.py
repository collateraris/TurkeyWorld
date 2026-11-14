"""Tiny CPU payloads only; no production checkpoint or worker is accessed."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from turkeyworld.rssm_motion_only_revision import motion_only_settings


PATH = Path(__file__).resolve().parents[1] / 'training_runs/diagnostic_tools/verify_motion_only_resume_20261114.py'
SPEC = importlib.util.spec_from_file_location('motion_resume_proof_test_module', PATH)
PROOF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROOF)


def payloads():
    step, protocol, plan_sha, source_sha = 100, PROOF.PROTOCOL, '7' * 64, '5' * 64
    nav = dict(protocol='nav_private', draws=56, rng=np.random.default_rng(17).bit_generator.state)
    enemy = dict(protocol='enemy_private', draws=80, rng=np.random.default_rng(18).bit_generator.state)
    q = dict(dynamics_trainable=10,
        moe_training=dict(last_optimizer_step=step, source_Beta_Gate={0: 'typed anchor'}),
        navigation_replay_revision=dict(sampler_state=nav, enemy_sampler_state=enemy),
        player_navigation_revision=dict(updates=12), motion_correction_revision=dict(updates=2),
        combat_rssm_revision=dict(step=50), respawn_timing_revision=dict(step=60))
    source = dict(training_steps=step, global_training_step=step, plan_sha256='6' * 64,
        run_state=dict(global_training_step=step, rssm_training=q),
        model={'w': torch.arange(6, dtype=torch.float32).reshape(2, 3),
            'b': torch.tensor(0.), 'frozen.buffer': torch.zeros(2, dtype=torch.bfloat16)},
        optimizer_parameter_names=['w', 'b'],
        optimizer=dict(state={0: dict(step=torch.tensor(31.), exp_avg=torch.full((2, 3), .001),
            exp_avg_sq=torch.full((2, 3), .002))}, param_groups=[
                dict(params=[0], lr=1e-4, weight_decay=.01, betas=(.9, .999), foreach=None),
                dict(params=[1], lr=1e-4, weight_decay=.01, betas=(.9, .999), foreach=None)]),
        torch_rng_state=torch.arange(32, dtype=torch.uint8),
        cuda_rng_state=[torch.arange(16, dtype=torch.uint8)],
        rng_state=deepcopy(np.random.default_rng(173).bit_generator.state),
        state_schema=dict(state_dim=20), config=dict(architecture='dreamer_rssm'),
        rssm_migration=dict(version=1, typed_metadata={1: 'one'}),
        rssm_combat_migration=dict(version=2))
    source['rssm_training'] = source['run_state']['rssm_training']
    initial = deepcopy(source)
    state = initial['run_state']
    state.update(objective_protocol=protocol, loss_protocol=protocol,
        active_objective_formula_revision=protocol,
        training_protocol=dict(motion_only_revision=dict(protocol=protocol,
            plan_sha256=plan_sha, **motion_only_settings())))
    r = dict(protocol=protocol, version=1, sha256=plan_sha, settings=motion_only_settings(),
        source_checkpoint_sha256=source_sha, source_optimizer_step=step, last_optimizer_step=step,
        source_rssm_training=deepcopy(q), updates=0, main_examples=0, main_transitions=0,
        round_history=[], initial_nav_draws=nav['draws'], initial_enemy_draws=enemy['draws'],
        nav_sampler_state=deepcopy(nav), enemy_sampler_state=deepcopy(enemy),
        independent_contact_draws=0, combat_draws=0, combat_examples=0, baseline_complete=True,
        gate=dict(round_index=0, source_optimizer_step=step, last_optimizer_step=step,
            updates=0, stage_index=0, stage_updates=0, current_horizon=16, completed=False,
            quality_passed=False, advance_history=[],
            horizon_update_counts={str(h): 0 for h in motion_only_settings()['horizons']}))
    state['rssm_training']['motion_only_revision'] = r
    state['rssm_training']['dynamics_trainable'] = 8  # Only the actual requires-grad count changes.
    state['updated_at'] = 'new operational timestamp'
    initial['plan_sha256'] = plan_sha
    return source, initial


def verify(source, initial, **kwargs):
    return PROOF.verify_resume_payloads(source, initial,
        expected_source_sha256='5' * 64, expected_plan_sha256='7' * 64,
        expected_model_tensors=3, expected_optimizer_states=1,
        expected_optimizer_groups=2, expected_cuda_devices=1, **kwargs)


def test_complete_tiny_boundary_preserves_mixed_age_adam_and_typed_history():
    source, initial = payloads()
    report = verify(source, initial)
    assert report['passed'] is True
    assert len(report['checks']) >= 25
    assert report['counts'] == dict(model_tensors=3, named_Adam_parameters=2,
        Adam_states=1, Adam_groups=2, CUDA_RNG_devices=1)
    assert report['model_constructed'] is False
    assert report['SGD_performed'] is False
    assert report['worker_launched'] is False
    assert report['arbitrary_operational_run_state_equality_claimed'] is False


MUTATIONS = [
    (lambda c: c['optimizer']['state'][0]['exp_avg'].add_(.1), 'full_named_Adam_state_bitexact'),
    (lambda c: c['optimizer']['state'][0]['step'].add_(1), 'full_named_Adam_state_bitexact'),
    (lambda c: c['optimizer']['param_groups'][1].update(betas=(.8, .999)), 'all_Adam_groups_and_scalars_bitexact'),
    (lambda c: c['optimizer_parameter_names'].reverse(), 'named_Adam_parameter_order_bitexact'),
    (lambda c: c['cuda_rng_state'][0].add_(1), 'all_CUDA_RNG_states_bitexact'),
    (lambda c: c['torch_rng_state'].add_(1), 'Torch_CPU_RNG_bitexact'),
    (lambda c: c['rng_state']['state'].update(state=1), 'private_global_PCG_bitexact'),
    (lambda c: c['model']['frozen.buffer'].fill_(-0.), 'all_model_tensors_bitexact'),
    (lambda c: c.pop('rssm_combat_migration'), 'all_migration_metadata_bitexact'),
    (lambda c: c['run_state']['rssm_training']['moe_training'].update(last_optimizer_step=101), 'opaque_parent_moe_training_bitexact'),
    (lambda c: c['run_state']['rssm_training']['motion_only_revision']['nav_sampler_state'].update(draws=64), 'raw_navigation_sampler_bitexact'),
    (lambda c: c['run_state']['rssm_training']['motion_only_revision']['enemy_sampler_state']['rng']['state'].update(state=1), 'raw_quiet_enemy_sampler_bitexact'),
    (lambda c: c['run_state']['rssm_training']['motion_only_revision']['gate'].update(stage_updates=1), 'initial_H16_round_without_SGD'),
    (lambda c: c['run_state']['rssm_training']['motion_only_revision'].update(source_checkpoint_sha256='other'), 'Source_file_hash_pinned_in_child'),
    (lambda c: c.update(global_training_step=101), 'same_complete_optimizer_step'),
]


def test_any_numerical_or_lineage_mutation_fails_precisely():
    for mutation, failed in MUTATIONS:
        source, initial = payloads()
        mutation(initial)
        report = verify(source, initial)
        assert report['passed'] is False, failed
        assert failed in report['failed_checks'], failed


def test_raw_key_types_signed_zero_container_types_and_dtype_are_not_json_normalized():
    digest = PROOF.typed_sha256
    assert digest({0: 'anchor'}) != digest({'0': 'anchor'})
    assert digest({True: 'anchor'}) != digest({1: 'anchor'})
    assert digest([1, 2]) != digest((1, 2))
    assert digest(0.) != digest(-0.)
    assert digest(torch.tensor(0.)) != digest(torch.tensor(-0.))
    assert digest(torch.ones(2, dtype=torch.float32)) != digest(torch.ones(2, dtype=torch.float64))
    assert digest({'a': 1, 'b': 2}) == digest({'b': 2, 'a': 1})
    assert digest(np.arange(6).reshape(2, 3)) == digest(np.asfortranarray(np.arange(6).reshape(2, 3)))


def test_invalid_named_group_coverage_and_wrong_real_counts_cannot_pass():
    source, initial = payloads()
    initial['optimizer']['param_groups'][1]['params'] = [0]
    assert verify(source, initial)['checks']['named_Adam_index_coverage'] is False
    source, initial = payloads()
    report = PROOF.verify_resume_payloads(source, initial, expected_optimizer_groups=7,
        expected_model_tensors=1599, expected_optimizer_states=442)
    assert report['passed'] is False
    assert set(report['failed_checks']) >= {'expected_Adam_groups', 'expected_model_tensors', 'expected_Adam_states'}


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
        if name.startswith('test_') and callable(value))


if __name__ == '__main__':
    unittest.main()
