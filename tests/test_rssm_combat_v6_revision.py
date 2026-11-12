"""Exact-source v6 opt-in, preserved Adam, sampler upgrade and durable EMA."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from test_rssm_combat_only import fixture as combat_only_fixture
from turkeyworld.rssm_batch_revision import batch_commit_size
from turkeyworld.rssm_combat_protocol import commit_combat_update
from turkeyworld.rssm_combat_v6 import bind_combat_v6, v6_boundary_valid
from turkeyworld.rssm_component_balance import SETTINGS, LOSS_PROTOCOL
from turkeyworld.rssm_combat_branches import normalize_combat_branch_protocol
from turkeyworld.rssm_spawn_curriculum import upgrade_attack_balanced_sampler_state
from turkeyworld.rssm_training_schedule import bind_combat_only_revision, commit_combat_only_update


def fixture(folder):
    state, schedule, schedule_kwargs, _, _, _ = combat_only_fixture(folder)
    bind_combat_only_revision(state, schedule, **schedule_kwargs)
    step = state['global_training_step']; training = state['rssm_training']
    training['combat_recovery_revision']['sampler_state'] = {
        'draws': 1216, 'negative_draws': 304, 'anchor_counts': dict.fromkeys(range(8), 38)}
    # A real previously initialized Adam state tests preservation beyond names.
    parameter = torch.nn.Parameter(state['model']['weight'].clone())
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    parameter.square().sum().backward(); optimizer.step()
    state['model'] = {'weight': parameter.detach().clone()}
    state['optimizer'] = deepcopy(optimizer.state_dict())
    state['optimizer_parameter_names'] = ['weight']
    calibration = {'passed': True, 'source_step': step, 'source_checkpoint_sha256': 'v6source',
        'optimizer_updated': False, 'model_weights_and_buffers_unchanged': True, 'settings': deepcopy(SETTINGS), 'components': {
            'spawn': {'enemy': {'raw': 2., 'chosen_priority': .25}},
            'recovery': {'player': {'raw': .5, 'chosen_priority': .2},
                'enemy': {'raw': 4., 'chosen_priority': .1},
                'interaction': {'raw': 1., 'chosen_priority': .15}}}}
    calibration_path = Path(folder)/'calibration.json'
    calibration_path.write_text(json.dumps(calibration, sort_keys=True), encoding='utf-8')
    calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    plan = {'version': 6, 'loss_protocol': LOSS_PROTOCOL, 'component_settings': deepcopy(SETTINGS),
        'base_plan_sha256': 'base', 'parent_loss_balance_sha256': training['loss_balance_revision']['sha256'],
        'parent_batch_sha256': training['batch_revision']['sha256'],
        'parent_combat_only_sha256': training['combat_only_revision']['sha256'],
        'source_checkpoint_sha256': 'v6source', 'gradient_calibration_sha256': calibration_sha,
        'attack_positive_balanced_sampling': True, 'optimizer_reset': False, 'source_future_feedback': False,
        'main_horizon': 450, 'tbptt_chunk': 16, 'validation_every': 250, 'preview_every': 10,
        'control_validation_updates': [100, 250], 'start_optimizer_step': step,
        'combat_branch_protocol': normalize_combat_branch_protocol({'version': 2})}
    path = Path(folder)/'combat_v6.json'; path.write_text(json.dumps(plan, sort_keys=True), encoding='utf-8')
    kwargs = {'path': path, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'calibration_sha256': calibration_sha, 'source_checkpoint_sha256': 'v6source',
        'loaded_checkpoint_sha256': 'v6source', 'base_plan_sha256': 'base'}
    return state, plan, calibration, kwargs, calibration_path


class CombatV6RevisionTests(unittest.TestCase):
    def test_initial_bind_preserves_model_adam_rng_actor_ema_nav_clocks_and_parent_files(self):
        with TemporaryDirectory() as folder:
            state, plan, calibration, kwargs, calibration_path = fixture(folder)
            before = deepcopy(state); rng = torch.get_rng_state().clone()
            bytes_before = {path: path.read_bytes() for path in Path(folder).glob('*.json')}
            revision, balancer, branches = bind_combat_v6(state, plan, calibration, **kwargs)
            self.assertEqual(state['loss_protocol'], LOSS_PROTOCOL)
            self.assertTrue(v6_boundary_valid(state)); self.assertEqual(batch_commit_size(state), 128)
            self.assertEqual(revision['updates'], 0)
            self.assertEqual(revision['legacy_sampler_state'], before['rssm_training']['combat_recovery_revision']['sampler_state'])
            self.assertEqual(branches['version'], 2); self.assertEqual(branches['max_branches'], 8)
            for name, value in before['model'].items(): self.assertTrue(torch.equal(state['model'][name], value))
            self.assertEqual(state['optimizer']['param_groups'], before['optimizer']['param_groups'])
            for name, value in before['optimizer']['state'][0].items():
                self.assertTrue(torch.equal(state['optimizer']['state'][0][name], value))
            for key in ('rng_state', 'global_training_step', 'current', 'completed_fragments', 'rssm_validation'):
                self.assertEqual(state[key], before[key])
            self.assertTrue(torch.equal(state['torch_rng_state'], before['torch_rng_state']))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key, value in before['rssm_training'].items():
                if key != 'loss_balance_state': self.assertEqual(state['rssm_training'][key], value, key)
            actor = {key: value for key, value in balancer.state_dict().items() if key != 'components'}
            self.assertEqual(actor, before['rssm_training']['loss_balance_state'])
            self.assertEqual(balancer.state_dict()['components']['committed_updates'], 0)
            self.assertEqual(state['training_protocol']['auxiliary_combat_quotas']['attack_positive'], 8)
            for path, value in bytes_before.items(): self.assertEqual(path.read_bytes(), value)
            history = len(state['objective_history'])
            again, restored, _ = bind_combat_v6(state, plan, calibration, **kwargs)
            self.assertIs(again, revision); self.assertEqual(restored.state_dict(), balancer.state_dict())
            self.assertEqual(len(state['objective_history']), history)

    def test_sampler_upgrade_is_explicit_retains_complete_legacy_ledger_and_no_rng_reset(self):
        with TemporaryDirectory() as folder:
            state, plan, calibration, kwargs, _ = fixture(folder)
            old = deepcopy(state['rssm_training']['combat_recovery_revision']['sampler_state'])
            rng = deepcopy(state['rng_state'])
            revision, _, _ = bind_combat_v6(state, plan, calibration, **kwargs)
            self.assertEqual(state['rssm_training']['combat_recovery_revision']['sampler_state'], old)
            upgraded = upgrade_attack_balanced_sampler_state(revision['legacy_sampler_state'])
            self.assertEqual(upgraded['retained_legacy_ledger'], old)
            for key in ('draws', 'negative_draws', 'anchor_counts'): self.assertEqual(upgraded[key], old[key])
            self.assertEqual(upgraded['v2_draws'], 0); self.assertEqual(upgraded['positive_draws'], 0)
            self.assertEqual(state['rng_state'], rng)
            self.assertEqual(revision['legacy_sampler_state'], old)

    def test_real_adam_once_then_parent_v6_ledgers_and_later_checkpoint_resume_exact(self):
        with TemporaryDirectory() as folder:
            state, plan, calibration, kwargs, _ = fixture(folder)
            revision, balancer, _ = bind_combat_v6(state, plan, calibration, **kwargs)
            parameter = torch.nn.Parameter(state['model']['weight'].clone())
            optimizer = torch.optim.AdamW([parameter], lr=.5); optimizer.load_state_dict(deepcopy(state['optimizer']))
            old_clock = float(optimizer.state_dict()['state'][0]['step'])
            nav = deepcopy(balancer.state_dict()['banks']['navigation'])
            step = state['global_training_step']+1; reports = []
            optimizer.zero_grad(set_to_none=True)
            for index in range(4):
                acc = balancer.new_microbatch('combat')
                loss = acc.combine({'player': parameter[0].square()+1.1, 'enemy': parameter[1].square()+1.1,
                    'interaction': parameter.square().sum()}, floors={'player': 1.1, 'enemy': 1.1})
                loss += acc.combine({'enemy': (parameter[0]-index/10).square()}, component='spawn')
                loss += acc.combine({'player': parameter[0].square(), 'enemy': parameter[1].square(),
                    'interaction': parameter.sum().square()}, component='recovery')
                loss.backward(); reports.append(acc.report())
            parameter.grad.div_(4)
            summary = balancer.summarize(reports, environment='combat', step=step)
            optimizer.step(); commit_combat_update(state, step=step, environment='combat', batch_size=128)
            self.assertFalse(v6_boundary_valid(state))
            with self.assertRaises(ValueError): batch_commit_size(state)
            for name in ('feedback_revision', 'combat_recovery_revision', 'batch_revision', 'loss_balance_revision'):
                state['rssm_training'][name]['last_optimizer_step'] = step
                state['rssm_training'][name]['updates'] += 1
            balancer.commit(summary, step=step)
            state['rssm_training']['loss_balance_state'] = balancer.state_dict()
            commit_combat_only_update(state, step=step, environment='combat')
            revision.update(last_optimizer_step=step, updates=1)
            self.assertTrue(v6_boundary_valid(state)); self.assertEqual(batch_commit_size(state), 128)
            self.assertEqual(float(optimizer.state_dict()['state'][0]['step']), old_clock+1)
            self.assertEqual(balancer.state_dict()['banks']['navigation'], nav)
            state['model']['weight'] = parameter.detach().clone(); state['optimizer'] = deepcopy(optimizer.state_dict())
            path = Path(folder)/'after.pt'; torch.save(state, path)
            restored = torch.load(path, map_location='cpu', weights_only=False)
            before = deepcopy(restored['rssm_training'])
            _, restored_balance, _ = bind_combat_v6(restored, plan, calibration,
                **{**kwargs, 'loaded_checkpoint_sha256': 'later_complete_checkpoint'})
            self.assertEqual(restored['rssm_training'], before)
            self.assertEqual(restored_balance.state_dict(), balancer.state_dict())
            self.assertEqual(batch_commit_size(restored), 128)
            self.assertTrue(torch.equal(restored['model']['weight'], parameter))
            for name, value in optimizer.state_dict()['state'][0].items():
                self.assertTrue(torch.equal(restored['optimizer']['state'][0][name], value))

    def test_settings_source_calibration_branch_budget_and_parent_pins_reject_before_binding(self):
        with TemporaryDirectory() as folder:
            for key, value in (('version', True), ('source_checkpoint_sha256', 'wrong'),
                ('gradient_calibration_sha256', 'wrong'), ('parent_batch_sha256', 'wrong'),
                ('parent_loss_balance_sha256', 'wrong'), ('parent_combat_only_sha256', 'wrong'),
                ('base_plan_sha256', 'wrong'), ('optimizer_reset', True), ('source_future_feedback', True),
                ('attack_positive_balanced_sampling', False), ('main_horizon', 16), ('validation_every', 100),
                ('control_validation_updates', [250]), ('start_optimizer_step', 1)):
                state, plan, calibration, kwargs, _ = fixture(folder)
                changed = deepcopy(plan); changed[key] = value
                with self.assertRaises(ValueError): bind_combat_v6(state, changed, calibration, **kwargs)
                self.assertNotIn('combat_v6_revision', state['rssm_training'])
            state, plan, calibration, kwargs, _ = fixture(folder)
            with self.assertRaises(ValueError): bind_combat_v6(state, plan, calibration,
                **{**kwargs, 'loaded_checkpoint_sha256': 'same_clock_wrong_weights'})
            for branch_change in ({'version': 1}, {'max_direct_source_queries': 64}):
                changed = deepcopy(plan); changed['combat_branch_protocol'].update(branch_change)
                with self.assertRaises(ValueError): bind_combat_v6(deepcopy(state), changed, calibration, **kwargs)
            for key, value in (('passed', False), ('source_step', 1), ('source_checkpoint_sha256', 'wrong'),
                ('optimizer_updated', True), ('model_weights_and_buffers_unchanged', False)):
                changed = deepcopy(calibration); changed[key] = value
                with self.assertRaises(ValueError): bind_combat_v6(deepcopy(state), plan, changed, **kwargs)

    def test_missing_component_state_priority_tampering_or_parent_clocks_fail_closed(self):
        with TemporaryDirectory() as folder:
            state, plan, calibration, kwargs, _ = fixture(folder)
            bind_combat_v6(state, plan, calibration, **kwargs)
            for mutation in ('missing_components', 'component_clock', 'revision_clock', 'v5_clock', 'priority'):
                broken = deepcopy(state); training = broken['rssm_training']
                if mutation == 'missing_components': training['loss_balance_state'].pop('components')
                elif mutation == 'component_clock': training['loss_balance_state']['components']['committed_updates'] = 1
                elif mutation == 'revision_clock': training['combat_v6_revision']['updates'] = 1
                elif mutation == 'v5_clock': training['loss_balance_revision']['last_optimizer_step'] -= 1
                else: training['loss_balance_state']['components']['priorities']['spawn']['enemy'] = .4
                with self.assertRaises(ValueError): bind_combat_v6(broken, plan, calibration, **kwargs)
                if mutation != 'priority':
                    if mutation != 'v5_clock': self.assertFalse(v6_boundary_valid(broken))
                    with self.assertRaises(ValueError): batch_commit_size(broken)

    def test_active_overbudget_calibration_is_rejected_before_revision_or_component_state_created(self):
        with TemporaryDirectory() as folder:
            state, plan, calibration, kwargs, _ = fixture(folder)
            old = deepcopy(state['rssm_training'])
            calibration['components']['recovery']['enemy'].update(active=True, chosen_priority=.05,
                expected_gradient_norm=6.72, target_gradient_norm=.416)
            with self.assertRaisesRegex(ValueError, 'gradient budget'):
                bind_combat_v6(state, plan, calibration, **kwargs)
            self.assertNotIn('combat_v6_revision', state['rssm_training'])
            self.assertEqual(state['rssm_training'], old)


if __name__ == '__main__': unittest.main()
