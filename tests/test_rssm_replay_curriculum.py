"""Long replay preserves old clocks, separate EMA values and exact Adam resume."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from test_rssm_batch256_revision import fixture as batch_fixture, update
from turkeyworld.rssm_batch256 import bind_batch256_revision
from turkeyworld.rssm_coordinate_curriculum import (
    coordinate_curriculum_plan, bind_coordinate_curriculum, commit_coordinate_update)
from turkeyworld.rssm_kinematic_losses import KINEMATIC_PROTOCOL_V1
from turkeyworld.rssm_memory_overshooting import MEMORY_OVERSHOOTING_PROTOCOL
from turkeyworld.rssm_late_prefix_recovery import LATE_PREFIX_RECOVERY_PROTOCOL
from turkeyworld.rssm_replay_curriculum import (EMA_KEY, ReplayComponentLossBalancer,
    replay_curriculum_plan, bind_replay_curriculum, replay_training_horizon,
    replay_coordinate_expectation, commit_replay_update)


def file_plan(path, plan):
    path.write_text(json.dumps(plan, sort_keys=True), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(folder, old_updates=46):
    state, batch, probe, kwargs, balance, parameter, optimizer = batch_fixture(folder)
    bind_batch256_revision(state, batch, probe, **kwargs)
    # The fixture retains all its genuine parent ledgers and initialized Adam;
    # translate only global metadata clocks to the production5014 seed boundary.
    offset = 5014-state['global_training_step']
    state['global_training_step'] += offset; state['current'] += offset
    q = state['rssm_training']
    q['combat_start_step'] += offset
    for value in q.values():
        if isinstance(value, dict):
            for key in ('start_optimizer_step', 'last_optimizer_step'):
                if key in value: value[key] += offset
    saved = balance.state_dict()
    saved['last_optimizer_step'] += offset
    for key in ('start_optimizer_step', 'last_optimizer_step', 'calibration_source_step'):
        saved['components'][key] += offset
    from turkeyworld.rssm_component_balance import ComponentLossBalancer
    balance = ComponentLossBalancer(saved)
    q['loss_balance_state'] = balance.state_dict()
    seed = deepcopy(q['loss_balance_state'])
    coordinate_plan = coordinate_curriculum_plan(state, source_checkpoint='source5014.pt',
        source_checkpoint_sha256='source5014', base_plan_sha256='base',
        kinematic_protocol=KINEMATIC_PROTOCOL_V1, overshooting_protocol=MEMORY_OVERSHOOTING_PROTOCOL)
    coordinate_path = Path(folder)/'coordinate.json'
    coordinate_kwargs = dict(path=coordinate_path, sha256=file_plan(coordinate_path, coordinate_plan),
        source_checkpoint_sha256='source5014', loaded_checkpoint_sha256='source5014', base_plan_sha256='base')
    bind_coordinate_curriculum(state, coordinate_plan, **coordinate_kwargs)
    for _ in range(old_updates):
        update(state, balance, parameter, optimizer, effective=256)
        commit_coordinate_update(state, step=state['global_training_step'], horizon=16,
            sampler_draws=q['combat_recovery_revision']['sampler_state']['draws'])
    plan = replay_curriculum_plan(state, source_checkpoint='source5060.pt',
        source_checkpoint_sha256='source5060', long_seed_checkpoint='source5014.pt',
        long_seed_sha256='source5014', base_plan_sha256='base', late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL)
    path = Path(folder)/'replay.json'
    kwargs = dict(path=path, sha256=file_plan(path, plan), source_checkpoint_sha256='source5060',
        loaded_checkpoint_sha256='source5060', long_seed_checkpoint_sha256='source5014',
        long_seed_loss_balance_state=seed, base_plan_sha256='base')
    return state, plan, kwargs, coordinate_plan, coordinate_kwargs, parameter, optimizer, seed


def advance(state, balancer, parameter, optimizer):
    choice = replay_training_horizon(state)
    balancer.select_update(horizon=choice['horizon'])
    report = update(state, balancer, parameter, optimizer, effective=256)
    q = state['rssm_training']
    commit_coordinate_update(state, step=state['global_training_step'], horizon=choice['horizon'],
        sampler_draws=q['combat_recovery_revision']['sampler_state']['draws'])
    commit_replay_update(state, step=state['global_training_step'], horizon=choice['horizon'],
        forced_long_replay=choice['forced_long_replay'])
    return choice, report


class ReplayCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_bind_preserves_source_adam_rng_counters_short_ema_and_exact5014_long_seed(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, _, _, parameter, optimizer, seed = setup(folder)
            old = deepcopy(state); rng = torch.get_rng_state().clone()
            files = {path: path.read_bytes() for path in Path(folder).glob('*.json')}
            revision, balance = bind_replay_curriculum(state, plan, **kwargs)
            self.assertEqual(state['global_training_step'], 5060)
            self.assertEqual(state['rssm_training']['coordinate_curriculum']['updates'], 46)
            self.assertEqual(revision['updates'], 0)
            self.assertTrue(torch.equal(state['model']['weight'], old['model']['weight']))
            for key, value in old['optimizer']['state'][0].items():
                self.assertTrue(torch.equal(value, state['optimizer']['state'][0][key]))
            self.assertEqual(state['optimizer']['param_groups'], old['optimizer']['param_groups'])
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(state['rng_state'], old['rng_state'])
            extended = balance.state_dict()
            self.assertEqual({key: value for key, value in extended.items() if key != EMA_KEY}, old['rssm_training']['loss_balance_state'])
            banks = extended[EMA_KEY]['banks']
            self.assertEqual(banks['short']['actor_ema'], old['rssm_training']['loss_balance_state']['banks']['combat']['ema'])
            self.assertEqual(banks['long450']['actor_ema'], seed['banks']['combat']['ema'])
            self.assertEqual(banks['long450']['component_ema'], seed['components']['banks']['combat']['ema'])
            for name, value in old['rssm_training'].items():
                if name != 'loss_balance_state': self.assertEqual(state['rssm_training'][name], value)
            for path, raw in files.items(): self.assertEqual(path.read_bytes(), raw)

    def test_fifth_update450_real_four_micro_one_adam_separate_bank_updates_actual_histogram(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, cp, ck, parameter, optimizer, _ = setup(folder)
            _, balance = bind_replay_curriculum(state, plan, **kwargs)
            clock = float(optimizer.state[parameter]['step'])
            nav = deepcopy(balance.state['banks']['navigation'])
            long_seed = deepcopy(balance.state[EMA_KEY]['banks']['long450'])
            choices = []
            for index in range(5):
                before = deepcopy(balance.state[EMA_KEY]['banks'])
                choice, report = advance(state, balance, parameter, optimizer)
                choices.append(choice['horizon'])
                inactive = 'short' if choice['horizon'] == 450 else 'long450'
                self.assertEqual(balance.state[EMA_KEY]['banks'][inactive], before[inactive])
                self.assertEqual(report['microbatches'], 4)
                self.assertEqual(report['horizon_ema_bank'], choice['ema_bank'])
                if index < 4: self.assertEqual(balance.state[EMA_KEY]['banks']['long450'], long_seed)
            self.assertEqual(choices, [16,16,16,16,450])
            self.assertEqual(float(optimizer.state[parameter]['step']), clock+5)
            q = state['rssm_training']; c = q['coordinate_curriculum']
            self.assertEqual(c['updates'], 51)
            self.assertEqual(c['horizon_update_counts']['16'], 50)
            self.assertEqual(c['horizon_update_counts']['450'], 1)
            self.assertEqual(c['main_transitions_per_example'], 50*16+450)
            self.assertEqual(c['main_transitions'], (50*16+450)*256)
            self.assertEqual(q['replay_curriculum']['forced_long_replay_updates'], 1)
            self.assertEqual(balance.state['banks']['navigation'], nav)
            self.assertEqual(q['replay_curriculum']['short_updates'], 4)
            self.assertEqual(q['replay_curriculum']['long450_updates'], 1)
            bind_coordinate_curriculum(state, cp, **{**ck, 'loaded_checkpoint_sha256':'continued'})
            bind_replay_curriculum(state, plan, **{**kwargs, 'loaded_checkpoint_sha256':'continued'})

    def test_source500_schedule_clock_not_reset_when_crossing100_with_twenty_percent_replay(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, _, _, parameter, optimizer, _ = setup(folder)
            _, balance = bind_replay_curriculum(state, plan, **kwargs)
            choices = [advance(state, balance, parameter, optimizer)[0] for _ in range(60)]
            self.assertEqual(sum(item['forced_long_replay'] for item in choices), 12)
            self.assertEqual(state['rssm_training']['coordinate_curriculum']['updates'], 106)
            self.assertEqual(choices[54]['curriculum_horizon'], 32)
            self.assertEqual(choices[54]['horizon'], 450)
            self.assertEqual(choices[55]['horizon'], 32)
            histogram, transitions, next_horizon = replay_coordinate_expectation(state, 106)
            self.assertEqual(histogram, state['rssm_training']['coordinate_curriculum']['horizon_update_counts'])
            self.assertEqual(sum(histogram.values()), 106)
            self.assertEqual(transitions, state['rssm_training']['coordinate_curriculum']['main_transitions_per_example'])
            self.assertEqual(next_horizon, 32)

    def test_exact_save_resume_after_long_bank_then_next_short_update(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, cp, ck, parameter, optimizer, _ = setup(folder)
            _, balance = bind_replay_curriculum(state, plan, **kwargs)
            for _ in range(5): advance(state, balance, parameter, optimizer)
            path = Path(folder)/'saved.pt'; torch.save(state, path)
            restored = torch.load(path, map_location='cpu', weights_only=False)
            bind_coordinate_curriculum(restored, cp, **{**ck, 'loaded_checkpoint_sha256':'later'})
            _, second = bind_replay_curriculum(restored, plan, **{**kwargs, 'loaded_checkpoint_sha256':'later'})
            parameter2 = torch.nn.Parameter(restored['model']['weight'].clone())
            optimizer2 = torch.optim.AdamW([parameter2], lr=1e-4); optimizer2.load_state_dict(restored['optimizer'])
            advance(state, balance, parameter, optimizer)
            advance(restored, second, parameter2, optimizer2)
            self.assertTrue(torch.equal(parameter, parameter2))
            self.assertEqual(state['rssm_training'], restored['rssm_training'])
            for key, value in optimizer.state[parameter].items():
                self.assertTrue(torch.equal(value, optimizer2.state[parameter2][key]))

    def test_report_bank_identity_partial_duplicate_and_nonfinite_rejected_without_ema_update(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, _, _, parameter, optimizer, _ = setup(folder)
            _, balance = bind_replay_curriculum(state, plan, **kwargs)
            before = deepcopy(balance.state_dict())
            with self.assertRaises(ValueError): commit_replay_update(state, step=5061, horizon=16, forced_long_replay=False)
            self.assertEqual(before, balance.state_dict())
            advance(state, balance, parameter, optimizer)
            old = deepcopy(state['rssm_training']['replay_curriculum'])
            with self.assertRaises(ValueError):commit_replay_update(state, step=5061, horizon=16, forced_long_replay=False)
            self.assertEqual(old, state['rssm_training']['replay_curriculum'])
            bad = deepcopy(balance.state_dict()); bad[EMA_KEY]['banks']['long450']['actor_ema']['enemy'] = float('nan')
            with self.assertRaises(ValueError): ReplayComponentLossBalancer(bad)
            bad = deepcopy(balance.state_dict()); bad['banks']['combat']['ema']['player'] += 1
            with self.assertRaises(ValueError): ReplayComponentLossBalancer(bad)

    def test_pins_cadence_seed_and_resume_counters_fail_closed(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, *_ = setup(folder)
            for key, value in [('source_checkpoint_sha256','wrong'),('coordinate_plan_sha256','wrong'),
                ('long_seed_checkpoint_sha256','wrong'),('settings',{}),('short_scout_microbatch_index',1)]:
                changed = deepcopy(plan); changed[key] = value
                with self.assertRaises(ValueError): bind_replay_curriculum(deepcopy(state), changed, **kwargs)
            for changes in ({'cli_microbatch':32},{'loaded_checkpoint_sha256':'wrong'}):
                with self.assertRaises(ValueError):bind_replay_curriculum(deepcopy(state),plan,**{**kwargs,**changes})
            bad_seed = deepcopy(kwargs['long_seed_loss_balance_state']); bad_seed['last_optimizer_step'] = 5013
            with self.assertRaises(ValueError): bind_replay_curriculum(deepcopy(state), plan, **{**kwargs,'long_seed_loss_balance_state':bad_seed})
            revision, _ = bind_replay_curriculum(state, plan, **kwargs)
            for key, value in [('updates',1),('coordinate_updates_at_start',0),('forced_long_replay_updates',1),('settings',{})]:
                bad = deepcopy(state); bad['rssm_training']['replay_curriculum'][key] = value
                with self.assertRaises(ValueError):bind_replay_curriculum(bad,plan,**kwargs)
            self.assertEqual(revision['updates'], 0)


if __name__ == '__main__': unittest.main()
