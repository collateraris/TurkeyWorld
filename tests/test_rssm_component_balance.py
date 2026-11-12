"""Measured component gradients bypass actor magnitude without losing clocks."""
from copy import deepcopy
import unittest

import torch
from torch import nn

from turkeyworld.rssm_loss_balance import GROUPS, GroupLossBalancer
from turkeyworld.rssm_component_balance import (
    SETTINGS, ComponentLossBalancer, initialize_component_balancer,
)


def fixture():
    actor = GroupLossBalancer().state_dict()
    actor.update(committed_updates=10, last_optimizer_step=4944)
    actor['banks']['combat'] = {'updates': 8, 'ema': {'player': 4., 'enemy': 50., 'interaction': 2.}}
    actor['banks']['navigation'] = {'updates': 2, 'ema': {'player': 1., 'enemy': .1, 'interaction': None}}
    calibration = {'passed': True, 'source_step': 4944, 'components': {
        'spawn': {'enemy': {'raw': 2., 'chosen_priority': .25}},
        'recovery': {'player': {'raw': .5, 'chosen_priority': .2},
                     'enemy': {'raw': 4., 'chosen_priority': .1},
                     'interaction': {'raw': 1., 'chosen_priority': .15}}}}
    return actor, calibration


def accumulator_losses(balance, parameter, factor=1.):
    accumulator = balance.new_microbatch('combat')
    base = {name: parameter[index].square()*factor+(1.1 if name != 'interaction' else 0.)
            for index, name in enumerate(GROUPS)}
    losses = [accumulator.combine(base, floors={'player': 1.1, 'enemy': 1.1})]
    losses.append(accumulator.combine({'enemy': parameter[3].square()*factor}, component='spawn'))
    losses.append(accumulator.combine({name: parameter[index+4].square()*factor
                                      for index, name in enumerate(GROUPS)}, component='recovery'))
    return accumulator, sum(losses)


class ComponentLossBalanceTests(unittest.TestCase):
    def test_measured_initialization_preserves_actor_banks_counters_and_source_inputs(self):
        actor, calibration = fixture(); old_actor, old_calibration = deepcopy(actor), deepcopy(calibration)
        state = initialize_component_balancer(actor, calibration, step=4944)
        self.assertEqual(actor, old_actor); self.assertEqual(calibration, old_calibration)
        self.assertEqual({key: value for key, value in state.items() if key != 'components'}, old_actor)
        balance = ComponentLossBalancer(state)
        comp = balance.state_dict()['components']
        self.assertEqual(comp['committed_updates'], 0)
        self.assertEqual(comp['last_optimizer_step'], 4944)
        self.assertEqual(comp['banks']['combat']['ema']['spawn']['enemy'], 2.)
        self.assertIsNone(comp['banks']['navigation']['ema']['spawn']['enemy'])
        self.assertEqual(balance.new_microbatch('combat').component_scales['spawn']['enemy'], 2.)
        self.assertEqual(balance.new_microbatch('combat').component_priorities['recovery']['enemy'], .1)

    def test_spawn_and_recovery_actual_gradients_do_not_shrink_with_large_actor_enemy_ema(self):
        measurements = []
        for magnitude in (50., 50_000.):
            actor, calibration = fixture(); actor['banks']['combat']['ema']['enemy'] = magnitude
            balance = ComponentLossBalancer(initialize_component_balancer(actor, calibration, step=4944))
            parameter = nn.Parameter(torch.ones(7, dtype=torch.float64))
            accumulator, loss = accumulator_losses(balance, parameter)
            loss.backward(); measurements.append(parameter.grad.clone())
            self.assertAlmostEqual(float(parameter.grad[1]), 2/magnitude)
            self.assertAlmostEqual(float(parameter.grad[3]), 2*.25/2.)
            self.assertAlmostEqual(float(parameter.grad[5]), 2*.1/4.)
            self.assertEqual(accumulator.scales['enemy'], magnitude)
        torch.testing.assert_close(measurements[0][3:], measurements[1][3:], rtol=0, atol=0)
        self.assertGreater(float(measurements[0][1]), float(measurements[1][1]))

    def test_raw_and_weighted_conservation_with_constant_unscaled_dual_kl_floor(self):
        actor, calibration = fixture()
        balance = ComponentLossBalancer(initialize_component_balancer(actor, calibration, step=4944))
        parameter = nn.Parameter(torch.tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64))
        accumulator, loss = accumulator_losses(balance, parameter)
        report = accumulator.report()
        for name in GROUPS:
            row = report[name]
            self.assertAlmostEqual(row['raw'], row['base_raw']+sum(value['raw'] for value in row['components'].values()))
            expected = row['base_raw']*row['weight']/row['scale']+row['kl_floor']
            expected += sum(value['raw']*value['priority']/value['scale'] for value in row['components'].values())
            self.assertAlmostEqual(row['weighted'], expected)
        self.assertAlmostEqual(report['total'], float(loss.detach()))
        self.assertAlmostEqual(report['total'], sum(report[name]['weighted'] for name in GROUPS))
        self.assertAlmostEqual(report['kl_floor'], 2.2)
        self.assertEqual(set(report['player']['components']), {'recovery'})
        self.assertEqual(set(report['enemy']['components']), {'spawn', 'recovery'})

    def test_four_microbatches_have_frozen_coefficients_one_commit_and_actor_base_only_ema(self):
        actor, calibration = fixture()
        balance = ComponentLossBalancer(initialize_component_balancer(actor, calibration, step=4944))
        before = balance.state_dict(); parameter = nn.Parameter(torch.arange(1., 8., dtype=torch.float64))
        reference = nn.Parameter(parameter.detach().clone())
        optimizer = torch.optim.AdamW([parameter], lr=.001)
        reference_optimizer = torch.optim.AdamW([reference], lr=.001)
        reports = []
        for factor in (1., 2., 3., 4.):
            accumulator, loss = accumulator_losses(balance, parameter, factor)
            loss.backward(); reports.append(accumulator.report())
            self.assertEqual(balance.state_dict(), before)
        parameter.grad.div_(4)
        # Independent full-batch mean with the fixed coefficients measured at4944.
        expected = 2.5*(reference[:3].square()/reference.new_tensor([4., 50., 2.])).sum()+2.2
        expected += 2.5*(reference[3].square()*.25/2.+reference[4].square()*.2/.5
                        +reference[5].square()*.1/4.+reference[6].square()*.15)
        expected.backward()
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-14, atol=1e-14)
        report = balance.summarize(reports, environment='combat', step=4945)
        self.assertEqual(report['microbatches'], 4)
        self.assertAlmostEqual(report['total'], float(expected.detach()))
        optimizer.step(); reference_optimizer.step(); balance.commit(report, step=4945)
        self.assertTrue(torch.equal(parameter, reference))
        after = balance.state_dict()
        self.assertEqual(after['committed_updates'], 11)
        self.assertEqual(after['components']['committed_updates'], 1)
        self.assertEqual(after['banks']['combat']['updates'], 9)
        self.assertEqual(after['components']['banks']['combat']['updates'], 1)
        for name in GROUPS:
            self.assertAlmostEqual(after['banks']['combat']['ema'][name],
                .95*before['banks']['combat']['ema'][name]+.05*report[name]['base_raw'])
        self.assertEqual(after['banks']['navigation'], before['banks']['navigation'])
        self.assertEqual(after['components']['banks']['navigation'], before['components']['banks']['navigation'])
        self.assertAlmostEqual(after['components']['banks']['combat']['ema']['spawn']['enemy'],
            .95*2.+.05*report['enemy']['components']['spawn']['raw'])
        unchanged = balance.state_dict()
        with self.assertRaises(ValueError): balance.commit(report, step=4945)
        self.assertEqual(balance.state_dict(), unchanged)

    def test_exact_next_adam_step_after_resume_including_component_and_actor_banks(self):
        actor, calibration = fixture()
        balance = ComponentLossBalancer(initialize_component_balancer(actor, calibration, step=4944))
        parameter = nn.Parameter(torch.arange(1., 8., dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=.001)

        def update(value, adam, balancer, step):
            adam.zero_grad(set_to_none=True)
            accumulator, loss = accumulator_losses(balancer, value, step-4943)
            loss.backward()
            report = balancer.summarize([accumulator.report()]*4, environment='combat', step=step)
            adam.step(); balancer.commit(report, step=step)

        update(parameter, optimizer, balance, 4945)
        saved = {'parameter': parameter.detach().clone(), 'adam': deepcopy(optimizer.state_dict()), 'balance': balance.state_dict()}
        restored = nn.Parameter(saved['parameter'].clone())
        restored_optimizer = torch.optim.AdamW([restored], lr=.5)
        restored_optimizer.load_state_dict(deepcopy(saved['adam']))
        restored_balance = ComponentLossBalancer(saved['balance'])
        update(parameter, optimizer, balance, 4946)
        update(restored, restored_optimizer, restored_balance, 4946)
        self.assertTrue(torch.equal(parameter, restored))
        self.assertEqual(balance.state_dict(), restored_balance.state_dict())
        for name, value in optimizer.state_dict()['state'][0].items():
            self.assertTrue(torch.equal(value, restored_optimizer.state_dict()['state'][0][name]))
        self.assertEqual(saved['balance']['components']['committed_updates'], 1)

    def test_nonfinite_component_rejected_before_adam_or_any_ema_mutation(self):
        actor, calibration = fixture()
        balance = ComponentLossBalancer(initialize_component_balancer(actor, calibration, step=4944))
        parameter = nn.Parameter(torch.ones(7, dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=.001)
        before = balance.state_dict(); accumulator = balance.new_microbatch('combat')
        original_report = accumulator.report()
        with self.assertRaises(FloatingPointError):
            accumulator.combine({'enemy': parameter[3]*float('nan')}, component='spawn').backward()
            optimizer.step()
        self.assertTrue(torch.equal(parameter, torch.ones(7, dtype=torch.float64)))
        self.assertFalse(optimizer.state)
        self.assertIsNone(parameter.grad)
        self.assertEqual(accumulator.report(), original_report)
        good, _ = accumulator_losses(balance, parameter)
        report = balance.summarize([good.report()]*4, environment='combat', step=4945)
        report['enemy']['components']['spawn']['raw'] = float('nan')
        with self.assertRaises(FloatingPointError): balance.commit(report, step=4945)
        self.assertEqual(balance.state_dict(), before)

    def test_invalid_calibration_identity_assignments_and_changed_micro_coefficients_rejected(self):
        actor, calibration = fixture()
        for field, value in (('passed', False), ('source_step', 4943)):
            changed = deepcopy(calibration); changed[field] = value
            with self.assertRaises(ValueError): initialize_component_balancer(actor, changed, step=4944)
        for priority in (.0001, 1.001, float('nan')):
            changed = deepcopy(calibration); changed['components']['spawn']['enemy']['chosen_priority'] = priority
            with self.assertRaises(ValueError): initialize_component_balancer(actor, changed, step=4944)
        state = initialize_component_balancer(actor, calibration, step=4944)
        with self.assertRaises(ValueError): initialize_component_balancer(state, calibration, step=4944)
        balance = ComponentLossBalancer(state)
        accumulator = balance.new_microbatch('combat')
        with self.assertRaises(ValueError): accumulator.combine({'player': torch.tensor(1.)}, component='spawn')
        with self.assertRaises(ValueError): accumulator.combine({'enemy': torch.tensor(1.)}, component='spawn', floors={'enemy': 1.1})
        parameter = nn.Parameter(torch.ones(7, dtype=torch.float64)); first, _ = accumulator_losses(balance, parameter)
        changed_report = first.report(); changed_report['enemy']['components']['spawn']['priority'] = .4
        with self.assertRaises(ValueError): balance.summarize([first.report(), changed_report], environment='combat', step=4945)
        for mutation in ('settings', 'clock', 'bank', 'priority'):
            broken = deepcopy(state)
            if mutation == 'settings': broken['components']['settings']['ema_decay'] = .5
            elif mutation == 'clock': broken['components']['last_optimizer_step'] -= 1
            elif mutation == 'bank': broken['components']['banks']['combat']['updates'] += 1
            else: broken['components']['priorities']['spawn']['enemy'] = 2.
            with self.assertRaises(ValueError): ComponentLossBalancer(broken)

    def test_active_calibration_cannot_overboost_gradient_budget_despite_valid_priority(self):
        actor, calibration = fixture()
        row = calibration['components']['recovery']['enemy']
        row.update(active=True, chosen_priority=.05, expected_gradient_norm=6.72,
                   target_gradient_norm=.416)
        before = deepcopy(actor)
        with self.assertRaisesRegex(ValueError, 'gradient budget'):
            initialize_component_balancer(actor, calibration, step=4944)
        self.assertEqual(actor, before)
        # A lower, now-permitted priority follows the measured norm budget.
        row.update(chosen_priority=.003, expected_gradient_norm=6.72*.003/.05)
        state = initialize_component_balancer(actor, calibration, step=4944)
        balance = ComponentLossBalancer(state)
        value = torch.tensor(2., dtype=torch.float64, requires_grad=True)
        balance.new_microbatch('combat').combine({'enemy': value}, component='recovery').backward()
        self.assertAlmostEqual(float(value.grad), .003/4.)
        self.assertLess(row['expected_gradient_norm'], row['target_gradient_norm']*1.1)
        for field in ('expected_gradient_norm', 'target_gradient_norm'):
            invalid = deepcopy(calibration); invalid['components']['recovery']['enemy'][field] = float('nan')
            with self.assertRaises(ValueError): initialize_component_balancer(actor, invalid, step=4944)


if __name__ == '__main__': unittest.main()
