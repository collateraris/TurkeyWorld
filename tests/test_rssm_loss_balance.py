"""Task routing, constant KL floors and optimizer-boundary EMA accounting."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.rssm import balanced_kl
from turkeyworld.rssm_accumulation import average_accumulated_gradients
from turkeyworld.rssm_loss_balance import (
    GROUPS, SETTINGS, GroupLossAccumulator, GroupLossBalancer,
    combat_loss_groups, split_state_loss,
)
from turkeyworld.train import state_loss


class GroupLossBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_disjoint_state_fields_preserve_old_weighted_value_and_gradient(self):
        expected = {'player': [0, 1, 2, 3], 'enemy': [8, 9, 10, 11, 17],
                    'interaction': [4, 5, 6, 7, 12, 13, 14, 15, 16, 18, 19]}
        target = torch.linspace(-.3, .7, 60, dtype=torch.float64).reshape(3, 20)
        prediction = target.flip(0).clone().requires_grad_()
        separated = split_state_loss(prediction, target)
        legacy = state_loss(prediction, target)
        torch.testing.assert_close(sum(separated.values()), legacy, rtol=0, atol=1e-15)
        original_gradient = torch.autograd.grad(legacy, prediction, retain_graph=True)[0]
        total_gradient = torch.autograd.grad(sum(separated.values()), prediction, retain_graph=True)[0]
        torch.testing.assert_close(total_gradient, original_gradient, rtol=0, atol=0)
        for group, ids in expected.items():
            gradient = torch.autograd.grad(separated[group], prediction, retain_graph=True)[0]
            other = sorted(set(range(20))-set(ids))
            self.assertEqual(float(gradient[:, other].abs().sum()), 0., group)
            torch.testing.assert_close(gradient[:, ids], original_gradient[:, ids], rtol=0, atol=0)
        # A single bad HP/clock field must not masquerade as motion of either actor.
        for index in range(20):
            query = torch.zeros(1, 20, requires_grad=True)
            query.data[0, index] = 1.
            terms = split_state_loss(query, torch.zeros_like(query))
            owner = next(name for name, ids in expected.items() if index in ids)
            self.assertGreater(float(terms[owner].detach()), 0.)
            self.assertTrue(all(float(value.detach()) == 0. for name, value in terms.items() if name != owner))

    def test_combat_damage_and_life_are_interaction_but_motion_and_spawn_are_enemy(self):
        names = ('combat_events', 'combat_target', 'combat_attack', 'combat_damage_npc',
                 'combat_damage_player', 'combat_death_npc', 'combat_spawn', 'combat_npc_motion')
        values = {name: torch.tensor(float(index+1), requires_grad=True)
                  for index, name in enumerate(names)}
        groups = combat_loss_groups(values)
        self.assertNotIn('player', groups)
        self.assertEqual(float(groups['interaction'].detach()), sum(range(1, 7)))
        self.assertEqual(float(groups['enemy'].detach()), 7.+8.)
        sum(groups.values()).backward()
        self.assertTrue(all(float(value.grad) == 1. for value in values.values()))
        with self.assertRaisesRegex(ValueError, 'Unassigned combat'):
            combat_loss_groups({'diagnostic_hit_count': torch.tensor(3.)})

    def test_actual_dual_kl_floor_is_constant_and_never_normalized(self):
        priors, posteriors, values = [], [], {}
        for name in ('player', 'enemy'):
            prior = torch.zeros(2, 4, 4, requires_grad=True)
            posterior = torch.zeros_like(prior, requires_grad=True)
            priors.append(prior); posteriors.append(posterior)
            values[name], _ = balanced_kl(posterior, prior)
        accumulator = GroupLossAccumulator(scales={'player': .1, 'enemy': 20.})
        loss = accumulator.combine(values, floors={'player': 1.1, 'enemy': 1.1})
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), 2.2, places=6)
        report = accumulator.report()
        self.assertAlmostEqual(report['kl_floor'], 2.2)
        for name in ('player', 'enemy'):
            self.assertEqual(report[name]['raw'], 0.)
            self.assertFalse(report[name]['active'])
            self.assertEqual(report[name]['weighted'], 1.1)
        self.assertTrue(all(float(value.grad.abs().sum()) == 0. for value in priors+posteriors))
        # The variable loss keeps its true inverse-scale gradient above the floor.
        variable = torch.tensor(3., requires_grad=True)
        loss = accumulator.combine({'player': variable+1.1}, floors={'player': 1.1})
        loss.backward()
        self.assertAlmostEqual(float(variable.grad), 10., places=6)
        self.assertAlmostEqual(float(loss.detach()), 31.1, places=5)

    def test_450_step_chunks_include_two_step_tail_and_conserve_gradients(self):
        coefficient = nn.Parameter(torch.tensor([.7, -.4, .2], dtype=torch.float64))
        source = torch.linspace(-1., 1., 450, dtype=torch.float64)
        scales = {'player': 4., 'enemy': .2, 'interaction': 3.}
        accumulator = GroupLossAccumulator(scales)
        for first in range(0, 450, 16):
            last = min(450, first+16)
            parts = {name: (coefficient[index]-source[first:last]).square().sum()/450
                     for index, name in enumerate(GROUPS)}
            accumulator.combine(parts).backward()
        chunk_gradient = coefficient.grad.clone()
        coefficient.grad = None
        direct = sum((coefficient[index]-source).square().mean()/scales[name]
                     for index, name in enumerate(GROUPS))
        direct.backward()
        self.assertEqual(accumulator.calls, 29)
        self.assertEqual(last-first, 2)
        torch.testing.assert_close(chunk_gradient, coefficient.grad, rtol=1e-13, atol=1e-13)
        self.assertAlmostEqual(accumulator.report()['total'], float(direct.detach()), places=12)

    def test_four_microbatches_use_frozen_scales_and_one_ema_commit(self):
        balancer = GroupLossBalancer()
        calibration = balancer.new_microbatch('combat')
        calibration.combine({name: torch.tensor(value) for name, value in zip(GROUPS, (4., .2, 3.))})
        balancer.commit(balancer.summarize([calibration.report()], environment='combat', step=101), step=101)
        before = balancer.state_dict()
        parameter = nn.Parameter(torch.tensor([.7, -.4, .2], dtype=torch.float64))
        targets = torch.tensor([[1., -.5, .3], [-1., .1, .6], [.3, 1., -.7], [.8, -.2, .1]], dtype=torch.float64)
        reports = []
        for row in targets:
            accumulator = balancer.new_microbatch('combat')
            accumulator.combine({name: (parameter[index]-row[index]).square()
                                 for index, name in enumerate(GROUPS)}).backward()
            reports.append(accumulator.report())
            self.assertEqual(balancer.state_dict(), before)
        average_accumulated_gradients([parameter], 4)
        obtained = parameter.grad.clone()
        parameter.grad = None
        direct = sum((parameter[index]-targets[:, index]).square().mean()/before['banks']['combat']['ema'][name]
                     for index, name in enumerate(GROUPS))
        direct.backward()
        torch.testing.assert_close(obtained, parameter.grad, rtol=1e-14, atol=1e-14)
        summary = balancer.summarize(reports, environment='combat', step=102)
        self.assertEqual(summary['microbatches'], 4)
        self.assertAlmostEqual(summary['total'], float(direct.detach()), places=12)
        balancer.commit(summary, step=102)
        self.assertEqual(balancer.state_dict()['committed_updates'], 2)
        self.assertEqual(balancer.state_dict()['banks']['combat']['updates'], 2)
        for name in GROUPS:
            expected = .95*before['banks']['combat']['ema'][name]+.05*summary[name]['raw']
            self.assertAlmostEqual(balancer.state_dict()['banks']['combat']['ema'][name], expected)
        unchanged = balancer.state_dict()
        with self.assertRaises(ValueError): balancer.commit(summary, step=102)
        self.assertEqual(balancer.state_dict(), unchanged)

    def test_environment_banks_empty_tasks_and_minimum_scale(self):
        balancer = GroupLossBalancer()
        first = balancer.new_microbatch('navigation')
        first.combine({'player': torch.tensor(.001)})
        balancer.commit(balancer.summarize([first.report()], environment='navigation', step=12), step=12)
        after = balancer.state_dict()
        self.assertEqual(after['banks']['combat']['updates'], 0)
        self.assertTrue(all(value is None for value in after['banks']['combat']['ema'].values()))
        self.assertIsNone(after['banks']['navigation']['ema']['enemy'])
        self.assertEqual(balancer.new_microbatch('navigation').scales['player'], SETTINGS['minimum_scale'])
        self.assertEqual(balancer.new_microbatch('combat').scales['player'], 1.)
        empty = balancer.new_microbatch('navigation')
        summary = balancer.summarize([empty.report()]*4, environment='navigation', step=13)
        self.assertEqual(summary['total'], 0.)
        balancer.commit(summary, step=13)
        self.assertEqual(balancer.state_dict()['banks']['navigation']['ema'], after['banks']['navigation']['ema'])
        with self.assertRaises(ValueError): balancer.summarize([], environment='combat', step=14)

    def test_resume_reproduces_next_adam_step_ema_and_preserves_private_state(self):
        parameter = nn.Parameter(torch.tensor([.2, -.4, .9], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=.01)
        balancer = GroupLossBalancer()

        def update(value, adam, balance, step):
            adam.zero_grad(set_to_none=True)
            accumulator = balance.new_microbatch('combat')
            accumulator.combine({name: (value[index]-.1*step).square()
                                 for index, name in enumerate(GROUPS)}).backward()
            report = balance.summarize([accumulator.report()], environment='combat', step=step)
            adam.step(); balance.commit(report, step=step)

        update(parameter, optimizer, balancer, 61)
        saved_parameter = parameter.detach().clone()
        saved_optimizer, saved_balancer = deepcopy(optimizer.state_dict()), balancer.state_dict()
        restored = nn.Parameter(saved_parameter.clone())
        restored_optimizer = torch.optim.AdamW([restored], lr=.5)
        restored_optimizer.load_state_dict(deepcopy(saved_optimizer))
        restored_balancer = GroupLossBalancer(saved_balancer)
        update(parameter, optimizer, balancer, 62)
        update(restored, restored_optimizer, restored_balancer, 62)
        self.assertTrue(torch.equal(parameter, restored))
        for key, tensor in optimizer.state_dict()['state'][0].items():
            self.assertTrue(torch.equal(tensor, restored_optimizer.state_dict()['state'][0][key]), key)
        self.assertEqual(optimizer.state_dict()['param_groups'], restored_optimizer.state_dict()['param_groups'])
        self.assertEqual(balancer.state_dict(), restored_balancer.state_dict())
        self.assertEqual(saved_balancer['committed_updates'], 1)
        self.assertEqual(saved_optimizer['state'][0]['step'], 1)
        copy_out = restored_balancer.state_dict()
        copy_out['banks']['combat']['ema']['player'] = 999.
        self.assertNotEqual(copy_out, restored_balancer.state_dict())

    def test_nonfinite_objectives_fail_atomically_before_adam_or_ema_changes(self):
        parameter = nn.Parameter(torch.tensor(2.))
        optimizer = torch.optim.AdamW([parameter], lr=.01)
        (parameter.square()).backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        before_parameter = parameter.detach().clone()
        before_optimizer = deepcopy(optimizer.state_dict())
        balancer = GroupLossBalancer()
        accumulator = balancer.new_microbatch('combat')
        before_accumulator = accumulator.report()
        with self.assertRaises(FloatingPointError):
            accumulator.combine({'player': parameter.square(), 'enemy': parameter*float('nan')}).backward()
            optimizer.step()
            balancer.commit(balancer.summarize([accumulator.report()], environment='combat', step=1), step=1)
        self.assertEqual(accumulator.report(), before_accumulator)
        self.assertEqual(balancer.state_dict()['committed_updates'], 0)
        self.assertTrue(torch.equal(parameter, before_parameter))
        self.assertIsNone(parameter.grad)
        for key, value in before_optimizer['state'][0].items():
            self.assertTrue(torch.equal(value, optimizer.state_dict()['state'][0][key]))
        bad_report = accumulator.report(); bad_report['interaction']['raw'] = float('inf')
        with self.assertRaises(FloatingPointError):
            balancer.summarize([bad_report], environment='combat', step=1)
        self.assertEqual(balancer.state_dict()['committed_updates'], 0)

    def test_changed_scales_inside_update_and_bad_checkpoint_settings_are_rejected(self):
        balancer = GroupLossBalancer()
        first = balancer.new_microbatch('combat')
        first.combine({'player': torch.tensor(1.)})
        changed = GroupLossAccumulator(scales={'player': 2.})
        changed.combine({'player': torch.tensor(1.)})
        with self.assertRaisesRegex(ValueError, 'weights changed'):
            balancer.summarize([first.report(), changed.report()], environment='combat', step=1)
        invalid = balancer.state_dict(); invalid['settings']['ema_decay'] = .5
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            GroupLossBalancer(invalid)
        with self.assertRaises(ValueError): GroupLossAccumulator(scales={'enemy': float('inf')})

    def test_actual_dual_rssm_first_calibration_preserves_legacy_gradients_and_causality(self):
        from test_dual_rssm_backend import model
        from test_dual_rssm_objectives import source_batch
        from turkeyworld.rssm_migration import configure_rssm_training
        from turkeyworld.rssm_objectives import rssm_dynamics_backward
        batch = source_batch()
        legacy = model(True)
        grouped = deepcopy(legacy)
        outputs, queries = [], []
        for world, accumulator in ((legacy, None), (grouped, GroupLossAccumulator())):
            configure_rssm_training(world)
            world.core.eval(); world.vjepa_backend.enemy_rssm.eval()
            captured = []
            transition = world.transition
            def capture(state, action, features, memory):
                captured.append((state.detach().clone(), memory.detach().clone()))
                return transition(state, action, features, memory)
            with patch.object(world, 'transition', side_effect=capture), \
                    patch.object(world, 'render', side_effect=AssertionError('No renderer')):
                outputs.append(rssm_dynamics_backward(world, batch, context=1, observed_context=1,
                    auxiliary_batch=batch, auxiliary_context=1, auxiliary_observed_context=1,
                    expected_horizon=4, chunk_size=2, numeric_state_feedback='detached',
                    combat_protocol={'target_supervision': 'shot_distance_balanced'}, loss_groups=accumulator))
            queries.append(captured)
        for old_query, new_query in zip(*queries):
            for old, new in zip(old_query, new_query):
                self.assertTrue(torch.equal(old, new))
        self.assertAlmostEqual(outputs[0]['train_loss'], outputs[1]['train_loss'], places=5)
        self.assertAlmostEqual(outputs[1]['loss_groups']['kl_floor'], 2.2, places=6)
        self.assertFalse(outputs[1]['rssm_posterior_feedback_to_prior'])
        self.assertEqual(outputs[1]['rssm_prior_future_rgb_inputs'], 0)
        for (name, old), (new_name, new) in zip(legacy.named_parameters(), grouped.named_parameters()):
            self.assertEqual(name, new_name)
            if old.grad is None:
                self.assertIsNone(new.grad, name)
            else:
                self.assertIsNotNone(new.grad, name)
                torch.testing.assert_close(old.grad, new.grad, rtol=2e-5, atol=2e-6, msg=name)


if __name__ == '__main__':
    unittest.main()
