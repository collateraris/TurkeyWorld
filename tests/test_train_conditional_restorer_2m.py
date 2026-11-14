"""Expanded-restorer driver: real loop/selection and named Adam adoption."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_conditional_restorer_goal import validation, PINS
from turkeyworld.conditional_decoder_restorer import ConditionalDecoderRestorer
from turkeyworld.conditional_restorer_expansion import expand_conditional_restorer
from turkeyworld.conditional_restorer_goal import update_restorer_best
from turkeyworld.renderer_native_phase import _same
from turkeyworld.train_conditional_restorer import ConditionalRestorerTraining
from turkeyworld.train_conditional_restorer_2m import (
    Restorer2MTraining, adopt_restorer_optimizer, validate_saved_goal_state)


class LoopHarness(Restorer2MTraining):
    """Keep production run/validate control flow; replace expensive numerics."""
    def __init__(self, root, *, start=1999, scores=None):
        super().__init__(SimpleNamespace(run_dir=Path(root), steps=None, probe=False, device='cpu'))
        self.start = start
        self.scores = scores or {2000: .85, 2100: .850001}
        self.updates, self.previews, self.exports, self.saves = [], [], [], []

    def prepare(self):
        self.step = self.start
        self.parent_updates = 1389
        self.phase = dict(panel_fingerprint=PINS['panel_fingerprint'],
                          inputs=dict(world=PINS['feature_world_sha256']))
        self.plan_sha = PINS['plan_sha256']
        self.training = dict(local_step=self.step)
        initial = validation(.74, 0)
        self.best = update_restorer_best(None, initial, **PINS)
        self.best['improved'] = False
        self.goal_met = False
        self.goal_history = [dict(step=0, ssim=.74, goal_met=False)]
        (self.root/'validation_initial.json').write_text(json.dumps(initial), encoding='utf-8')

    def update(self):
        self.step += 1
        self.updates.append(self.step)
        return dict(step=self.step, train_loss=.1)

    def payload(self):
        return dict(step=self.step, best=self.best, goal_met=self.goal_met,
                    goal_history=self.goal_history)

    def save(self):
        self.saves.append(self.step)

    def export(self, report=None, path=None):
        self.exports.append((self.step, str(path) if path is not None else None))

    def preview(self):
        self.previews.append(self.step)


def parent_validation(trainer, baseline):
    return validation(trainer.scores[trainer.step], trainer.step)


def trained_parent():
    old = ConditionalDecoderRestorer()
    optimizer = torch.optim.AdamW(old.parameters(), lr=.0007, betas=(.8, .95),
                                  eps=1e-6, weight_decay=.03)
    codes = torch.randn(1, 16, 8, 8)
    rgb = torch.rand(1, 3, 32, 32) * .6 + .2
    state = torch.rand(1, 20)
    grid = torch.randint(0, 16, (1, 16, 16))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        old(codes, rgb, state, grid).square().mean().backward()
        optimizer.step()
    parent = dict(step=2, optimizer=deepcopy(optimizer.state_dict()),
        optimizer_parameter_names=['flow_renderer.conditional_restorer.' + name
                                   for name, _ in old.named_parameters()])
    return old, parent, (codes, rgb, state, grid)


class Restorer2MDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_loop_crosses2000_equal85_does_not_stop_then2100_strict_success(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = LoopHarness(directory)
            with patch.object(ConditionalRestorerTraining, 'validate', parent_validation):
                trainer.run()
            self.assertEqual(trainer.updates, list(range(2000, 2101)))
            self.assertEqual([row['step'] for row in trainer.goal_history], [0, 2000, 2100])
            self.assertFalse(trainer.goal_history[1]['goal_met'])
            self.assertTrue(trainer.goal_history[2]['goal_met'])
            self.assertEqual(trainer.state['status'], 'completed')
            self.assertIsNone(trainer.state['total'])
            self.assertEqual(trainer.best['step'], 2100)
            reached = json.loads((Path(directory)/'goal_reached.json').read_text(encoding='utf-8'))
            self.assertEqual(reached['step'], 2100)
            self.assertTrue(reached['decision']['goal_met'])

    def test_explicit_user_stop_is_stopped_not_goal_achievement_even_beyond_old_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'stop_training').write_text('user stop', encoding='ascii')
            trainer = LoopHarness(directory, start=2050)
            trainer.run()
            self.assertEqual(trainer.state['status'], 'stopped')
            self.assertFalse(trainer.goal_met)
            self.assertEqual(trainer.updates, [])
            self.assertFalse((Path(directory)/'goal_reached.json').exists())

    def test_actual_validate_best_file_only_on_strict_SSIM_improvement_not_changed_false_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = LoopHarness(directory, start=0, scores={0: .75, 100: .74, 200: .75, 300: .751})
            trainer.prepare()
            with patch.object(ConditionalRestorerTraining, 'validate', parent_validation):
                trainer.validate(None)
                best_path = Path(directory)/'checkpoints/restorer_best.pt'
                first = hashlib.sha256(best_path.read_bytes()).hexdigest()
                first_json = (Path(directory)/'best.json').read_bytes()
                for step in (100, 200):
                    trainer.step = step
                    trainer.validate(None)
                    self.assertEqual(hashlib.sha256(best_path.read_bytes()).hexdigest(), first)
                    self.assertEqual((Path(directory)/'best.json').read_bytes(), first_json)
                self.assertEqual(len(trainer.exports), 1)
                trainer.step = 300
                trainer.validate(None)
                self.assertEqual(len(trainer.exports), 2)
                self.assertEqual(trainer.best['step'], 300)
                self.assertNotEqual(hashlib.sha256(best_path.read_bytes()).hexdigest(), first)

    def test_actual_expansion_adopts65_old_Adam_options_moments_then65_fresh_states(self):
        legacy, parent, inputs = trained_parent()
        expanded, _ = expand_conditional_restorer(legacy)
        before = deepcopy(parent)
        rng = torch.get_rng_state().clone()
        optimizer = adopt_restorer_optimizer(expanded, parent, lr=.0001)
        state = optimizer.state_dict()
        self.assertEqual(len(state['state']), 65)
        self.assertTrue(_same(state['state'], before['optimizer']['state']))
        self.assertTrue(_same(state['param_groups'][0], before['optimizer']['param_groups'][0]))
        self.assertEqual(len(state['param_groups']), 2)
        self.assertEqual(len(state['param_groups'][1]['params']), 65)
        self.assertTrue(set(state['param_groups'][1]['params']).isdisjoint(state['state']))
        self.assertEqual(state['param_groups'][1]['lr'], .0001)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        optimizer.zero_grad(set_to_none=True)
        expanded(*inputs).square().mean().backward()
        optimizer.step()
        after = optimizer.state_dict()
        self.assertEqual(len(after['state']), 130)
        self.assertEqual({float(after['state'][i]['step']) for i in range(65)}, {3.})
        self.assertEqual({float(after['state'][i]['step']) for i in range(65, 130)}, {1.})
        self.assertTrue(all(torch.isfinite(v).all() for row in after['state'].values()
                            for v in row.values() if torch.is_tensor(v)))
        self.assertTrue(_same(parent, before), 'The loaded parent snapshot must remain immutable in memory')

    def test_bad_named_old_Adam_order_age_shape_or_nonfinite_state_rejected(self):
        legacy, parent, _ = trained_parent()
        expanded, _ = expand_conditional_restorer(legacy)
        changes = [lambda p: p['optimizer_parameter_names'].reverse(),
                   lambda p: p['optimizer']['param_groups'][0]['params'].reverse(),
                   lambda p: p['optimizer']['param_groups'][0]['params'].pop(),
                   lambda p: p['optimizer']['state'][0]['step'].fill_(1),
                   lambda p: p['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
                   lambda p: p['optimizer']['state'][0].__setitem__('exp_avg', torch.zeros(1))]
        for index, change in enumerate(changes):
            bad = deepcopy(parent); change(bad)
            with self.subTest(index=index), self.assertRaises((ValueError, AssertionError)):
                adopt_restorer_optimizer(expanded, bad, lr=.0001)

    def test_resume_goal_is_rebuilt_from_pinned_full_reports_not_a_truthy_boolean(self):
        pins = {key: PINS[key] for key in ('panel_fingerprint', 'plan_sha256', 'feature_world_sha256')}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'validation').mkdir()
            best, history = None, []
            for step, score in [(0, .74), (100, .75), (2000, .85)]:
                report = validation(score, step)
                path = root/'validation'/f'step_{step:06d}.json'
                path.write_text(json.dumps(report), encoding='utf-8')
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                best = update_restorer_best(best, report, report_sha256=sha, **PINS)
                history.append(dict(step=step, ssim=score, goal_met=score>.85, report_sha256=sha))
            saved = dict(step=2050, best=best, goal_history=history, goal_met=False)
            restored = validate_saved_goal_state(saved, run_dir=root, **pins)
            self.assertEqual(restored, (best, history, False))
            self.assertFalse(restored[2])
            for change in ('truthy_string', 'unmeasured_true', 'history_wrong_score',
                           'history_partial_step', 'history_wrong_SHA', 'best_wrong_max', 'future_measurement'):
                bad = deepcopy(saved)
                if change == 'truthy_string': bad['goal_met'] = 'false'
                if change == 'unmeasured_true': bad['goal_met'] = True
                if change == 'history_wrong_score': bad['goal_history'][-1]['ssim'] = .99
                if change == 'history_partial_step': bad['goal_history'][-1]['step'] = 2001
                if change == 'history_wrong_SHA': bad['goal_history'][-1]['report_sha256'] = 'e'*64
                if change == 'best_wrong_max': bad['best']['ssim'] = .99
                if change == 'future_measurement': bad['step'] = 1999
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate_saved_goal_state(bad, run_dir=root, **pins)
            # A full, exact-pinned >.85 measurement is sufficient by itself.
            report = validation(.850001, 2100)
            path = root/'validation/step_002100.json'; path.write_text(json.dumps(report), encoding='utf-8')
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            best = update_restorer_best(best, report, report_sha256=sha, **PINS)
            history.append(dict(step=2100, ssim=.850001, goal_met=True, report_sha256=sha))
            complete = dict(step=2100, best=best, goal_history=history, goal_met=True)
            self.assertTrue(validate_saved_goal_state(complete, run_dir=root, **pins)[2])
            # Replacing a stored full report cannot preserve its original pin.
            path.write_text(json.dumps(validation(.9, 2100)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'exact complete validation'):
                validate_saved_goal_state(complete, run_dir=root, **pins)

    def test_resume_cannot_continue_past_a_successful_goal_or_fabricate_unmeasured_success(self):
        pins = {key: PINS[key] for key in ('panel_fingerprint', 'plan_sha256', 'feature_world_sha256')}
        self.assertEqual(validate_saved_goal_state(dict(step=0, best=None, goal_history=[], goal_met=False),
                                                  **pins), (None, [], False))
        for bad in [dict(step=1, best=None, goal_history=[], goal_met=False),
                    dict(step=0, best=None, goal_history=[], goal_met=True)]:
            with self.assertRaises(ValueError): validate_saved_goal_state(bad, **pins)
        best = update_restorer_best(None, validation(.86, 0), **PINS)
        history = [dict(step=0, ssim=.86, goal_met=True), dict(step=100, ssim=.87, goal_met=True)]
        with self.assertRaises(ValueError):
            validate_saved_goal_state(dict(step=100, best=best, goal_history=history, goal_met=True), **pins)


if __name__ == '__main__':
    unittest.main()
