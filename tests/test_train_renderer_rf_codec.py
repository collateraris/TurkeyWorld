"""Real RF driver loop, best-file decisions and report-backed resume state."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from test_renderer_rf_codec_goal import PINS, validation
from turkeyworld.renderer_rf_codec_goal import update_rf_codec_best, validate_saved_rf_codec_goal
from turkeyworld.train_renderer_rf_codec import RFCodecTraining


class Loop(RFCodecTraining):
    def __init__(self, root, *, start=1999, scores=None):
        super().__init__(SimpleNamespace(run_dir=Path(root), device='cpu', probe=False))
        self.start = start
        self.scores = scores or {2000: .85, 2100: .850001}
        self.updates, self.saves, self.exports, self.previews = [], [], [], []

    def prepare(self):
        self.step = self.start
        self.phase = dict(panel_fingerprint=PINS['panel_fingerprint'],
                          inputs=dict(world=PINS['feature_world_sha256']))
        self.plan_sha = PINS['plan_sha256']
        self.training = dict(local_step=self.step)
        self.model, self.reference = object(), object()
        self.profiled = dict(val=object())
        self.panel, self.geometry_panel = list(range(256)), list(range(128))
        report = validation(.74, 0)
        payload = json.dumps(report).encode()
        (self.root/'validation').mkdir(exist_ok=True)
        (self.root/'validation/step_000000.json').write_bytes(payload)
        (self.root/'validation_initial.json').write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        self.best = update_rf_codec_best(None, report, report_sha256=sha, **PINS)
        self.best['improved'] = False
        self.goal_history = [dict(step=0, ssim=.74, goal_met=False, report_sha256=sha)]
        self.goal_met = False

    def update(self):
        self.step += 1; self.updates.append(self.step)
        return dict(step=self.step, train_loss=.1)

    def save(self):
        self.saves.append(self.step)

    def payload(self):
        return dict(step=self.step, best=self.best, goal_history=self.goal_history, goal_met=self.goal_met)

    def export(self, report=None, path=None):
        self.exports.append((self.step, str(path) if path is not None else None))

    def preview(self):
        self.previews.append(self.step)


def evaluator(trainer):
    def evaluate(model, reference, data, panel, geometry_panel, *, step, batch_size, baseline):
        return validation(trainer.scores[step], step), None
    return evaluate


class RFCodecDriverTests(unittest.TestCase):
    def test_actual_loop_continues_after2000_equal85_then_stops_at2100_strict_success(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Loop(directory)
            with patch('turkeyworld.train_renderer_rf_codec.evaluate_rf_codec', side_effect=evaluator(trainer)):
                trainer.run()
            self.assertEqual(trainer.updates, list(range(2000, 2101)))
            self.assertEqual([x['step'] for x in trainer.goal_history], [0, 2000, 2100])
            self.assertFalse(trainer.goal_history[1]['goal_met'])
            self.assertTrue(trainer.goal_history[2]['goal_met'])
            self.assertEqual(trainer.state['status'], 'completed')
            self.assertIsNone(trainer.state['total'])
            reached = json.loads((Path(directory)/'goal_reached.json').read_text(encoding='utf-8'))
            self.assertEqual(reached['step'], 2100)
            self.assertTrue(reached['decision']['goal_met'])

    def test_explicit_stop_beyond_old_budget_is_not_quality_or_goal_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'stop_training').write_text('explicit stop', encoding='ascii')
            trainer = Loop(directory, start=2500)
            trainer.run()
            self.assertEqual(trainer.updates, [])
            self.assertEqual(trainer.state['status'], 'stopped')
            self.assertFalse(trainer.goal_met)
            self.assertFalse((Path(directory)/'goal_reached.json').exists())

    def test_actual_best_checkpoint_is_untouched_on_lower_or_tied_SSIM(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Loop(directory, start=0, scores={0: .75, 100: .74, 200: .75, 300: .751})
            trainer.prepare(); trainer.best=None; trainer.goal_history=[]
            with patch('turkeyworld.train_renderer_rf_codec.evaluate_rf_codec', side_effect=evaluator(trainer)):
                trainer.validate(None)
                best = Path(directory)/'checkpoints/renderer_best.pt'
                digest = hashlib.sha256(best.read_bytes()).hexdigest()
                json_before = (Path(directory)/'best.json').read_bytes()
                for step in (100, 200):
                    trainer.step=step; trainer.validate(None)
                    self.assertEqual(hashlib.sha256(best.read_bytes()).hexdigest(), digest)
                    self.assertEqual((Path(directory)/'best.json').read_bytes(), json_before)
                self.assertEqual(len([x for x in trainer.exports if x[1] is not None]), 1)
                trainer.step=300; trainer.validate(None)
                self.assertEqual(trainer.best['step'], 300)
                self.assertEqual(len([x for x in trainer.exports if x[1] is not None]), 2)
                self.assertNotEqual(hashlib.sha256(best.read_bytes()).hexdigest(), digest)

    def test_actual_driver_reports_reconstruct_resume_goal_and_best_SHA_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Loop(directory, start=0, scores={0: .74, 100: .75, 2000: .85, 2100: .850001})
            trainer.prepare(); trainer.best=None; trainer.goal_history=[]
            with patch('turkeyworld.train_renderer_rf_codec.evaluate_rf_codec', side_effect=evaluator(trainer)):
                for step in (0, 100, 2000):
                    trainer.step=step; trainer.validate(None)
                saved=trainer.payload()
                self.assertFalse(validate_saved_rf_codec_goal(saved, run_dir=directory, **PINS)[2])
                self.assertEqual(saved['best']['step'], 2000)
                for key, value in [('goal_met', 'false'), ('goal_met', True)]:
                    bad=deepcopy(saved); bad[key]=value
                    with self.assertRaises(ValueError):
                        validate_saved_rf_codec_goal(bad, run_dir=directory, **PINS)
                bad=deepcopy(saved); bad['best']['ssim']=.99
                with self.assertRaises(ValueError):
                    validate_saved_rf_codec_goal(bad, run_dir=directory, **PINS)
                trainer.step=2100; trainer.validate(None)
                complete=trainer.payload()
                self.assertTrue(validate_saved_rf_codec_goal(complete, run_dir=directory, **PINS)[2])
                report_path=Path(directory)/'validation/step_002100.json'
                report=json.loads(report_path.read_text(encoding='utf-8'))
                report['models']['rf_new']['summary']['ssim_mean']=.99
                report_path.write_text(json.dumps(report), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'SHA changed'):
                    validate_saved_rf_codec_goal(complete, run_dir=directory, **PINS)


if __name__=='__main__':
    unittest.main()
