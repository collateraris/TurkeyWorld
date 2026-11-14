"""Actual Base driver budget/stop/best behavior, with no production assets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.renderer_student import _hash_tensors
from turkeyworld.train_base_geometry import BaseGeometryTraining


PANEL, PLAN, WORLD = 'a'*64, 'b'*64, 'c'*64


def report(ssim, step):
    return dict(kind='base_geometry_validation_v1', protocol='base_expanded_geometry_v1',
        step=step, frames=256, source_held_out=True, renderer_only=True,
        world_rollout=False, restorer_active=False, quality_passed=False,native_resolution=32,
        RF_forward_calls=0,Source_Q_image_branch=False,Source_RGB_into_Base_conditioning=False,
        future_source_rgb_in_prior=False,future_source_pose_in_prior=False,
        data_identity_sha256='d'*64,codec_sha256='e'*64,frozen_flow_sha256='f'*64,
        models=dict(base_new=dict(summary=dict(frames=256, ssim_mean=ssim, mae_mean=.9))),
        panel_fingerprint=PANEL, val_loss=.1,
        validation_objective=dict(loss=.1, protocol='base_expanded_geometry_v1'))


class Loop(BaseGeometryTraining):
    def __init__(self, root, *, start=1999, steps=2000, scores=None, initial_exists=True):
        super().__init__(SimpleNamespace(run_dir=Path(root), device='cpu', probe=False, steps=steps))
        self.start, self.scores, self.initial_exists = start, scores or {2000: .75}, initial_exists
        self.updates, self.saves, self.exports, self.previews, self.events = [], [], [], [], []

    def prepare(self):
        self.events.append(('prepare', self.start))
        self.step = self.start
        self.phase = dict(panel_fingerprint=PANEL, inputs=dict(world=WORLD))
        self.plan_sha = PLAN; self.training = dict(local_step=self.step)
        self.model, self.reference = object(), object()
        self.profiled = dict(val=object())
        self.panel, self.geometry_panel = list(range(256)), list(range(128))
        if self.initial_exists:
            initial = report(.74, 0)
            (self.root/'validation_initial.json').write_text(json.dumps(initial), encoding='utf-8')

    def update(self):
        self.step += 1; self.updates.append(self.step); self.events.append(('update', self.step))
        return dict(step=self.step, train_loss=.1, renderer_training_steps=30717,
                    base_training_steps=29209+self.step)

    def save(self):
        self.saves.append(self.step); self.events.append(('save', self.step))

    def payload(self):
        return dict(step=self.step, best=self.best, renderer_training_steps=30717,
                    base_training_steps=29209+self.step,validation_history=deepcopy(self.validation_history))

    def export(self, value=None, path=None):
        self.exports.append((self.step, str(path) if path is not None else None))

    def preview(self):
        self.previews.append(self.step)


def evaluator(trainer):
    def evaluate(model, reference, data, panel, *geometry, step, batch_size, **kwargs):
        trainer.events.append(('validate', step))
        return report(trainer.scores[step], step), None
    return evaluate


class BaseGeometryDriverTests(unittest.TestCase):
    def test_actual_loop_stops_at2000_as_budget_without_claiming_quality_or_RF_training(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = Loop(root)
            with patch('turkeyworld.train_base_geometry.evaluate_base_geometry', side_effect=evaluator(trainer)):
                trainer.run()
            self.assertEqual(trainer.updates, [2000])
            self.assertEqual(trainer.state['status'], 'completed')
            self.assertEqual(trainer.state['total'], 2000)
            self.assertFalse(trainer.state['gameplay_qualified'])
            self.assertFalse(trainer.state['validation']['quality_passed'])
            self.assertEqual(trainer.state['base_training_steps'], 31209)
            self.assertEqual(trainer.state['latest_metrics']['renderer_training_steps'], 30717)
            self.assertEqual(trainer.best['ssim'], .75)
            self.assertFalse((Path(root)/'goal_reached.json').exists())
            self.assertIn(2000, trainer.saves)

    def test_explicit_user_stop_saves_boundary_and_never_claims_budget_or_quality_complete(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'stop_training').write_text('explicit stop', encoding='ascii')
            trainer = Loop(root, start=123)
            trainer.run()
            self.assertEqual(trainer.updates, [])
            self.assertEqual(trainer.state['status'], 'stopped')
            self.assertFalse(trainer.state['gameplay_qualified'])
            self.assertIn(123, trainer.saves)
            self.assertEqual(trainer.exports, [])

    def test_initial_validation_precedes_every_update_and_completed_resume_does_no_extra_SGD(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = Loop(root, start=0, steps=2, scores={0: .74}, initial_exists=False)
            with patch('turkeyworld.train_base_geometry.evaluate_base_geometry', side_effect=evaluator(trainer)):
                trainer.run()
            self.assertLess(trainer.events.index(('validate', 0)), trainer.events.index(('update', 1)))
            self.assertEqual(trainer.updates, [1, 2])
            self.assertTrue((Path(root)/'validation_initial.json').exists())
            self.assertEqual(trainer.best['step'], 0)
        with tempfile.TemporaryDirectory() as root:
            trainer = Loop(root, start=2000)
            trainer.run()
            self.assertEqual(trainer.updates, [])
            self.assertEqual(trainer.state['status'], 'completed')

    def test_best_file_uses_raw_Base_SSIM_strict_improvement_and_earliest_tie(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = Loop(root, start=0, scores={0: .75, 100: .74, 200: .75, 300: .751})
            trainer.prepare()
            with patch('turkeyworld.train_base_geometry.evaluate_base_geometry', side_effect=evaluator(trainer)):
                trainer.validate(None)
                best = Path(root)/'checkpoints/base_best.pt'
                sha = hashlib.sha256(best.read_bytes()).hexdigest()
                best_json = (Path(root)/'best.json').read_bytes()
                for step in (100, 200):
                    trainer.step=step; trainer.validate(None)
                    self.assertEqual(hashlib.sha256(best.read_bytes()).hexdigest(), sha)
                    self.assertEqual((Path(root)/'best.json').read_bytes(), best_json)
                self.assertEqual(trainer.best['step'], 0)
                trainer.step=300; trainer.validate(None)
                self.assertEqual(trainer.best['step'], 300)
                self.assertEqual(trainer.best['metric'], 'models.base_new.summary.ssim_mean')
                path = Path(root)/'validation/step_000300.json'
                self.assertEqual(trainer.best['report_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(len([entry for entry in trainer.exports if entry[1] is not None]), 2)

    def test_nonfinite_heldout_Base_score_cannot_overwrite_best(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = Loop(root, start=0, scores={0: .75, 100: float('nan')})
            trainer.prepare()
            with patch('turkeyworld.train_base_geometry.evaluate_base_geometry', side_effect=evaluator(trainer)):
                trainer.validate(None)
                before = (Path(root)/'best.json').read_bytes()
                trainer.step=100
                with self.assertRaises(FloatingPointError):trainer.validate(None)
                self.assertEqual((Path(root)/'best.json').read_bytes(), before)

    def test_actual_update_sums_three_independent_streams_only_moves_Base_and_advances_its_clock(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as root:
            trainer = BaseGeometryTraining(SimpleNamespace(run_dir=Path(root), device='cpu', probe=False, steps=2000))
            trainer.step=0; trainer.plan_sha=PLAN; trainer.phase=dict(inputs=dict(world=WORLD))
            trainer.model=nn.Module(); trainer.model.base_student=nn.Module()
            trainer.model.base_student.output_moe=nn.Module()
            experts=[]
            for _ in range(5):
                expert=nn.Module(); expert.private_transformer=nn.Linear(1, 1, bias=False)
                expert.private_transformer.weight.data.fill_(.1); experts.append(expert)
            trainer.model.base_student.output_moe.experts=nn.ModuleList(experts)
            trainer.model.flow_renderer=nn.Linear(1, 1).eval().requires_grad_(False)
            trainer.reference=deepcopy(trainer.model).eval().requires_grad_(False)
            trainer.reference_hash=_hash_tensors(trainer.reference.state_dict())
            trainer.frozen=_hash_tensors(trainer.model.flow_renderer.state_dict())
            trainer.optimizer=torch.optim.AdamW(trainer.model.base_student.parameters(), lr=1e-4)
            seen=[]
            class Frames:
                def __init__(self, kind):self.kind=kind
                def batch(self, size):seen.append((self.kind, size)); return dict(kind=self.kind, size=size)
            trainer.profiled={kind:Frames(kind) for kind in ('main','far','geometry')}
            factors=dict(main=1.,far=2.,geometry=3.)
            def loss(model, batch, *, kind):
                value=sum(p.square().sum() for p in model.base_student.parameters())*factors[kind]
                return value, dict(train_loss=float(value.detach()))
            old_flow=_hash_tensors(trainer.model.flow_renderer.state_dict())
            with patch('turkeyworld.train_base_geometry.configure_base_geometry_training'), \
                 patch('turkeyworld.train_base_geometry.base_geometry_refinement_loss', side_effect=loss), \
                 patch('turkeyworld.train_base_geometry.base_geometry_frozen_fingerprint', side_effect=lambda m:_hash_tensors(m.flow_renderer.state_dict())), \
                 patch.object(trainer.model.flow_renderer,'forward', side_effect=AssertionError('No RF forward')):
                row=trainer.update()
            self.assertEqual(seen,[('main',16),('far',4),('geometry',8)])
            self.assertAlmostEqual(row['train_loss'], row['main']['train_loss']+.25*row['far']['train_loss']+row['geometry']['train_loss'])
            self.assertEqual(trainer.step,1); self.assertEqual(row['renderer_training_steps'],30717)
            self.assertEqual(row['base_training_steps'],29210)
            self.assertTrue(all(value>0 for value in row['private_transformer_gradient_abs'].values()))
            self.assertEqual(old_flow,_hash_tensors(trainer.model.flow_renderer.state_dict()))


if __name__=='__main__':unittest.main()
