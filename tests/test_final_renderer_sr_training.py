"""Actual SR/Adam updates on a tiny parent; cache and long validation are isolated."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_atlas_enemy_sr import small
from test_sr_moe_pipeline import Pairs
from turkeyworld import train_final_renderer_sr as worker
from turkeyworld.renderer_native_phase import _same
from turkeyworld.sr_moe_objectives import sr_moe_loss
from turkeyworld.sr_transformer_moe import enable_sr_transformer_moe, configure_sr_transformer_moe, sr_moe_metadata


CACHE = dict(world_sha256='1'*64, world_step=5906, renderer_sha256='2'*64,
    renderer_step=33701, renderer_format_version=8, restorer_present=False,
    solver_steps=3, base_training_steps=32000, decoder_refinement_updates=4725,
    future_rgb_features=False, paired_same_pose_map_tick=True)
RECEIPT = dict(cache_identity=CACHE, paired_manifest_sha256='3'*64,
               hypotheses_manifest_sha256='4'*64, source_manifest_sha256='5'*64)


class Data(Pairs):
    instances = []

    def __init__(self, root=None, split='val', *, hypotheses_root=None, seed=173, device='cpu'):
        super().__init__()
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.identity = dict(fingerprint='6'*64, base=dict(cache_identity=deepcopy(CACHE)))
        self.state.requires_grad_()
        self.instances.append(self)

    def fixed_panel(self, count, *, seed):
        return list(range(count))

    def batch(self, count, indices=None, rng=None):
        if indices is None:
            draw_rng = self.rng if rng is None else rng
            # Four actual visible and four true-empty examples, with all distances.
            indices = [int(draw_rng.choice([0, 1, 2, 3, 6, 7])) for _ in range(4)] + [4, 5, 4, 5]
            draw_rng.shuffle(indices)
        result = super().batch(count, indices=indices)
        result['row_index'] = torch.as_tensor(indices, dtype=torch.long)
        return result


def evaluation(model, reference, data, panel, *, step, batch_size, baseline_report=None):
    report = dict(kind='sr_moe_five_validation_v1', step=step, source_held_out=True,
        panel_fingerprint=data.panel_fingerprint(panel), data_identity_sha256='7'*64,
        models=dict(current=dict(summary=dict(frames=len(panel), ssim_mean=.5+step*.001)),
                    frozen_atlas_parent=dict(summary=dict(frames=len(panel), ssim_mean=.5))),
        validation_objective=dict(protocol=worker.PROTOCOL, loss=.5, frames=len(panel), input_kind='npr_rgb'),
        atlas_validation_identity=dict(step=0))
    preview = dict(metadata=dict(kind='sr_moe_five_preview_v1', row_count=4, step=step), cases={})
    return report, preview


class FinalRendererSRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.source = cls.root / 'parent.pt'
        torch.random.default_generator.manual_seed(19)
        model = small().requires_grad_(True)
        old_names = list(dict(model.named_parameters()))
        cls.old_count = len(old_names)
        old_params = list(model.parameters())
        optimizer = torch.optim.AdamW([dict(params=old_params[:80], lr=7e-5),
                                       dict(params=old_params[80:], lr=5e-5)], lr=1e-4)
        sum(p.square().mean() + p.mean()*.001 for p in old_params).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        migration = enable_sr_transformer_moe(model)
        configure_sr_transformer_moe(model)
        added = [p for name, p in model.named_parameters() if name.startswith('sr_moe.')]
        optimizer.add_param_group(dict(params=added, lr=1e-4))
        data = Data()
        for _ in range(2):
            batch = data.batch(8, indices=list(range(8)))
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_with_aux(batch['lr_npr'].detach(), batch['hypothesis'].detach())
            loss, _ = sr_moe_loss(output, batch, label_renderer=model.render_label_atlas)
            loss.backward()
            optimizer.step()
        new_names = [name for name, _ in model.named_parameters() if name.startswith('sr_moe.')]
        cls.saved = dict(stage='sr', config=deepcopy(model.config), model=deepcopy(model.state_dict()),
            optimizer=deepcopy(optimizer.state_dict()), optimizer_parameter_names=[old_names[:80], old_names[80:], new_names],
            sr_moe_migration=migration, sr_moe_binding=sr_moe_metadata(model),
            training_steps=108105, parent_training_steps=108103, sr_moe_updates=2,
            sampler=dict(protocol='old_cache_sampler', draws=16000),
            rng_numpy=deepcopy(np.random.default_rng(81).bit_generator.state),
            rng_torch=torch.get_rng_state(), rng_python=random.getstate(), rng_cuda=[])
        torch.save(cls.saved, cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def args(self, name, *, steps=2, probe=False):
        return Namespace(run_dir=self.root/name, parent_checkpoint=self.source,
            data_root=self.root/'paired', hypotheses_root=self.root/'hypotheses', seed=173,
            device='cpu', steps=steps, validation_frames=8, validation_batch=8,
            validate_every=100, prediction_every=15, probe=probe)

    def invoke(self, args, *, stop_after_one=False):
        original = worker.clip_finite_grad_norm_
        def clip(*a, **kw):
            value = original(*a, **kw)
            if stop_after_one:
                (args.run_dir/'stop_training').write_text('test boundary', encoding='ascii')
            return value
        with patch.object(worker, 'EnemyDecoderData', Data), patch.object(worker, 'evaluate_sr_moe', evaluation), \
                patch.object(worker, 'clip_finite_grad_norm_', clip):
            return worker.run_training(args, deepcopy(RECEIPT))

    def test_actual_two_SGD_full_resume_preserves_parent_Adam_and_counters(self):
        before = self.source.read_bytes()
        continuous = self.args('continuous')
        self.invoke(continuous)
        resumed = self.args('resumed')
        self.invoke(resumed, stop_after_one=True)
        self.assertEqual(json.loads((resumed.run_dir/'status.json').read_text())['status'], 'stopped')
        (resumed.run_dir/'stop_training').unlink()
        self.invoke(resumed)
        a = torch.load(continuous.run_dir/'checkpoints/sr_last.pt', weights_only=False)
        b = torch.load(resumed.run_dir/'checkpoints/sr_last.pt', weights_only=False)
        for key in ('model', 'optimizer', 'sampler', 'rng_numpy', 'rng_torch', 'rng_python', 'rng_cuda'):
            self.assertTrue(_same(a[key], b[key]), key)
        self.assertEqual(a['step'], 2)
        self.assertEqual(a['sequential_sr_updates'], 2)
        self.assertEqual(a['training_steps'], 108107)
        self.assertEqual(a['sr_moe_updates'], 4)
        self.assertEqual(a['parent_training_steps'], 108103)
        self.assertEqual(a['sampler']['draws'], 16)
        self.assertEqual(a['continuation_source_sampler'], self.saved['sampler'])
        self.assertEqual(len(a['optimizer']['state']), len(self.saved['optimizer']['state']))
        self.assertEqual(a['optimizer_parameter_names'], self.saved['optimizer_parameter_names'])
        for name, value in self.saved['model'].items():
            if not name.startswith('sr_moe.'):
                self.assertTrue(_same(value, a['model'][name]), name)
        for index in range(self.old_count):
            self.assertTrue(_same(self.saved['optimizer']['state'][index], a['optimizer']['state'][index]), str(index))
        self.assertTrue(any(not _same(value, a['model'][name]) for name, value in self.saved['model'].items()
                            if name.startswith('sr_moe.')))
        best = json.loads((continuous.run_dir/'best.json').read_text())
        selected = torch.load(continuous.run_dir/'checkpoints/sr_best.pt', weights_only=False)
        exported = torch.load(continuous.run_dir/'checkpoints/sr_for_game.pt', weights_only=False)
        self.assertEqual(selected['step'], best['step'])
        self.assertTrue(_same(selected['model'], exported['model']))
        self.assertNotIn('optimizer', exported)
        self.assertTrue((continuous.run_dir/'validation/step_000000.json').is_file())
        handoff = json.loads((continuous.run_dir/'handoff.json').read_text())
        self.assertEqual(handoff['completed_updates'], 2)
        self.assertFalse(handoff['automatic_promotion'])
        rows = [json.loads(row) for row in (continuous.run_dir/'metrics.jsonl').read_text().splitlines()]
        for row in rows:
            self.assertEqual(row['input_domain_counts'], dict(npr=6, clean=1, codec=1))
            self.assertEqual(row['hypotheses_deliberately_dropped'], 4)
            self.assertEqual(row['mismatched_predicted_hypotheses'], 2)
        self.assertEqual(before, self.source.read_bytes())
        for data in Data.instances:
            self.assertIsNone(data.hr.grad)
            self.assertIsNone(data.hr_mask.grad)
            self.assertIsNone(data.npr.grad)
            self.assertIsNone(data.state.grad)

    def test_actual_disposable_three_updates_emit_supervisor_gate_and_zero_local_baseline(self):
        args = self.args('probe', steps=3, probe=True)
        self.invoke(args)
        report = json.loads((args.run_dir/'gpu_warmstart.json').read_text())
        self.assertTrue(report['passed'])
        self.assertTrue(all(report['checks'].values()))
        self.assertEqual(report['discarded_optimizer_updates'], 3)
        self.assertEqual(report['production_optimizer_updates'], 0)
        self.assertTrue(all(v > 0 for v in report['last_metrics']['expert_gradient_norms'].values()))
        payload = torch.load(args.run_dir/'checkpoints/sr_last.pt', weights_only=False)
        self.assertEqual(payload['step'], 3)
        self.assertTrue(payload['disposable_probe'])
        baseline = json.loads((args.run_dir/'validation/step_000000.json').read_text())
        self.assertEqual(baseline['step'], 0)
        self.assertEqual(baseline['reference_training_steps'], 108105)
        self.assertTrue(baseline['source_held_out'])
        self.assertFalse(baseline['restorer_active'])

    def test_chain_header_rejects_restorer_nonfinal_format_and_invalid_source_clock(self):
        header = dict(format_version=8, kind='decoder_rf_chain_bundle_v1', restorer_present=False,
            solver='euler', solver_steps=3, training_steps=33000, base_training_steps=32000,
            decoder_refinement_updates=4725)
        kwargs = dict(world_sha256='1'*64, world_step=5906, renderer_sha256='2'*64)
        identity = worker._chain_identity(header, **kwargs)
        self.assertEqual(identity['renderer_step'], 33000)
        self.assertEqual(identity['decoder_refinement_updates'], 4725)
        for key, value in (('restorer_present', True), ('format_version', 3),
                           ('solver_steps', 5), ('training_steps', True)):
            with self.assertRaises(ValueError):
                worker._chain_identity({**header, key:value}, **kwargs)
        with self.assertRaises(ValueError):
            worker._chain_identity(header, **{**kwargs, 'world_step':5947})

    def test_real_validation_accepts_trained_five_expert_reference_without_advancing_sampler(self):
        model = worker._initial_model(self.saved, torch.device('cpu'))
        reference = worker._initial_model(self.saved, torch.device('cpu')).eval().requires_grad_(False)
        data = Data()
        sampler = deepcopy(data.sampler_state())
        rng = torch.get_rng_state().clone()
        from turkeyworld.sr_moe_validation import evaluate_sr_moe
        report, preview = evaluate_sr_moe(model, reference, data, list(range(8)), step=0, batch_size=4)
        self.assertEqual(report['validation_objective']['frames'], 8)
        self.assertEqual(report['validation_objective']['protocol'], worker.PROTOCOL)
        self.assertEqual(report['models']['current']['summary']['frames'], 8)
        self.assertEqual(report['models']['frozen_atlas_parent']['summary']['frames'], 8)
        self.assertEqual(preview['metadata']['step'], 0)
        self.assertTrue(_same(sampler, data.sampler_state()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
