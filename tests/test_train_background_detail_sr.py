"""CPU-only checks for fresh-architecture clocks, Adam and exact continuation."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from turkeyworld import train_background_detail_sr as runner


class TinyDetail(nn.Module):
    def __init__(self):
        super().__init__()
        self.background = nn.Conv2d(3, 3, 1)
        self.detail_residual = nn.Conv2d(3, 3, 1)
        nn.init.zeros_(self.background.weight)
        nn.init.constant_(self.background.bias, .2)
        nn.init.zeros_(self.detail_residual.weight)
        nn.init.zeros_(self.detail_residual.bias)
        self.config = dict(architecture=runner.ARCHITECTURE, detail_channels=48, detail_blocks=5, residual_bound=.25)
        self.configure_training()

    def configure_training(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(runner.TRAINABLE_PREFIX))
        return self

    def forward_with_aux(self, low):
        high = F.interpolate(low, size=(128, 128), mode='bilinear', align_corners=False)
        background = self.background(high).detach()
        residual = self.detail_residual(high)
        return dict(image=background+residual, background=background, residual=residual)


class FakeData:
    def __init__(self, root, split, *, seed=173, device='cpu'):
        self.split, self.device, self.draws = split, torch.device(device), 0
        self.scene_keys = {split}
        self.base = SimpleNamespace(identity={'split': split}, manifest={'cache_identity': dict(
            future_rgb_features=False, paired_same_pose_map_tick=True, restorer_present=False, solver_steps=3)})
        self.identity = {'base': self.base.identity, 'fingerprint': split+'-fixed'}

    def __len__(self): return 8192 if self.split == 'train' else 1024
    def fixed_panel(self, count, seed): return list(range(count))
    def panel_fingerprint(self, panel): return 'panel-'+str(len(panel))
    def sampler_state(self): return {'draws': self.draws, 'identity': self.identity}
    def restore_sampler(self, state):
        if state['identity'] != self.identity: raise ValueError('fake corpus changed')
        self.draws = state['draws']

    def batch(self, size):
        rows = torch.arange(self.draws, self.draws+size)
        self.draws += size
        low = .3 + .01*(rows % 4).float()[:, None, None, None].expand(size, 3, 32, 32)
        return dict(lr_npr=low, hr_source=torch.full((size, 3, 128, 128), .4), row_index=rows)


class BackgroundDetailTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / 'source5308.pt'
        self.reference = self.directory / 'reference.json'
        self.detail = dict(fingerprint='same-detail120', frames=120)
        parent = dict(training_steps=5308, model={'immutable': torch.ones(1)},
            training_plan=dict(train={'base': {'split': 'train'}}, val={'base': {'split': 'val'}}, panel=list(range(256))))
        torch.save(parent, self.source)
        self.reference.write_text(json.dumps(dict(panel=list(range(256)), detail_animation_panel=self.detail)), encoding='utf-8')
        self.args = SimpleNamespace(run_dir=self.directory/'run_DISCARDED', source_checkpoint=self.source,
            reference_plan=self.reference, data_root=self.directory/'data', device='cpu', seed=173,
            detail_channels=48, detail_blocks=5, residual_bound=.25, batch=8, steps=3000,
            validation_frames=256, validation_batch=8, prediction_every=15, validate_every=100,
            learning_rate=1e-4, prepare_only=False, probe=True, max_updates=3)
        self.source_sha = runner.file_sha(self.source)

    def fixture_patches(self):
        def initialize(parent, **kw):
            return TinyDetail(), dict(kind='fake_new_branch', old_Adam_reused=False,
                copied_background_sha256='unused', kwargs=kw)
        return patch.multiple(runner, SOURCE_SHA=self.source_sha,
            BACKGROUND_SHA=runner._hash_tensors({name: value for name, value in TinyDetail().state_dict().items()
                if not name.startswith(runner.TRAINABLE_PREFIX)}),
            initialize_background_detail=initialize, NeuralEnemyRestorationData=FakeData,
            build_detail_animation_panel=lambda data, seed: deepcopy(self.detail))

    def test_fresh_adam_starts_local0_and_owns_only_new_branch(self):
        model = TinyDetail()
        optimizer, names = runner.restore_detail_optimizer(model, None, step=0)
        self.assertTrue(names); self.assertTrue(all(name.startswith('detail_residual.') for name in names))
        self.assertEqual(len(optimizer.state), 0)
        with self.assertRaises(ValueError): runner.restore_detail_optimizer(model, None, step=5308)
        model.background.weight.requires_grad_(True)
        with self.assertRaises(ValueError): runner.restore_detail_optimizer(model, None, step=0)

    def optimizer_payload(self):
        model = TinyDetail()
        optimizer, names = runner.restore_detail_optimizer(model, None, step=0)
        loss = model.forward_with_aux(torch.full((1, 3, 32, 32), .3))['image'].square().mean()
        loss.backward(); optimizer.step()
        return model, dict(optimizer=deepcopy(optimizer.state_dict()), optimizer_parameter_names=names)

    def test_complete_adam_resume_retains_exact_moments_and_hyperparameters(self):
        model, payload = self.optimizer_payload()
        optimizer, names = runner.restore_detail_optimizer(model, payload, step=1)
        self.assertEqual(len(optimizer.state), len(names))
        for actual, expected in zip(optimizer.state_dict()['state'].values(), payload['optimizer']['state'].values()):
            for name in actual: torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_adam_rejects_old_names_wrong_steps_hyperparameters_and_corrupt_moments(self):
        model, original = self.optimizer_payload()
        for mutation in ('names', 'step', 'beta', 'nan', 'shape', 'negative_second_moment'):
            with self.subTest(mutation=mutation):
                payload = deepcopy(original)
                state = next(iter(payload['optimizer']['state'].values()))
                if mutation == 'names': payload['optimizer_parameter_names'][0] = 'enemy_restoration.old_head'
                elif mutation == 'step': state['step'] = torch.tensor(5309.)
                elif mutation == 'beta': payload['optimizer']['param_groups'][0]['betas'] = (.1, .2)
                elif mutation == 'nan': state['exp_avg'].flatten()[0] = float('nan')
                elif mutation == 'shape': state['exp_avg'] = torch.zeros(1)
                else: state['exp_avg_sq'].flatten()[0] = -1
                with self.assertRaises(ValueError): runner.restore_detail_optimizer(model, payload, step=1)

    def test_zero_residual_initialization_must_equal_preferred_row_exactly(self):
        model = TinyDetail()
        runner.zero_residual_probe(model)
        with torch.no_grad(): model.detail_residual.bias.fill_(.01)
        with self.assertRaises(ValueError): runner.zero_residual_probe(model)

    def test_sampler_exposure_uses_new_local_steps_not_source5308(self):
        runner.validate_sampler_exposure(SimpleNamespace(draws=0), step=0, batch=8)
        runner.validate_sampler_exposure(SimpleNamespace(draws=24), step=3, batch=8)
        for draws, step in ((24, 5308), (25, 3), (24., 3), (0, -1)):
            with self.assertRaises(ValueError): runner.validate_sampler_exposure(SimpleNamespace(draws=draws), step=step, batch=8)

    def test_rng_resume_reproduces_all_cpu_random_streams(self):
        runner.restore_rng(None, seed=173, device=torch.device('cpu'))
        payload = dict(rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state())
        expected = random.random(), np.random.rand(), torch.rand(3)
        runner.restore_rng(payload, seed=0, device=torch.device('cpu'))
        self.assertEqual(expected[0], random.random()); self.assertEqual(expected[1], np.random.rand())
        torch.testing.assert_close(expected[2], torch.rand(3), rtol=0, atol=0)

    def test_production_and_discarded_probe_have_identical_immutable_objective_plan(self):
        with self.fixture_patches():
            plan, _ = runner.prepare_experiment(self.args)
            other = deepcopy(self.args); other.run_dir = self.directory/'production'; other.probe=False; other.max_updates=0
            production, _ = runner.prepare_experiment(other)
        self.assertEqual(plan, production)
        self.assertEqual(plan['steps'], 3000); self.assertEqual(plan['source_training_steps'], 5308)
        self.assertFalse(plan['inherited_optimizer_states'])
        self.assertEqual(runner.file_sha(self.source), self.source_sha)

    def test_preparation_rejects_configuration_or_data_receipt_drift(self):
        with self.fixture_patches():
            runner.prepare_experiment(self.args)
            changed = deepcopy(self.args); changed.batch=4
            with self.assertRaises(ValueError): runner.prepare_experiment(changed)
            reference = json.loads(self.reference.read_text()); reference['panel'][0]=999
            self.reference.write_text(json.dumps(reference))
            with self.assertRaises(ValueError): runner.prepare_experiment(self.args)

    def test_three_updates_then_two_resume_match_five_uninterrupted_updates(self):
        def objective(output, batch):
            loss=(output['image']-batch['hr_source']).square().mean()
            return loss, dict(train_loss=float(loss.detach()))
        def evaluate(model, data, panel, *, step, **kw):
            return dict(objective_protocol=runner.PROTOCOL, step=step, panel_fingerprint='panel-256',
                        validation_objective={'loss': .1}), {'metadata': {}}
        def metadata(model):
            return dict(trainable_parameters=sum(p.numel() for p in model.detail_residual.parameters()))
        with self.fixture_patches(), patch.multiple(runner,
                background_detail_metadata=metadata, background_detail_loss=objective,
                evaluate_background_detail_sr=evaluate,
                save_background_detail_sr_preview=lambda preview, path, every: preview['metadata'],
                background_detail_candidate=lambda report, baseline: dict(eligible=False, score=0)):
            runner.run(self.args)
            first = torch.load(self.args.run_dir/'checkpoints/sr_last.pt', weights_only=False)
            self.assertEqual(first['training_steps'], 3)
            self.assertEqual(first['sampler']['draws'], 24)
            self.assertEqual(first['source_training_steps'], 5308)
            self.assertTrue((self.args.run_dir/'checkpoints/sr_initial.pt').is_file())
            self.assertTrue((self.args.run_dir/'checkpoints/sr_first.pt').is_file())
            continued = deepcopy(self.args); continued.max_updates=2
            runner.run(continued)
            resumed = torch.load(self.args.run_dir/'checkpoints/sr_last.pt', weights_only=False)
            direct = deepcopy(self.args); direct.run_dir=self.directory/'direct_DISCARDED'; direct.max_updates=5
            runner.run(direct)
            whole = torch.load(direct.run_dir/'checkpoints/sr_last.pt', weights_only=False)
        self.assertEqual(resumed['training_steps'], 5); self.assertEqual(resumed['sampler']['draws'], 40)
        self.assertEqual(resumed['training_plan'], whole['training_plan'])
        for name, value in resumed['model'].items(): torch.testing.assert_close(value, whole['model'][name], rtol=0, atol=0)
        for index, state in resumed['optimizer']['state'].items():
            for name, value in state.items(): torch.testing.assert_close(value, whole['optimizer']['state'][index][name], rtol=0, atol=0)
        self.assertEqual(runner.file_sha(self.source), self.source_sha)


if __name__ == '__main__': unittest.main()
