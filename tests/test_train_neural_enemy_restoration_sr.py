from copy import deepcopy
from contextlib import contextmanager
import functools
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld import train_neural_enemy_restoration_sr as trainer
from turkeyworld.neural_enemy_restoration_sr import (
    NeuralEnemyRestorationSR, initialize_neural_enemy_restoration)
from turkeyworld.sr_neural_pairs_data import fingerprint
from test_neural_enemy_restoration_sr import parent_payload


class FakePairs:
    """Preparation-only immutable identity fixture, never a model input."""
    changed = False
    overlap = False
    panel_shift = False
    cache_change = None

    def __init__(self, root, split, *, seed, device):
        if device != 'cpu': raise AssertionError('Preparation must stay on CPU')
        self.split = split
        self.scene_keys = {'same' if self.overlap else split}
        cache = dict(future_rgb_features=False, paired_same_pose_map_tick=True,
                     restorer_present=False, solver_steps=3)
        if self.cache_change: cache.update(self.cache_change)
        base = dict(protocol='cached_NPR_RGB_fixture', split=split,
                    fingerprint='changed' if self.changed else 'original', cache_identity=cache)
        self.identity = dict(protocol='paired_RGB_fixture', split=split,
                             fingerprint='changed' if self.changed else 'original', base=base)
        self.base = SimpleNamespace(manifest={'cache_identity': cache}, identity=base)

    def fixed_panel(self, count, *, seed):
        values = list(range(count))
        return values[1:] + values[:1] if self.panel_shift else values

    def panel_fingerprint(self, panel):
        return fingerprint(dict(identity=self.identity, indices=panel))


def small_model():
    return NeuralEnemyRestorationSR(channels=8, blocks=1, semantic_channels=8,
                                    semantic_blocks=1, restoration_channels=8)


def optimizer_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().mean() for parameter in model.parameters() if parameter.requires_grad)
    loss.backward(); optimizer.step()


class NeuralEnemyRestorationTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        FakePairs.changed = FakePairs.overlap = FakePairs.panel_shift = False
        FakePairs.cache_change = None

    def preparation_fixture(self, root):
        parent = parent_payload(); parent['protocol'] = dict(panel=list(range(256)),
            train={'base': FakePairs(root / 'cache', 'train', seed=173, device='cpu').base.identity},
            val={'base': FakePairs(root / 'cache', 'val', seed=174, device='cpu').base.identity})
        path = root / 'pinned_parent.pt'; torch.save(parent, path)
        args = SimpleNamespace(parent_checkpoint=path, data_root=root / 'cache',
            run_dir=root / 'new_run', seed=173, batch=8, steps=3000,
            validation_frames=256, validation_batch=8, prediction_every=15,
            validate_every=100, prepare_only=True, device='cpu', probe=False)
        return args, parent

    def prepare(self, args, *, channels=8):
        initialize = functools.partial(initialize_neural_enemy_restoration, restoration_channels=channels)
        with patch.object(trainer, 'NeuralEnemyRestorationData', FakePairs), \
             patch.object(trainer, 'initialize_neural_enemy_restoration', initialize):
            return trainer.prepare_experiment(args)

    def test_runtime_initializes_cuda_then_thread_budget_without_touching_cuda_for_cpu(self):
        events=[]
        with patch.object(trainer.torch.cuda,'init',side_effect=lambda:events.append('cuda')),\
             patch.object(trainer.torch,'set_num_threads',side_effect=lambda n:events.append(('threads',n))):
            self.assertEqual(trainer.initialize_runtime('cuda').type,'cuda')
            self.assertEqual(events,['cuda',('threads',2)])
            events.clear();self.assertEqual(trainer.initialize_runtime('cpu').type,'cpu')
            self.assertEqual(events,[('threads',2)])

    def test_direct_run_initializes_runtime_before_checkpoint_preparation(self):
        events=[]
        def prepare(args):events.append('prepare');raise LookupError('stop before file/model work')
        with patch.object(trainer,'initialize_runtime',side_effect=lambda device:events.append('runtime')),\
             patch.object(trainer,'prepare_experiment',side_effect=prepare):
            with self.assertRaisesRegex(LookupError,'stop before file/model work'):
                trainer.run(SimpleNamespace(device='cuda'))
        self.assertEqual(events,['runtime','prepare'])

    def test_cli_initializes_runtime_before_keepawake_and_run(self):
        events=[]
        @contextmanager
        def guard():
            events.append('guard');yield
        with tempfile.TemporaryDirectory()as folder,\
             patch.object(trainer,'initialize_runtime',side_effect=lambda device:events.append('runtime')),\
             patch.object(trainer,'KeepAwake',side_effect=guard),\
             patch.object(trainer,'run',side_effect=lambda args:events.append('run')):
            trainer.main(['--run-dir',folder,'--parent-checkpoint','unused.pt','--data-root','unused',
                          '--device','cuda','--probe'])
        self.assertEqual(events,['runtime','guard','run'])

    def test_fresh_adam_contains_only_new_decoder_and_exact_resume_continues_next_update(self):
        model = small_model()
        optimizer, names = trainer.restore_decoder_optimizer(model, None, step=0)
        self.assertTrue(names); self.assertTrue(all(name.startswith('enemy_restoration.') for name in names))
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual({id(p) for group in optimizer.param_groups for p in group['params']},
                         {id(p) for name, p in model.named_parameters() if name in names})
        frozen = {name: value.detach().clone() for name, value in model.named_parameters()
                  if not value.requires_grad}
        optimizer_step(model, optimizer); optimizer_step(model, optimizer)
        payload = dict(optimizer_parameter_names=list(names), optimizer=deepcopy(optimizer.state_dict()))
        resumed = deepcopy(model)
        restored, restored_names = trainer.restore_decoder_optimizer(resumed, payload, step=2)
        self.assertEqual(restored_names, names)
        self.assertEqual(len(restored.state), len(names))
        self.assertTrue(all(float(state['step']) == 2 for state in restored.state.values()))
        optimizer_step(model, optimizer); optimizer_step(resumed, restored)
        for first, second in zip(model.state_dict().values(), resumed.state_dict().values()):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertTrue(all(torch.equal(model.state_dict()[name], value) for name, value in frozen.items()))

    def test_resume_rejects_parameter_order_step_missing_states_and_altered_adam_state(self):
        model = small_model(); optimizer, names = trainer.restore_decoder_optimizer(model, None, step=0)
        optimizer_step(model, optimizer)
        base = dict(optimizer_parameter_names=names, optimizer=deepcopy(optimizer.state_dict()))
        first_id = next(iter(base['optimizer']['state']))
        mutations = [
            lambda value: value.update(optimizer_parameter_names=list(reversed(names))),
            lambda value: value['optimizer']['param_groups'][0].update(lr=2e-4),
            lambda value: value['optimizer']['param_groups'][0].update(betas=(.5, .9)),
            lambda value: value['optimizer']['param_groups'][0].update(eps=1e-4),
            lambda value: value['optimizer']['state'][first_id].update(step=torch.tensor(2.)),
            lambda value: value['optimizer']['state'].pop(first_id),
            lambda value: value['optimizer']['state'][first_id].update(exp_avg=torch.full_like(
                value['optimizer']['state'][first_id]['exp_avg'], float('nan'))),
            lambda value: value['optimizer']['state'][first_id].update(exp_avg=torch.zeros(1)),
            lambda value: value['optimizer']['state'][first_id].update(exp_avg_sq=-torch.ones_like(
                value['optimizer']['state'][first_id]['exp_avg_sq'])),
        ]
        for mutate in mutations:
            altered = deepcopy(base); mutate(altered)
            with self.assertRaises(ValueError):
                trainer.restore_decoder_optimizer(deepcopy(model), altered, step=1)
        with self.assertRaises(ValueError): trainer.restore_decoder_optimizer(model, base, step=0)

    def test_sampler_exposure_must_match_complete_optimizer_clock_without_mutation(self):
        data = SimpleNamespace(draws=24, sentinel=['unchanged'])
        before = deepcopy(vars(data))
        trainer.validate_resume_sampler(data, step=3, batch=8)
        self.assertEqual(vars(data), before)
        for step, batch, draws in ((3, 8, 16), (3, 8, 32), (0, 8, 8)):
            data.draws = draws
            with self.assertRaises(ValueError): trainer.validate_resume_sampler(data, step=step, batch=batch)
        data.draws = 0; trainer.validate_resume_sampler(data, step=0, batch=8)

    def test_rng_restore_reproduces_python_numpy_and_torch_after_constructor_noise(self):
        device = torch.device('cpu'); trainer._rng(seed=175, device=device)
        payload = dict(rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                       rng_torch=torch.get_rng_state().clone(), rng_cuda=[])
        expected = (random.random(), np.random.rand(5), torch.rand(5))
        random.seed(999); np.random.seed(999); torch.manual_seed(999)
        small_model(); random.random(); np.random.rand(7); torch.rand(7)
        trainer._rng(payload, seed=1, device=device)
        self.assertEqual(random.random(), expected[0]); np.testing.assert_array_equal(np.random.rand(5), expected[1])
        torch.testing.assert_close(torch.rand(5), expected[2], atol=0, rtol=0)
        trainer._rng(seed=175, device=device)
        self.assertEqual(random.random(), expected[0]); np.testing.assert_array_equal(np.random.rand(5), expected[1])
        torch.testing.assert_close(torch.rand(5), expected[2], atol=0, rtol=0)

    def test_preparation_pins_exact_parent_data_config_panel_and_fresh_architecture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); args, parent = self.preparation_fixture(root)
            source_before = args.parent_checkpoint.read_bytes()
            rng = torch.get_rng_state().clone()
            plan, loaded, model = self.prepare(args)
            self.assertEqual(plan['panel'], parent['protocol']['panel'])
            self.assertTrue(plan['parent_panel_indices_equal'])
            self.assertTrue(plan['parent_paired_data_equal'])
            self.assertEqual(plan['parent_checkpoint_sha256'], trainer.file_sha(args.parent_checkpoint))
            self.assertEqual(plan['parent_training_steps'], 1234)
            self.assertEqual(plan['optimizer'], 'fresh_Adam_for_new_architecture')
            self.assertFalse(plan['hypothesis_sidecar_required']); self.assertFalse(plan['atlas_in_new_model'])
            self.assertEqual(plan['inference_inputs'], 'actual_NPR_RGB32_only')
            self.assertFalse(any('atlas' in name for name in model.state_dict()))
            self.assertEqual(args.parent_checkpoint.read_bytes(), source_before)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            repeated, _, _ = self.prepare(args)
            self.assertEqual(repeated, plan)
            self.assertEqual(json.loads((args.run_dir / 'training_plan.json').read_text()), plan)
            FakePairs.changed = True
            with self.assertRaises(ValueError): self.prepare(args)
            FakePairs.changed = False
            with self.assertRaises(ValueError): self.prepare(args, channels=16)
            changed = deepcopy(args); changed.steps += 1
            with self.assertRaises(ValueError): self.prepare(changed)

    def test_preparation_rejects_overlap_shifted_parent_panel_and_nonfinal_or_oracle_cache(self):
        for kind in ('overlap', 'panel', 'parent_data', 'future', 'tick', 'restorer', 'solver'):
            with tempfile.TemporaryDirectory() as folder:
                self.setUp(); args, parent = self.preparation_fixture(Path(folder))
                if kind == 'overlap': FakePairs.overlap = True
                elif kind == 'panel': FakePairs.panel_shift = True
                elif kind == 'parent_data':
                    parent['protocol']['val']['base']['fingerprint'] = 'different_cache_same_indices'
                    torch.save(parent, args.parent_checkpoint)
                else:
                    FakePairs.cache_change = {'future': {'future_rgb_features': True},
                        'tick': {'paired_same_pose_map_tick': False}, 'restorer': {'restorer_present': True},
                        'solver': {'solver_steps': 1}}[kind]
                with self.assertRaises(ValueError): self.prepare(args)
                self.assertFalse((args.run_dir / 'training_plan.json').exists())


if __name__ == '__main__': unittest.main()
