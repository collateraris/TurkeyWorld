from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from turkeyworld.neural_enemy_restoration_continuation import (
    continuation_source, validate_continuation_checkpoint)
from turkeyworld import train_neural_enemy_restoration_sr as trainer
from turkeyworld.renderer_student import _hash_tensors
import test_train_neural_enemy_restoration_sr as fixtures

optimizer_step = fixtures.optimizer_step
small_model = fixtures.small_model


def fixture():
    model = small_model()
    optimizer, names = trainer.restore_decoder_optimizer(model, None, step=0)
    optimizer_step(model, optimizer); optimizer_step(model, optimizer)
    frozen = {name: value for name, value in model.state_dict().items()
              if not name.startswith('enemy_restoration.')}
    plan = dict(kind='neural_enemy_restoration_sr_v1', config=deepcopy(model.config),
        train={'id': 'train'}, val={'id': 'val'}, panel=[0, 1], panel_fingerprint='panel',
        batch=8, seed=173, parent_checkpoint_sha256='parent',
        migration={'copied_background_sha256': _hash_tensors(frozen)})
    source = dict(training_steps=2, neural_enemy_restoration_updates=2,
        training_plan=deepcopy(plan), plan_sha256='source_plan', config=deepcopy(model.config),
        model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
        optimizer_parameter_names=names, sampler={'draws': 16},
        standalone_RGB_only=True, atlas_in_new_model=False, disposable_probe=False,
        rng_python=(), rng_numpy=(), rng_torch=torch.get_rng_state(), rng_cuda=[],
        best={'step': 1, 'loss': .2})
    return model, source, plan


class RestorationContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_source_validation_preserves_all_weights_adam_and_source_best_without_reusing_score(self):
        model, source, plan = fixture()
        before = deepcopy(source); weights = _hash_tensors(model.state_dict())
        binding = continuation_source(source, plan, model)
        self.assertEqual(binding['training_steps'], 2)
        self.assertEqual(binding['sampler_draws'], 16)
        self.assertEqual(binding['optimizer_state_count'], 58)
        self.assertEqual(binding['previous_best'], source['best'])
        self.assertFalse(binding['previous_best_used_for_new_objective'])
        for name in ('optimizer_reset', 'weights_reset', 'sampler_reset', 'RNG_reset'):
            self.assertFalse(binding[name])
        self.assertEqual(_hash_tensors(model.state_dict()), weights)
        self.assertEqual(_hash_tensors(source['model']), _hash_tensors(before['model']))
        for key, state in source['optimizer']['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, before['optimizer']['state'][key][name], atol=0, rtol=0)
        resumed = deepcopy(model)
        optimizer, _ = trainer.restore_decoder_optimizer(resumed, source, step=2)
        self.assertTrue(all(float(value['step']) == 2 for value in optimizer.state.values()))

    def test_source_rejects_data_panel_batch_model_background_or_exposure_change(self):
        model, source, plan = fixture()
        mutations = [
            lambda s: s['training_plan'].update(train={'id': 'different'}),
            lambda s: s['training_plan'].update(panel=[1, 0]),
            lambda s: s['training_plan'].update(batch=16),
            lambda s: s.update(disposable_probe=True),
            lambda s: s.update(neural_enemy_restoration_updates=1),
            lambda s: s['sampler'].update(draws=8),
            lambda s: s.update(optimizer_parameter_names=list(reversed(s['optimizer_parameter_names']))),
            lambda s: s['model'].pop('enemy_restoration.rgb.bias'),
            lambda s: s['model']['input.bias'].add_(1),
            lambda s: s['model']['enemy_restoration.rgb.bias'].fill_(float('nan')),
            lambda s: s.pop('rng_torch'),
        ]
        for mutate in mutations:
            changed = deepcopy(source); mutate(changed)
            with self.assertRaises(ValueError): continuation_source(changed, plan, model)

    def test_checkpoint_replacement_is_rejected_after_binding(self):
        model, source, plan = fixture()
        plan['resume_source'] = continuation_source(source, plan, model)
        validate_continuation_checkpoint(source, plan)
        changed = deepcopy(source); changed['model']['enemy_restoration.rgb.bias'].add_(1)
        with self.assertRaises(ValueError): validate_continuation_checkpoint(changed, plan)
        changed = deepcopy(source); changed['plan_sha256'] = 'different'
        with self.assertRaises(ValueError): validate_continuation_checkpoint(changed, plan)

    def test_prepare_new_phase_pins_complete_source_and_does_not_reinterpret_old_plan(self):
        helper = fixtures.NeuralEnemyRestorationTrainerTests()
        helper.setUp()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); args, _ = helper.preparation_fixture(root)
            old_plan, _, model = helper.prepare(args)
            optimizer, names = trainer.restore_decoder_optimizer(model, None, step=0)
            optimizer_step(model, optimizer); optimizer_step(model, optimizer)
            trainer._rng(seed=173, device=torch.device('cpu'))
            source = dict(training_steps=2, neural_enemy_restoration_updates=2,
                training_plan=old_plan, plan_sha256=trainer.file_sha(args.run_dir/'training_plan.json'),
                config=deepcopy(model.config), model=model.state_dict(),
                optimizer=optimizer.state_dict(), optimizer_parameter_names=names,
                sampler={'draws': 16}, standalone_RGB_only=True, atlas_in_new_model=False,
                disposable_probe=False, rng_python=(), rng_numpy=(),
                rng_torch=torch.get_rng_state(), rng_cuda=[], best={'step': 1, 'loss': .1})
            path = root/'source.pt'; torch.save(source, path)
            old_bytes = (args.run_dir/'training_plan.json').read_bytes()
            args.run_dir = root/'correction'; args.resume_checkpoint = path
            args.loss_protocol = trainer.FOREGROUND_PROTOCOL
            plan, _, _ = helper.prepare(args)
            self.assertEqual(plan['kind'], trainer.FOREGROUND_PROTOCOL)
            self.assertEqual(plan['phase_start_step'], 2)
            self.assertEqual(plan['phase_budget'], 3000)
            self.assertEqual(plan['target_step'], 3002)
            self.assertEqual(plan['foreground_topk_pixels'], 32)
            self.assertEqual(plan['edge_ssim_background_context'], 'same_frozen_display_background_both_sides')
            self.assertEqual(plan['resume_source']['checkpoint_sha256'], trainer.file_sha(path))
            self.assertEqual(plan['resume_source']['optimizer_state_count'], 58)
            self.assertFalse(plan['resume_source']['previous_best_used_for_new_objective'])
            self.assertEqual((root/'new_run/training_plan.json').read_bytes(), old_bytes)
            self.assertNotEqual(plan['loss_weights'], old_plan['loss_weights'])
            args.run_dir=root/'detail';args.loss_protocol=trainer.DETAIL_PROTOCOL
            receipt={'kind':'detail_fixture','fingerprint':'fixed_detail','frames':120,'indices':list(range(120))}
            with patch.object(trainer,'build_detail_animation_panel',return_value=receipt)as build:
                detail,_,_=helper.prepare(args)
            build.assert_called_once()
            self.assertEqual(detail['version'],3)
            self.assertEqual(detail['detail_animation_panel'],receipt)
            self.assertEqual(detail['detail_loss_settings'],trainer.DETAIL_SETTINGS)
            self.assertEqual(detail['target_step'],3002)
            self.assertFalse(detail['detail_selection_policy']['automatic_deployment'])


if __name__ == '__main__': unittest.main()
