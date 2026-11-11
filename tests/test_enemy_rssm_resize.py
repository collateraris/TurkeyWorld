import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_rssm_combat_training import raw_model
from turkeyworld.enemy_rssm import EnemyRSSM, EnemyRSSMConfig
from turkeyworld.enemy_rssm_resize import (ENEMY_PREFIX, ENEMY_20M_OVERRIDES,
    _copy_overlap, prepare_enemy_rssm_resize, resize_enemy_rssm_checkpoint)
from turkeyworld.models import NeuralWorldModel, STATE_SCHEMA, WorldModelConfig
from turkeyworld.rssm_migration import configure_rssm_training


class EnemyRSSMResizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_tree(self, actual, expected):
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_tree(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_tree(value, reference)
        else:
            self.assertEqual(actual, expected)

    def fixture(self):
        reference = raw_model()
        config = replace(reference.config, navigation_only=False, enemy_rssm_enabled=True,
            enemy_rssm_deter_dim=16, enemy_rssm_hidden_dim=16, enemy_rssm_stoch=2,
            enemy_rssm_classes=4, enemy_rssm_embed_dim=8, enemy_rssm_action_dim=8, enemy_rssm_blocks=2)
        model = NeuralWorldModel(config)
        model.flow_renderer.codec_loaded.fill_(True)
        configure_rssm_training(model)
        names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        parameters = dict(model.named_parameters())
        split = next(i for i, name in enumerate(names) if name.startswith(ENEMY_PREFIX))
        optimizer = torch.optim.AdamW([
            {'params': [parameters[name] for name in names[:split]], 'lr': 1e-4},
            {'params': [parameters[name] for name in names[split:]], 'lr': 2e-4}], weight_decay=.01)
        sum(parameters[name].square().sum()*1e-6 for name in names).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        run_state = {'run_id': 'source', 'global_training_step': 4482, 'current': 4482,
            'loss_protocol': 'rssm_dual_group_balance_v5',
            'training_protocol': {'microbatch_size': 32, 'gradient_accumulation_steps': 4},
            'rssm_training': {'source_completed': 4282, 'combat_completed': 17,
                'loss_balance_revision': {'committed_updates': 28, 'last_optimizer_step': 4482}},
            'objective_history': [{'kind': 'historical', 'step': 4454}]}
        saved = {'stage': 'world', 'state_schema': dict(STATE_SCHEMA), 'config': asdict(config),
            'model': {name: value.clone() for name, value in model.state_dict().items()},
            'optimizer': copy.deepcopy(optimizer.state_dict()), 'optimizer_parameter_names': names,
            'training_steps': 4482, 'global_training_step': 4482, 'run_state': run_state,
            'rssm_training': copy.deepcopy(run_state['rssm_training']),
            'dual_rssm_migration': {'old_append_lineage': True}, 'rssm_migration': {'original': True},
            'training_counters': {'pending_microbatches': 0, 'fragments': 286848},
            'rng_state': np.random.default_rng(19).bit_generator.state,
            'torch_rng_state': torch.get_rng_state().clone(), 'cuda_rng_state': [],
            'loss_balance_state': {'banks': {'combat': {'updates': 28}}},
            'ema': {'retained': torch.arange(4.)}, 'adversarial': {'state': torch.arange(3.)}}
        return model, optimizer, saved

    def resized(self, saved, **kwargs):
        return resize_enemy_rssm_checkpoint(saved,
            enemy_config={'deter_dim': 24, 'hidden_dim': 32}, **kwargs)

    def test_migration_retains_player_adam_rng_clocks_and_all_enemy_channels(self):
        source, optimizer, saved = self.fixture()
        before = copy.deepcopy(saved); rng = torch.get_rng_state().clone()
        with patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('No CUDA RNG use')):
            migrated, report = self.resized(saved, run_id='new')
        self.assert_tree(saved, before)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(migrated['optimizer_parameter_names'], saved['optimizer_parameter_names'])
        self.assert_tree(migrated['optimizer']['param_groups'], saved['optimizer']['param_groups'])
        for key in ('rng_state', 'torch_rng_state', 'cuda_rng_state', 'training_counters',
                    'loss_balance_state', 'dual_rssm_migration', 'rssm_migration', 'ema', 'adversarial'):
            self.assert_tree(migrated[key], saved[key])
        for key in ('global_training_step', 'current', 'training_protocol'):
            self.assert_tree(migrated['run_state'][key], saved['run_state'][key])
        self.assert_tree(migrated['run_state']['objective_history'][:-1], saved['run_state']['objective_history'])
        self.assertEqual(migrated['run_state']['run_id'], 'new')
        self.assertFalse(report['inference_functions_bitexact_claimed'])
        reset = set(report['enemy_adam_reset_parameter_ids'])
        self.assertTrue(reset)
        for identifier, name in enumerate(saved['optimizer_parameter_names']):
            if identifier in reset:
                self.assertTrue(name.startswith(ENEMY_PREFIX))
                self.assertNotIn(identifier, migrated['optimizer']['state'])
            else:
                self.assert_tree(migrated['optimizer']['state'][identifier], saved['optimizer']['state'][identifier])
        for name, value in saved['model'].items():
            if not name.startswith(ENEMY_PREFIX):
                self.assertTrue(torch.equal(value, migrated['model'][name]), name)
            else:
                detail = report['enemy_tensor_mappings'][name]
                axes = detail.get('destination_axis_indices')
                selected = migrated['model'][name]
                if axes:
                    selected = selected[tuple(torch.meshgrid(*[torch.tensor(a) for a in axes], indexing='ij'))]
                torch.testing.assert_close(selected, value, rtol=0, atol=0)
        restored = NeuralWorldModel(WorldModelConfig(**migrated['config']))
        restored.load_state_dict(migrated['model'], strict=True); configure_rssm_training(restored)
        params = dict(restored.named_parameters()); names = migrated['optimizer_parameter_names']
        groups = [{**{k: v for k, v in group.items() if k != 'params'},
            'params': [params[names[i]] for i in group['params']]} for group in migrated['optimizer']['param_groups']]
        new_optimizer = torch.optim.AdamW(groups); new_optimizer.load_state_dict(migrated['optimizer'])
        for model, opt in ((source, optimizer), (restored, new_optimizer)):
            opt.zero_grad(set_to_none=True)
            sum(p.square().sum()*1e-6 for p in model.parameters() if p.requires_grad).backward(); opt.step()
        source_params = dict(source.named_parameters())
        for i, name in enumerate(names):
            if i not in reset:
                torch.testing.assert_close(params[name], source_params[name], rtol=0, atol=0)
                self.assert_tree(new_optimizer.state[params[name]], optimizer.state[source_params[name]])
            else:
                self.assertEqual(float(new_optimizer.state[params[name]]['step']), 1)
        # The expanded recurrence really executes and learns; renderer features
        # retain the player's128D contract rather than the larger enemy memory.
        enemy = restored.vjepa_backend.enemy_rssm
        memory = enemy.initial(2).requires_grad_()
        result = enemy.prior(memory, torch.zeros(2, 20), torch.zeros(2, 4), torch.zeros(2, 2),
            torch.zeros(2, enemy.width), torch.zeros(2, 2, enemy.width), sample=False)
        self.assertEqual(result['memory'].shape, (2, 32))
        self.assertTrue(torch.isfinite(result['motion']).all())
        (result['motion'].square().sum()+result['memory'].square().mean()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in enemy.core.parameters()))

    def test_semantic_mapping_respects_groups_three_gate_offsets_and_latent_tail(self):
        old = EnemyRSSMConfig(deter_dim=8, hidden_dim=8, blocks=2)
        new = replace(old, deter_dim=12, hidden_dim=12)
        source = torch.arange(2*4*28.).reshape(2, 4, 28)
        target = torch.full((2, 6, 42), -1.)
        mapped, _ = _copy_overlap('core.recurrent_hidden.weight', source, target, old, new)
        self.assertEqual(mapped[1, 3, 6+12+7], source[1, 3, 4+8+7])
        self.assertEqual(mapped[1, 3, 6+24+7], source[1, 3, 4+16+7])
        self.assertEqual(mapped[1, 5, 0], -1)
        gates = torch.arange(2*12*4.).reshape(2, 12, 4)
        mapped, _ = _copy_overlap('core.recurrent_gates.weight', gates, torch.full((2, 18, 6), -1.), old, new)
        self.assertEqual(mapped[1, 12+3, 3], gates[1, 8+3, 3])
        self.assertEqual(mapped[1, 6+3, 3], gates[1, 4+3, 3])
        readout = torch.arange(8*264.).reshape(8, 264)
        mapped, _ = _copy_overlap('readout.0.weight', readout, torch.full((12, 268), -1.), old, new)
        self.assertEqual(mapped[7, 6+3], readout[7, 4+3])
        self.assertEqual(mapped[7, 12+255], readout[7, 8+255])

    def test_exact_production_count_and_default_legacy_profile_remain(self):
        with torch.device('meta'):
            legacy = EnemyRSSM()
            enlarged = EnemyRSSM(EnemyRSSMConfig(deter_dim=1344, hidden_dim=1408))
        self.assertEqual(sum(p.numel() for p in legacy.parameters()), 2438019)
        self.assertEqual(sum(p.numel() for p in enlarged.parameters()), 20000323)
        self.assertEqual(enlarged.memory_size, 1600)
        self.assertEqual(ENEMY_20M_OVERRIDES['enemy_rssm_deter_dim'], 1344)
        _, _, saved = self.fixture()
        first, report = self.resized(saved, seed=173)
        same, _ = self.resized(saved, seed=173)
        other, _ = self.resized(saved, seed=174)
        self.assertTrue(all(torch.equal(v, same['model'][name]) for name, v in first['model'].items()))
        self.assertTrue(any(not torch.equal(first['model'][name], other['model'][name])
            for name in report['enemy_resized_parameter_names']))
        self.assertTrue(all(torch.equal(first['model'][name], other['model'][name])
            for name in saved['model'] if not name.startswith(ENEMY_PREFIX)))

    def test_rejects_partial_or_corrupted_sources_and_unapproved_config_changes(self):
        _, _, saved = self.fixture()
        for kind in ('pending', 'shape', 'names', 'moment', 'nonfinite', 'wrong_schema'):
            bad = copy.deepcopy(saved)
            if kind == 'pending': bad['training_counters']['pending_microbatches'] = 1
            elif kind == 'shape': bad['model'][next(iter(bad['model']))] = torch.ones(1)
            elif kind == 'names': bad['optimizer_parameter_names'][0] = 'wrong'
            elif kind == 'moment': bad['optimizer']['state'][0]['exp_avg'] = torch.ones(1)
            elif kind == 'nonfinite': bad['model'][next(iter(bad['model']))].flatten()[0] = float('nan')
            else: bad['state_schema'] = {}
            with self.assertRaises(ValueError): self.resized(bad)
        for overrides in ({'classes': 8, 'deter_dim': 24}, {'deter_dim': 8}, {'deter_dim': 16, 'hidden_dim': 16}):
            with self.assertRaises(ValueError): resize_enemy_rssm_checkpoint(saved, enemy_config=overrides)

    def test_separate_atomic_file_and_report_never_overwrite_source(self):
        _, _, saved = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder)/'old.pt', Path(folder)/'new.pt'
            torch.save(saved, source); original = source.read_bytes()
            report = prepare_enemy_rssm_resize(source, destination,
                enemy_config={'deter_dim': 24, 'hidden_dim': 32}, run_id='new')
            self.assertEqual(original, source.read_bytes())
            self.assertEqual(json.loads(destination.with_suffix('.resize.json').read_text())['kind'],
                'resize_independent_enemy_rssm')
            result = torch.load(destination, weights_only=False)
            self.assertEqual(result['enemy_rssm_resize']['source_step'], 4482)
            self.assertEqual(result['dual_rssm_migration'], saved['dual_rssm_migration'])
            self.assertTrue(report['player_and_shared_adam_states_bitexact'])
            with self.assertRaises(FileExistsError): prepare_enemy_rssm_resize(source, destination)


if __name__ == '__main__':
    unittest.main()
