"""Full named-Adam continuation and persistent-memory append regressions."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_rssm_fusion_integration as fusion_fixture
from turkeyworld.combat_rssm import CombatRSSM, COMBAT_RSSM_PARAMETERS
from turkeyworld.models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
from turkeyworld.renderer_native_phase import _same
from turkeyworld.rssm_fusion_migration import enable_rssm_fusion_checkpoint
from turkeyworld.rssm_combat_memory_migration import (COMBAT_PREFIX,
    enable_combat_memory_checkpoint, prepare_combat_memory_continuation)
from turkeyworld.rssm_migration import configure_rssm_training
from turkeyworld.rssm_moe_protocol import (moe_training_plan, bind_moe_training,
    commit_moe_update, moe_training_horizon)


SMALL_COMBAT = {'combat_rssm_deter_dim': 16, 'combat_rssm_hidden_dim': 16,
    'combat_rssm_stoch': 2, 'combat_rssm_classes': 4, 'combat_rssm_embed_dim': 8,
    'combat_rssm_action_dim': 8, 'combat_rssm_blocks': 2}


def restore(saved):
    model = NeuralWorldModel(WorldModelConfig(**saved['config']))
    model.load_state_dict(saved['model'], strict=True)
    model.rssm_fusion_migration = saved['rssm_fusion_migration']
    if 'combat_rssm_migration' in saved:
        model.combat_rssm_migration = saved['combat_rssm_migration']
    configure_rssm_training(model)
    parameters = dict(model.named_parameters()); names = saved['optimizer_parameter_names']
    groups = [{**{k: v for k, v in group.items() if k != 'params'},
        'params': [parameters[names[i]] for i in group['params']]} for group in saved['optimizer']['param_groups']]
    optimizer = torch.optim.AdamW(groups)
    optimizer.load_state_dict(saved['optimizer'])
    return model, optimizer


class CombatMemoryMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def fixture(self):
        _, _, source = fusion_fixture.RSSMFusionIntegrationTests().fixture()
        source, _ = enable_rssm_fusion_checkpoint(source, config_overrides=fusion_fixture.SMALL_FUSION)
        model, optimizer = restore(source)
        # Train every old head once, including formerly zero MoE readouts.
        sum(p.sum()*1e-6 for p in model.parameters() if p.requires_grad).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        source['model'] = {n: v.detach().clone() for n, v in model.state_dict().items()}
        source['optimizer'] = copy.deepcopy(optimizer.state_dict())
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        path = Path(temporary.name)/'moe_plan.json'
        state = source['run_state']
        plan = moe_training_plan(state, source_checkpoint='initial_moe.pt',
            source_checkpoint_sha256='a'*64, architecture_migration_sha256='b'*64)
        path.write_text(json.dumps(plan), encoding='utf-8')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        bind_moe_training(state, plan, path=path, sha256=sha, loaded_checkpoint_sha256='a'*64)
        # Already at H32 / streak7: adding memory must not restart the curriculum.
        for _ in range(37):
            h = moe_training_horizon(state); state['global_training_step'] += 1
            commit_moe_update(state, step=state['global_training_step'], horizon=h, train_loss=.1)
        source['training_steps'] = source['global_training_step'] = state['global_training_step']
        source['rssm_training'] = copy.deepcopy(state['rssm_training'])
        return model, optimizer, source

    def migrate(self, saved):
        return enable_combat_memory_checkpoint(saved, config_overrides=SMALL_COMBAT)

    def test_full_old_tensors_Adam_mixed_ages_rng_and_H32_streak7_remain_exact(self):
        old, optimizer, saved = self.fixture(); original = copy.deepcopy(saved)
        rng = torch.get_rng_state().clone()
        with patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('CPU migration only')):
            new_saved, report = self.migrate(saved)
        self.assertTrue(_same(saved, original)); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, new_saved['model'][name]) for name, value in saved['model'].items()))
        self.assertTrue(_same(new_saved['optimizer']['state'], saved['optimizer']['state']))
        self.assertTrue(_same(new_saved['optimizer']['param_groups'][:-1], saved['optimizer']['param_groups']))
        self.assertEqual(new_saved['optimizer_parameter_names'][:len(saved['optimizer_parameter_names'])], saved['optimizer_parameter_names'])
        for field in ('rng_state', 'torch_rng_state', 'cuda_rng_state', 'plan_sha256', 'rssm_fusion_migration'):
            self.assertTrue(_same(new_saved[field], saved[field]), field)
        self.assertTrue(_same(new_saved['rssm_training']['moe_training'], saved['rssm_training']['moe_training']))
        self.assertEqual(moe_training_horizon(new_saved['run_state']), 32)
        self.assertEqual(new_saved['rssm_training']['moe_training']['consecutive_below_threshold'], 7)
        self.assertNotIn('combat_rssm_revision', new_saved['rssm_training'])
        self.assertEqual(report['new_optimizer_state_entries'], 0)
        self.assertEqual(report['combined_memory_size'], report['source_memory_size']+report['combat_memory_size'])
        new, new_optimizer = restore(new_saved)
        for model, opt in ((old, optimizer), (new, new_optimizer)):
            sum(p.sum()*1e-6 for p in model.parameters() if p.requires_grad).backward(); opt.step()
        old_parameters, new_parameters = dict(old.named_parameters()), dict(new.named_parameters())
        for name in saved['optimizer_parameter_names']:
            self.assertTrue(torch.equal(old_parameters[name], new_parameters[name]), name)
            self.assertTrue(_same(optimizer.state[old_parameters[name]], new_optimizer.state[new_parameters[name]]))
        self.assertTrue(all(float(new_optimizer.state[new_parameters[n]]['step']) == 1 for n in report['added_parameter_names']))

    def test_nonregistration_source_name_order_retains_named_group_mapping(self):
        _, _, saved = self.fixture()
        # A saved group can follow a valid named order different from Module registration.
        saved = copy.deepcopy(saved); names = saved['optimizer_parameter_names']
        names[0], names[1] = names[1], names[0]
        moments = saved['optimizer']['state']; moments[0], moments[1] = moments[1], moments[0]
        migrated, report = self.migrate(saved)
        self.assertEqual(migrated['optimizer_parameter_names'][:len(names)], names)
        self.assertTrue(_same(migrated['optimizer']['state'], moments))
        self.assertFalse(report['old_parameter_order_is_prefix'])
        restore(migrated)

    def test_default_production_count_meta_and_no_new_attention_token(self):
        with torch.device('meta'):
            module = CombatRSSM()
        self.assertEqual(sum(p.numel() for p in module.parameters()), COMBAT_RSSM_PARAMETERS)
        self.assertEqual(COMBAT_RSSM_PARAMETERS, 20_007_020)
        self.assertEqual(module.memory_size, 1600)
        _, _, saved = self.fixture(); migrated, report = self.migrate(saved)
        self.assertTrue(all(n.startswith(COMBAT_PREFIX) for n in report['added_parameter_names']))
        self.assertTrue(all(not n.startswith('vjepa_backend.fusion_moe.') for n in set(migrated['model'])-set(saved['model'])))

    def test_rejected_partial_source_tampering_and_unrelated_config_leave_source_unchanged(self):
        _, _, saved = self.fixture()
        for label in ('partial', 'names', 'moment', 'schema', 'gate'):
            bad = copy.deepcopy(saved)
            if label == 'partial': bad['training_steps'] -= 1
            elif label == 'names': bad['optimizer_parameter_names'][0] = 'wrong'
            elif label == 'moment': next(iter(bad['optimizer']['state'].values()))['exp_avg'] = torch.zeros(1)
            elif label == 'schema': bad['state_schema'] = {}
            else: bad['run_state']['rssm_training']['moe_training']['consecutive_below_threshold'] = 10
            before = copy.deepcopy(bad)
            with self.assertRaises(ValueError): self.migrate(bad)
            self.assertTrue(_same(before, bad))
        with self.assertRaises(ValueError):
            enable_combat_memory_checkpoint(saved, config_overrides={'rssm_fusion_moe_width': 32})

    def test_atomic_new_checkpoint_and_strict_composed_renderer_lineage(self):
        _, _, saved = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source, target = root/'source.pt', root/'new.pt'
            torch.save(saved, source); before = source.read_bytes()
            report = prepare_combat_memory_continuation(source, target, config_overrides=SMALL_COMBAT)
            self.assertEqual(source.read_bytes(), before)
            self.assertTrue(target.with_suffix('.pt.combat-memory.json').is_file())
            self.assertEqual(report['checkpoint_sha256'], hashlib.sha256(target.read_bytes()).hexdigest())
            model, cp = load_world_checkpoint(target, 'cpu')
            self.assertTrue(model.config.combat_rssm_enabled)
            self.assertEqual(model.vjepa_backend.memory_size, report['combined_memory_size'])
            self.assertEqual(moe_training_horizon(cp['run_state']), 32)
            invalid = torch.load(target, weights_only=False)
            del invalid['combat_rssm_migration']; torch.save(invalid, root/'bad.pt')
            with self.assertRaisesRegex(ValueError, 'Combat RSSM migration lineage'):
                load_world_checkpoint(root/'bad.pt', 'cpu')
            with self.assertRaises(FileExistsError):
                prepare_combat_memory_continuation(source, target, config_overrides=SMALL_COMBAT)


if __name__ == '__main__': unittest.main()
