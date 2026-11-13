import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_rssm_combat_memory_migration as combat_fixture
import test_rssm_fusion_integration as fusion_fixture
from turkeyworld.models import NeuralWorldModel, WorldModelConfig, load_world_checkpoint
from turkeyworld.renderer_native_phase import _same
from turkeyworld.rssm_combat_memory_migration import enable_combat_memory_checkpoint
from turkeyworld.rssm_five_migration import (FIVE_PREFIX, enable_rssm_five_checkpoint,
    prepare_rssm_five_continuation, FiveRoleDiagnostics, report_sha)
from turkeyworld.rssm_migration import configure_rssm_training
from turkeyworld.train_rssm_moe import restore_optimizer


class RSSMFiveMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def fixture(self):
        helper = combat_fixture.CombatMemoryMigrationTests()
        self.addCleanup(helper.doCleanups)
        _, _, saved = helper.fixture()
        saved, _ = enable_combat_memory_checkpoint(saved, config_overrides=combat_fixture.SMALL_COMBAT)
        model = NeuralWorldModel(WorldModelConfig(**saved['config']))
        model.load_state_dict(saved['model'], strict=True); configure_rssm_training(model)
        optimizer, _, _ = restore_optimizer(model, saved)
        # Create every new combat moment and retain differing old group ages.
        sum(p.square().mean() + p.mean() * .001 for p in model.parameters() if p.requires_grad).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        saved['model'] = {n: p.detach().clone() for n, p in model.state_dict().items()}
        saved['optimizer'] = copy.deepcopy(optimizer.state_dict())
        return model, optimizer, saved

    @staticmethod
    def restore(saved):
        model = NeuralWorldModel(WorldModelConfig(**saved['config']))
        model.load_state_dict(saved['model'], strict=True)
        for key in ('rssm_fusion_migration', 'combat_rssm_migration', 'rssm_five_migration'):
            if key in saved: setattr(model, key, saved[key])
        configure_rssm_training(model)
        optimizer, _, _ = restore_optimizer(model, saved)
        return model, optimizer

    def test_old_tensors_all_named_mixed_Adam_groups_rng_curriculum_and_samplers_preserved(self):
        old, oldopt, saved = self.fixture(); before = copy.deepcopy(saved)
        rng = torch.get_rng_state().clone()
        with patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('No GPU in CPU migration')):
            migrated, report = enable_rssm_five_checkpoint(saved)
        self.assertTrue(_same(saved, before)); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, migrated['model'][n]) for n, v in saved['model'].items()))
        self.assertTrue(_same(saved['optimizer']['state'], migrated['optimizer']['state']))
        self.assertTrue(_same(saved['optimizer']['param_groups'], migrated['optimizer']['param_groups'][:-1]))
        self.assertEqual(saved['optimizer_parameter_names'],
                         migrated['optimizer_parameter_names'][:len(saved['optimizer_parameter_names'])])
        for key in ('rng_state', 'torch_rng_state', 'cuda_rng_state', 'rssm_training', 'plan_sha256'):
            self.assertTrue(_same(saved[key], migrated[key]), key)
        self.assertTrue(_same(saved['run_state']['rssm_training'], migrated['run_state']['rssm_training']))
        self.assertEqual(report['revision_sha256'], report_sha(report))
        self.assertEqual(report['memory_size'], old.vjepa_backend.memory_size)
        self.assertTrue(all(n.startswith(FIVE_PREFIX) for n in report['added_parameter_names']))
        self.assertEqual(report['new_optimizer_state_entries'], 0)
        new, newopt = self.restore(migrated)
        for model, opt in ((old, oldopt), (new, newopt)):
            opt.zero_grad(set_to_none=True)
            sum(p.square().mean() + p.mean() * .001 for p in model.parameters() if p.requires_grad).backward()
            opt.step()
        a, b = dict(old.named_parameters()), dict(new.named_parameters())
        for name in saved['optimizer_parameter_names']:
            self.assertTrue(torch.equal(a[name], b[name]), name)
            self.assertTrue(_same(oldopt.state[a[name]], newopt.state[b[name]]))
        self.assertTrue(all(float(newopt.state[b[n]]['step']) == 1 for n in report['added_parameter_names']))

    def test_zero_task_heads_preserve_one_step_state_memory_features_rgb_and_stochastic_rng(self):
        old, _, saved = self.fixture(); migrated, _ = enable_rssm_five_checkpoint(saved)
        new, _ = self.restore(migrated)
        state, grid, rgb, action = fusion_fixture.RSSMFusionIntegrationTests.controls()
        for mode in (False, True):
            for model in (old, new): configure_rssm_training(model); model.train(mode)
            with torch.no_grad():
                maps = [model.encode_map(grid) for model in (old, new)]
                initial_rng = torch.get_rng_state().clone(); memories = []
                for model, fmap in zip((old, new), maps):
                    torch.set_rng_state(initial_rng)
                    memories.append(model.initialize_from_image(rgb, state, fmap))
                self.assertTrue(torch.equal(memories[0], memories[1]))
                torch.set_rng_state(initial_rng); a = old.transition(state, action, maps[0], memories[0])
                after = torch.get_rng_state().clone()
                torch.set_rng_state(initial_rng); b = new.transition(state, action, maps[1], memories[1])
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                for key in ('state', 'memory', 'features'):
                    self.assertTrue(torch.equal(a[key], b[key]), (mode, key))
                torch.set_rng_state(initial_rng); image_a = old.render(a['state'], a['features'], maps[0])
                torch.set_rng_state(initial_rng); image_b = new.render(b['state'], b['features'], maps[1])
                self.assertTrue(torch.equal(image_a, image_b))

    def test_actual_forward_diagnostics_weight_actor_queries_without_grad_or_rng_side_effects(self):
        _, _, saved = self.fixture(); migrated, _ = enable_rssm_five_checkpoint(saved)
        model, _ = self.restore(migrated); module = model.vjepa_backend.five_role_fusion
        collector = FiveRoleDiagnostics(); hook = module.register_forward_hook(collector)
        collector.begin(); rng = torch.get_rng_state().clone()
        a = module(torch.ones(2, 3, module.width), torch.zeros(2, 20), torch.zeros(2, 4))
        b = module(torch.ones(1, 3, module.width), torch.zeros(1, 20), torch.zeros(1, 4))
        report = collector.summary(); hook.remove()
        expected = (a['router_probabilities'].sum((0, 1)) + b['router_probabilities'].sum((0, 1))) / 15
        self.assertTrue(torch.allclose(torch.tensor(report['router_importance']), expected.detach()))
        self.assertEqual(report['forward_calls'], 2); self.assertEqual(report['role_query_count'], 15)
        self.assertEqual(report['player_mode_actor_ticks'], 3)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_atomic_separate_checkpoint_strict_lineage_and_reject_tamper_without_source_mutation(self):
        _, _, saved = self.fixture()
        for mode in ('partial', 'missing_name', 'wrong_moment', 'already_enabled', 'changed_gate', 'reordered_names'):
            bad = copy.deepcopy(saved)
            if mode == 'partial': bad['global_training_step'] -= 1
            elif mode == 'missing_name': bad['optimizer_parameter_names'][0] = 'wrong'
            elif mode == 'wrong_moment': next(iter(bad['optimizer']['state'].values()))['exp_avg'] = torch.zeros(1)
            elif mode == 'already_enabled': bad['config']['rssm_five_role_enabled'] = True
            elif mode == 'changed_gate': bad['run_state']['rssm_training']['moe_training']['consecutive_below_threshold'] = 10
            else:
                names = bad['optimizer_parameter_names']; names[0], names[1] = names[1], names[0]
                moments = bad['optimizer']['state']; moments[0], moments[1] = moments[1], moments[0]
            before = copy.deepcopy(bad)
            with self.subTest(mode=mode), self.assertRaises(ValueError): enable_rssm_five_checkpoint(bad)
            self.assertTrue(_same(before, bad))
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source.pt', Path(directory) / 'target.pt'
            torch.save(saved, source); before = source.read_bytes()
            proof = prepare_rssm_five_continuation(source, target)
            self.assertEqual(source.read_bytes(), before); self.assertTrue(target.with_suffix('.pt.five.json').exists())
            model, metadata = load_world_checkpoint(target, 'cpu')
            self.assertEqual(metadata['rssm_five_migration']['source_checkpoint_sha256'], proof['source_checkpoint_sha256'])
            self.assertTrue(model.config.rssm_five_role_enabled)
            broken = torch.load(target, weights_only=False); broken.pop('rssm_five_migration')
            torch.save(broken, Path(directory) / 'bad.pt')
            with self.assertRaises(ValueError): load_world_checkpoint(Path(directory) / 'bad.pt', 'cpu')
            with self.assertRaises(FileExistsError): prepare_rssm_five_continuation(source, target)


if __name__ == '__main__': unittest.main()
