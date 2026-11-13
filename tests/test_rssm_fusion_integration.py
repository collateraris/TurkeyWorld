"""Real dual-RSSM fusion migration, causal ordering and frozen-renderer lineage."""
import copy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.base_student import build_base_student, export_base_student
from turkeyworld.engine import Game
from turkeyworld.models import NeuralWorldModel, WorldModelConfig, STATE_SCHEMA, load_world_checkpoint, parameter_counts
from turkeyworld.renderer_native_phase import _same
from turkeyworld.renderer_student import (build_renderer_student, export_renderer_student,
                                         load_renderer_student, _hash_tensors)
from turkeyworld.rssm_fusion_migration import (enable_rssm_fusion_checkpoint,
                                             prepare_rssm_fusion_continuation, FUSION_PREFIX)
from turkeyworld.rssm_migration import configure_rssm_training


SMALL_FUSION = {'rssm_fusion_moe_width': 16, 'rssm_fusion_moe_depth': 1,
                'rssm_fusion_moe_heads': 2, 'rssm_fusion_moe_expert_hidden': 32}


def raw_dual():
    config = WorldModelConfig(kind='teacher', architecture='dreamer_rssm', image_size=32,
        endless=True, navigation_only=False, spatial_entities=True, spatial_combat=True,
        player_motion_mode='axis_fraction', game_training_v5=True,
        renderer_mode='flow_matching', renderer_codec='vqgan',
        renderer_codec_config=dict(image_size=32, base_channels=4, latent_channels=8,
                                  embedding_dim=4, n_embed=8, data_init=False),
        flow_steps=1, flow_width=16, flow_depth=1, flow_heads=2,
        vjepa_width=16, vjepa_readout_mlp_ratio=2, rssm_deter_dim=16, rssm_hidden_dim=16,
        rssm_stoch=2, rssm_classes=4, rssm_embed_dim=16, rssm_action_dim=20, rssm_blocks=2,
        enemy_rssm_enabled=True, enemy_rssm_deter_dim=16, enemy_rssm_hidden_dim=16,
        enemy_rssm_stoch=2, enemy_rssm_classes=4, enemy_rssm_embed_dim=8,
        enemy_rssm_action_dim=8, enemy_rssm_blocks=2)
    model = NeuralWorldModel(config)
    model.flow_renderer.codec_loaded.fill_(True)
    return model


class RSSMFusionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        model = raw_dual(); configure_rssm_training(model)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        sum(p.square().sum()*1e-6 for p in model.parameters() if p.requires_grad).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        q = {'combat_completed': 210, 'memory_size': model.vjepa_backend.memory_size,
             'loss_gated_curriculum': {'updates': 210, 'current_horizon': 128, 'consecutive_below_threshold': 7},
             'combat_recovery_revision': {'sampler_state': {'draws': 34567, 'rng': np.random.default_rng(9).bit_generator.state}},
             'loss_balance_state': {'frozen_old_combat_bank': [1., 2., 3.]}}
        run = {'run_id': 'old', 'global_training_step': 37, 'current': 37,
               'rssm_training': copy.deepcopy(q), 'training_protocol': {'old_batch': '64x4'},
               'loss_protocol': 'old_pure_xy', 'objective_history': [{'kind': 'old', 'step': 35}]}
        saved = {'stage': 'world', 'state_schema': dict(STATE_SCHEMA), 'config': asdict(model.config),
                 'model': {k: v.detach().clone() for k, v in model.state_dict().items()},
                 'optimizer': copy.deepcopy(optimizer.state_dict()), 'optimizer_parameter_names': names,
                 'training_steps': 37, 'global_training_step': 37, 'run_state': run,
                 'rssm_training': copy.deepcopy(q), 'plan_sha256': 'old_immutable_plan',
                 'rng_state': np.random.default_rng(11).bit_generator.state,
                 'torch_rng_state': torch.get_rng_state().clone(), 'cuda_rng_state': []}
        return model, optimizer, saved

    def migrate(self, saved):
        return enable_rssm_fusion_checkpoint(saved, config_overrides=SMALL_FUSION)

    def restore(self, saved):
        model = NeuralWorldModel(WorldModelConfig(**saved['config']))
        model.load_state_dict(saved['model'], strict=True)
        model.rssm_fusion_migration = saved['rssm_fusion_migration']
        configure_rssm_training(model)
        params = dict(model.named_parameters()); names = saved['optimizer_parameter_names']
        groups = [{**{k: v for k, v in group.items() if k != 'params'},
                   'params': [params[names[i]] for i in group['params']]}
                  for group in saved['optimizer']['param_groups']]
        optimizer = torch.optim.AdamW(groups); optimizer.load_state_dict(saved['optimizer'])
        return model, optimizer

    @staticmethod
    def controls():
        game = Game(seed=4, endless=True)
        state = torch.from_numpy(np.stack([game.state_vector(), game.state_vector()])).float()
        # Include a ready respawn in the second actor, exercising the hierarchy.
        state[1, 12:14] = 0; state[1, 16] = 5.
        grid = torch.from_numpy(np.repeat(game.map[None], 2, axis=0)).long()
        rgb = torch.from_numpy(np.repeat(game.render(32, overlays=False)[None], 2, axis=0)).permute(0, 3, 1, 2).float()/255
        action = torch.tensor([[.6, .2, .1, 1.], [.2, -.1, -.2, 0.]])
        return state, grid, rgb, action

    def test_migration_preserves_all_old_tensors_mixed_Adam_ages_rng_clocks_and_next_old_Adam_step(self):
        old, optimizer, saved = self.fixture(); source_before = copy.deepcopy(saved)
        rng = torch.get_rng_state().clone()
        with patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('CPU-only initialization')):
            migrated, report = self.migrate(saved)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(_same(saved, source_before))
        self.assertTrue(all(torch.equal(v, migrated['model'][n]) for n, v in saved['model'].items()))
        self.assertTrue(_same(saved['optimizer']['state'], migrated['optimizer']['state']))
        self.assertTrue(_same(saved['optimizer']['param_groups'], migrated['optimizer']['param_groups'][:-1]))
        self.assertEqual(migrated['optimizer_parameter_names'][:len(saved['optimizer_parameter_names'])], saved['optimizer_parameter_names'])
        self.assertEqual(migrated['rssm_training'], saved['rssm_training'])
        for field in ('rng_state', 'torch_rng_state', 'cuda_rng_state', 'plan_sha256'):
            self.assertTrue(_same(migrated[field], saved[field]))
        self.assertEqual(report['combined_memory_size'], old.vjepa_backend.memory_size)
        self.assertEqual(report['new_optimizer_state_entries'], 0)
        new, new_optimizer = self.restore(migrated)
        self.assertTrue(all(n.startswith(FUSION_PREFIX) for n in report['added_parameter_names']))
        for model, opt in ((old, optimizer), (new, new_optimizer)):
            opt.zero_grad(set_to_none=True)
            sum(p.square().sum()*1e-6 for p in model.parameters() if p.requires_grad).backward(); opt.step()
        p_old, p_new = dict(old.named_parameters()), dict(new.named_parameters())
        for name in saved['optimizer_parameter_names']:
            self.assertTrue(torch.equal(p_old[name], p_new[name]))
            self.assertTrue(_same(optimizer.state[p_old[name]], new_optimizer.state[p_new[name]]))
        self.assertTrue(all(float(new_optimizer.state[p_new[n]]['step']) == 1 for n in report['added_parameter_names']))

    def test_zero_residual_16_tick_own_rollout_and_stochastic_rng_stream_are_exact(self):
        old, _, saved = self.fixture(); migrated, _ = self.migrate(saved); new, _ = self.restore(migrated)
        old.eval(); new.eval()
        state, grid, rgb, action = self.controls()
        with torch.no_grad():
            maps = [m.encode_map(grid) for m in (old, new)]
            memory = [m.initialize_from_image(rgb, state, mp) for m, mp in zip((old, new), maps)]
            self.assertTrue(torch.equal(memory[0], memory[1]))
            states = [state.clone(), state.clone()]
            for _ in range(16):
                a = old.transition(states[0], action, maps[0], memory[0])
                b = new.transition(states[1], action, maps[1], memory[1])
                self.assertTrue(all(_same(value, b[key]) for key, value in a.items()))
                states, memory = [a['state'], b['state']], [a['memory'], b['memory']]
        old.train(); new.train()
        rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng); a = old.transition(state, action, maps[0], memory[0])
        after_a = torch.get_rng_state().clone()
        torch.set_rng_state(rng); b = new.transition(state, action, maps[1], memory[1])
        self.assertTrue(all(_same(value, b[key]) for key, value in a.items()))
        self.assertTrue(torch.equal(torch.get_rng_state(), after_a))

    def test_previous_memories_fused_once_before_Prior_and_corrected_player_pose_conditions_enemy(self):
        _, _, saved = self.fixture(); migrated, _ = self.migrate(saved); model, _ = self.restore(migrated)
        model.eval(); backend = model.vjepa_backend
        with torch.no_grad(): backend.fusion_moe.player_head.bias.copy_(torch.tensor([4., -4.]))
        state, grid, _, action = self.controls(); maps = model.encode_map(grid)
        memory = backend.initial_memory(2)
        order = []
        fusion, player, enemy = backend.fusion_moe.forward, backend.core.imagine_step, backend.enemy_rssm.prior
        def fuse(*a, **kw): order.append('fusion'); return fusion(*a, **kw)
        def play(*a, **kw): order.append('player'); return player(*a, **kw)
        def foe(*a, **kw): order.append('enemy'); return enemy(*a, **kw)
        with patch.object(backend.fusion_moe, 'forward', side_effect=fuse) as probe, \
             patch.object(backend.core, 'imagine_step', side_effect=play), \
             patch.object(backend.enemy_rssm, 'prior', side_effect=foe) as foe_probe:
            result = model.transition(state, action, maps, memory)
        self.assertEqual(order, ['fusion', 'player', 'enemy'])
        self.assertEqual(probe.call_count, 1)
        self.assertTrue(torch.equal(probe.call_args.args[0], memory[:, :backend.core.memory_size]))
        self.assertTrue(torch.equal(probe.call_args.args[1], memory[:, backend.core.memory_size:]))
        expected = state[:, :2]+result['raw_player_delta']
        self.assertTrue(torch.equal(foe_probe.call_args.args[3], expected.detach()))
        self.assertTrue(torch.equal(result['enemy_rssm_conditioned_player_xy'], expected.detach()))

    def test_future_RGB_only_changes_posterior_and_never_recomputes_or_changes_prior_fusion(self):
        _, _, saved = self.fixture(); migrated, _ = self.migrate(saved); model, _ = self.restore(migrated)
        model.eval(); backend = model.vjepa_backend
        state, grid, rgb, action = self.controls(); maps = model.encode_map(grid); memory = backend.initial_memory(2)
        with patch.object(backend.fusion_moe, 'forward', wraps=backend.fusion_moe.forward) as fusion:
            a = backend.observe_step(state, action, maps, memory, rgb, sample=False)
            b = backend.observe_step(state, action, maps, memory, 1-rgb, sample=False)
        self.assertEqual(fusion.call_count, 2)
        self.assertTrue(_same(a['prior_result'], b['prior_result']))
        self.assertIs(a['rssm_fusion_moe'], a['prior_result']['rssm_fusion_moe'])
        self.assertFalse(torch.equal(a['rssm_posterior_logits'], b['rssm_posterior_logits']))
        self.assertTrue(torch.equal(a['enemy_rssm_conditioned_player_xy'], a['prior_result']['enemy_rssm_conditioned_player_xy']))

    def test_interaction_residual_can_send_gradients_to_both_continuous_RSSM_cores(self):
        _, _, saved = self.fixture(); migrated, _ = self.migrate(saved); model, _ = self.restore(migrated)
        model.eval(); backend = model.vjepa_backend
        with torch.no_grad():
            # A constant row would intentionally annihilate LayerNorm's
            # centered role features and cannot test the memory gradient.
            backend.fusion_moe.combat_head.weight[0].copy_(torch.linspace(.01, .07, 16))
            backend.fusion_moe.combat_head.weight[1].copy_(torch.linspace(.04, -.02, 16))
        state, grid, _, action = self.controls(); maps = model.encode_map(grid)
        a = model.transition(state, action, maps, backend.initial_memory(2))
        b = model.transition(a['state'].detach(), action, maps, a['memory'])
        b['target_logits'][:, 0].sum().backward()
        for core in (backend.core, backend.enemy_rssm.core):
            gradients = [p.grad for p in core.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        self.assertTrue(all(p.grad is None for p in model.flow_renderer.parameters()))

    def test_strict_legacy_renderer_reuse_requires_explicit_fusion_proof_and_retains_moe_config(self):
        model, _, saved = self.fixture(); migrated, report = self.migrate(saved)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = build_base_student(model); export_base_student(model, base, root/'base.pt', training_steps=3)
            model.base_student = base; model.base_student_bundle = torch.load(root/'base.pt', weights_only=False)
            student = build_renderer_student(model, steps=1, width=16, depth=1, heads=2, mlp_hidden=32)
            export_renderer_student(model, student, root/'renderer.pt', training_steps=9, metrics={'source_quality_passed': False})
            target, _ = self.restore(migrated); del target.rssm_fusion_migration
            initial_hash = _hash_tensors(target.state_dict())
            with self.assertRaisesRegex(ValueError, 'fusion MoE migration lineage'):
                load_renderer_student(target, root/'renderer.pt')
            self.assertEqual(_hash_tensors(target.state_dict()), initial_hash)
            target.rssm_fusion_migration = report
            preserved = {n: v.clone() for n, v in target.state_dict().items() if not n.startswith('flow_renderer.')}
            load_renderer_student(target, root/'renderer.pt')
            self.assertTrue(target.config.rssm_fusion_moe_enabled)
            self.assertTrue(all(torch.equal(v, target.state_dict()[n]) for n, v in preserved.items()))
            self.assertEqual(target.vjepa_backend.memory_size, report['combined_memory_size'])

    def test_source_tampering_or_nonappend_architecture_changes_rejected_and_file_migration_separate(self):
        _, _, saved = self.fixture()
        for label in ('names', 'moment', 'partial', 'extra', 'schema'):
            bad = copy.deepcopy(saved)
            if label == 'names': bad['optimizer_parameter_names'][0] = 'wrong'
            elif label == 'moment': next(iter(bad['optimizer']['state'].values()))['exp_avg'] = torch.ones(1)
            elif label == 'partial': bad['training_steps'] -= 1
            elif label == 'extra': bad['model'][FUSION_PREFIX+'fake'] = torch.zeros(1)
            else: bad['state_schema'] = {}
            original = copy.deepcopy(bad)
            with self.assertRaises(ValueError): self.migrate(bad)
            self.assertTrue(_same(bad, original))
        with self.assertRaises(ValueError): enable_rssm_fusion_checkpoint(saved, config_overrides={'enemy_rssm_deter_dim': 32})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); old, new = root/'old.pt', root/'new.pt'; torch.save(saved, old)
            old_bytes = old.read_bytes()
            report = prepare_rssm_fusion_continuation(old, new, config_overrides=SMALL_FUSION)
            self.assertEqual(old.read_bytes(), old_bytes)
            restored, cp = load_world_checkpoint(new, 'cpu')
            self.assertTrue(restored.config.rssm_fusion_moe_enabled)
            self.assertEqual(cp['training_steps'], 37)
            self.assertEqual(report['combined_memory_size'], restored.vjepa_backend.memory_size)
            invalid = torch.load(new, weights_only=False); del invalid['rssm_fusion_migration']; torch.save(invalid, root/'bad.pt')
            with self.assertRaisesRegex(ValueError, 'fusion MoE migration lineage'): load_world_checkpoint(root/'bad.pt', 'cpu')
            with self.assertRaises(FileExistsError): prepare_rssm_fusion_continuation(old, new, config_overrides=SMALL_FUSION)

    def test_production_default_adds3118099_parameters_and_keeps3632_memory_without_allocation(self):
        config = raw_dual().config
        config = replace(config, vjepa_width=512, rssm_deter_dim=1008, rssm_hidden_dim=1024,
            rssm_stoch=32, rssm_classes=32, rssm_embed_dim=512, rssm_action_dim=516, rssm_blocks=8,
            enemy_rssm_deter_dim=1344, enemy_rssm_hidden_dim=1408, enemy_rssm_stoch=8,
            enemy_rssm_classes=32, enemy_rssm_embed_dim=128, enemy_rssm_action_dim=128,
            enemy_rssm_blocks=4, rssm_fusion_moe_enabled=True)
        with torch.device('meta'): model = NeuralWorldModel(config)
        self.assertEqual(model.vjepa_backend.memory_size, 3632)
        self.assertEqual(sum(p.numel() for p in model.vjepa_backend.fusion_moe.parameters()), 3118099)
        self.assertEqual(parameter_counts(model)['rssm_fusion_moe'], 3118099)


if __name__ == '__main__':
    unittest.main()
