"""Persistent third memory integration, causal readouts and strict image lineage."""
import copy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from turkeyworld.base_student import build_base_student, export_base_student
from turkeyworld.models import NeuralWorldModel, STATE_SCHEMA, load_world_checkpoint, parameter_counts
from turkeyworld.renderer_native_phase import _same
from turkeyworld.renderer_student import (build_renderer_student, export_renderer_student,
    load_renderer_student, renderer_condition_identity, _hash_tensors)
from turkeyworld.rssm_objectives import _valid_result
import test_rssm_fusion_integration as fusion_helpers


COMBAT_SMALL = dict(combat_rssm_enabled=True, combat_rssm_deter_dim=16,
    combat_rssm_hidden_dim=16, combat_rssm_stoch=2, combat_rssm_classes=4,
    combat_rssm_embed_dim=8, combat_rssm_action_dim=8, combat_rssm_blocks=2)


class CombatRSSMBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        helper = fusion_helpers.RSSMFusionIntegrationTests()
        _, _, saved = helper.fixture()
        cls.saved, _ = helper.migrate(saved)

    def pair(self):
        helper = fusion_helpers.RSSMFusionIntegrationTests()
        old, _ = helper.restore(self.saved)
        new = NeuralWorldModel(replace(old.config, **COMBAT_SMALL))
        missing, extra = new.load_state_dict(old.state_dict(), strict=False)
        fusion_helpers.configure_rssm_training(new)
        self.assertFalse(extra)
        self.assertTrue(missing and all(n.startswith('vjepa_backend.combat_rssm.') for n in missing))
        new.rssm_fusion_migration = copy.deepcopy(old.rssm_fusion_migration)
        backend = new.vjepa_backend
        previous, migrated = asdict(old.config), asdict(new.config)
        new.combat_rssm_migration = dict(format_version=1, kind='append_persistent_combat_rssm',
            old_model_tensors_bitexact=True, frozen_render_assets_bitexact=True,
            zero_residual_initialization=True, source_config=previous, target_config=migrated,
            allowed_config_changes=sorted(k for k in previous if previous[k] != migrated[k]),
            added_parameter_names=['vjepa_backend.combat_rssm.'+n for n, _ in backend.combat_rssm.named_parameters()],
            added_parameter_count=sum(p.numel() for p in backend.combat_rssm.parameters()),
            source_memory_size=old.vjepa_backend.memory_size, combined_memory_size=backend.memory_size,
            source_renderer_conditioning_identity=renderer_condition_identity(new))
        return old, new

    def test_zero_heads_keep_16_tick_old_prefix_outputs_and_global_sampling_stream_exact(self):
        old, new = self.pair()
        old_size = old.vjepa_backend.memory_size
        old_names = list(dict(old.named_parameters()))
        self.assertEqual(list(dict(new.named_parameters()))[:len(old_names)], old_names)
        self.assertTrue(all(torch.equal(v, new.state_dict()[n]) for n, v in old.state_dict().items()))
        state, grid, rgb, action = fusion_helpers.RSSMFusionIntegrationTests.controls()
        for training, observed in ((False, 1), (True, 1), (True, 2)):
            old.train(training); new.train(training)
            maps = [m.encode_map(grid) for m in (old, new)]
            rng = torch.get_rng_state().clone()
            observed_rgb = rgb if observed == 1 else torch.stack([rgb, 1-rgb], 1)
            observed_state = state if observed == 1 else state[:, None].expand(-1, 2, -1)
            with torch.no_grad():
                memory_a = old.initialize_from_image(observed_rgb, observed_state, maps[0]); after_a = torch.get_rng_state().clone()
                torch.set_rng_state(rng)
                memory_b = new.initialize_from_image(observed_rgb, observed_state, maps[1])
                self.assertTrue(torch.equal(after_a, torch.get_rng_state()))
                self.assertTrue(torch.equal(memory_a, memory_b[:, :old_size]))
                current_a, current_b = state.clone(), state.clone()
                for step in range(16):
                    rng = torch.get_rng_state().clone()
                    a = old.transition(current_a, action, maps[0], memory_a)
                    after_a = torch.get_rng_state().clone(); torch.set_rng_state(rng)
                    b = new.transition(current_b, action, maps[1], memory_b)
                    self.assertTrue(torch.equal(after_a, torch.get_rng_state()), (training, step))
                    for key in a:
                        if key != 'memory': self.assertTrue(_same(a[key], b[key]), (training, step, key))
                    self.assertTrue(torch.equal(a['memory'], b['memory'][:, :old_size]))
                    current_a, current_b, memory_a, memory_b = a['state'], b['state'], a['memory'], b['memory']

    def test_causal_order_uses_current_attached_actor_contexts_and_previous_combat_fusion_only(self):
        _, model = self.pair(); model.eval(); backend = model.vjepa_backend
        state, grid, _, action = fusion_helpers.RSSMFusionIntegrationTests.controls()
        maps = model.encode_map(grid); memory = model.initial_memory(2)
        calls = []
        originals = [backend.fusion_moe.forward, backend.core.imagine_step,
                     backend.enemy_rssm.prior, backend.combat_rssm.prior]
        def traced(index):
            def call(*args, **kwargs):
                calls.append(index); return originals[index](*args, **kwargs)
            return call
        with patch.object(backend.fusion_moe, 'forward', side_effect=traced(0)) as fusion, \
             patch.object(backend.core, 'imagine_step', side_effect=traced(1)), \
             patch.object(backend.enemy_rssm, 'prior', side_effect=traced(2)), \
             patch.object(backend.combat_rssm, 'prior', side_effect=traced(3)) as combat:
            result = model.transition(state, action, maps, memory)
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertTrue(torch.equal(fusion.call_args.kwargs['combat_context'], torch.zeros(2, 16)))
        self.assertTrue(torch.equal(combat.call_args.args[0], memory[:, -24:]))
        self.assertIs(combat.call_args.kwargs['enemy_context'], result['enemy_rssm_latent']['pooled'])
        self.assertTrue(combat.call_args.kwargs['player_context'].requires_grad)
        self.assertTrue(combat.call_args.kwargs['enemy_context'].requires_grad)
        self.assertTrue(torch.equal(combat.call_args.args[3], result['enemy_rssm_conditioned_player_xy']))
        player, enemy = backend.split_memory(result['memory'])
        p, e, c = backend.split_all_memory(result['memory'])
        self.assertTrue(torch.equal(player, p)); self.assertTrue(torch.equal(enemy, e))
        self.assertEqual(c.shape, (2, 24)); self.assertEqual(enemy.shape, (2, 24))

    def test_combat_memory_advances_when_both_actors_dead_without_observation_or_reset(self):
        _, model = self.pair(); model.eval(); backend = model.vjepa_backend
        state, grid, _, action = fusion_helpers.RSSMFusionIntegrationTests.controls()
        state[:, 4] = 0; state[:, 12:14] = 0
        maps = model.encode_map(grid); memory = model.initial_memory(2)
        with patch.object(backend.combat_rssm, 'observe_initial', side_effect=AssertionError('No reset')), \
             patch.object(backend.obs_encoder, 'forward', side_effect=AssertionError('No future observation')):
            with torch.no_grad():
                a = model.transition(state, action, maps, memory)
                b = model.transition(a['state'], action, maps, a['memory'])
        self.assertFalse(torch.equal(a['combat_rssm_h'], b['combat_rssm_h']))
        self.assertEqual(b['memory'].shape, (2, 72))
        self.assertEqual(b['memory'].dtype, torch.float32)
        _valid_result(b, (2, 72), 'continuous empty combat')
        bad = {**b, 'combat_rssm_h': b['combat_rssm_h']*float('nan')}
        with self.assertRaisesRegex(FloatingPointError, 'combat_rssm_h'): _valid_result(bad, (2, 72), 'probe')

    def test_combat_posterior_reuses_exact_single_prior_and_future_rgb_cannot_change_it(self):
        _, model = self.pair(); model.eval(); backend = model.vjepa_backend
        state, grid, rgb, action = fusion_helpers.RSSMFusionIntegrationTests.controls()
        maps = model.encode_map(grid); memory = model.initial_memory(2)
        with patch.object(backend.combat_rssm, 'prior', wraps=backend.combat_rssm.prior) as prior, \
             patch.object(backend.combat_rssm, 'posterior', wraps=backend.combat_rssm.posterior) as post:
            a = backend.observe_step(state, action, maps, memory, rgb, sample=False)
        self.assertEqual(prior.call_count, 1); self.assertEqual(post.call_count, 1)
        self.assertIs(post.call_args.args[0], a['prior_result']['combat_rssm_latent'])
        b = backend.observe_step(state, action, maps, memory, 1-rgb, sample=False)
        self.assertTrue(_same(a['prior_result'], b['prior_result']))
        self.assertFalse(torch.equal(a['combat_rssm_posterior_logits'], b['combat_rssm_posterior_logits']))
        self.assertTrue(torch.equal(a['combat_rssm_prior_logits'], a['prior_result']['combat_rssm_prior_logits']))

    def test_previous_combat_memory_reaches_existing_shared_query_and_actor_head_after_adapter_learns(self):
        _, model = self.pair(); model.eval(); backend = model.vjepa_backend
        with torch.no_grad():
            backend.combat_rssm.fusion_adapter.weight.copy_(torch.eye(16)*.2)
            backend.fusion_moe.player_head.weight[0].copy_(torch.linspace(.01, .12, 16))
        state, grid, _, action = fusion_helpers.RSSMFusionIntegrationTests.controls()
        maps = model.encode_map(grid); memory = model.initial_memory(2).detach().requires_grad_()
        result = model.transition(state, action, maps, memory)
        result['player_pass_logits'][:, 0].sum().backward()
        self.assertGreater(float(memory.grad[:, -24:].abs().sum()), 0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in backend.combat_rssm.readout.parameters()
                               if p.grad is not None), 0)
        self.assertEqual(backend.fusion_moe.token_types.shape, (10, 16))

    def test_nonzero_combat_head_backpropagates_through_both_current_cores_and_own_memory(self):
        _, model = self.pair(); model.eval(); backend = model.vjepa_backend
        with torch.no_grad():
            backend.combat_rssm.task_head[-1].weight[0].copy_(torch.linspace(.01, .12, 16))
        state, grid, _, action = fusion_helpers.RSSMFusionIntegrationTests.controls()
        maps = model.encode_map(grid); memory = model.initial_memory(2).detach().requires_grad_()
        result = model.transition(state, action, maps, memory)
        result['combat_rssm_latent']['target_residual'][:, 0].sum().backward()
        for core in (backend.core, backend.enemy_rssm.core, backend.combat_rssm.core):
            grads = [p.grad for p in core.parameters() if p.grad is not None]
            self.assertTrue(grads); self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
        self.assertGreater(float(memory.grad[:, -24:].abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.flow_renderer.parameters()))

    def test_strict_checkpoint_and_old_compact_base_rf_require_composite_migration_lineage(self):
        old, model = self.pair()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = build_base_student(old); export_base_student(old, base, root/'base.pt', training_steps=3)
            old.base_student = base; old.base_student_bundle = torch.load(root/'base.pt', weights_only=False)
            renderer = build_renderer_student(old, steps=1, width=16, depth=1, heads=2, mlp_hidden=32)
            export_renderer_student(old, renderer, root/'renderer.pt', training_steps=9)
            report = model.combat_rssm_migration; del model.combat_rssm_migration
            before = _hash_tensors(model.state_dict())
            with self.assertRaisesRegex(ValueError, 'persistent Combat RSSM migration lineage'):
                load_renderer_student(model, root/'renderer.pt')
            self.assertEqual(before, _hash_tensors(model.state_dict()))
            model.combat_rssm_migration = report
            load_renderer_student(model, root/'renderer.pt')
            self.assertTrue(model.config.combat_rssm_enabled)
            cp = dict(stage='world', state_schema=STATE_SCHEMA, config=asdict(model.config), model=model.state_dict(),
                rssm_fusion_migration=model.rssm_fusion_migration, combat_rssm_migration=report)
            # A standalone world checkpoint retains the original flow config;
            # the compact renderer remains its separately pinned asset.
            fresh = NeuralWorldModel(model.config)
            cp['model'] = {n: v for n, v in cp['model'].items() if not n.startswith('base_student.')}
            fresh.load_state_dict(cp['model'], strict=True)
            torch.save(cp, root/'world.pt')
            loaded, _ = load_world_checkpoint(root/'world.pt', 'cpu')
            self.assertTrue(loaded.config.combat_rssm_enabled)
            self.assertEqual(loaded.vjepa_backend.memory_size, 72)
            invalid = copy.deepcopy(cp); invalid['combat_rssm_migration']['source_memory_size'] += 1
            torch.save(invalid, root/'bad.pt')
            with self.assertRaisesRegex(ValueError, 'Persistent Combat RSSM lineage'):
                load_world_checkpoint(root/'bad.pt', 'cpu')

    def test_production_meta_count_and_5232_memory(self):
        config = replace(fusion_helpers.raw_dual().config, vjepa_width=512,
            rssm_deter_dim=1008, rssm_hidden_dim=1024, rssm_stoch=32, rssm_classes=32,
            rssm_embed_dim=512, rssm_action_dim=516, rssm_blocks=8,
            enemy_rssm_deter_dim=1344, enemy_rssm_hidden_dim=1408, enemy_rssm_stoch=8,
            enemy_rssm_classes=32, enemy_rssm_embed_dim=128, enemy_rssm_action_dim=128,
            enemy_rssm_blocks=4, rssm_fusion_moe_enabled=True, combat_rssm_enabled=True)
        with torch.device('meta'): model = NeuralWorldModel(config)
        self.assertEqual(model.vjepa_backend.memory_size, 5232)
        self.assertEqual(parameter_counts(model)['combat_rssm'], 20_007_020)


if __name__ == '__main__': unittest.main()
