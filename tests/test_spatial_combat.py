"""Pairwise combat and explicit hit-count rendering retain trained spatial weights."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F

import test_spatial_vjepa as spatial_fixture


COMBAT_PREFIXES = ('vjepa_backend.spatial_combat_head.',
                   'vjepa_backend.renderer.spatial_combat_counts.')
NEW_PREFIXES = COMBAT_PREFIXES+('vjepa_backend.spatial_spawn_',)


class SpatialCombatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def models(self, trained=True):
        from turkeyworld.models import NeuralWorldModel
        fixture = spatial_fixture.SpatialVJEPATests()
        old = fixture.model(True).eval()
        if trained:
            with torch.no_grad():
                old.vjepa_backend.spatial_visual_attention.output.weight.normal_(std=.02)
                old.vjepa_backend.spatial_motion_head[-1].weight.normal_(std=.02)
                old.vjepa_backend.renderer.spatial_render.attention.out_proj.weight.normal_(std=.02)
                old.vjepa_backend.hit_target_head[-1].weight.normal_(std=.02)
                old.vjepa_backend.attack_head[-1].weight.normal_(std=.02)
                old.vjepa_backend.spawn_head[-1].weight.normal_(std=.02)
        new = NeuralWorldModel(replace(old.config, spatial_combat=True)).eval()
        missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(all(name.startswith(NEW_PREFIXES) for name in missing), missing)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
        return old, new, fixture

    def test_config_requires_spatial_entities_and_false_adds_no_modules(self):
        from turkeyworld.models import WorldModelConfig
        with self.assertRaisesRegex(ValueError, 'requires spatial_entities'):
            WorldModelConfig(spatial_combat=True)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            WorldModelConfig(spatial_combat=1)
        old, _, _ = self.models()
        self.assertFalse(old.config.spatial_combat)
        self.assertFalse(any('spatial_combat' in name for name in old.state_dict()))

    def test_zero_residuals_preserve_all_trained_spatial_outputs_exactly(self):
        old, new, fixture = self.models()
        with torch.inference_mode():
            state, grid, fmap, rgb, memory = fixture.inputs(old, batch=2)
            new_map = new.encode_map(grid)
            new_memory = new.initialize_from_image(rgb, state, new_map)
            action = torch.tensor([[.5, .2, .1, 1.], [.2, -.1, -.2, 0.]])
            first = old.step(state, action, memory=memory, map_features=fmap)
            second = new.step(state, action, memory=new_memory, map_features=new_map)
        for key in ('state', 'memory', 'latent_prediction', 'target_logits', 'attack_logits', 'features', 'rgb',
                    'spawn_logits', 'spawn_positions', 'spawn_valid'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        self.assertEqual(second['state'].shape, (2, 90))
        self.assertEqual(second['memory'].shape, (2, 264, 32))
        self.assertEqual(second['combat_context']['spatial'].shape, (2, 9, 32))

    def test_all_eight_pairs_include_player_npc_pooled_and_both_count_features(self):
        from turkeyworld.combat_objectives import combat_count_features
        _, model, fixture = self.models()
        state, _, fmap, _, _ = fixture.inputs(model, batch=2)
        state[0, 4], state[1, 4] = .84, .92
        state[0, 12], state[1, 12] = .25, .35
        spatial = model.vjepa_backend.spatial_tokens(state, fmap)
        pooled = torch.randn(2, 32)
        captured = []
        handle = model.vjepa_backend.spatial_combat_head.register_forward_pre_hook(lambda module, values: captured.append(values[0]))
        try:
            model.vjepa_backend.combat_logits(state, pooled, spatial)
        finally:
            handle.remove()
        pair = captured[0]
        counts = combat_count_features(state)
        self.assertEqual(pair.shape, (2, 8, 3*32+2*counts.shape[-1]))
        torch.testing.assert_close(pair[:, :, :32], spatial[:, :1].expand(-1, 8, -1))
        torch.testing.assert_close(pair[:, :, 32:64], spatial[:, 1:])
        torch.testing.assert_close(pair[:, :, 64:96], pooled[:, None].expand(-1, 8, -1))
        torch.testing.assert_close(pair[:, :, 96:102], counts[:, :1].expand(-1, 8, -1))
        torch.testing.assert_close(pair[:, :, 102:], counts[:, 1:])
        with self.assertRaisesRegex(ValueError, 'explicit nine'):
            model.vjepa_backend.combat_logits(state, pooled)

    def test_activated_pair_branch_sees_coordinates_directions_counts_and_keeps_none_class(self):
        _, model, fixture = self.models()
        state, _, fmap, _, _ = fixture.inputs(model)
        backend, pooled = model.vjepa_backend, torch.zeros(1, 32)
        with torch.no_grad():
            backend.spatial_combat_head[-1].weight.normal_(std=.05)
            original_tokens = backend.spatial_tokens(state, fmap)
            targets, attacks = backend.combat_logits(state, pooled, original_tokens)
        variants = []
        moved = state.clone(); moved[:, 8] += .3/16; variants.append(moved)
        directed = state.clone(); directed[:, 10:12] = torch.tensor([0., .2]); variants.append(directed)
        damaged = state.clone(); damaged[:, 12] = .25; variants.append(damaged)
        player_moved = state.clone(); player_moved[:, 1] += .3/16; variants.append(player_moved)
        player_turned = state.clone(); player_turned[:, 2:4] = torch.tensor([.6, .8]); variants.append(player_turned)
        player_damaged = state.clone(); player_damaged[:, 4] = .84; variants.append(player_damaged)
        with torch.no_grad():
            for variant in variants:
                changed = backend.combat_logits(variant, pooled, backend.spatial_tokens(variant, fmap))
                torch.testing.assert_close(changed[0][:, 8], targets[:, 8], rtol=0, atol=0)
                self.assertGreater(float((changed[0][:, :8]-targets[:, :8]).abs().sum()
                                         +(changed[1]-attacks).abs().sum()), 0)
            shifted_target, shifted_attack = backend.combat_logits(moved, pooled, backend.spatial_tokens(moved, fmap))
            torch.testing.assert_close(shifted_target[:, 1:], targets[:, 1:], rtol=0, atol=0)
            torch.testing.assert_close(shifted_attack[:, 1:], attacks[:, 1:], rtol=0, atol=0)

    def test_light_combat_recompute_after_primary_backward_does_not_retain_visual_graph(self):
        from turkeyworld.engine import Game
        _, model, fixture = self.models()
        model.train()
        state, _, fmap, _, memory = fixture.inputs(model, batch=2)
        state.requires_grad_()
        result = model.transition(state, torch.tensor([[.5, 0., .1, 1.], [.2, 0., 0., 0.]]), fmap, memory)
        context = result['combat_context']
        self.assertTrue(context['state'].requires_grad)
        self.assertIsNotNone(context['pooled'].grad_fn)
        self.assertIsNotNone(context['spatial'].grad_fn)
        result['latent_prediction'].square().mean().backward()
        model.zero_grad(set_to_none=True)
        detached = {name: value.detach() for name, value in context.items()}
        with patch.object(Game, 'step', side_effect=AssertionError('No engine combat')), \
                patch.object(Game, 'render', side_effect=AssertionError('No future image')), \
                patch.object(model.core.encoder, 'forward', side_effect=AssertionError('No visual recomputation')):
            target, attack = model.vjepa_backend.combat_logits(detached['state'], detached['pooled'], detached['spatial'])
            spawn = model.vjepa_backend.spawn_logits(detached['state'], detached['pooled'],
                detached['spatial'], detached['map_features'])
            (F.cross_entropy(target, torch.tensor([0, 8]))+F.binary_cross_entropy_with_logits(attack, torch.ones_like(attack))
                +F.cross_entropy(spawn.flatten(0, 1), torch.arange(16))).backward()
        backend = model.vjepa_backend
        for parameter in (backend.hit_target_head[-1].weight, backend.attack_head[-1].weight,
                          backend.spatial_combat_head[-1].weight, backend.spatial_spawn_output.weight,
                          backend.spawn_head[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.core.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in backend.spatial_encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in backend.renderer.map_encoder.parameters()))

    def test_explicit_count_adapter_changes_actual_rgb_with_all_prior_conditioning_held_fixed(self):
        _, model, fixture = self.models()
        state, _, fmap, _, _ = fixture.inputs(model)
        renderer = model.vjepa_backend.renderer
        features = torch.zeros(1, 128)
        with torch.no_grad():
            geometry = renderer.geometry(state.to(fmap.dtype), fmap)
            weights = renderer.spatial_render_weights(state, geometry)
            tokens = renderer.spatial_encoder(state, fmap)
            for parameter in renderer.render_condition.parameters():
                parameter.zero_()
            renderer.spatial_render.attention.out_proj.weight.copy_(torch.eye(renderer.width))
        damaged = state.clone(); damaged[:, 4], damaged[:, 12] = .84, .25
        with patch.object(renderer, 'geometry', return_value=geometry), \
                patch.object(renderer.spatial_encoder, 'forward', return_value=tokens), \
                patch.object(renderer, 'spatial_render_weights', return_value=weights):
            with torch.no_grad():
                first = renderer.render_with_aux(state, features, fmap)['rgb']
                zero = renderer.render_with_aux(damaged, features, fmap)['rgb']
                torch.testing.assert_close(first, zero, rtol=0, atol=0)
                renderer.spatial_combat_counts.weight.normal_(std=.03)
                first = renderer.render_with_aux(state, features, fmap)['rgb']
                changed = renderer.render_with_aux(damaged, features, fmap)['rgb']
                self.assertGreater(float((first-changed).abs().sum()), 0)
            model.zero_grad(set_to_none=True)
            image = renderer.render_with_aux(damaged, features, fmap)['rgb']
            F.mse_loss(image, torch.rand_like(image)).backward()
            self.assertGreater(float(renderer.spatial_combat_counts.weight.grad.abs().sum()), 0)

    def test_teacher_budget_includes_only_new_pair_and_count_adapters(self):
        from turkeyworld.models import NeuralWorldModel
        from turkeyworld.vqgan import VQGANConfig
        from turkeyworld.vq_world import teacher175_vq_config
        with torch.device('meta'):
            model = NeuralWorldModel(teacher175_vq_config(VQGANConfig(image_size=32, n_embed=1488),
                spatial_entities=True, spatial_combat=True))
        count = sum(parameter.numel() for parameter in model.parameters())
        # The separately owned Flow spatial adapters also contribute when enabled.
        self.assertGreaterEqual(count, 175_785_704)
        self.assertEqual(sum(parameter.numel() for name, parameter in model.named_parameters()
                             if name.startswith(COMBAT_PREFIXES)), 199_874)
        self.assertEqual(sum(parameter.numel() for name, parameter in model.named_parameters()
                             if name.startswith('vjepa_backend.spatial_spawn_')), 180_544)
        self.assertEqual(model.core.memory_tokens, 264)

    def test_flow_receives_original_count_state_only_when_combat_conditioning_is_enabled(self):
        from turkeyworld.models import render_state_view
        old, new, fixture = self.models()
        state, _, fmap, _, _ = fixture.inputs(old)
        state[:, 4] = .84
        base = torch.zeros(1, 3, 32, 32)
        for model, enabled in ((old, False), (new, True)):
            model.config.renderer_mode = 'flow_matching'
            sample = Mock(return_value=base)
            model.flow_renderer = SimpleNamespace(sample=sample)
            with patch.object(model, 'render_base', return_value=base):
                model.render(state, torch.zeros(1, 128), fmap)
                model.render_with_aux(state, torch.zeros(1, 128), fmap)
            self.assertEqual(sample.call_count, 2)
            for call in sample.call_args_list:
                expected = state if enabled else render_state_view(state)
                torch.testing.assert_close(call.args[1], expected, rtol=0, atol=0)
                if enabled:
                    self.assertIs(call.args[1], state)

    def test_spawn_has_eight_distinct_queries_and_reads_other_entities_clock_and_map_cells(self):
        _, model, fixture = self.models()
        state, _, fmap, _, _ = fixture.inputs(model)
        backend = model.vjepa_backend
        pooled = torch.zeros(1, 32)
        with torch.no_grad():
            for parameter in backend.spawn_head.parameters():
                parameter.zero_()
            backend.spatial_spawn_output.weight.copy_(torch.eye(32))
            spatial = backend.spatial_tokens(state, fmap)
            baseline = backend.spawn_logits(state, pooled, spatial, fmap)
            self.assertEqual(baseline.shape, (1, 8, 256))
            self.assertEqual(torch.unique(baseline[0], dim=0).shape[0], 8)
            # Slot0's own token/context stay fixed; a hidden/dead OTHER slot
            # still changes its query through the all-nine entity attention.
            changed = state.clone(); changed[:, 78:80] += torch.tensor([.2/16, -.3/16])
            other = backend.spatial_tokens(changed, fmap)
            torch.testing.assert_close(spatial[:, :8], other[:, :8], rtol=0, atol=0)
            other_logits = backend.spawn_logits(changed, pooled, other, fmap)
            self.assertGreater(float((baseline[:, 0]-other_logits[:, 0]).abs().sum()), 0)
            variants = []
            moved = state.clone(); moved[:, 8] += .25/16; variants.append(moved)
            directed = state.clone(); directed[:, 10:12] = torch.tensor([0., .2]); variants.append(directed)
            player = state.clone(); player[:, 2:4] = torch.tensor([.6, .8]); variants.append(player)
            for value in variants:
                logits = backend.spawn_logits(value, pooled, backend.spatial_tokens(value, fmap), fmap)
                self.assertGreater(float((baseline-logits).abs().sum()), 0)
            later = state.clone(); later[:, -1] += 3/300
            self.assertGreater(float((baseline-backend.spawn_logits(later, pooled, spatial, fmap)).abs().sum()), 0)
            cell = fmap.clone(); cell[:, :, 5, 5] += .5
            map_logits = backend.spawn_logits(state, pooled, spatial, cell)
            self.assertGreater(float((baseline[:, :, 85]-map_logits[:, :, 85]).abs().sum()), 0)
            keep = torch.arange(256) != 85
            torch.testing.assert_close(baseline[:, :, keep], map_logits[:, :, keep], rtol=0, atol=0)

    def test_spawn_zero_projection_opens_before_entity_query_and_map_gradients(self):
        _, model, fixture = self.models()
        model.train()
        state, grid, fmap, _, _ = fixture.inputs(model, batch=2)
        backend = model.vjepa_backend
        pooled = torch.randn(2, 32, requires_grad=True)
        spatial = backend.spatial_tokens(state, fmap)
        logits = backend.spawn_logits(state, pooled, spatial, fmap)
        F.cross_entropy(logits.flatten(0, 1), torch.arange(16)*3).backward()
        self.assertGreater(float(backend.spatial_spawn_output.weight.grad.abs().sum()), 0)
        self.assertGreater(float(backend.spawn_head[-1].weight.grad.abs().sum()), 0)
        with torch.no_grad():
            backend.spatial_spawn_output.weight.copy_(torch.eye(32))
        model.zero_grad(set_to_none=True)
        fmap = model.encode_map(grid)
        pooled = torch.randn(2, 32, requires_grad=True)
        spatial = backend.spatial_tokens(state, fmap)
        F.cross_entropy(backend.spawn_logits(state, pooled, spatial, fmap).flatten(0, 1), torch.arange(16)*3).backward()
        for layer in (backend.spatial_spawn_query[0], backend.spatial_spawn_entity_key,
                      backend.spatial_spawn_entity_value, backend.spatial_spawn_map_key):
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0)
        self.assertGreater(float(backend.spatial_encoder.descriptor_projection[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(pooled.grad.abs().sum()), 0)

    def test_finite_world_has_no_new_spawn_modules(self):
        from turkeyworld.models import NeuralWorldModel
        _, model, _ = self.models()
        finite = NeuralWorldModel(replace(model.config, endless=False))
        self.assertFalse(any(name.startswith('vjepa_backend.spatial_spawn_') for name in finite.state_dict()))
        with self.assertRaisesRegex(ValueError, 'endless'):
            finite.vjepa_backend.spawn_logits(torch.zeros(1, 90), torch.zeros(1, 32), None, None)


if __name__ == '__main__':
    unittest.main()
