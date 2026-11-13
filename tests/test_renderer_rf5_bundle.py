"""Real strict RF1 parent binding and transactional RF5-expanded bundle loading."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_renderer_enemy_objectives import batch
from test_renderer_student import tiny_config
from turkeyworld.base_student import build_base_student
from turkeyworld.models import NeuralWorldModel, WorldModelConfig
from turkeyworld.renderer_geometry_opacity import enable_geometry_opacity
from turkeyworld.renderer_moe import enable_renderer_moe, migrate_renderer_moe_five
from turkeyworld.renderer_refinement import export_refined_bundle
from turkeyworld.renderer_rf5_bundle import (
    export_rf5_renderer, parent_bundle_fingerprint, PROTOCOL,
)
from turkeyworld.renderer_rf5_experts import upgrade_rf5_experts
from turkeyworld.renderer_student import (
    _hash_tensors, build_renderer_student, load_renderer_student,
)


class RendererRF5BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        codec = dict(image_size=32, base_channels=4, latent_channels=16,
                     embedding_dim=16, n_embed=8, data_init=False)
        config = tiny_config(architecture='dreamer_rssm', flow_steps=1,
            flow_width=16, flow_heads=2, renderer_codec_config=codec,
            navigation_only=True, rssm_deter_dim=16, rssm_hidden_dim=16,
            rssm_stoch=2, rssm_classes=4, rssm_embed_dim=16,
            rssm_action_dim=20, rssm_blocks=2)
        source = NeuralWorldModel(config).eval()
        source.flow_renderer.codec_loaded.fill_(True)
        cls.unloaded = deepcopy(source)
        working = deepcopy(source)
        working.flow_renderer = build_renderer_student(working, steps=1,
            width=256, depth=1, heads=8, mlp_hidden=128)
        working.base_student = build_base_student(working)
        working.config = WorldModelConfig(**working.flow_renderer._renderer_student_config)
        enable_geometry_opacity(working)
        enable_renderer_moe(working)
        migrate_renderer_moe_five(working)
        with torch.no_grad():
            working.flow_renderer.output_moe.output.weight.normal_(std=.01)
            working.base_student.output_moe.output.weight.normal_(std=.01)
            working.flow_renderer.geometry_opacity[-1].bias.add_(.2)
        export_refined_bundle(source, working, cls.root/'parent', updates=3,
            report={'passed': False}, initial_renderer_steps=29606,
            initial_base_steps=29206, provenance={})
        cls.parent_path = cls.root/'parent'/'renderer_student.pt'
        cls.expanded = deepcopy(cls.unloaded)
        load_renderer_student(cls.expanded, cls.parent_path)
        upgrade_rf5_experts(cls.expanded.flow_renderer.output_moe)
        cls.expanded.flow_renderer.steps = 5
        with torch.no_grad():
            cls.expanded.flow_renderer.output.bias.add_(.01)
            for expert in cls.expanded.flow_renderer.output_moe.experts:
                expert.private_output.weight.normal_(std=.005)
        cls.path = cls.root/'rf5.pt'
        cls.metadata = export_rf5_renderer(cls.expanded, cls.parent_path, cls.path,
            training_steps=29611, local_updates=2)
        cls.payload = torch.load(cls.path, map_location='cpu', weights_only=False)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_strict_roundtrip_is_self_contained_RF1_config_truthful_and_RF5_runtime_exact(self):
        target = deepcopy(self.unloaded)
        before = {n: p.clone() for n, p in target.state_dict().items()
                  if not n.startswith('flow_renderer.')}
        rng = torch.get_rng_state().clone()
        with patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No Source at inference')):
            metadata = load_renderer_student(target, self.path)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(target.config.flow_steps, 1)
        self.assertEqual(target.flow_renderer._renderer_student_config['flow_steps'], 1)
        self.assertEqual(target.flow_renderer.steps, 5)
        self.assertEqual(metadata['solver_steps'], 5)
        self.assertEqual(metadata['objective_protocol'], PROTOCOL)
        self.assertEqual(metadata['base_training_steps'], 29209)
        self.assertEqual(metadata['parent_renderer_training_steps'], 29609)
        self.assertEqual(metadata['training_steps'], 29611)
        self.assertFalse(metadata['source_quality_passed'])
        self.assertFalse(metadata['gameplay_qualified'])
        self.assertTrue(all(torch.equal(v, target.state_dict()[n]) for n, v in before.items()))
        self.assertEqual(_hash_tensors(target.base_student.state_dict()),
                         self.payload['base_student_identity'])
        self.assertEqual(_hash_tensors(target.flow_renderer.state_dict()), self.payload['model_sha256'])
        working = deepcopy(self.expanded).eval().requires_grad_(False)
        target.eval().requires_grad_(False)
        query = batch((False,))
        with torch.inference_mode():
            features = torch.zeros(1, 128)
            expected = working.render(query['state'], features, working.encode_map(query['map']))
            actual = target.render(query['state'], features, target.encode_map(query['map']))
        self.assertTrue(torch.equal(expected, actual))

    def test_corrupt_parent_solver_protocol_weights_and_expansion_fail_before_any_live_mutation(self):
        mutations = {
            'protocol': lambda p: p.update(objective_protocol='old'),
            'no_expansion': lambda p: p.pop('expansion'),
            'private_width': lambda p: p['expansion'].update(private_width=32),
            'name_map': lambda p: p['expansion']['legacy_parameter_name_map'].popitem(),
            'parent_hash': lambda p: p.update(parent_bundle_sha256='a'*64),
            'parent_source_pin': lambda p: p.update(parent_source_sha256=None),
            'parent_weight': lambda p: p['parent_bundle']['model']['output.weight'].add_(.1),
            'foreign_Base': lambda p: p.update(base_student_identity='f'*64),
            'config_step5': lambda p: p['config'].update(flow_steps=5),
            'solver1': lambda p: p.update(solver_steps=1),
            'bad_clock': lambda p: p.update(local_updates=3),
            'missing_private': lambda p: p['model'].pop('output_moe.experts.0.private_output.weight'),
            'nonfinite_private': lambda p: p['model']['output_moe.experts.0.private_output.weight'].fill_(float('nan')),
            'unhashed_private': lambda p: p['model']['output_moe.experts.0.private_output.weight'].add_(.1),
            'false_qualification': lambda p: p.update(gameplay_qualified=True),
        }
        target = deepcopy(self.unloaded)
        before = _hash_tensors(target.state_dict())
        config = deepcopy(target.config)
        ids = [id(p) for p in target.parameters()]
        for name, mutate in mutations.items():
            bad = deepcopy(self.payload)
            mutate(bad)
            path = self.root/'bad.pt'
            torch.save(bad, path)
            rng = torch.get_rng_state().clone()
            with self.subTest(name=name), self.assertRaises((ValueError, RuntimeError)):
                load_renderer_student(target, path)
            self.assertEqual(_hash_tensors(target.state_dict()), before, name)
            self.assertEqual(target.config, config, name)
            self.assertEqual([id(p) for p in target.parameters()], ids, name)
            self.assertFalse(hasattr(target, 'base_student'))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()), name)

    def test_even_rehashed_codec_or_private_opacity_changes_are_rejected(self):
        for key in ['latent_mean', 'fixed_noise', 'geometry_opacity.2.bias',
                    'codec.encoder.conv_in.weight']:
            bad = deepcopy(self.payload)
            if key not in bad['model']:
                key = next(n for n in bad['model'] if n.startswith('codec.') and bad['model'][n].is_floating_point())
            bad['model'][key].add_(.01)
            bad['model_sha256'] = _hash_tensors(bad['model'])
            path = self.root/'bad-fixed.pt'
            torch.save(bad, path)
            target = deepcopy(self.unloaded)
            before = _hash_tensors(target.state_dict())
            with self.assertRaises(ValueError):
                load_renderer_student(target, path)
            self.assertEqual(before, _hash_tensors(target.state_dict()))

    def test_local_zero_export_allowed_but_Base_and_frozen_assets_must_match_parent(self):
        working = deepcopy(self.expanded)
        path = self.root/'initial.pt'
        export_rf5_renderer(working, self.parent_path, path, training_steps=29609, local_updates=0)
        self.assertEqual(load_renderer_student(deepcopy(self.unloaded), path)['local_updates'], 0)
        with torch.no_grad():
            working.base_student.output_moe.output.bias.add_(.1)
        with self.assertRaises(ValueError):
            export_rf5_renderer(working, self.parent_path, path, training_steps=29609, local_updates=0)
        bad = deepcopy(self.payload)
        bad['parent_bundle']['config']['flow_steps'] = 5
        bad['parent_bundle_sha256'] = parent_bundle_fingerprint(bad['parent_bundle'])
        torch.save(bad, path)
        with self.assertRaises(ValueError):
            load_renderer_student(deepcopy(self.unloaded), path)

    def test_existing_exact_Base_and_codec_objects_are_reused_on_success(self):
        target = deepcopy(self.unloaded)
        load_renderer_student(target, self.parent_path)
        base, codec = target.base_student, target.flow_renderer.codec
        load_renderer_student(target, self.path)
        self.assertIs(target.base_student, base)
        self.assertIs(target.flow_renderer.codec, codec)

    def test_three_step_roundtrip_declares_actual_solver_and_rejects_mismatched_profile(self):
        working = deepcopy(self.expanded)
        working.flow_renderer.steps = 3
        path = self.root/'rf3.pt'
        metadata = export_rf5_renderer(working, self.parent_path, path,
                                      training_steps=29612, local_updates=3)
        self.assertEqual(metadata['solver_steps'], 3)
        self.assertEqual(metadata['solver_profile']['velocity_evaluation_times'], [0., 1/3, 2/3])
        target = deepcopy(self.unloaded)
        load_renderer_student(target, path)
        self.assertEqual(target.config.flow_steps, 1)
        self.assertEqual(target.flow_renderer.steps, 3)
        query = batch((False,))
        working.eval().requires_grad_(False)
        target.eval().requires_grad_(False)
        with torch.inference_mode():
            features = torch.zeros(1, 128)
            expected = working.render(query['state'], features, working.encode_map(query['map']))
            actual = target.render(query['state'], features, target.encode_map(query['map']))
        self.assertTrue(torch.equal(expected, actual))
        saved = torch.load(path, map_location='cpu', weights_only=False)
        for steps in (0, 1, 2, 4, True, 5):
            bad = deepcopy(saved)
            bad['solver_steps'] = steps  # Profile still explicitly records three.
            torch.save(bad, self.root/'bad-profile.pt')
            with self.assertRaises(ValueError):
                load_renderer_student(deepcopy(self.unloaded), self.root/'bad-profile.pt')

    def test_export_after_differentiable_forward_keeps_live_diagnostic_cache_untouched(self):
        working = deepcopy(self.expanded)
        module = working.flow_renderer.output_moe
        module(torch.randn(1, 16, 256), torch.zeros(1, 20),
               role_priors=torch.rand(1, 16, 3))
        original = module.last_aux
        self.assertTrue(original['expert_tokens'].requires_grad)
        path = self.root/'after-forward.pt'
        rng = torch.get_rng_state().clone()
        export_rf5_renderer(working, self.parent_path, path,
                            training_steps=29612, local_updates=3)
        self.assertIs(module.last_aux, original)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_explicit_original_source_teacher_revalidates_parent_and_rejects_changed_working_features(self):
        working = deepcopy(self.expanded)
        path = self.root/'with-source.pt'
        source = deepcopy(self.unloaded)
        before = _hash_tensors(source.state_dict())
        export_rf5_renderer(working, self.parent_path, path,
            training_steps=29612, local_updates=3, source_model=source)
        self.assertEqual(before, _hash_tensors(source.state_dict()))
        with torch.no_grad():
            working.vjepa_backend.feature_head.bias.add_(.01)
        with self.assertRaises(ValueError):
            export_rf5_renderer(working, self.parent_path, path,
                training_steps=29612, local_updates=3, source_model=source)
        self.assertEqual(before, _hash_tensors(source.state_dict()))


if __name__ == '__main__':
    unittest.main()
