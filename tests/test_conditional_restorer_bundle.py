from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_decoder_rf_adaptation_bundle as support
from test_renderer_enemy_objectives import batch
from turkeyworld.conditional_decoder_restorer import ConditionalDecoderRestorer
from turkeyworld.conditional_restorer_bundle import export_conditional_restorer, PROTOCOL
from turkeyworld.conditional_restorer_expansion import expand_conditional_restorer
from turkeyworld.decoder_rf_adaptation_bundle import export_decoder_adaptation
from turkeyworld.renderer_student import _hash_tensors, load_renderer_student


class ConditionalRestorerBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.support = support.DecoderRFAdaptationBundleTests
        cls.support.setUpClass()
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        cls.source = deepcopy(cls.support.source)
        cls.parent = cls.support.path
        cls.model = deepcopy(cls.support.adapted)
        cls.model.flow_renderer.conditional_restorer = ConditionalDecoderRestorer()
        cls.path = cls.root/'restorer.pt'
        export_conditional_restorer(cls.model, cls.parent, cls.path,
            restorer_updates=0, source_model=cls.source)
        cls.payload = torch.load(cls.path, map_location='cpu', weights_only=False)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()
        cls.support.tearDownClass()

    def test_zero_activation_is_exact_and_sample_uses_current_state_and_same_raw_map_once(self):
        old, new = deepcopy(self.support.adapted), deepcopy(self.model)
        old.eval().requires_grad_(False)
        new.eval().requires_grad_(False)
        query = batch((False,))
        features = torch.zeros(1, 128)
        with patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No runtime Source')):
            with torch.inference_mode():
                expected = old.render(query['state'], features, old.encode_map(query['map']))
                actual = new.render(query['state'], features, new.encode_map(query['map']))
            self.assertTrue(torch.equal(expected, actual))
            with torch.no_grad():
                new.flow_renderer.conditional_restorer.output.bias.add_(.2)
            with torch.inference_mode():
                fmap = new.encode_map(query['map'])
                base = new.render_base(query['state'], features, fmap)
                flow = new.flow_renderer
                z = flow.latent_sample(base, query['state'], noise=flow.fixed_noise,
                                      map_condition=query['map'])
                q = flow.codec.quantize(flow.denormalize_latents(z))['quantized']
                decoded = flow.codec.decode(q)
                manually_once = flow.conditional_restorer(q, decoded, query['state'], query['map'])
                sampled = flow.sample(base, query['state'], noise=flow.fixed_noise)
                explicit = flow.sample(base, query['state'], noise=flow.fixed_noise,
                                       map_condition=query['map'])
                self.assertTrue(torch.equal(sampled, manually_once))
                self.assertTrue(torch.equal(explicit, manually_once))
                self.assertFalse(torch.equal(decoded, sampled))
                self.assertTrue(torch.equal(decoded, flow.codec.decode(q)))
            self.assertEqual(query['map'].dtype, torch.int64)

    def test_trained_format4_roundtrip_keeps_parent_tensors_clocks_and_unqualified_scope(self):
        working = deepcopy(self.model)
        with torch.no_grad():
            working.flow_renderer.conditional_restorer.output.weight.normal_(std=.01)
        path = self.root/'trained.pt'
        export_conditional_restorer(working, self.parent, path,
            restorer_updates=3, source_model=self.source)
        target = deepcopy(self.source)
        parent_hash = _hash_tensors(self.support.adapted.flow_renderer.state_dict())
        rng = torch.get_rng_state().clone()
        metadata = load_renderer_student(target, path)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(metadata['format_version'], 4)
        self.assertEqual(metadata['objective_protocol'], PROTOCOL)
        self.assertEqual(metadata['restorer_updates'], 3)
        self.assertEqual(metadata['restorer_refinement_updates'], 3)
        self.assertEqual(metadata['training_steps'], 29612)
        self.assertEqual(metadata['decoder_refinement_updates'], 20)
        self.assertEqual(metadata['base_training_steps'], 29209)
        self.assertEqual(metadata['restorer_metadata']['parameters'], 247539)
        self.assertEqual(metadata['restorer_metadata']['parameter_tensors'], 65)
        self.assertEqual(target.flow_renderer.steps, 3)
        self.assertEqual(target.config.flow_steps, 1)
        self.assertFalse(metadata['restorer_quality_qualified'])
        self.assertFalse(metadata['parent_codec_qualification_applies_to_restorer'])
        self.assertFalse(metadata['Source_code_decode_automatically_restored'])
        frozen = {key:value for key,value in target.flow_renderer.state_dict().items()
                  if not key.startswith('conditional_restorer.')}
        self.assertEqual(_hash_tensors(frozen), parent_hash)
        self.assertEqual(_hash_tensors(target.base_student.state_dict()),
                         _hash_tensors(working.base_student.state_dict()))
        working.eval().requires_grad_(False)
        target.eval().requires_grad_(False)
        query = batch((False,))
        with torch.inference_mode():
            a = working.render(query['state'], torch.zeros(1,128), working.encode_map(query['map']))
            b = target.render(query['state'], torch.zeros(1,128), target.encode_map(query['map']))
        self.assertTrue(torch.equal(a, b))

    def test_tampered_restorer_parent_or_runtime_contract_rejects_before_live_mutation(self):
        def key(p):
            return next(iter(p['restorer_state']))
        mutations = {
            'parent_hash': lambda p:p.update(parent_bundle_sha256='a'*64),
            'RFclock': lambda p:p.update(training_steps=p['training_steps']+1),
            'Dclock': lambda p:p.update(decoder_refinement_updates=1726),
            'solver': lambda p:p.update(solver_steps=5),
            'protocol': lambda p:p.update(objective_protocol='decoder_rf_source_paired_adaptation_v1'),
            'missing_config': lambda p:p.pop('restorer_config'),
            'missing_metadata': lambda p:p.pop('restorer_metadata'),
            'metadata_Source': lambda p:p['restorer_metadata'].update(future_Source_inputs=True),
            'config': lambda p:p['restorer_config'].update(state_dim=21),
            'missing_weight': lambda p:p['restorer_state'].pop(key(p)),
            'extra_RF_weight': lambda p:p['restorer_state'].update({'flow.output.weight':torch.zeros(1)}),
            'nonfinite': lambda p:p['restorer_state'][key(p)].fill_(float('nan')),
            'changed_weight': lambda p:p['restorer_state'][key(p)].add_(.1),
            'old_format3': lambda p:p.update(format_version=3),
            'old_quality': lambda p:p.update(parent_codec_qualification_applies_to_restorer=True),
        }
        target = deepcopy(self.source)
        before = _hash_tensors(target.state_dict())
        identities = [id(p) for p in target.parameters()]
        for name, mutate in mutations.items():
            broken = deepcopy(self.payload)
            mutate(broken)
            path = self.root/'bad.pt'
            torch.save(broken, path)
            rng = torch.get_rng_state().clone()
            with self.subTest(name=name), self.assertRaises((ValueError, RuntimeError, TypeError)):
                load_renderer_student(target, path)
            self.assertEqual(_hash_tensors(target.state_dict()), before)
            self.assertEqual([id(p) for p in target.parameters()], identities)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(hasattr(target, 'base_student'))

    def test_frozen_parent_changes_and_legacy_decoder_export_cannot_hide_restorer(self):
        for mutate in [lambda m:m.flow_renderer.output.bias.add_(.01),
                       lambda m:m.flow_renderer.codec.post_quant_conv.bias.add_(.01),
                       lambda m:m.base_student.output_moe.output.bias.add_(.01)]:
            bad = deepcopy(self.model)
            with torch.no_grad():
                mutate(bad)
            with self.assertRaises(ValueError):
                export_conditional_restorer(bad,self.parent,self.root/'invalid.pt',
                    restorer_updates=1,source_model=self.source)
        with self.assertRaises(ValueError):
            export_decoder_adaptation(self.model,self.support.parent,self.root/'wrong-format3.pt',
                decoder_updates=20,source_model=self.source)

    def test_expanded_format4_roundtrip_keeps_trained_legacy_and_actual_runtime_function(self):
        working = deepcopy(self.model).eval().requires_grad_(False)
        with torch.no_grad():
            working.flow_renderer.conditional_restorer.output.bias.fill_(.07)
        query = batch((False,))
        features = torch.zeros(1, 128)
        with torch.inference_mode():
            legacy_image = working.render(query['state'], features, working.encode_map(query['map']))
        legacy_hash = _hash_tensors(working.flow_renderer.conditional_restorer.state_dict())
        rng = torch.get_rng_state().clone()
        expanded, _ = expand_conditional_restorer(working.flow_renderer.conditional_restorer)
        working.flow_renderer.conditional_restorer = expanded
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(_hash_tensors(expanded.legacy.state_dict()), legacy_hash)
        with torch.inference_mode():
            zero_image = working.render(query['state'], features, working.encode_map(query['map']))
        self.assertTrue(torch.equal(zero_image, legacy_image))
        with torch.no_grad():
            expanded.expansion.output.bias.fill_(.11)
        path = self.root/'expanded.pt'
        export_conditional_restorer(working, self.parent, path,
            restorer_updates=1410, source_model=self.source)
        target = deepcopy(self.source)
        before_rng = torch.get_rng_state().clone()
        metadata = load_renderer_student(target, path)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(metadata['restorer_metadata']['kind'], 'expanded_conditional_decoder_restorer_v1')
        self.assertEqual(metadata['restorer_metadata']['parameters'], 1997798)
        self.assertEqual(metadata['restorer_metadata']['parameter_tensors'], 130)
        self.assertEqual(metadata['restorer_updates'], 1410)
        self.assertEqual(metadata['decoder_refinement_updates'], 20)
        loaded = target.flow_renderer.conditional_restorer
        self.assertEqual(_hash_tensors(loaded.legacy.state_dict()), legacy_hash)
        self.assertEqual(_hash_tensors(loaded.state_dict()), _hash_tensors(expanded.state_dict()))
        self.assertEqual(_hash_tensors({k:v for k,v in target.flow_renderer.state_dict().items()
            if not k.startswith('conditional_restorer.')}), self.payload['frozen_parent_flow_sha256'])
        target.eval().requires_grad_(False)
        with torch.inference_mode():
            a = working.render(query['state'], features, working.encode_map(query['map']))
            b = target.render(query['state'], features, target.encode_map(query['map']))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(b, legacy_image))
        self.assertFalse(metadata['restorer_quality_qualified'])

    def test_expanded_architecture_and_branch_state_tampering_rejects_transactionally(self):
        working = deepcopy(self.model)
        working.flow_renderer.conditional_restorer, _ = expand_conditional_restorer(
            working.flow_renderer.conditional_restorer)
        path = self.root/'expanded-check.pt'
        export_conditional_restorer(working, self.parent, path,
            restorer_updates=1, source_model=self.source)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        target = deepcopy(self.source)
        before = _hash_tensors(target.state_dict())
        ids = [id(p) for p in target.parameters()]
        mutations = {
            'legacy_kind': lambda p:p['restorer_metadata'].update(kind='conditional_decoder_restorer_v1'),
            'missing_expansion_config': lambda p:p['restorer_config'].pop('expansion_config'),
            'wrong_expansion_width': lambda p:p['restorer_config']['expansion_config'].update(width=64),
            'wrong_count': lambda p:p['restorer_metadata'].update(parameters=247539),
            'missing_legacy': lambda p:p['restorer_state'].pop('legacy.output.bias'),
            'missing_expansion': lambda p:p['restorer_state'].pop('expansion.output.bias'),
        }
        for name, mutate in mutations.items():
            broken = deepcopy(payload)
            mutate(broken)
            # Bind the altered bytes too: rejection must check the architecture,
            # not only notice an old checksum after a missing tensor.
            broken['restorer_state_sha256'] = _hash_tensors(broken['restorer_state'])
            broken['restorer_metadata']['state_sha256'] = broken['restorer_state_sha256']
            broken['restorer_metadata']['config'] = deepcopy(broken['restorer_config'])
            torch.save(broken, path)
            rng = torch.get_rng_state().clone()
            with self.subTest(name=name), self.assertRaises((ValueError, RuntimeError, TypeError)):
                load_renderer_student(target, path)
            self.assertEqual(_hash_tensors(target.state_dict()), before)
            self.assertEqual([id(p) for p in target.parameters()], ids)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_7m_format4_preserves_trained_2m_zero_function_then_reloads_trained_wide_runtime(self):
        from turkeyworld.conditional_restorer_7m import expand_conditional_restorer_7m
        working = deepcopy(self.model).eval().requires_grad_(False)
        previous, _ = expand_conditional_restorer(working.flow_renderer.conditional_restorer)
        with torch.no_grad():
            previous.legacy.output.bias.fill_(.03)
            previous.expansion.output.bias.fill_(.04)
        working.flow_renderer.conditional_restorer = previous
        previous_hash = _hash_tensors(previous.state_dict())
        query, features = batch((False,)), torch.zeros(1, 128)
        with torch.inference_mode():
            previous_rgb = working.render(query['state'], features, working.encode_map(query['map']))
        rng = torch.get_rng_state().clone()
        large, _ = expand_conditional_restorer_7m(previous)
        working.flow_renderer.conditional_restorer = large
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(_hash_tensors(large.previous.state_dict()), previous_hash)
        with torch.inference_mode():
            zero_rgb = working.render(query['state'], features, working.encode_map(query['map']))
        self.assertTrue(torch.equal(zero_rgb, previous_rgb))
        with torch.no_grad():
            large.expansion.output.bias.fill_(.09)
        path = self.root/'large-trained.pt'
        export_conditional_restorer(working, self.parent, path,
            restorer_updates=2389, source_model=self.source)
        target, rng = deepcopy(self.source), torch.get_rng_state().clone()
        metadata = load_renderer_student(target, path)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(metadata['restorer_metadata']['kind'], 'large_conditional_decoder_restorer_v1')
        self.assertEqual(metadata['restorer_metadata']['capacity_variant'], 'expanded_7m_v1')
        self.assertEqual(metadata['restorer_metadata']['parameters'], 7000313)
        self.assertEqual(metadata['restorer_metadata']['parameter_tensors'], 195)
        self.assertEqual(metadata['restorer_updates'], 2389)
        self.assertEqual(metadata['training_steps'], self.payload['training_steps'])
        self.assertEqual(_hash_tensors(target.flow_renderer.conditional_restorer.previous.state_dict()), previous_hash)
        self.assertEqual(_hash_tensors(target.flow_renderer.conditional_restorer.state_dict()),
                         _hash_tensors(large.state_dict()))
        target.eval().requires_grad_(False)
        with torch.inference_mode():
            a = working.render(query['state'], features, working.encode_map(query['map']))
            b = target.render(query['state'], features, target.encode_map(query['map']))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(b, previous_rgb))
        self.assertEqual(_hash_tensors({k:v for k,v in target.flow_renderer.state_dict().items()
            if not k.startswith('conditional_restorer.')}), self.payload['frozen_parent_flow_sha256'])
        self.assertFalse(metadata['gameplay_qualified'])

    def test_7m_format4_rejects_old_capacity_or_incomplete_branch_state_before_mutation(self):
        from turkeyworld.conditional_restorer_7m import expand_conditional_restorer_7m
        working = deepcopy(self.model)
        previous, _ = expand_conditional_restorer(working.flow_renderer.conditional_restorer)
        working.flow_renderer.conditional_restorer, _ = expand_conditional_restorer_7m(previous)
        path = self.root/'large-check.pt'
        export_conditional_restorer(working, self.parent, path,
            restorer_updates=1, source_model=self.source)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        target, before = deepcopy(self.source), _hash_tensors(self.source.state_dict())
        ids = [id(p) for p in target.parameters()]
        mutations = {
            'old_2m_kind': lambda p:p['restorer_metadata'].update(kind='expanded_conditional_decoder_restorer_v1'),
            'wrong_capacity': lambda p:p['restorer_metadata'].update(capacity_variant='expanded_2m_v1'),
            'missing_previous_config': lambda p:p['restorer_config'].pop('previous_config'),
            'wrong_width': lambda p:p['restorer_config']['expansion_config'].update(width=224),
            'missing_previous_weight': lambda p:p['restorer_state'].pop('previous.legacy.output.bias'),
            'missing_new_weight': lambda p:p['restorer_state'].pop('expansion.output.bias'),
            'old_2m_count': lambda p:p['restorer_metadata'].update(parameters=1997798),
        }
        for name, mutate in mutations.items():
            broken = deepcopy(payload)
            mutate(broken)
            broken['restorer_state_sha256'] = _hash_tensors(broken['restorer_state'])
            broken['restorer_metadata']['state_sha256'] = broken['restorer_state_sha256']
            broken['restorer_metadata']['config'] = deepcopy(broken['restorer_config'])
            torch.save(broken, path)
            rng = torch.get_rng_state().clone()
            with self.subTest(name=name), self.assertRaises((ValueError, RuntimeError, TypeError)):
                load_renderer_student(target, path)
            self.assertEqual(_hash_tensors(target.state_dict()), before)
            self.assertEqual([id(p) for p in target.parameters()], ids)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
