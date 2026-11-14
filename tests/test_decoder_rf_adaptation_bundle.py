from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

import test_renderer_rf5_bundle as support
from test_renderer_enemy_objectives import batch
from turkeyworld.decoder_rf_adaptation_bundle import export_decoder_adaptation, PROTOCOL
from turkeyworld.renderer_refinement import refinement_forward
from turkeyworld.renderer_rf5_bundle import export_rf5_renderer
from turkeyworld.renderer_student import _hash_tensors, load_renderer_student


class DecoderRFAdaptationBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.support = support.RendererRF5BundleTests
        cls.support.setUpClass()
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        cls.source = deepcopy(cls.support.unloaded)
        model = deepcopy(cls.support.expanded)
        model.flow_renderer.steps = 3
        cls.parent = cls.root/'parent-rf3.pt'
        export_rf5_renderer(model, cls.support.parent_path, cls.parent,
            training_steps=29612, local_updates=3, source_model=cls.source)
        cls.reference = deepcopy(cls.source)
        load_renderer_student(cls.reference, cls.parent)
        cls.adapted = deepcopy(cls.reference)
        with torch.no_grad():
            cls.adapted.flow_renderer.codec.decoder.conv_out.bias.add_(.15)
            cls.adapted.flow_renderer.codec.post_quant_conv.weight.add_(.025)
        cls.path = cls.root/'adapted.pt'
        cls.metadata = export_decoder_adaptation(cls.adapted, cls.parent, cls.path,
            decoder_updates=20, source_model=cls.source)
        cls.payload = torch.load(cls.path, map_location='cpu', weights_only=False)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()
        cls.support.tearDownClass()

    def test_only_decoding_changes_while_EQ_RF_base_and_terminal_codes_are_exact(self):
        query = batch((False,))
        old, new = deepcopy(self.reference), deepcopy(self.source)
        load_renderer_student(new, self.path)
        old.eval().requires_grad_(False)
        new.eval().requires_grad_(False)
        with torch.inference_mode():
            features = torch.zeros(1, 128)
            a = {'state': query['state'], 'features': features,
                 'geometry': old.render_geometry(query['state'], old.encode_map(query['map'])),
                 'map': query['map']}
            b = {**a, 'geometry': new.render_geometry(query['state'], new.encode_map(query['map']))}
            old_base, old_rgb, old_z = refinement_forward(old, a)
            new_base, new_rgb, new_z = refinement_forward(new, b)
            old_encoding = old.flow_renderer.codec.encode(query['target'])
            new_encoding = new.flow_renderer.codec.encode(query['target'])
            old_reconstruction = old.flow_renderer.codec.decode(old_encoding['quantized'])
            new_reconstruction = new.flow_renderer.codec.decode(new_encoding['quantized'])
        self.assertTrue(torch.equal(old_base, new_base))
        self.assertTrue(torch.equal(old_z, new_z))
        self.assertFalse(torch.equal(old_rgb, new_rgb))
        self.assertFalse(torch.equal(old_reconstruction, new_reconstruction))
        for key in ('prequant', 'quantized', 'indices'):
            self.assertTrue(torch.equal(old_encoding[key], new_encoding[key]), key)
        self.assertEqual(_hash_tensors(old.base_student.state_dict()), _hash_tensors(new.base_student.state_dict()))

    def test_format3_strict_roundtrip_preserves_original_codec_object_and_frozen_RF_clocks(self):
        target = deepcopy(self.source)
        original_codec = target.flow_renderer.codec
        original_hash = _hash_tensors(original_codec.state_dict())
        outside = _hash_tensors({k:v for k,v in target.state_dict().items()
                                if not k.startswith('flow_renderer.')})
        rng = torch.get_rng_state().clone()
        metadata = load_renderer_student(target, self.path)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(metadata['format_version'], 3)
        self.assertEqual(metadata['objective_protocol'], PROTOCOL)
        self.assertEqual(metadata['training_steps'], 29612)
        self.assertEqual(metadata['renderer_training_steps'], 29612)
        self.assertEqual(metadata['refinement_updates'], 3)
        self.assertEqual(metadata['decoder_refinement_updates'], 20)
        self.assertEqual(metadata['base_training_steps'], 29209)
        self.assertEqual(target.flow_renderer.steps, 3)
        self.assertEqual(target.config.flow_steps, 1)
        self.assertFalse(metadata['decoder_quality_qualified'])
        self.assertFalse(metadata['original_codec_qualification_applies_to_new_decoder'])
        self.assertFalse(metadata['gameplay_qualified'])
        self.assertIsNot(target.flow_renderer.codec, original_codec)
        self.assertEqual(original_hash, _hash_tensors(original_codec.state_dict()))
        self.assertEqual(outside, _hash_tensors({k:v for k,v in target.state_dict().items()
                                               if not k.startswith(('flow_renderer.', 'base_student.'))}))
        self.assertEqual(_hash_tensors(target.flow_renderer.state_dict()),
                         _hash_tensors(self.adapted.flow_renderer.state_dict()))
        self.assertFalse(any(p.requires_grad for p in target.flow_renderer.codec.parameters()))

    def test_complete_override_and_parent_hashes_reject_transactionally(self):
        def first(p):
            return next(iter(p['decoder_override']))
        mutations = {
            'missing_allowed': lambda p: p['decoder_override'].pop(first(p)),
            'extra_RF': lambda p: p['decoder_override'].update({'output.weight': torch.zeros(1)}),
            'extra_encoder': lambda p: p['decoder_override'].update({'codec.encoder.conv_in.weight': torch.zeros(1)}),
            'extra_codebook': lambda p: p['decoder_override'].update({'codec.quantize.embedding.weight': torch.zeros(1)}),
            'shape': lambda p: p['decoder_override'].__setitem__(first(p), torch.zeros(1)),
            'dtype': lambda p: p['decoder_override'].__setitem__(first(p), p['decoder_override'][first(p)].double()),
            'nonfinite': lambda p: p['decoder_override'][first(p)].fill_(float('nan')),
            'bad_parent_hash': lambda p: p.update(parent_bundle_sha256='f'*64),
            'wrong_protocol': lambda p: p.update(objective_protocol='renderer_rf_expanded_experts_v1'),
            'wrong_steps': lambda p: p.update(solver_steps=5),
            'wrong_RF_clock': lambda p: p.update(training_steps=29632),
            'wrong_Base': lambda p: p.update(base_student_identity='e'*64),
            'geometry': lambda p: p.update(codec_config={}),
            'qualification': lambda p: p.update(decoder_quality_qualified=True),
            'old_format2': lambda p: p.update(format_version=2),
        }
        for name, mutate in mutations.items():
            target = deepcopy(self.source)
            before = _hash_tensors(target.state_dict())
            ids = [id(p) for p in target.parameters()]
            bad = deepcopy(self.payload)
            mutate(bad)
            path = self.root/'bad.pt'
            torch.save(bad, path)
            rng = torch.get_rng_state().clone()
            with self.subTest(name=name), self.assertRaises((ValueError, RuntimeError)):
                load_renderer_student(target, path)
            self.assertEqual(before, _hash_tensors(target.state_dict()), name)
            self.assertEqual(ids, [id(p) for p in target.parameters()], name)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()), name)
            self.assertFalse(hasattr(target, 'base_student'))

    def test_export_rejects_changed_RF_EQ_Base_or_solver_and_legacyRF3_stays_immutable(self):
        modifications = [lambda m: m.flow_renderer.output.bias.add_(.1),
            lambda m: m.flow_renderer.codec.encoder.conv_in.weight.add_(.1),
            lambda m: m.flow_renderer.codec.quantize.embedding.weight.add_(.1),
            lambda m: m.base_student.output_moe.output.bias.add_(.1)]
        for modify in modifications:
            model = deepcopy(self.adapted)
            with torch.no_grad():
                modify(model)
            with self.assertRaises(ValueError):
                export_decoder_adaptation(model, self.parent, self.root/'invalid.pt',
                    decoder_updates=1, source_model=self.source)
        with self.assertRaises(ValueError):
            export_rf5_renderer(self.adapted, self.support.parent_path,
                self.root/'wrong-format2.pt', training_steps=29612, local_updates=3,
                source_model=self.source)

    def test_zero_update_private_export_with_existing_Base_and_distinct_postquant(self):
        path = self.root/'zero.pt'
        export_decoder_adaptation(self.reference, self.parent, path,
                                  decoder_updates=0, source_model=self.source)
        target = deepcopy(self.source)
        load_renderer_student(target, self.parent)
        base = target.base_student
        load_renderer_student(target, path)
        self.assertIs(target.base_student, base)
        self.assertEqual(_hash_tensors(target.flow_renderer.state_dict()),
                         _hash_tensors(self.reference.flow_renderer.state_dict()))


if __name__ == '__main__':
    unittest.main()
