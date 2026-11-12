import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_renderer_student import TinyTeacher
from turkeyworld.base_student import build_base_student, export_base_student
from turkeyworld.models import WorldModelConfig
from turkeyworld.renderer_student import (build_renderer_student, export_renderer_student,
                                         load_renderer_student, _hash_tensors)
from turkeyworld.renderer_feature_inputs import (prepare_renderer_feature_profile,
                                                prepare_current_renderer_inputs)


class CurrentRendererFeatureInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root):
        source = TinyTeacher().eval()
        source.config = replace(source.config, architecture='dreamer_rssm', navigation_only=False)
        source.vjepa_backend.config = source.config
        base = build_base_student(source)
        export_base_student(source, base, root/'base.pt', training_steps=3)
        source.base_student_bundle = torch.load(root/'base.pt', weights_only=False)
        source.base_student = base
        student = build_renderer_student(source, steps=1, width=32, depth=1, heads=4, mlp_hidden=64)
        export_renderer_student(source, student, root/'renderer.pt', training_steps=16200,
                                metrics={'passed': False, 'refinement_updates': 6600})
        del source.base_student
        del source.base_student_bundle
        return source

    def test_current_core_can_change_but_selected_assets_and_all_current_world_tensors_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = self.fixture(root)
            # The current dynamics differ from the historical renderer world.
            # They must be copied, never rolled back to the historical core.
            with torch.no_grad():
                source.vjepa_backend.core.weight.add_(.3)
            world_hash = _hash_tensors(source.state_dict())
            old_bytes = (root/'renderer.pt').read_bytes()
            rng = torch.get_rng_state().clone()
            nav, payload, proof = prepare_renderer_feature_profile(source, root/'renderer.pt')
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(source.config.navigation_only)
            self.assertFalse(source.vjepa_backend.config.navigation_only)
            self.assertTrue(nav.config.navigation_only)
            self.assertTrue(nav.vjepa_backend.config.navigation_only)
            self.assertIsNot(nav.vjepa_backend, source.vjepa_backend)
            self.assertEqual(_hash_tensors(source.state_dict()), world_hash)
            self.assertEqual(_hash_tensors(nav.state_dict()), world_hash)
            self.assertFalse(hasattr(source, 'base_student'))
            self.assertEqual((root/'renderer.pt').read_bytes(), old_bytes)
            old = torch.load(root/'renderer.pt', weights_only=False)
            for key in ('model', 'base_student_identity', 'student_codec_identity', 'metrics',
                        'codec_quality_qualified', 'experimental_preview', 'gameplay_qualified'):
                if key == 'model':
                    self.assertEqual(_hash_tensors(payload[key]), _hash_tensors(old[key]))
                elif key in old:
                    self.assertEqual(payload[key], old[key])
            self.assertFalse(payload['metrics']['passed'])
            self.assertEqual(proof['allowed_config_changes'], ['navigation_only'])
            self.assertFalse(proof['automatic_active_promotion'])
            torch.save(payload, root/'new.pt')
            load_renderer_student(nav, root/'new.pt')
            self.assertEqual(_hash_tensors(nav.flow_renderer.state_dict()), proof['flow_tensor_sha256'])

    def test_invalid_original_binding_or_incorrect_profile_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = self.fixture(root)
            original = _hash_tensors(source.state_dict())
            payload = torch.load(root/'renderer.pt', weights_only=False)
            payload['conditioning_identity']['appearance_sha256'] = 'invalid'
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                prepare_renderer_feature_profile(source, payload)
            self.assertEqual(_hash_tensors(source.state_dict()), original)
            bad = copy.deepcopy(source)
            bad.config = replace(bad.config, navigation_only=True)
            with self.assertRaisesRegex(ValueError, 'raw combat'):
                prepare_renderer_feature_profile(bad, root/'renderer.pt')

    def test_saved_inputs_have_current_step_and_exact_hash_pins_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = self.fixture(root)
            saved = {'stage': 'world', 'training_steps': 5150, 'config': asdict(source.config),
                     'model': source.state_dict(), 'optimizer': {'must_not_be_copied': True}}
            torch.save(saved, root/'snapshot.pt')
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            arguments = dict(source_checkpoint=root/'snapshot.pt', renderer_checkpoint=root/'renderer.pt',
                             output_directory=root/'output', source_checkpoint_sha256=digest(root/'snapshot.pt'),
                             renderer_checkpoint_sha256=digest(root/'renderer.pt'),
                             expected_codec_sha256=_hash_tensors(source.flow_renderer.codec.state_dict()))

            def tiny_loader(path, device):
                contents = torch.load(path, map_location=device, weights_only=False)
                model = TinyTeacher().eval()
                model.config = WorldModelConfig(**contents['config'])
                model.vjepa_backend.config = model.config
                model.load_state_dict(contents['model'])
                return model, contents

            with patch('turkeyworld.renderer_feature_inputs.load_world_checkpoint', side_effect=tiny_loader):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    prepare_current_renderer_inputs(**{**arguments, 'source_checkpoint_sha256': 'bad'})
                report = prepare_current_renderer_inputs(**arguments)
                self.assertEqual(report['source_saved_step'], 5150)
                self.assertTrue(report['strict_saved_inputs_reload_verified'])
                emitted = torch.load(root/'output/world_navigation.pt', weights_only=False)
                self.assertNotIn('optimizer', emitted)
                self.assertEqual(_hash_tensors(emitted['model']), _hash_tensors(saved['model']))
                self.assertTrue(emitted['config']['navigation_only'])
                persisted = json.loads((root/'output/feature_inputs_verification.json').read_text())
                self.assertEqual(persisted['source_checkpoint_sha256'], arguments['source_checkpoint_sha256'])
                with self.assertRaises(FileExistsError):
                    prepare_current_renderer_inputs(**arguments)
            self.assertEqual(digest(root/'snapshot.pt'), arguments['source_checkpoint_sha256'])


if __name__ == '__main__':
    unittest.main()
