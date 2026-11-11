import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from test_renderer_student import TinyTeacher
from turkeyworld.base_student import build_base_student, export_base_student
from turkeyworld.renderer_student import (build_renderer_student, export_renderer_student,
                                         load_renderer_student, _hash_tensors)
from turkeyworld.renderer_profile_rebind import prepare_renderer_profile_rebind


class RendererProfileRebindTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root):
        source = TinyTeacher().eval()
        source.config = replace(source.config, architecture='dreamer_rssm', navigation_only=True)
        base = build_base_student(source)
        export_base_student(source, base, root/'base.pt', training_steps=3)
        source.base_student_bundle = torch.load(root/'base.pt', weights_only=False)
        source.base_student = base
        student = build_renderer_student(source, steps=1, width=32, depth=1, heads=4, mlp_hidden=64)
        # Bundle creation consumes the approved Base but the raw world must not
        # have an extra compact head during the exact source/target comparison.
        export_renderer_student(source, student, root/'old.pt', training_steps=9,
                                metrics={'passed': False})
        del source.base_student
        del source.base_student_bundle
        target = copy.deepcopy(source)
        target.config = replace(target.config, navigation_only=False)
        return source, target

    def test_new_profile_loads_with_identical_assets_and_preserves_inputs_rng_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, target = self.fixture(root)
            old_bytes = (root/'old.pt').read_bytes()
            source_hash, target_hash = _hash_tensors(source.state_dict()), _hash_tensors(target.state_dict())
            rng = torch.get_rng_state().clone()
            with self.assertRaises(ValueError):
                load_renderer_student(target, root/'old.pt')
            payload, proof = prepare_renderer_profile_rebind(source, target, root/'old.pt')
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual((root/'old.pt').read_bytes(), old_bytes)
            self.assertEqual(_hash_tensors(source.state_dict()), source_hash)
            self.assertEqual(_hash_tensors(target.state_dict()), target_hash)
            self.assertEqual(proof['allowed_config_changes'], ['navigation_only'])
            self.assertFalse(payload['metrics']['passed'])
            self.assertFalse(payload['gameplay_qualified'])
            torch.save(payload, root/'new.pt')
            load_renderer_student(target, root/'new.pt')
            self.assertFalse(target.config.navigation_only)
            self.assertEqual(_hash_tensors(target.flow_renderer.state_dict()), _hash_tensors(payload['model']))
            self.assertEqual(_hash_tensors(target.base_student.state_dict()), payload['base_student_identity'])

    def test_other_configuration_or_world_tensor_changes_rejected_transactionally(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, target = self.fixture(root)
            for invalid in (replace(target.config, simulation_hz=60),
                            replace(target.config, navigation_only=True)):
                bad = copy.deepcopy(target); bad.config = invalid
                with self.assertRaises(ValueError): prepare_renderer_profile_rebind(source, bad, root/'old.pt')
            for name in ('vjepa_backend.core.weight', 'vjepa_backend.renderer.weight',
                         'flow_renderer.latent_mean', 'flow_renderer.fixed_noise'):
                bad = copy.deepcopy(target)
                with torch.no_grad(): bad.state_dict()[name].flatten()[0].add_(.01)
                with self.assertRaises(ValueError): prepare_renderer_profile_rebind(source, bad, root/'old.pt')

    def test_invalid_old_binding_is_not_laundered_by_a_profile_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, target = self.fixture(root)
            payload = torch.load(root/'old.pt', weights_only=False)
            payload['conditioning_identity']['appearance_sha256'] = 'invalid'
            with self.assertRaises(ValueError): prepare_renderer_profile_rebind(source, target, payload)

    def test_original_jepa_bundle_requires_existing_rssm_lineage_before_profile_rebind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,target=self.fixture(root)
            payload=torch.load(root/'old.pt',weights_only=False)
            original={**payload['source_teacher_config'],'architecture':'vjepa2_ac'}
            payload['source_teacher_config']=original
            payload['base_student_bundle']['source_teacher_config']=copy.deepcopy(original)
            with self.assertRaisesRegex(ValueError,'lineage'):
                prepare_renderer_profile_rebind(source,target,payload)
            source.rssm_migration={'format_version':1,'source_architecture':'vjepa2_ac',
                'architecture':'dreamer_rssm','frozen_render_assets_bitexact':True,
                'source_teacher_config':original,
                'source_renderer_conditioning_identity':payload['conditioning_identity']}
            rebound,proof=prepare_renderer_profile_rebind(source,target,payload)
            self.assertEqual(proof['original_source_teacher_config'],original)
            self.assertEqual(rebound['source_teacher_config']['architecture'],'dreamer_rssm')
            self.assertFalse(rebound['source_teacher_config']['navigation_only'])
            self.assertEqual(rebound['base_student_bundle']['conditioning_identity'],
                             payload['base_student_bundle']['conditioning_identity'])


if __name__ == '__main__': unittest.main()
