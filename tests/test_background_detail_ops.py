"""No-process CPU tests for the new-architecture launch/probe conservation guards."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('background_detail_operator',
    Path(__file__).resolve().parents[1]/'training_runs/diagnostic_tools/background_detail_ops_20261114.py')
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


class BackgroundDetailOperatorTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        source = self.root/'source.pt'; source.write_bytes(b'full-source5308-readonly')
        (self.root/'training_plan.json').write_text('{}')
        self.source_sha = operator.runner.file_sha(source)
        self.preparation = dict(source_checkpoint=str(source), new_branch_parameters=526131)
        self.code = {'train_background_detail_sr.py': 'same-code', operator.BOOTSTRAP.name: 'same-bootstrap'}
        self.proof = dict(kind='background_detail_discarded_three_step_proof_v1', passed=True,
            plan_sha256=operator.runner.file_sha(self.root/'training_plan.json'),
            source_checkpoint_sha256=self.source_sha, preferred_background_sha256=operator.runner.BACKGROUND_SHA,
            code_sha256=self.code, checks={name: True for name in operator.REQUIRED_PROBE_CHECKS},
            budget=3000, batch=8, local_steps=3, new_Adam_states=42, source_5308_Adam_reused=False,
            trainable_parameters=526131, production_modified=False, weights_promoted=False)

    def test_bootstrap_only_changes_entrypoint_preserving_all_canonical_settings(self):
        canonical = operator.canonical_argv(self.root)
        before = canonical.copy()
        actual = operator.actual_argv(canonical)
        self.assertEqual(canonical, before)
        self.assertEqual(actual[:4], canonical[:4])
        self.assertEqual(actual[4:7], [str(operator.BOOTSTRAP), '--module', operator.MODULE])
        self.assertEqual(actual[7:], canonical[6:])
        for flag, value in (('--batch', '8'), ('--steps', '3000'), ('--validate-every', '100'), ('--prediction-every', '15')):
            self.assertEqual(canonical[canonical.index(flag)+1], value)

    def test_complete_proof_passes_without_initializing_CUDA_or_spawning_a_process(self):
        with patch.object(operator.runner, 'SOURCE_SHA', self.source_sha):
            operator.require_probe_proof(self.root, self.preparation, self.proof, self.code)

    def test_partial_or_changed_conservation_evidence_cannot_launch(self):
        for kind in ('missing_check', 'failed_check', 'changed_source', 'changed_plan', 'changed_code',
                     'wrong_steps', 'wrong_batch', 'old_Adam', 'promoted_weights'):
            with self.subTest(kind=kind), patch.object(operator.runner, 'SOURCE_SHA', self.source_sha):
                proof = deepcopy(self.proof)
                if kind == 'missing_check': proof['checks'].pop(operator.REQUIRED_PROBE_CHECKS[0])
                elif kind == 'failed_check': proof['checks'][operator.REQUIRED_PROBE_CHECKS[1]] = False
                elif kind == 'changed_source': proof['source_checkpoint_sha256'] = 'old-source'
                elif kind == 'changed_plan': proof['plan_sha256'] = 'old-plan'
                elif kind == 'changed_code': proof['code_sha256']['train_background_detail_sr.py'] = 'old-code'
                elif kind == 'wrong_steps': proof['local_steps'] = 2
                elif kind == 'wrong_batch': proof['batch'] = 4
                elif kind == 'old_Adam': proof['source_5308_Adam_reused'] = True
                else: proof['weights_promoted'] = True
                with self.assertRaises(ValueError): operator.require_probe_proof(self.root, self.preparation, proof, self.code)

    def test_new_source_bytes_invalidate_an_old_successful_probe(self):
        Path(self.preparation['source_checkpoint']).write_bytes(b'changed')
        with patch.object(operator.runner, 'SOURCE_SHA', self.source_sha):
            with self.assertRaises(ValueError): operator.require_probe_proof(self.root, self.preparation, self.proof, self.code)

    def test_code_pins_include_model_objective_data_validation_and_safe_bootstrap(self):
        code = operator.code_sha256()
        for name in ('background_detail_sr.py', 'background_detail_objectives.py',
                     'background_detail_sr_validation.py', 'train_background_detail_sr.py',
                     'neural_enemy_restoration_data.py', operator.BOOTSTRAP.name):
            self.assertIn(name, code); self.assertEqual(len(code[name]), 64)


if __name__ == '__main__': unittest.main()
