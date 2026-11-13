import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from turkeyworld.models import WorldModelConfig

from training_runs.diagnostic_tools.prepare_rssm_five_roles_20261113 import (
    launch_file_pins, prepare_same_run, verify_prepared_payloads)


class PrepareRSSMFiveTests(unittest.TestCase):
    def test_cli_plan_pins_and_pid_guard_do_not_write_or_replace_any_production_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); plan = root / 'old_plan.json'; plan.write_text('{"version":1}')
            argv = ['python', '-m', 'turkeyworld.train_rssm_moe', '--moe-plan', str(plan), '--updates', '0']
            self.assertEqual(launch_file_pins(argv), {str(plan.resolve()): hashlib.sha256(plan.read_bytes()).hexdigest()})
            (root / 'training.pid').write_text('123')
            before = sorted(p.name for p in root.iterdir())
            with patch('turkeyworld.vqgan_pipeline.process_alive', return_value=True):
                with self.assertRaises(RuntimeError): prepare_same_run(root)
            self.assertEqual(before, sorted(p.name for p in root.iterdir()))
            self.assertEqual((root / 'training.pid').read_text(), '123')

    def test_payload_verifier_checks_whole_curriculum_pcg_rng_preview_and_old_name_order(self):
        config = asdict(WorldModelConfig(architecture='dreamer_rssm', enemy_rssm_enabled=True,
            rssm_fusion_moe_enabled=True, combat_rssm_enabled=True, spatial_entities=True, spatial_combat=True,
            player_motion_mode='axis_fraction', endless=True))
        old = dict(training_steps=5937, config=config, model={'old': torch.ones(1)},
            optimizer_parameter_names=['old'], optimizer={'state': {0: {'step': torch.tensor(2.)}},
                'param_groups': [{'params': [0], 'lr': .0001}]},
            torch_rng_state=torch.tensor([1, 2], dtype=torch.uint8), rng_state={'pcg': 7},
            run_state={'rssm_training': {'moe_training': {'current_horizon': 208, 'streak': 2},
                'samplers': {'navigation': {'rng': 'old', 'draws': 888}}},
                'training_protocol': {'old': True}, 'preview_assets_revision': {'sha256': 'old'},
                'model_parameters': {'old': 1}})
        new = copy.deepcopy(old); new['config']['rssm_five_role_enabled'] = True
        new['model']['vjepa_backend.five_role_fusion.new'] = torch.zeros(1)
        new['optimizer_parameter_names'].append('vjepa_backend.five_role_fusion.new')
        new['optimizer']['param_groups'].append({'params': [1], 'lr': .0001})
        new['run_state']['training_protocol']['rssm_five_role'] = {'sha256': 'new'}
        new['run_state']['rssm_five_training'] = {'enabled': True}
        self.assertTrue(verify_prepared_payloads(old, new)['passed'])
        legacy = copy.deepcopy(old)
        legacy['config'].pop('flow_output_moe_role_schema'); legacy['config'].pop('flow_output_moe_experts')
        result = verify_prepared_payloads(legacy, new)
        self.assertTrue(result['passed'])
        self.assertEqual(result['added_default_config_fields'], ['flow_output_moe_experts', 'flow_output_moe_role_schema'])
        for field in ('gate', 'pcg', 'rng', 'preview', 'moments', 'names'):
            broken = copy.deepcopy(new)
            if field == 'gate': broken['run_state']['rssm_training']['moe_training']['current_horizon'] = 4
            elif field == 'pcg': broken['run_state']['rssm_training']['samplers']['navigation']['draws'] = 0
            elif field == 'rng': broken['torch_rng_state'][0] = 0
            elif field == 'preview': broken['run_state']['preview_assets_revision']['sha256'] = 'other'
            elif field == 'moments': broken['optimizer']['state'][0]['step'].fill_(0)
            else: broken['optimizer_parameter_names'][0] = 'wrong'
            with self.subTest(field=field), self.assertRaises(AssertionError): verify_prepared_payloads(old, broken)


if __name__ == '__main__': unittest.main()
