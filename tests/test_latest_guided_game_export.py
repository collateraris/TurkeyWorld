"""Offline export/import checks; importing the launcher must never launch it."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from turkeyworld.atlas_enemy_sr import AtlasEnemyEDSRUpscaler
from turkeyworld.models import build_upscaler


def launcher():
    path=Path(__file__).resolve().parents[1]/'training_runs/diagnostic_tools/launch_latest_guided_game_20261113.py'
    spec=importlib.util.spec_from_file_location('offline_latest_guided_export',path)
    module=importlib.util.module_from_spec(spec)
    with patch('torch.cuda.init')as cuda_init,patch('subprocess.Popen')as spawn:
        spec.loader.exec_module(module)
        cuda_init.assert_not_called();spawn.assert_not_called()
    return module


class LatestGuidedGameExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_old_decoder_and_patchgan_clocks_remain_explicit_and_import_has_no_gpu_or_process_side_effect(self):
        module=launcher()
        source=dict(stage='sr',config={'architecture':'enemy_decoder_edsr','enemy_decoder_stable':True},
            model={'unchanged':torch.tensor([1.])},training_steps=53107,decoder_updates=7,
            enemy_decoder_migration={'kind':'lineage'},optimizer={'old':'moments'},
            optimizer_parameter_names=[['unchanged']],rng_torch=torch.get_rng_state(),
            sampler={'draws':56},parent_sampler={'draws':44000})
        export,facts=module.prepare_sr_play_export(source,'immutable.pt','a'*64)
        self.assertEqual((facts['sr_local_update_field'],facts['sr_local_updates']),('decoder_updates',7))
        self.assertEqual(export['training_steps'],53107)
        self.assertEqual(export['enemy_decoder_migration'],source['enemy_decoder_migration'])
        self.assertFalse(any(key.startswith(('optimizer','rng_'))for key in export))
        self.assertNotIn('sampler',export);self.assertNotIn('parent_sampler',export)
        self.assertIn('optimizer',source)
        patchgan={**source,'training_steps':98033,'decoder_updates':1853,'patchgan_updates':3,
            'patchgan_state':{'control_optimizer':{'large':'discard'}},'patchgan_migration':{'preserved':True}}
        export,facts=module.prepare_sr_play_export(patchgan,'immutable.pt','b'*64)
        self.assertEqual((facts['sr_local_update_field'],facts['sr_local_updates']),('patchgan_updates',3))
        self.assertEqual(facts['decoder_updates'],1853)
        self.assertNotIn('patchgan_state',export);self.assertIn('patchgan_migration',export)

    def test_embedded_atlas_config_pixels_provenance_and_local_cumulative_counters_strict_reload(self):
        module=launcher()
        model=AtlasEnemyEDSRUpscaler(channels=4,blocks=1,semantic_channels=4,
            semantic_blocks=1,enemy_decoder_channels=4)
        source=dict(stage='sr',config=deepcopy(model.config),model=deepcopy(model.state_dict()),
            training_steps=98507,atlas_updates=7,semantic_updates=7,
            atlas_migration={'kind':'fixed_atlas_enemy_sr_v1','source_sha256':'a'*64},
            restoration_migration={'preserved':True},optimizer={'source':'moments'},
            sampler={'draws':391656},rng_torch=torch.get_rng_state())
        export,facts=module.prepare_sr_play_export(source,'immutable_atlas.pt','c'*64)
        self.assertEqual((facts['sr_training_steps'],facts['sr_local_updates'],facts['atlas_updates']),(98507,7,7))
        self.assertEqual(facts['sr_local_update_field'],'atlas_updates')
        self.assertTrue(facts['atlas_embedded'])
        self.assertEqual(export['config'],source['config'])
        self.assertEqual(export['atlas_migration'],source['atlas_migration'])
        self.assertEqual(export['restoration_migration'],source['restoration_migration'])
        self.assertEqual(export['model']['atlas_frames_rgba'].dtype,torch.uint8)
        loaded=build_upscaler(export['config']);loaded.load_state_dict(export['model'],strict=True)
        for name,value in source['model'].items():
            torch.testing.assert_close(value,loaded.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(export['latest_play_snapshot']['sr_local_updates'],7)
        self.assertFalse(export['gameplay_qualified'])

    def test_damaged_atlas_pixels_or_missing_local_stage_clock_reject(self):
        module=launcher();model=AtlasEnemyEDSRUpscaler(channels=4,blocks=1,
            semantic_channels=4,semantic_blocks=1,enemy_decoder_channels=4)
        source=dict(stage='sr',config=model.config,model=model.state_dict(),training_steps=98500,
            atlas_updates=0,atlas_migration={'kind':'proof'})
        broken={**source,'model':deepcopy(source['model'])}
        broken['model']['atlas_frames_rgba'].view(-1)[0]^=1
        with self.assertRaisesRegex(ValueError,'embedded native pixels'):
            module.prepare_sr_play_export(broken,'source.pt','d'*64)
        missing={k:v for k,v in source.items()if k!='atlas_updates'}
        with self.assertRaisesRegex(ValueError,'local update clock'):
            module.prepare_sr_play_export(missing,'source.pt','d'*64)


if __name__=='__main__':unittest.main()
