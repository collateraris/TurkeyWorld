"""The optional-hint protocol measures every held-out positive and negative."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from test_enemy_decoder_validation import GuidedPairs
from test_semantic_sr_validation import TinySR
from turkeyworld.enemy_decoder_sr import EnemyDecoderEDSRUpscaler
from turkeyworld.semantic_sr_validation import (evaluate_semantic_sr, preview_semantic_sr,
    save_semantic_sr_preview, _current_forward)
from turkeyworld.train_semantic_sr import mix_visual_first_hypotheses


class VisualFirstValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = EnemyDecoderEDSRUpscaler(channels=8, blocks=1, semantic_channels=4,
            semantic_blocks=1, enemy_decoder_channels=4, enemy_decoder_stable=True,
            enemy_decoder_visual_first=True)
        with torch.no_grad():
            model.enemy_decoder_global_presence.weight.zero_()
            model.enemy_decoder_global_presence.bias.fill_(10)
            model.enemy_decoder_visual_confirmation[-1].weight.zero_()
            model.enemy_decoder_visual_confirmation[-1].bias.fill_(10)
            model.enemy_decoder_visual_gain.weight.fill_(1)
        return model

    def test_all_hint_modes_full_denominators_exact_high_visual_invariance_and_six_rows(self):
        model, data = self.model(), GuidedPairs()
        sampler = data.sampler_state(); rng = torch.get_rng_state().clone()
        report, preview = evaluate_semantic_sr(model, data, list(range(8)), step=100, batch_size=4,
            baselines={'sr38100': TinySR(.04), 'sr40100': TinySR(.02)})
        self.assertEqual(report['kind'], 'visual_first_enemy_decoder_super_resolution_v2')
        self.assertEqual(report['validation_objective']['protocol'], 'visual_first_enemy_decoder_sr_v2')
        comparison = report['enemy_decoder']['visual_first']
        self.assertEqual(set(comparison['domains']), {'with_hint','no_hint','wrong_hint'})
        for values in comparison['domains'].values():
            self.assertEqual((values['frames'], values['positive_frames'], values['negative_frames']), (8,6,2))
            self.assertEqual((values['visual_route_frames'],values['guided_route_frames']), (8,0))
            self.assertEqual((values['tp'],values['fp']), (6,2))
        invariant = comparison['high_visual_output_invariance']
        self.assertEqual(invariant['frames'],8);self.assertTrue(invariant['passed'])
        self.assertEqual(sampler,data.sampler_state());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        light = preview_semantic_sr(model,data,list(range(8)),step=15,baseline_report=report)
        self.assertEqual(light['metadata']['row_count'],6)
        for case in light['cases'].values():
            np.testing.assert_array_equal(case['predictions']['current'],case['predictions']['no_hint'])
        with tempfile.TemporaryDirectory() as temporary:
            metadata = save_semantic_sr_preview(light,Path(temporary)/'preview.png',every=15)
            self.assertEqual(metadata['comparison_modes'],['with_hint','no_hint'])
            with Image.open(Path(temporary)/'preview.png') as image:
                self.assertEqual(image.size,(1024,2552))
        json.dumps(report,allow_nan=False)

    def test_missing_hint_and_future_label_poison_do_not_change_visual_forward(self):
        model, data = self.model().eval(), GuidedPairs()
        batch = data.batch(4,indices=list(range(4)))
        with torch.no_grad():
            no_hint = _current_forward(model,batch['lr_npr'],{})['image']
            altered = _current_forward(model,batch['lr_npr'],{'state':torch.full((4,20),float('nan')),
                'hr_source':torch.ones_like(batch['hr_source']),'mask_hr':torch.zeros_like(batch['mask_hr'])})['image']
        torch.testing.assert_close(no_hint,altered,rtol=0,atol=0)

    def test_half_dropped_quarter_wrong_predictions_preserve_inputs_and_rng_repeatability(self):
        data=GuidedPairs();batch=data.batch(8,indices=list(range(8)));original=batch['hypothesis'].clone()
        first, dropped, wrong = mix_visual_first_hypotheses(original,np.random.default_rng(17))
        second, d2, w2 = mix_visual_first_hypotheses(original,np.random.default_rng(17))
        self.assertEqual((dropped,wrong,d2,w2),(4,2,4,2))
        self.assertEqual(int((first[:,-1]==0).sum()),4)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        torch.testing.assert_close(original,batch['hypothesis'],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
