from copy import deepcopy
import unittest

import numpy as np
import torch

from test_codec_enemy_objectives import fixture
from test_renderer_enemy_objectives import tiny_model, batch
from turkeyworld.compact_codec import CompactCodec
from turkeyworld.codec_enemy_objectives import codec_enemy_loss
from turkeyworld.train_codec_enemy import (configure_codec_phase, prepare_references,
    evaluate_codec_enemy, codec_assessment, attach_terminal_assessment)


class CodecEnemyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_warmup_then_joint_one_adam_preserves_quantizer_and_updates_requested_modules(self):
        student, original, old, source, refs = fixture()
        e = list(student.encoder.parameters())+list(student.quant_conv.parameters())
        d = list(student.decoder.parameters())+list(student.post_quant_conv.parameters())
        optimizer = torch.optim.AdamW([{'params': e, 'lr': 2e-5}, {'params': d, 'lr': 1e-4}])
        fixed = {key: value.clone() for key, value in student.quantize.state_dict().items()}
        before_e = [value.clone() for value in e]
        configure_codec_phase(student, True)
        loss, _ = codec_enemy_loss(student, source, refs, decoder_only=True)
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_e, e)))
        self.assertTrue(all(value not in optimizer.state for value in e))
        d_step = [int(optimizer.state[value]['step']) for value in d]
        configure_codec_phase(student, False)
        loss, _ = codec_enemy_loss(student, source, refs)
        loss.backward(); optimizer.step()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_e, e)))
        self.assertTrue(all(int(optimizer.state[value]['step']) == 1 for value in e))
        self.assertTrue(all(int(optimizer.state[value]['step']) == step+1 for value, step in zip(d, d_step)))
        self.assertTrue(all(torch.equal(value, student.quantize.state_dict()[key]) for key, value in fixed.items()))

    def test_full_new_encoder_rf_validation_restores_parent_and_preserves_rng_and_trainable_student(self):
        model = tiny_model().eval().requires_grad_(False)
        old = model.flow_renderer.codec
        student = CompactCodec(old.config)
        student.load_state_dict(old.state_dict())
        configure_codec_phase(student, False)
        original = deepcopy(old).eval().requires_grad_(False)
        source = batch((False, True, False))
        class Data:
            rng = np.random.default_rng(173)
            rows = [{'category': category} for category in ('large', 'empty_hidden', 'corpse')]
            def batch(self, count, *, indices):
                return {key: value[indices] for key, value in source.items()}
            def panel_fingerprint(self, panel): return 'fixed-panel'
        data = Data()
        conditions, refs = prepare_references(model, original, source)
        self.assertTrue(all(not value.requires_grad for value in refs.values()))
        q_before = {key: value.clone() for key, value in student.quantize.state_dict().items()}
        rng = torch.get_rng_state().clone()
        report, arrays = evaluate_codec_enemy(student, model, original, data, [0, 1, 2], batch_size=2)
        self.assertIs(model.flow_renderer.codec, old)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(student.training)
        self.assertTrue(all(p.requires_grad for p in student.encoder.parameters()))
        self.assertFalse(student.quantize.training)
        self.assertTrue(all(torch.equal(v, student.quantize.state_dict()[k]) for k, v in q_before.items()))
        self.assertTrue(torch.equal(arrays['full_rf1'], arrays['old_full_rf1']))
        self.assertEqual(report['base_encoder_index_agreement_with_parent'], 1.)
        self.assertFalse(report['gameplay_qualified'])
        self.assertEqual(report['frames'], 3)

    def test_empty_regression_cannot_be_hidden_by_good_enemy_reconstruction(self):
        visible = {'foreground_mae': .03, 'closer_to_source_than_without_npc_rate': 1.}
        def row(mae):
            return {'overall': {'mae_mean': mae, 'ssim_mean': .95}, 'visible_enemy': visible,
                'categories': {key: {'frames': 8, 'mae_mean': mae, 'ssim_mean': .95}
                    for key in ('empty_navigation', 'empty_hidden')}}
        baseline = {'source_reconstruction': row(.01), 'full_rf1': row(.02)}
        current = deepcopy(baseline)
        self.assertTrue(codec_assessment(current, baseline)['passed'])
        current['full_rf1']['categories']['empty_hidden']['mae_mean'] = .03
        assessed = codec_assessment(current, baseline)
        self.assertTrue(assessed['source_quality_passed'])
        self.assertFalse(assessed['passed'])
        self.assertIn('full_rf1_empty_hidden_regression', assessed['failure_reasons'])

    def test_hp0_gain_cannot_hide_alive_npc_regression_inside_better_global_average(self):
        def prediction(roi):
            return {'overall': {'frames': 256, 'mae_mean': .02, 'ssim_mean': .95},
                'visible_enemy': {'foreground_mae': roi, 'closer_to_source_than_without_npc_rate': .8}}
        baseline = {'passed': True, 'rf_and_empty_guards_passed': True,
            'source_reconstruction': prediction(.10), 'full_rf1': prediction(.12),
            'player_health_cohorts': {'alive': {'source_reconstruction': prediction(.09), 'full_rf1': prediction(.11)}},
            'terminal_supplement': {'source_reconstruction': {'overall': {'frames': 18, 'mae_mean': .12, 'ssim_mean': .6}}}}
        current = deepcopy(baseline)
        current['source_reconstruction']['visible_enemy']['foreground_mae'] = .08
        current['player_health_cohorts']['alive']['source_reconstruction']['visible_enemy']['foreground_mae'] = .10
        terminal = {'source_reconstruction': {'overall': {'frames': 18, 'mae_mean': .02, 'ssim_mean': .95}}}
        assessed = attach_terminal_assessment(current, terminal, baseline=baseline)
        self.assertFalse(assessed['passed'])
        self.assertFalse(assessed['rf_and_empty_guards_passed'])
        self.assertIn('source_reconstruction_alive_npc_regression', assessed['terminal_focus']['failure_reasons'])


if __name__ == '__main__': unittest.main()
