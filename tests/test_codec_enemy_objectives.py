from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.codec_enemy_objectives import codec_enemy_loss
from turkeyworld.compact_codec import CompactCodec
from turkeyworld.engine import Game
from turkeyworld.vqgan import VQGAN, VQGANConfig


def fixture(empty=False):
    config = VQGANConfig(image_size=32, base_channels=4, channel_multipliers=(1, 2, 3),
        num_res_blocks=1, latent_channels=8, embedding_dim=4, n_embed=8, attention=False, data_init=False)
    student = CompactCodec(config)
    original = VQGAN(config).eval().requires_grad_(False)
    original.quantize.load_state_dict(student.quantize.state_dict())
    old = deepcopy(student).eval().requires_grad_(False)
    images, backgrounds, masks = [], [], []
    for seed in range(2):
        game = Game(seed=seed, endless=True)
        image = game.render(size=32, overlays=False)
        game.enemies = []
        background = game.render(size=32, overlays=False)
        mask = np.any(image != background, axis=-1)
        if empty:
            image, mask = background, np.zeros_like(mask)
        images.append(image); backgrounds.append(background); masks.append(mask)
    target = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()/255
    background = torch.from_numpy(np.stack(backgrounds)).permute(0, 3, 1, 2).float()/255
    mask = torch.from_numpy(np.stack(masks))[:, None].float()
    if not empty:
        assert bool(mask.any()), 'The actual Source fixture must contain visible NPC pixels'
    with torch.no_grad():
        teacher_z = original.encode(target)
        base_rgb = .8*target+.1
        base_z = old.encode(base_rgb)
        rf_z = old.quantize(base_z['prequant']+.02)['quantized']
        references = {'original_prequant': teacher_z['prequant'], 'original_quantized': teacher_z['quantized'],
            'base_rgb': base_rgb, 'old_base_prequant': base_z['prequant'], 'rf_quantized': rf_z,
            'old_rf_rgb': old.decode(rf_z), 'original_rf_rgb': original.decode(rf_z)}
    batch = {name: value.detach().requires_grad_(True) for name, value in
             {'target': target, 'background': background, 'object_mask': mask}.items()}
    references = {name: value.detach().requires_grad_(True) for name, value in references.items()}
    return student, original, old, batch, references


def same_state(before, module):
    return all(torch.equal(value, module.state_dict()[name]) for name, value in before.items())


class CodecEnemyObjectivesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_decoder_warmup_uses_original_codes_without_encoder_or_quantizer_forward(self):
        student, original, old, batch, references = fixture()
        student.train().requires_grad_(False)
        student.decoder.requires_grad_(True)
        student.post_quant_conv.requires_grad_(True)
        q_before = {key: value.clone() for key, value in student.quantize.state_dict().items()}
        before = {key: value.clone() for key, value in student.state_dict().items()}
        original_before = {key: value.clone() for key, value in original.state_dict().items()}
        old_before = {key: value.clone() for key, value in old.state_dict().items()}
        optimizer = torch.optim.AdamW([value for value in student.parameters() if value.requires_grad], lr=1e-4)
        with patch.object(student, 'encode', side_effect=AssertionError('No E warmup')), \
                patch.object(student.quantize, 'forward', side_effect=AssertionError('No Q warmup')):
            loss, metrics = codec_enemy_loss(student, batch, references, decoder_only=True)
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(value.grad is None for value in student.encoder.parameters()))
        self.assertTrue(all(value.grad is None for value in student.quant_conv.parameters()))
        self.assertGreater(float(student.decoder.conv_out.weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertTrue(same_state(q_before, student.quantize))
        self.assertTrue(all(torch.equal(value, student.state_dict()[key]) for key, value in before.items()
                            if key.startswith(('encoder.', 'quant_conv.'))))
        self.assertFalse(torch.equal(before['decoder.conv_out.weight'], student.decoder.conv_out.weight))
        self.assertTrue(same_state(original_before, original))
        self.assertTrue(same_state(old_before, old))
        self.assertTrue(all(value.grad is None for value in (*batch.values(), *references.values())))
        self.assertTrue(metrics['decoder_only'])
        self.assertFalse(metrics['encoder_forward_active'])
        self.assertNotIn('encoder_prequant_mse', metrics)
        self.assertFalse(metrics['gan_active'])

    def test_joint_real_source_roi_updates_e_and_d_but_preserves_every_quantizer_tensor(self):
        student, _, _, batch, references = fixture()
        student.requires_grad_(True).train()
        self.assertFalse(student.quantize.training)
        q_before = {key: value.clone() for key, value in student.quantize.state_dict().items()}
        e_before, d_before = student.quant_conv.weight.clone(), student.decoder.conv_out.weight.clone()
        optimizer = torch.optim.AdamW([value for value in student.parameters() if value.requires_grad], lr=1e-4)
        ids = {id(value) for group in optimizer.param_groups for value in group['params']}
        self.assertFalse(ids & {id(value) for value in student.quantize.parameters()})
        with patch('turkeyworld.engine.Game.step', side_effect=AssertionError('Only recorded Source labels')), \
                patch('turkeyworld.engine.Game.render', side_effect=AssertionError('Only recorded Source labels')):
            loss, metrics = codec_enemy_loss(student, batch, references)
            loss.backward()
        self.assertGreater(float(student.quant_conv.weight.grad.abs().sum()), 0)
        self.assertGreater(float(student.decoder.conv_out.weight.grad.abs().sum()), 0)
        self.assertTrue(all(value.grad is None for value in student.quantize.parameters()))
        self.assertTrue(all(value.grad is None for value in (*batch.values(), *references.values())))
        optimizer.step()
        self.assertFalse(torch.equal(e_before, student.quant_conv.weight))
        self.assertFalse(torch.equal(d_before, student.decoder.conv_out.weight))
        self.assertTrue(same_state(q_before, student.quantize))
        self.assertGreater(metrics['visible_enemy_examples'], 0)
        self.assertEqual(metrics['weight_encoder_prequant_mse'], .25)
        self.assertEqual(metrics['weight_encoder_vq_loss'], .1)
        self.assertEqual(metrics['weight_base_encoder_compat_mse'], .5)
        self.assertEqual(metrics['weight_rf_old_replay_l1'], .25)
        self.assertEqual(metrics['weight_rf_original_replay_l1'], .05)
        self.assertAlmostEqual(metrics['train_loss'], sum(value for key, value in metrics.items()
                                                       if key.startswith('weighted_')), places=5)

    def test_empty_npc_labels_are_finite_and_foreground_contrast_is_exactly_zero(self):
        student, _, _, batch, references = fixture(empty=True)
        student.requires_grad_(True).train()
        for decoder_only in (True, False):
            with self.subTest(decoder_only=decoder_only):
                student.zero_grad(set_to_none=True)
                loss, metrics = codec_enemy_loss(student, batch, references, decoder_only=decoder_only)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(metrics['visible_enemy_examples'], 0)
                self.assertEqual(metrics['decoder_source_contrast'], 0)
                if not decoder_only:
                    self.assertEqual(metrics['encoder_source_contrast'], 0)

    def test_training_or_unfrozen_or_uninitialized_quantizer_is_rejected_before_forward(self):
        student, _, _, batch, references = fixture()
        student.train().requires_grad_(True)
        student.quantize.train()
        with self.assertRaisesRegex(ValueError, 'frozen quantizer'):
            codec_enemy_loss(student, batch, references)
        student.quantize.eval()
        student.quantize.embedding.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen quantizer'):
            codec_enemy_loss(student, batch, references)
        student.quantize.requires_grad_(False)
        student.quantize.initialized.fill_(False)
        with self.assertRaisesRegex(ValueError, 'initialized frozen'):
            codec_enemy_loss(student, batch, references)


if __name__ == '__main__':
    unittest.main()
