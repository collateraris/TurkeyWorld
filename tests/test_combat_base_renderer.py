import unittest
from unittest.mock import patch

import torch

from turkeyworld.base_student import CompactBaseHead, BaseStudentConfig
from turkeyworld.combat_base_renderer import CombatBaseReadout, combat_bundle, load_combat_renderer


class CombatBaseReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_inactive_rows_in_mixed_batch_and_navigation_profile(self):
        head = CompactBaseHead(BaseStudentConfig(width=32, depth=1, heads=2))
        q = torch.randn(3, 256, 32)
        state = torch.zeros(3, 20)
        state[:, 3] = 1
        state[:, 8:10] = .5
        state[:, 13] = 1
        state[1, 7] = 1  # muzzle flash
        state[2, [13, 16, 18]] = torch.tensor([0., 0., 1.])  # current-tick death
        original = head._rgb_logits(q, state)
        head.combat_readout = CombatBaseReadout(32)
        self.assertTrue(torch.equal(head._rgb_logits(q, state), original))
        with torch.no_grad():
            head.combat_readout.mlp[-1].bias.fill_(.25)
        actual = head._rgb_logits(q, state)
        self.assertTrue(torch.equal(actual[0], original[0]))
        self.assertTrue(torch.equal(actual[1:], original[1:] + .25))
        self.assertTrue(torch.equal(head._rgb_logits(q, state, navigation_only=True), original))

    def test_invalid_fingerprint_is_rejected_before_touching_model(self):
        parent = dict(format_version=8, base_training_steps=31709, base_student_identity='1' * 64)
        bundle = combat_bundle(parent, CombatBaseReadout(32), steps=42)
        bundle['readout']['mlp.0.bias'][0] += 1
        with patch('turkeyworld.combat_base_renderer.load_decoder_rf_chain_payload') as load:
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                load_combat_renderer(None, bundle, checkpoint='dummy.pt')
            load.assert_not_called()
