"""Independent Game/typed-state event timing, without a neural checkpoint."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

_path = Path(__file__).resolve().with_name('respawn_clock_validation_support.py')
_spec = importlib.util.spec_from_file_location('clock_coverage_audit', _path)
audit = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(audit)


class RespawnClockCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_same_canonical_badq_roundtrip_now_has_same_birth(self):
        result = audit.canonical_probe(badq=True)
        self.assertEqual(result['typed_birth_offsets'], [1])
        self.assertTrue(result['canonical_roundtrip_exact'])

    def test_all_four_future_offsets_with_one_and_two_observed_frames(self):
        for context in (1, 2):
            for offset in (1, 2, 3, 4):
                with self.subTest(context=context, offset=offset):
                    result = audit.canonical_probe(context=context, offset=offset)
                    self.assertEqual(result['Source_birth_offsets'], [offset])
                    self.assertEqual(result['typed_birth_offsets'], [offset])

    def test_full_death90_ticks_not89_or91(self):
        self.assertEqual(audit.canonical_probe(ticks=90)['typed_birth_offsets'], [90])

    def test_legacy_reference_keeps_actual_failed_label_and_revision_restores_it(self):
        initial = np.float32((3.-2/30)/.6)
        before, after = audit.clock_birth(initial, 4, revised=False), audit.clock_birth(initial, 4, revised=True)
        self.assertEqual(before['offset'], 2); self.assertFalse(before['source_label'])
        self.assertEqual(after['offset'], 2); self.assertTrue(after['source_label'])


if __name__ == '__main__': unittest.main()
