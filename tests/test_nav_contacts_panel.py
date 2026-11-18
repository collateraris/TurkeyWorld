import unittest
import numpy as np
from turkeyworld.validate_nav_contacts import denominator, manifest, nav_contacts, PROBE_OFFSETS


class ContactValidationTests(unittest.TestCase):
    def test_registered_panel_is_stable_valid_quiet_and_has_sufficient_contacts(self):
        episodes=nav_contacts()
        self.assertEqual(manifest(episodes)['fingerprint'],'2d4a440fa7fb3a2df55b0e765fcd7217d1e62c29dbadfa5a78dd5a38eeacd50f')
        self.assertEqual(len({e['trajectory_sha256'] for e in episodes}),24)
        for e in episodes:
            self.assertEqual(e['states'].shape,(184,20))
            self.assertFalse(e['actions'][:,3].any())
            self.assertTrue(np.isfinite(e['states']).all())
            np.testing.assert_array_equal(e['states'][:4,:2],np.tile(e['states'][3,:2],(4,1)))
            self.assertTrue(np.all(e['states'][:,13]==0)) # disabled enemy
        counts=denominator(episodes)
        for group in counts.values():
            self.assertGreater(group['blocked_axes'],80)
            self.assertGreater(group['free_axes'],170)
            self.assertEqual(group['free_axes']+group['partial_axes']+group['blocked_axes'], 12*len(PROBE_OFFSETS)*2)
            self.assertEqual(group['false_stops']+group['false_passes'],0)
