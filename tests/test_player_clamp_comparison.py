import unittest
from turkeyworld.report_player_clamp_ab import compare, INDEPENDENT_FINGERPRINT, STREAMS


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.plan = dict(start_step=10, target_step=11)
        self.receipts = {name: [dict(step=11, horizon=180, main_actors=16, wall_coefficient=.25,
                    samples=[{'episode':i} for i in range(16)],player_clamp_gradient=name)] for name in ('hard','ste')}
        self.reports = {}
        for name, drift, stops in [('hard', 1., 3), ('ste', .9, 2)]:
            self.reports[name] = dict(step=11, panel_fingerprint=INDEPENDENT_FINGERPRINT, panel='independent',
                       feedback_mode='on', horizon=180, objective_wall_coefficient=.25,
                       objective_auxiliary_mode='blocks16',Source_future_feedback=False)
            for stream in STREAMS:
                self.reports[name][stream] = dict(count=8,player_XY_H180_mean_tiles=drift,
                       player_wall_overlap_ticks_mean=5,enemy_XY_H180_mean_tiles=.8,
                       common_start_axis=dict(free_axes=16,blocked_axes=8,false_stops=stops,false_passes=1))

    def test_safety_gates_and_matched_replay(self):
        self.assertTrue(compare(self.plan,self.reports,self.receipts)['passes_pilot'])
        self.reports['ste']['enemy_motion']['player_wall_overlap_ticks_mean'] += .1
        self.assertFalse(compare(self.plan,self.reports,self.receipts)['passes_pilot'])
        self.receipts['ste'][0]['samples'].reverse()
        with self.assertRaisesRegex(ValueError,'TRAIN windows'): compare(self.plan,self.reports,self.receipts)

    def test_mismatched_probe_denominator(self):
        self.reports['ste']['navigation']['common_start_axis']['blocked_axes'] += 1
        with self.assertRaisesRegex(ValueError,'denominators'): compare(self.plan,self.reports,self.receipts)
