"""Explicit B32x1 -> B16x1 preserves every earlier replay segment."""
from copy import deepcopy
import json
import unittest

import test_rssm_moe_micro32_batch_revision as legacy
from turkeyworld.rssm_moe_batch_revision import (
    MICRO32_PROTOCOL, MICRO16_SINGLE_PROTOCOL, MICRO16_SINGLE_SETTINGS,
    micro16_single_batch_revision_plan, bind_moe_batch_revision,
    moe_batch_settings, microbatch_layout,
)
from turkeyworld.rssm_moe_protocol import moe_training_horizon
from turkeyworld.rssm_player_navigation_revision import file_sha
from turkeyworld.renderer_native_phase import _same


class Micro16SingleRevisionTests(unittest.TestCase):
    def setUp(self):
        self.old=legacy.Micro32BatchRevisionTests('runTest');self.old.setUp()
        self.addCleanup(self.old.doCleanups);self.old.bind();self.old.update()
        self.state,self.root,self.source=self.old.state,self.old.root,self.old.source
        self.before=deepcopy(self.state)
        self.plan=micro16_single_batch_revision_plan(self.state,source_checkpoint=self.source)
        self.path=self.old.write('micro16_single.json',self.plan)

    def bind(self,state=None,plan=None,path=None):
        path=self.path if path is None else path
        return bind_moe_batch_revision(self.state if state is None else state,
            self.plan if plan is None else plan,path=path,sha256=file_sha(path),
            loaded_checkpoint_sha256=file_sha(self.source))

    def test_activation_retains_raw_old_gate_rng_and_samplers(self):
        record=self.bind();gate=self.state['rssm_training']['moe_training']
        old_gate=self.before['rssm_training']['moe_training']
        self.assertEqual(record['source_gate'],old_gate)
        restored=deepcopy(gate);restored['batch_revision']=old_gate['batch_revision']
        restored['batch_protocol']=MICRO32_PROTOCOL
        self.assertEqual(restored,old_gate)
        self.assertEqual(self.state['rssm_training']['navigation_replay_revision'],
                         self.before['rssm_training']['navigation_replay_revision'])
        for key in('_model','_optimizer','_rng'):
            self.assertTrue(_same(self.state[key],self.before[key]))
        self.assertEqual(moe_batch_settings(self.state),MICRO16_SINGLE_SETTINGS)
        layout=microbatch_layout(self.state,0)
        self.assertEqual((layout['combat'],layout['enemy_motion'],layout['navigation']), (4,8,4))
        self.assertAlmostEqual(layout['main_player_weight'],4/16)
        self.assertAlmostEqual(layout['main_enemy_weight'],4/12)
        self.assertAlmostEqual(layout['enemy_motion_enemy_weight'],8/12)
        self.assertEqual((layout['combat_auxiliary'],layout['combat_late'],layout['discarded_combat_rng_draws']), (16,2,12))
        self.assertEqual(gate['batch_protocol'],MICRO16_SINGLE_PROTOCOL)

    def test_new_update_counts_sixteen_without_rewriting_earlier_segments(self):
        self.bind();before=deepcopy(self.state)
        self.old.update()
        gate=self.state['rssm_training']['moe_training']
        self.assertEqual(gate['main_examples'],before['rssm_training']['moe_training']['main_examples']+16)
        horizon=moe_training_horizon(before)
        self.assertEqual(gate['main_transitions'],before['rssm_training']['moe_training']['main_transitions']+16*horizon)
        old=before['rssm_training']['navigation_replay_revision'];current=self.old.navigation_record
        for key,increment in(('combat_main_actors',4),('enemy_motion_main_actors',8),
                              ('navigation_main_actors',4),('discarded_combat_rng_draws',12)):
            self.assertEqual(current[key],old[key]+increment)
        restored=deepcopy(self.state);self.bind(state=restored)
        self.assertTrue(_same(restored,self.state))

    def test_wrong_parent_or_tampered_weights_fail_closed(self):
        with self.assertRaises(ValueError):
            micro16_single_batch_revision_plan(self.old.before,source_checkpoint=self.source)
        for mutate in(lambda p:p['settings'].update(gradient_accumulation_steps=4),
                       lambda p:p.update(previous_batch_revision_sha256='f'*64)):
            bad=deepcopy(self.plan);mutate(bad);path=self.old.write('bad16.json',bad)
            with self.assertRaises(ValueError):self.bind(plan=bad,path=path)


if __name__=='__main__':unittest.main()
