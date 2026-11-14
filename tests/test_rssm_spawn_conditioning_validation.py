import json
import unittest
from unittest.mock import patch

import torch

from turkeyworld.anchor_spawn import hierarchical_spawn_logits
from turkeyworld.rssm_spawn_conditioning import RSSMSpawnConditioning
from turkeyworld.rssm_spawn_conditioning_validation import audit_spawn_conditioning_inputs,spawn_confusion


class SpawnConditioningValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_all_eight_phases_safety_masks_phase_boundaries_and_death_boundary(self):
        module=RSSMSpawnConditioning(width=16)
        with patch('turkeyworld.engine.Game.step',side_effect=AssertionError('No Source step')),\
             patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No Source render')):
            report=audit_spawn_conditioning_inputs(module,expect_zero=True)
        self.assertTrue(report['passed'],report['checks']);self.assertEqual(report['synthetic_cases'],80)
        self.assertEqual(report['phase_boundary_cases'],24);json.dumps(report,allow_nan=False)

    def test_candidate_geometry_matches_existing_hierarchy_without_readiness_or_clock_oracle(self):
        module=RSSMSpawnConditioning(width=16);state=torch.zeros(8,20)
        state[:,:2]=torch.tensor([8.5,8.5])/16;state[:,4]=1;state[:,8:10]=torch.tensor([2.5,2.5])/16
        state[:,16]=89/18;state[:,19]=torch.arange(8)*3/300
        state[2:4,12:14]=1;state[3,8:10]=torch.tensor([13.5,13.5])/16
        floor=torch.ones(8,16,16,dtype=torch.bool);floor[1,2,2]=False;floor[4]=False
        out=module(state,torch.zeros(8,16),torch.zeros(8,2,16),floor)
        old=hierarchical_spawn_logits(torch.zeros(8,1,256),torch.zeros(8,1,9),state,floor,1/30)
        self.assertTrue(torch.equal(out['candidate_legal'],old['spawn_anchor_legal'][:,0]))
        poison=state.clone();poison[:,16]=0
        changed=module(poison,torch.zeros(8,16),torch.zeros(8,2,16),floor)
        self.assertTrue(torch.equal(out['candidate_legal'],changed['candidate_legal']))
        self.assertTrue(torch.equal(out['phase_index'],changed['phase_index']))

    def test_confusion_uses_all9_classes_and_never_fills_missing_phase_support_with_zero_accuracy(self):
        logits=torch.full((3,1,9),-3.);logits[0,0,0]=4;logits[1,0,2]=4;logits[2,0,8]=4
        legal=torch.ones_like(logits,dtype=torch.bool);legal[1,0,2]=False
        report=spawn_confusion(dict(spawn_anchor_logits=logits,spawn_anchor_legal=legal),[
            dict(anchor=0,clock_phase=0),dict(anchor=1,clock_phase=0),dict(anchor=8,clock_phase=7)])
        self.assertEqual(report['raw_anchor_confusion'][0][0],1)
        self.assertEqual(report['raw_anchor_confusion'][1][2],1)
        self.assertEqual(report['raw_anchor_confusion'][8][8],1)
        self.assertAlmostEqual(report['raw_anchor_top1_accuracy'],2/3)
        self.assertIsNone(report['per_phase']['3']['raw_anchor_top1_accuracy'])
        self.assertEqual(report['Source_anchor_own_legal_count'],3);json.dumps(report,allow_nan=False)


if __name__=='__main__':unittest.main()
