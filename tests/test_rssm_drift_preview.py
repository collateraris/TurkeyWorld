import copy
import json
import math
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.rssm_drift_preview import build_drift_preview


def fixture(context=1, horizon=4):
    source = np.zeros((1, context+horizon, 20), dtype=np.float64)
    source[0, :, 0] = np.arange(context+horizon)/100+.1
    source[0, :, 1] = .2
    source[0, :, 3:5] = 1.
    source[0, :, 8:10] = (.5, .6)
    source[0, :, 10:12] = (.3, .4)
    source[0, :, 12:14] = 1.
    grid = np.zeros((1,16,16), dtype=np.int64)
    grid[0,2,7], grid[0,11,4] = 3, 1
    batch = {'states': source, 'map': grid}
    spec = {'context': context, 'horizon': horizon,
            'display_offsets': np.rint(np.linspace(1,horizon,4)).astype(int).tolist()}
    return batch, spec, source[0,context-1:context+horizon].copy()


class DriftPreviewTests(unittest.TestCase):
    def test_exact_map_axes_tile_scaling_signed_errors_and_direction_semantics(self):
        batch, spec, own = fixture()
        own[1, :2] += (3/16, 4/16)
        own[1, 2:4] = (1., 0.)
        own[1, 8:10] += (0., 1/16)
        result = build_drift_preview(batch, spec, own, step=5816)
        self.assertEqual(result['wall_cells'], [[7,2,3], [4,11,1]])
        self.assertEqual(result['world_map'][2][7], 3)
        self.assertEqual(result['reference']['player_xy'][0], [1.6, 3.2])
        self.assertAlmostEqual(result['errors']['player_xy_error_tiles'][1], 5.)
        self.assertAlmostEqual(result['errors']['player_dx_tiles'][1], 3.)
        self.assertAlmostEqual(result['errors']['player_dy_tiles'][1], 4.)
        self.assertEqual(result['reference']['player_heading'][1], 0.)
        self.assertEqual(result['prediction']['player_heading'][1], math.pi/2)
        self.assertAlmostEqual(result['errors']['player_heading_error_degrees'][1], 90.)
        self.assertEqual(result['reference']['npc_velocity_xy'][1], [.6,.8])
        self.assertAlmostEqual(result['reference']['npc_direction'][1], math.atan2(.8,.6))
        self.assertAlmostEqual(result['errors']['npc_xy_error_tiles'][1], 1.)
        self.assertEqual(result['display_frames'][0]['offset'], 1)
        self.assertEqual(result['display_frames'][0]['errors']['player_xy_error_tiles'], 5.)

    def test_context_one_and_two_use_same_initial_frame_and_exact_full_horizon(self):
        for context in (1,2):
            for horizon in (4,112,450):
                with self.subTest(context=context,horizon=horizon):
                    batch, spec, own = fixture(context,horizon)
                    result = build_drift_preview(batch,spec,own,step=9)
                    self.assertEqual(len(result['reference']['player_xy']),horizon+1)
                    self.assertEqual(result['source_frame_indices'],list(range(context-1,context+horizon)))
                    self.assertEqual(result['reference']['player_xy'][0],(batch['states'][0,context-1,:2]*16).tolist())
                    self.assertEqual(result['reference'],result['prediction'])
                    self.assertEqual([row['offset']for row in result['display_frames']],spec['display_offsets'])
                    self.assertEqual(result['time_seconds'][-1],horizon/30)
                    self.assertEqual(result['transition_hz'],30)
                    self.assertEqual(result['summary']['player_xy_error_tiles']['count'],horizon)

    def test_alive_disagreement_is_null_and_exact_birth_is_not_continuing_drift(self):
        batch,spec,own=fixture()
        source=batch['states'][0]
        source[1:3,12:14]=0.
        own[1:4,12:14]=0.  # Prediction misses the Source birth at3.
        own[4,8:10]=(.7,.8)
        result=build_drift_preview(batch,spec,own,step=10)
        error=result['errors']
        self.assertEqual(result['reference']['npc_respawn'],[False,False,False,True,False])
        self.assertEqual(result['prediction']['npc_respawn'],[False,False,False,False,True])
        self.assertEqual(result['reference']['npc_life_index'],[0,0,0,1,1])
        self.assertEqual(result['prediction']['npc_life_index'],[0,0,0,0,1])
        self.assertEqual(error['npc_xy_error_tiles'][1:4],[None,None,None])
        self.assertTrue(error['npc_alive_mismatch'][3])
        self.assertIsNone(error['npc_continuing_xy_error_tiles'][3])
        self.assertIsNotNone(error['npc_continuing_xy_error_tiles'][4])
        self.assertEqual(result['summary']['npc_common_alive_xy_error_tiles']['count'],1)
        self.assertEqual(result['summary']['npc_alive_mismatch_frames'],1)
        self.assertIsNone(result['reference']['npc_motion_direction'][3])
        self.assertIsNone(result['prediction']['npc_motion_direction'][4])

    def test_matching_birth_error_disclosed_but_excluded_from_continuing_summary(self):
        batch,spec,own=fixture()
        source=batch['states'][0]
        source[:2,12:14]=0.;own[:2,12:14]=0.
        source[2:,8:10]=(.8,.2);own[2:,8:10]=(.9,.2)
        result=build_drift_preview(batch,spec,own,step=10)
        self.assertEqual(result['reference']['npc_life_index'],[-1,-1,0,0,0])
        self.assertAlmostEqual(result['errors']['npc_xy_error_tiles'][2],1.6)
        self.assertIsNone(result['errors']['npc_continuing_xy_error_tiles'][2])
        self.assertEqual(result['summary']['npc_common_alive_xy_error_tiles']['count'],3)
        self.assertEqual(result['summary']['npc_continuing_xy_error_tiles']['count'],2)
        self.assertIsNone(result['prediction']['npc_motion_direction'][2])

    def test_zero_support_directions_and_wrapped_heading_are_honest_json_null(self):
        batch,spec,own=fixture()
        source=batch['states'][0]
        source[:,12:14]=0.;own[:,12:14]=0.
        source[1,2:4]=[math.sin(math.radians(179)),math.cos(math.radians(179))]
        own[1,2:4]=[math.sin(math.radians(-179)),math.cos(math.radians(-179))]
        source[2,2:4]=0.;own[2,2:4]=0.
        result=build_drift_preview(batch,spec,own,step=0)
        self.assertAlmostEqual(result['errors']['player_heading_error_degrees'][1],2.)
        self.assertIsNone(result['errors']['player_heading_error_degrees'][2])
        self.assertIsNone(result['prediction']['player_heading'][2])
        self.assertFalse(result['prediction']['player_heading_valid'][2])
        self.assertTrue(all(value is None for value in result['prediction']['npc_direction']))
        self.assertEqual(result['summary']['npc_common_alive_xy_error_tiles'],{'mean':None,'max':None,'count':0})
        json.dumps(result,allow_nan=False)

    def test_no_extra_engine_model_randomness_feedback_or_input_aliases(self):
        batch,spec,own=fixture(context=2)
        batch={key:torch.from_numpy(value)for key,value in batch.items()}
        own=torch.from_numpy(own).requires_grad_(True)
        before=copy.deepcopy(batch);initial=own.detach().clone()
        torch_rng=torch.get_rng_state().clone();np_rng=np.random.get_state();py_rng=random.getstate()
        with patch('turkeyworld.engine.Game',side_effect=AssertionError('No engine')), \
             patch('turkeyworld.models.NeuralWorldModel.transition',side_effect=AssertionError('No new rollout')):
            result=build_drift_preview(batch,spec,own,step=5)
        self.assertTrue(torch.equal(torch_rng,torch.get_rng_state()))
        current_np=np.random.get_state()
        self.assertEqual(np_rng[0],current_np[0])
        np.testing.assert_array_equal(np_rng[1],current_np[1])
        self.assertEqual(np_rng[2:],current_np[2:])
        self.assertEqual(py_rng,random.getstate())
        self.assertTrue(all(torch.equal(value,batch[name])for name,value in before.items()))
        self.assertTrue(torch.equal(initial,own.detach()));self.assertIsNone(own.grad)
        result['reference']['player_xy'][0][0]=999
        self.assertTrue(torch.equal(initial,own.detach()))
        self.assertTrue(torch.equal(batch['states'],before['states']))
        self.assertEqual(result['additional_model_transitions'],0)
        self.assertEqual(result['additional_engine_queries'],0)
        self.assertFalse(result['source_future_feedback'])
        json.dumps(result,allow_nan=False)

    def test_different_life_positions_are_disclosed_and_excluded_from_same_life_error(self):
        batch,spec,own=fixture()
        batch['states'][0,1,12:14]=0.
        own[2:,8]+=.1  # Own NPC never died; Source has a new life at2.
        result=build_drift_preview(batch,spec,own,step=9)
        self.assertEqual(result['reference']['npc_life_index'],[0,0,1,1,1])
        self.assertEqual(result['prediction']['npc_life_index'],[0,0,0,0,0])
        self.assertTrue(result['errors']['npc_life_mismatch'][2])
        self.assertAlmostEqual(result['errors']['npc_xy_error_tiles'][2],1.6)
        self.assertIsNone(result['errors']['npc_same_life_xy_error_tiles'][2])
        self.assertEqual(result['summary']['npc_common_alive_xy_error_tiles']['count'],3)
        self.assertEqual(result['summary']['npc_same_life_xy_error_tiles']['count'],0)

    def test_misaligned_partial_nonfinite_or_transposed_maps_fail_without_repair(self):
        batch,spec,own=fixture()
        for change in ('initial','short','nonfinite','wrong_map','fractional_map','bad_offsets'):
            b,s,p=copy.deepcopy(batch),copy.deepcopy(spec),own.copy()
            if change=='initial':p[0,0]+=.1
            elif change=='short':p=p[:-1]
            elif change=='nonfinite':p[1,0]=float('nan')
            elif change=='wrong_map':b['map']=np.zeros((16,16))
            elif change=='fractional_map':b['map']=b['map'].astype(float);b['map'][0,0,0]=.5
            else:s['display_offsets']=[1,3,2,4]
            with self.subTest(change=change),self.assertRaises(ValueError):
                build_drift_preview(b,s,p,step=1)


if __name__=='__main__':unittest.main()
