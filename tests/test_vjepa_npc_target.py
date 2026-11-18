import unittest
import numpy as np
import torch
from test_base_student import tiny_teacher, states
from turkeyworld.audit_vjepa_npc_target import paired_step, source_masks
from turkeyworld.vjepa_policy_world import VJEPAImaginationAdapter


class PairedTargetTests(unittest.TestCase):
    def test_matches_runtime_until_first_xy_correction_and_renders_corrected_state(self):
        torch.set_num_threads(2)
        adapter = VJEPAImaginationAdapter(tiny_teacher())
        state = states()
        current = adapter.initialize(state, torch.zeros(2,16,16,dtype=torch.long), torch.zeros(2,3,32,32))
        class Fake:
            def observation(self, s, grid, navigation_only):
                self.last = s.clone()
                return s[:,0,None,None,None].expand(-1,3,32,32)
        adapter.feedback_renderer = fake = Fake()
        adapter.feedback_settings = dict(every=1,blend=.5)
        action = torch.tensor([[1.,0.,0.,0.]]).repeat(2,1)
        prod = adapter.transition_prior(current,action)
        identity = paired_step(adapter,current,action,prod.state[1:,:2])
        torch.testing.assert_close(prod.state,identity.state,rtol=0,atol=0)
        torch.testing.assert_close(prod.memory,identity.memory)
        target = prod.state[1:,:2] + .1
        changed = paired_step(adapter,current,action,target)
        torch.testing.assert_close(changed.state[0],prod.state[0],rtol=0,atol=0)
        torch.testing.assert_close(changed.state[1,2:],prod.state[1,2:],rtol=0,atol=0)
        torch.testing.assert_close(fake.last[1:,:2],target,rtol=0,atol=0)
        n = adapter.world.core.block_tokens*adapter.world.core.width
        torch.testing.assert_close(changed.memory[:,:-n],prod.memory[:,:-n],rtol=0,atol=0)
        self.assertFalse(torch.equal(changed.memory[1,-n:],prod.memory[1,-n:]))

    def test_occlusion_and_direction_labels(self):
        grid = np.zeros((16,16)); grid[1,3] = 1
        path = np.zeros((4,20)); path[:,:2] = np.array([5.5,1.5])/16
        path[:,8:10] = np.array([[1.5,1.5],[1.5,2],[1.5,2.5],[2,2.5]])/16
        m = source_masks(grid,path[0],path[1:])
        self.assertTrue(m['occluded'][0])
        np.testing.assert_array_equal(m['occluded_recent_turn'] | m['occluded_no_recent_turn'],m['occluded'])
        self.assertFalse((m['occluded_recent_turn'] & m['occluded_no_recent_turn']).any())
        self.assertEqual(m['within_8_ticks_of_turn_ge30deg'].tolist(), [False,False,True])
