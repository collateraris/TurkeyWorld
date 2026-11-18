import unittest
import torch
from turkeyworld.pilot_npc_direction import features, velocity_labels, evaluate


class LocalNPCTests(unittest.TestCase):
    def test_local_wall_axis_subtile_and_translation(self):
        g=torch.zeros(16,16);g[6,8]=1
        s=torch.zeros(1,20);s[0,:2]=torch.tensor([4.5,3.5])/16
        s[0,8:10]=torch.tensor([7.1,6.9])/16;s[0,10:12]=torch.tensor([-.3,.2])
        x=features(s,g)
        self.assertEqual(x.shape,(1,34))
        self.assertEqual(x[0,:25].sum().item(),1)
        self.assertEqual(x[0,13].item(),1)
        torch.testing.assert_close(x[0,25:27],torch.tensor([-.4,.4]))
        t=s.clone();t[:,[0,1,8,9]]+=1/16
        torch.testing.assert_close(features(t,torch.roll(g,(1,1),(0,1))),x)

    def test_velocity_units_and_stationary_direction_error(self):
        before=torch.zeros(2,20);after=before.clone();after[1,8]+=.5/480
        y=velocity_labels(before,after);p=evaluate(torch.zeros_like(y),y)
        self.assertAlmostEqual(p['vector_error_tiles_s'],.25)
        self.assertEqual(p['moving_ticks'],1)
        self.assertAlmostEqual(p['angle_deg'],90)

    def test_conditional_roll_integrates_own_velocity_on_each_tick(self):
        from turkeyworld.pilot_npc_direction_rollout import roll
        s=torch.zeros(20);s[8:10]=.5;s[10]=.35
        player=torch.rand(16,2)
        seen=[]
        def estimator(state,_):
            seen.append(state[0,:2].clone())
            return state[:,10:12]*2
        trajectory=roll(s,player,torch.zeros(16,16),estimator)
        torch.testing.assert_close(torch.stack(seen),player)
        self.assertAlmostEqual(trajectory[-1,0].item(),.5+16*.7/480,places=5)
        self.assertAlmostEqual(s[8].item(),.5)
