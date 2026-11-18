"""Only the player command clamp's backward pass changes in the STE branch."""
import unittest
import torch
from turkeyworld.engine import Game
from turkeyworld.structured_state import structured_transition


class PlayerClampGradientTests(unittest.TestCase):
    def test_clipped_free_axis_can_learn_both_directions(self):
        state = torch.from_numpy(Game(seed=3).state_vector()).expand(4,-1).clone()
        state[:, :4] = torch.tensor([.4, .4, 0., 1.])
        # Wrong sign, excessive displacement, valid displacement, and idle.
        candidate = state.clone()
        candidate[:, 0] += torch.tensor([-.02, .02, .002, .02])
        candidate[:, 1] += .10
        candidate[:, 8:10] += .04
        action = torch.zeros(4,4)
        action[:3,0] = 1
        target = state[:,0] + torch.tensor([.005, 0, .004, .001])
        def run(override):
            raw = candidate.clone().requires_grad_()
            result = structured_transition(state, raw, action, torch.zeros(4,2),
                torch.full((4,1), -10.), endless=True, training=False, player_clamp_ste=override)
            nxt = result['state']
            loss = ((nxt[:,0] - target)**2).sum()+nxt[:,1].sum()+nxt[:,8:10].sum()
            grad, = torch.autograd.grad(loss, raw)
            return nxt.detach(), grad
        old, old_grad = run(None)
        base, hard_grad = run(False)
        treated, ste_grad = run(True)
        for v in (base,treated):
            self.assertTrue(torch.equal(v,old))
        self.assertTrue(torch.equal(old_grad,hard_grad))
        self.assertEqual(hard_grad[:2,0].abs().sum().item(),0)
        self.assertLess(ste_grad[0,0].item(),0)
        self.assertGreater(ste_grad[1,0].item(),0)
        self.assertTrue(torch.equal(hard_grad[2:,0],ste_grad[2:,0]))
        self.assertEqual(ste_grad[:,1].abs().sum().item(),0)
        self.assertEqual(ste_grad[3,0].item(),0)
        self.assertTrue(torch.equal(hard_grad[:,2:],ste_grad[:,2:]))
