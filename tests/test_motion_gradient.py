"""NPC slowing must remain learnable through the hard runtime speed bound."""
import unittest

import torch
from torch.nn import functional as F

from turkeyworld.structured_state import (
    MAP_EXTENT, NPC_SPEED_BOUND, _bounded_delta, structured_transition,
)


class MotionGradientTests(unittest.TestCase):
    dt = 1 / 30
    limit = NPC_SPEED_BOUND * dt / MAP_EXTENT

    def motion_loss(self, candidate, target, *, training):
        old = torch.full_like(candidate, .5)
        movement = _bounded_delta(old, candidate, NPC_SPEED_BOUND, self.dt,
                                  training=training)
        return F.smooth_l1_loss(movement / self.limit, target / self.limit)

    def test_overspeed_npc_can_learn_to_stop(self):
        target = torch.zeros(1, 2, dtype=torch.float64)
        inference_proposal = torch.tensor([[.52, .5]], dtype=torch.float64, requires_grad=True)
        self.motion_loss(inference_proposal, target, training=False).backward()
        torch.testing.assert_close(inference_proposal.grad, torch.zeros_like(target), atol=1e-12, rtol=0)

        proposal = torch.nn.Parameter(torch.tensor([[.52, .5]], dtype=torch.float64))
        optimizer = torch.optim.SGD([proposal], lr=1e-4)
        initial = self.motion_loss(proposal, target, training=True)
        initial.backward()
        self.assertGreater(float(proposal.grad[0, 0]), 0)
        self.assertEqual(float(proposal.grad[0, 1]), 0)
        self.assertTrue(bool(torch.isfinite(proposal.grad).all()))
        before = float(torch.linalg.vector_norm(proposal.detach() - .5))
        optimizer.step()
        self.assertLess(float(torch.linalg.vector_norm(proposal.detach() - .5)), before)
        self.assertLess(float(self.motion_loss(proposal, target, training=True).detach()), float(initial.detach()))

    def test_collinear_slower_target_has_radial_gradient(self):
        target = torch.tensor([[self.limit / 2, 0]], dtype=torch.float64)
        proposal = torch.nn.Parameter(torch.tensor([[.52, .5]], dtype=torch.float64))
        optimizer = torch.optim.SGD([proposal], lr=1.8e-4)
        initial = self.motion_loss(proposal, target, training=True)
        initial.backward()
        self.assertGreater(float(proposal.grad[0, 0]), 0)
        self.assertEqual(float(proposal.grad[0, 1]), 0)
        optimizer.step()
        self.assertLess(float(proposal.detach()[0, 0] - .5), .02)
        self.assertLess(float(self.motion_loss(proposal, target, training=True).detach()), float(initial.detach()))

    def test_default_forward_is_exact_original_cap(self):
        rng = torch.Generator().manual_seed(1393)
        old = torch.rand(4, 8, 2, generator=rng)
        candidate = torch.rand(4, 8, 2, generator=rng) * 1.4 - .2
        delta = candidate.clamp(0, 1) - old
        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        factor = (self.limit / norm.clamp_min(max(torch.finfo(delta.dtype).tiny, 1e-6))).clamp_max(1)
        original = delta * factor
        for training in (False, True):
            result = _bounded_delta(old, candidate, NPC_SPEED_BOUND, self.dt, training=training)
            self.assertTrue(torch.equal(result.view(torch.int32), original.view(torch.int32)))
            self.assertLessEqual(float(torch.linalg.vector_norm(result, dim=-1).max()), self.limit + 1e-8)

    def test_transition_forward_and_dead_masks_unchanged(self):
        state = torch.zeros(2, 90)
        state[:, :2] = torch.tensor([.15, .15])
        state[:, 3] = 1
        state[0, 4] = 1  # The second batch item is terminal.
        state[:, 5] = .8
        old = state[:, 8:88].reshape(2, 8, 10)
        old[:, :, 0] = torch.arange(8) * .06 + .2
        old[:, :, 1] = .5
        old[:, :, 4:6] = torch.tensor([.6, 1.])
        old[:, 2, 4:6] = 0  # This corpse must neither move nor receive gradients.
        candidate = state.clone()
        proposed = candidate[:, 8:88].reshape(2, 8, 10)
        proposed[:, 0, :2] += torch.tensor([.001, -.001])
        proposed[:, 1:3, :2] += torch.tensor([.02, .03])
        candidate.requires_grad_()
        targets = torch.full((2, 9), -10.)
        targets[:, 8] = 10.
        attacks = torch.full((2, 8), -10.)
        for endless in (False, True):
            with self.subTest(endless=endless):
                train = structured_transition(state, candidate, torch.zeros(2, 4), targets, attacks,
                                              training=True, endless=endless)
                evaluate = structured_transition(state, candidate, torch.zeros(2, 4), targets, attacks,
                                                 training=False, endless=endless)
                self.assertTrue(torch.equal(train['state'].detach().view(torch.int32),
                                            evaluate['state'].detach().view(torch.int32)))
                slots = train['state'][:, 8:88].reshape(2, 8, 10)
                displacement = slots[:, :, :2] - old[:, :, :2]
                self.assertLessEqual(float(torch.linalg.vector_norm(displacement, dim=-1).max().detach()),
                                     self.limit + 1e-7)
                self.assertTrue(torch.equal(displacement[1], torch.zeros_like(displacement[1])))
                self.assertTrue(torch.equal(displacement[0, 2], torch.zeros(2)))
                gradient, = torch.autograd.grad(slots[:, :, :2].sum(), candidate)
                npc_gradient = gradient[:, 8:88].reshape(2, 8, 10)[:, :, :2]
                torch.testing.assert_close(npc_gradient[0, 1], torch.ones(2), rtol=0, atol=0)
                self.assertTrue(torch.equal(npc_gradient[0, 2], torch.zeros(2)))
                self.assertTrue(torch.equal(npc_gradient[1], torch.zeros_like(npc_gradient[1])))
                # Supervision is used only by the loss; transition inputs stay intact.
                self.assertTrue(torch.equal(state[:, :2], torch.tensor([[.15, .15], [.15, .15]])))


if __name__ == '__main__':
    unittest.main()
