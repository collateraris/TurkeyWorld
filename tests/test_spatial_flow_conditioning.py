import copy
import unittest

import torch

from turkeyworld.latent_flow_renderer import LatentVelocityNetwork, SpatialFlowCondition


class SpatialFlowConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def state(self, batch=2):
        state = torch.zeros(batch, 90)
        state[:, :2] = torch.tensor([.1, .5])
        state[:, 3:5] = 1
        npc = state[:, 8:88].reshape(batch, 8, 10)
        npc[:, :, 0] = torch.linspace(.35, .8, 8)
        npc[:, :, 1] = torch.linspace(.47, .53, 8)
        npc[:, :, 2:4] = torch.tensor([.1, .2])
        npc[:, :, 4] = .6;npc[:, :, 5] = 1
        return state

    def network(self):
        model = LatentVelocityNetwork(4, 8, 32, 1, 4)
        with torch.no_grad():
            model.output.weight.normal_(std=.02)
        return model

    def test_opt_in_zero_bridge_preserves_trained_velocity_weights_and_rng(self):
        torch.manual_seed(23)
        model = self.network().eval()
        old = copy.deepcopy(model.state_dict())
        noisy, base = torch.rand(2, 4, 8, 8), torch.rand(2, 4, 8, 8)
        time, state = torch.tensor([.3, .8]), self.state()
        with torch.no_grad():before = model(noisy, time, base, state)
        rng = torch.get_rng_state().clone()
        model.enable_spatial_conditioning(32)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with torch.no_grad():after = model(noisy, time, base, state)
        self.assertTrue(torch.equal(before, after))
        for name, value in old.items():self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        self.assertEqual(tuple(model.spatial_condition.entity_tokens(state).shape), (2, 9, 32))

    def test_all_eight_npc_coordinates_direction_and_hp_receive_actual_velocity_gradients(self):
        model = self.network().train();model.enable_spatial_conditioning(32)
        with torch.no_grad():
            for layer in model.state_input.modules():
                if isinstance(layer, torch.nn.Linear):layer.weight.zero_();layer.bias.zero_()
            model.spatial_condition.output.weight.normal_(std=.02)
        state = self.state().requires_grad_()
        output = model(torch.rand(2, 4, 8, 8), torch.tensor([.3, .8]), torch.rand(2, 4, 8, 8), state)
        output.square().mean().backward()
        gradients = state.grad[:, 8:88].reshape(2, 8, 10)
        for field in [0, 1, 2, 3, 4]:
            self.assertTrue(bool((gradients[:, :, field].abs().sum(0) > 0).all()), field)
        self.assertGreater(float(state.grad[:, :4].abs().sum()), 0)
        self.assertGreater(float(model.spatial_condition.features[0].weight.grad.abs().sum()), 0)

    def test_offscreen_participant_tokens_persist_but_image_projection_excludes_them(self):
        module = SpatialFlowCondition(32, 2, 32)
        state = self.state()
        state[:, 8] = .01
        self.assertEqual(tuple(module.entity_tokens(state).shape), (2, 9, 32))
        weights = module.projection_weights(state)
        self.assertTrue(torch.equal(weights[:, :, 0], torch.ones(2, 4)))
        self.assertEqual(float(weights[:, :, 1].abs().sum()), 0)
        output = module(torch.rand(2, 4, 32), state)
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_bfloat16_training_forward_and_backward_are_finite(self):
        module = SpatialFlowCondition(32, 2, 32)
        with torch.no_grad():module.output.weight.normal_(std=.01)
        state = self.state().requires_grad_()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = module(torch.rand(2, 4, 32), state)
            loss = output.float().square().mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(state.grad).all()))


if __name__ == '__main__':
    unittest.main()
