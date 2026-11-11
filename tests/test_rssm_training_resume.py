from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from test_rssm_world_backend import tiny_config, inputs
from turkeyworld.models import NeuralWorldModel, load_world_checkpoint
from turkeyworld.rssm_migration import configure_rssm_training
from turkeyworld.train_rssm_dynamics import RSSMTraining


class RSSMResumeTests(unittest.TestCase):
    def test_checkpoint_restores_categorical_rng_adam_and_next_actual_update(self):
        torch.set_num_threads(2)
        torch.manual_seed(77)
        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(run_dir=Path(tmp), seed=77, device='cpu',
                batch=16, teacher=Path(tmp)/'archived_teacher.pt', updates=2000,
                distill_updates=200)
            experiment = RSSMTraining(args)
            experiment.plan_sha = 'immutable_test_plan'
            model = NeuralWorldModel(tiny_config())
            configure_rssm_training(model)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-4)
            def update(actual_model, actual_optimizer):
                actual_optimizer.zero_grad(set_to_none=True)
                state, rgb, fmap = inputs(actual_model)
                memory = actual_model.initialize_from_image(rgb, state, fmap)
                action = state.new_tensor([[1., .5, -.5, 0.]])
                result = actual_model.vjepa_backend.observe_step(state, action, fmap, memory, rgb)
                loss = (F.binary_cross_entropy_with_logits(result['prior_result']['player_pass_logits'],
                    torch.ones(1, 2)) + .01*result['rssm_prior_logits'].square().mean()
                    + F.l1_loss(result['rssm_reconstruction'], rgb))
                loss.backward()
                actual_optimizer.step()
                return result['memory'].detach().clone()
            update(model, optimizer)
            experiment.state['global_training_step'] = 1
            experiment.state['rssm_training']['distillation_completed'] = 1
            experiment.save(model, optimizer, {'rssm_migration': {'source_global_training_step': 1633}})
            expected_memory = update(model, optimizer)
            expected = {name: value.clone() for name, value in model.state_dict().items()}
            restored, checkpoint = load_world_checkpoint(experiment.checkpoint)
            configure_rssm_training(restored)
            resumed_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=1e-4)
            resumed_optimizer.load_state_dict(checkpoint['optimizer'])
            torch.set_rng_state(checkpoint['torch_rng_state'])
            actual_memory = update(restored, resumed_optimizer)
            self.assertTrue(torch.equal(actual_memory, expected_memory))
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]), name)
            self.assertEqual(checkpoint['rssm_training']['distillation_completed'], 1)
            self.assertEqual(checkpoint['training_steps'], 1)
            self.assertNotIn('ema', checkpoint)
            self.assertNotIn('ema_target_encoder', checkpoint)


if __name__ == '__main__':
    unittest.main()
