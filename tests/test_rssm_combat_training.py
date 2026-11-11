import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_renderer_enemy_objectives import tiny_model
from test_rssm_combat_objectives import source_batch
from turkeyworld.engine import Game
from turkeyworld.models import NeuralWorldModel
from turkeyworld.rssm_combat_protocol import (prepare_combat_continuation, continued_combat_state,
    environment_profile, commit_combat_update)
from turkeyworld.rssm_migration import configure_rssm_training
from turkeyworld.rssm_objectives import rssm_dynamics_backward, _npc_xy_mask
from test_rssm_objectives import SpyRSSM
from test_motion_recovery_training import source_batch as motion_batch


def raw_model():
    reference = tiny_model()
    model = NeuralWorldModel(copy.deepcopy(reference.config))
    return model


class RSSMCombatTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_semantic_migration_preserves_all_weights_adam_rng_and_global_clocks(self):
        model = raw_model(); configure_rssm_training(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        model.vjepa_backend.player_pass_head[-1].weight.sum().backward(); optimizer.step()
        state = {'global_training_step': 217, 'current': 217, 'loss_protocol': 'old_navigation',
            'training_protocol': {}, 'rssm_training': {'phase': 'source', 'source_completed': 17,
                'distillation_completed': 200, 'distillation_updates': 200, 'auxiliary_draws': 3472,
                'navigation_revision': {'revision_updates': 3}, 'last_validation_passed': False}}
        payload = {'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict(),
            'training_steps': 217, 'global_training_step': 217, 'run_state': state,
            'rssm_training': state['rssm_training'], 'rssm_migration': {'format_version': 1},
            'optimizer': optimizer.state_dict(), 'optimizer_parameter_names': [n for n,p in model.named_parameters() if p.requires_grad],
            'rng_state': np.random.default_rng(17).bit_generator.state,
            'torch_rng_state': torch.get_rng_state().clone(), 'cuda_rng_state': []}
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder)/'old.pt', Path(folder)/'new.pt'; torch.save(payload, old)
            bytes_before = old.read_bytes(); rng_before = torch.get_rng_state().clone()
            proof = prepare_combat_continuation(old, new, run_id='combat_test')
            migrated = torch.load(new, weights_only=False)
            self.assertEqual(old.read_bytes(), bytes_before)
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            self.assertFalse(migrated['config']['navigation_only'])
            self.assertTrue(all(torch.equal(v, migrated['model'][n]) for n,v in payload['model'].items()))
            self.assertEqual(migrated['optimizer_parameter_names'], payload['optimizer_parameter_names'])
            for key, values in payload['optimizer']['state'].items():
                self.assertTrue(all(torch.equal(v, migrated['optimizer']['state'][key][n]) for n,v in values.items()))
            self.assertTrue(torch.equal(migrated['torch_rng_state'], payload['torch_rng_state']))
            self.assertEqual(migrated['rng_state'], payload['rng_state'])
            self.assertEqual(migrated['run_state']['rssm_training']['combat_start_step'], 217)
            self.assertEqual(migrated['run_state']['rssm_training']['source_completed'], 17)
            self.assertEqual(migrated['run_state']['navigation_training_history']['navigation_revision']['revision_updates'], 3)
            self.assertTrue(proof['all_existing_adam_states_preserved'])
            with self.assertRaises(FileExistsError): prepare_combat_continuation(old, new, run_id='changed')

    def test_known_environment_scope_restores_config_even_on_error_and_commits_only_success(self):
        model = raw_model(); original_model, original_backend = model.config, model.vjepa_backend.config
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with environment_profile(model, navigation_only=False):
                self.assertFalse(model.config.navigation_only)
                self.assertFalse(model.vjepa_backend.config.navigation_only)
                raise RuntimeError('test')
        self.assertIs(model.config, original_model); self.assertIs(model.vjepa_backend.config, original_backend)
        state = continued_combat_state({'global_training_step': 200, 'training_protocol': {},
            'rssm_training': {'source_completed': 0, 'auxiliary_draws': 3200}},
            run_id='new', source_checkpoint='old', source_sha256='fixed')
        with self.assertRaises(ValueError): commit_combat_update(state, step=202, environment='combat', batch_size=16)
        self.assertEqual(state['rssm_training']['combat_completed'], 0)
        commit_combat_update(state, step=201, environment='combat', batch_size=16)
        self.assertEqual(state['global_training_step'], 201)
        self.assertEqual(state['rssm_training']['combat_completed'], 1)
        self.assertEqual(state['rssm_training']['source_completed'], 1)
        self.assertEqual(state['rssm_training']['combat_auxiliary_draws'], 16)

    def test_each_life_gets_xy_gradients_after_respawn_only_exact_teleport_is_excluded(self):
        alive = torch.tensor([[[False], [True], [True], [False], [True], [True]]])
        mask = _npc_xy_mask(torch.tensor([[False]]), alive, combat=True)
        self.assertEqual(mask.flatten().tolist(), [False, False, True, False, False, True])
        self.assertFalse(_npc_xy_mask(torch.tensor([[False]]), alive).any())
        prediction = torch.ones(1, 6, 1, 2, requires_grad=True)
        ((prediction.square().sum(-1))*mask).sum().backward()
        self.assertEqual(prediction.grad.abs().sum(-1).gt(0).flatten().tolist(), mask.flatten().tolist())

    def test_real_rssm_compound_backward_trains_combat_before_source_rgb_and_never_renders(self):
        model = raw_model(); configure_rssm_training(model)
        data = source_batch(4)
        data['frames'] = torch.full((4, 5, 3, 32, 32), .2)
        default = Game().map
        maps = np.repeat(default[None], 4, 0)
        maps[-1] = 0; maps[-1, [0,-1], :] = 1; maps[-1, :, [0,-1]] = 1; maps[-1, 2, 3] = 1
        data['map'] = torch.from_numpy(maps).long()
        frozen = {n:v.clone() for n,v in model.state_dict().items()
            if n.startswith(('flow_renderer.', 'vjepa_backend.renderer.', 'vjepa_backend.feature_head.'))}
        with environment_profile(model, navigation_only=False), \
                patch.object(model, 'render', side_effect=AssertionError('No frame generator in dynamics')):
            result = rssm_dynamics_backward(model, data, context=1, observed_context=1,
                auxiliary_batch=data, auxiliary_context=1, auxiliary_observed_context=1,
                expected_horizon=4, chunk_size=2, combat_protocol={})
        self.assertTrue(np.isfinite(result['train_loss']))
        self.assertGreater(result['rssm_aux_combat_source_hit_count'], 0)
        self.assertGreater(result['rssm_aux_combat_attack_positive_count'], 0)
        self.assertGreater(result['rssm_aux_combat_spawn_event_count'], 0)
        self.assertFalse(result['rssm_posterior_feedback_to_prior'])
        self.assertEqual(result['rssm_prior_future_rgb_inputs'], 0)
        self.assertGreater(float(model.vjepa_backend.hit_target_head[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.vjepa_backend.attack_head[-1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(torch.equal(v, model.state_dict()[n]) for n,v in frozen.items()))
        self.assertTrue(all(p.grad is None for p in model.flow_renderer.parameters()))

    def test_numeric_feedback_detach_is_forward_exact_keeps_latent_temporal_gradients(self):
        initial = SpyRSSM(); data = motion_batch(context=1, horizon=4, batch_size=1)
        outcomes = {}
        for mode in ('attached', 'detached'):
            model = copy.deepcopy(initial); original = model.transition
            rows, state_flags, memory_flags = [], [], []
            def transition(state, action, features, memory):
                state_flags.append(state.requires_grad); memory_flags.append(memory.requires_grad)
                result = original(state, action, features, memory)
                rows.append((result['state'].detach().clone(), result['memory'].detach().clone()))
                return result
            model.transition = transition
            with patch('turkeyworld.rssm_objectives._posterior_backward', return_value={'loss': 0.}):
                report = rssm_dynamics_backward(model, data, context=1, observed_context=1,
                    expected_horizon=4, chunk_size=4, numeric_state_feedback=mode)
            self.assertTrue(all(memory_flags))
            self.assertGreater(float(model.obs_encoder.weight.grad.abs().sum()), 0)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.core.parameters() if p.grad is not None), 0)
            outcomes[mode] = rows, report['rssm_main_loss']
            if mode == 'detached': self.assertFalse(any(state_flags))
            else: self.assertTrue(all(state_flags[1:]))
        self.assertEqual(outcomes['attached'][1], outcomes['detached'][1])
        self.assertTrue(all(torch.equal(a,b) for first, second in zip(outcomes['attached'][0], outcomes['detached'][0])
            for a,b in zip(first, second)))


if __name__ == '__main__': unittest.main()
