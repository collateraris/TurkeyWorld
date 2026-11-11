"""Player compatibility, independent enemy recurrence and observation causality."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from turkeyworld.engine import Game
from turkeyworld.models import NeuralWorldModel
from test_rssm_world_backend import tiny_config,inputs


def config(enabled):
    return replace(tiny_config(),enemy_rssm_enabled=enabled,
        enemy_rssm_deter_dim=16,enemy_rssm_hidden_dim=16,enemy_rssm_stoch=2,
        enemy_rssm_classes=4,enemy_rssm_embed_dim=16,enemy_rssm_action_dim=16,enemy_rssm_blocks=4)


def model(enabled):
    torch.manual_seed(173)
    return NeuralWorldModel(config(enabled))


class DualRSSMBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_disabled_legacy_keys_rng_weights_and_enabled_player_forward_remain_exact(self):
        old=model(False).eval();old_rng=torch.get_rng_state().clone()
        explicit=model(False).eval()
        self.assertTrue(torch.equal(old_rng,torch.get_rng_state()))
        self.assertEqual(old.state_dict().keys(),explicit.state_dict().keys())
        self.assertFalse(any('enemy_rssm' in key for key in old.state_dict()))
        self.assertTrue(all(torch.equal(value,explicit.state_dict()[key])for key,value in old.state_dict().items()))
        dual=model(True).eval()
        added=set(dual.state_dict())-set(old.state_dict())
        self.assertTrue(added);self.assertTrue(all(key.startswith('vjepa_backend.enemy_rssm.')for key in added))
        self.assertTrue(all(torch.equal(value,dual.state_dict()[key])for key,value in old.state_dict().items()))
        state,image,fmap=inputs(old);newmap=dual.encode_map(torch.tensor(Game(seed=0,endless=True,enemies_enabled=False).map)[None])
        with torch.no_grad():
            # Frozen appearance is deliberately nonzero so this tests a real path.
            old.vjepa_backend.feature_head.weight.normal_(std=.02)
            dual.vjepa_backend.feature_head.load_state_dict(old.vjepa_backend.feature_head.state_dict())
            initial=old.initialize_from_image(image,state,fmap)
            packed=dual.initialize_from_image(image,state,newmap)
            self.assertEqual(packed.shape,(1,72));self.assertEqual(packed.dtype,torch.float32)
            self.assertTrue(torch.equal(initial,packed[:,:48]))
            action=state.new_tensor([[1.,.4,.3,0.]])
            a=old.transition(state,action,fmap,initial);b=dual.transition(state,action,newmap,packed)
        for key in ('features','raw_player_delta','player_pass_logits','rssm_prior_logits','rssm_h','rssm_z'):
            self.assertTrue(torch.equal(a[key],b[key]),key)
        self.assertTrue(torch.equal(a['state'][:,:8],b['state'][:,:8]))
        self.assertTrue(torch.equal(a['memory'],b['memory'][:,:48]))

    def test_priors_never_observe_rgb_and_enemy_memory_survives_offscreen_and_death(self):
        world=model(True).eval();state,image,fmap=inputs(world)
        backend=world.vjepa_backend
        with torch.no_grad():memory=world.initialize_from_image(image,state,fmap)
        # Its numerical state is deliberately behind the camera, then dead.
        state=state.clone();state[:,8:10]=state[:,:2]+state.new_tensor([[-.04,0.]])
        state[:,12:14]=state.new_tensor([[.6,1.]])
        action=state.new_zeros(1,4)
        with patch.object(backend.obs_encoder,'forward',side_effect=AssertionError('No future RGB')), \
                patch.object(backend.enemy_rssm,'observe_initial',side_effect=AssertionError('No offscreen reset')), \
                patch.object(Game,'step',side_effect=AssertionError('No engine during prior')), \
                patch.object(Game,'render',side_effect=AssertionError('No engine during prior')), \
                patch.object(backend.enemy_rssm.core,'imagine_step',wraps=backend.enemy_rssm.core.imagine_step)as imagine:
            with torch.no_grad():
                first=world.transition(state,action,fmap,memory)
                state=first['state'].clone();state[:,12:14]=0
                second=world.transition(state,action,fmap,first['memory'])
        self.assertEqual(imagine.call_count,2)
        self.assertEqual(second['memory'].shape,(1,72))
        self.assertTrue(torch.isfinite(second['memory']).all())
        self.assertGreater(float(second['enemy_rssm_h'].abs().sum()),0)
        self.assertFalse(torch.equal(first['enemy_rssm_h'],second['enemy_rssm_h']))

    def test_next_rgb_changes_only_posteriors_and_reuses_exact_pre_rgb_enemy_prior(self):
        world=model(True).eval();state,image,fmap=inputs(world);backend=world.vjepa_backend
        with torch.no_grad():memory=world.initialize_from_image(image,state,fmap)
        action=state.new_tensor([[1.,-.3,.2,0.]])
        with patch.object(backend.obs_encoder,'forward',wraps=backend.obs_encoder.forward)as observe, \
                patch.object(backend.enemy_rssm,'prior',wraps=backend.enemy_rssm.prior)as prior:
            with torch.no_grad():a=backend.observe_step(state,action,fmap,memory,image,sample=False)
            self.assertEqual(observe.call_count,1);self.assertEqual(prior.call_count,1)
        with torch.no_grad():b=backend.observe_step(state,action,fmap,memory,1-image,sample=False)
        for key in ('state','memory','features','rssm_prior_logits','enemy_rssm_prior_logits','enemy_rssm_motion','enemy_rssm_attack_logits'):
            self.assertTrue(torch.equal(a['prior_result'][key],b['prior_result'][key]),key)
        self.assertTrue(torch.equal(a['enemy_rssm_prior_logits'],a['prior_result']['enemy_rssm_prior_logits']))
        self.assertTrue(torch.equal(a['enemy_rssm_conditioned_player_xy'],a['prior_result']['enemy_rssm_conditioned_player_xy']))
        self.assertTrue(torch.equal(a['enemy_rssm_conditioned_player_xy'],b['enemy_rssm_conditioned_player_xy']))
        self.assertFalse(torch.equal(a['enemy_rssm_posterior_logits'],b['enemy_rssm_posterior_logits']))
        self.assertFalse(torch.equal(a['rssm_posterior_logits'],b['rssm_posterior_logits']))

    def test_enemy_gradient_reaches_only_own_parameters_and_previous_enemy_memory(self):
        world=model(True).train();backend=world.vjepa_backend
        # Runner latent modes switch the established core; both priors must
        # honor that exact mode even while the enemy module remains training.
        backend.core.eval();self.assertTrue(backend.enemy_rssm.training)
        state,image,fmap=inputs(world);state=state.detach().requires_grad_()
        memory=world.initial_memory(1).detach().requires_grad_()
        action=state.new_tensor([[1.,0.,0.,0.]])
        with patch.object(backend.enemy_rssm,'prior',wraps=backend.enemy_rssm.prior)as prior:
            result=world.transition(state,action,fmap,memory)
        self.assertIs(prior.call_args.kwargs['sample'],False)
        loss=result['enemy_rssm_motion'].square().sum()+result['enemy_rssm_attack_logits'].sum()
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum())for p in backend.enemy_rssm.parameters()if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for name,p in world.named_parameters()if not name.startswith('vjepa_backend.enemy_rssm.')))
        self.assertIsNone(state.grad)
        self.assertEqual(float(memory.grad[:,:48].abs().sum()),0)
        self.assertGreater(float(memory.grad[:,48:].abs().sum()),0)

    def test_render_features_ignore_enemy_memory_and_wrong_memory_shape_is_rejected(self):
        world=model(True).eval();backend=world.vjepa_backend
        with torch.no_grad():backend.feature_head.weight.normal_(std=.03)
        memory=torch.randn(2,72)
        changed=memory.clone();changed[:,48:]=torch.randn_like(changed[:,48:])*10
        self.assertTrue(torch.equal(backend.latent_features(memory),backend.latent_features(changed)))
        with self.assertRaises(ValueError):backend.split_memory(torch.zeros(2,48))
        self.assertEqual(backend.memory_size,72)


if __name__=='__main__':unittest.main()
