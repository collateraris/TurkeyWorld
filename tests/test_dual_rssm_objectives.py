"""Both object memories learn from causal rollouts and separate Source posteriors."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_dual_rssm_backend import model
from turkeyworld.engine import Game
from turkeyworld.rssm_migration import configure_rssm_training
from turkeyworld.rssm_objectives import rssm_dynamics_backward


def source_batch():
    rows=[]
    for seed in range(2):
        game=Game(seed=seed,endless=True,enemies_enabled=True)
        if seed:game.enemies[0]['x']=3.25
        states=[game.state_vector()];frames=[game.render(size=32,overlays=False)]
        actions=[];events=[];hits=[];attacks=[]
        for index in range(4):
            action=np.asarray([.25,0,.1,1. if not seed else 0.],np.float32)
            event=game.step(action,1/30);states.append(game.state_vector());frames.append(game.render(size=32,overlays=False))
            actions.append(action);events.append(Game.event_vector(event));hits.append(event.get('hit_id',-1))
            attacks.append([float(0 in event.get('attacker_ids',[]))])
        rows.append(dict(states=np.stack(states),frames=np.stack(frames),actions=np.stack(actions),
            events=np.stack(events),hit_ids=np.asarray(hits,np.int64),attack_flags=np.asarray(attacks,np.float32),map=game.map))
    batch={key:torch.from_numpy(np.stack([row[key]for row in rows]))for key in rows[0]}
    batch['frames']=batch['frames'].permute(0,1,4,2,3).float()/255
    batch['states']=batch['states'].float();batch['target_states']=batch['states'][:,1:]
    batch['context_frames']=batch['frames'][:,:1];batch['map']=batch['map'].long()
    return batch


class DualRSSMObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def run_model(self,world,batch):
        configure_rssm_training(world);world.core.eval()
        queries=[];transition=world.transition
        def capture(state,action,features,memory):
            queries.append((state.detach().clone(),memory.detach().clone()))
            return transition(state,action,features,memory)
        with patch.object(world,'transition',side_effect=capture),\
                patch.object(world,'render',side_effect=AssertionError('No renderer during dynamics training')),\
                patch.object(Game,'step',side_effect=AssertionError('No engine labels during the recorded rollout')):
            metrics=rssm_dynamics_backward(world,batch,context=1,observed_context=1,
                auxiliary_batch=batch,auxiliary_context=1,auxiliary_observed_context=1,
                expected_horizon=4,chunk_size=2,numeric_state_feedback='detached',
                combat_protocol={'target_supervision':'shot_distance_balanced'})
        return metrics,queries

    def test_both_cores_receive_gradients_with_separate_kl_and_all_frozen_assets_unchanged(self):
        world=model(True);batch=source_batch()
        configure_rssm_training(world)
        frozen={name:value.clone()for name,value in world.state_dict().items()if name.startswith(
            ('flow_renderer.','vjepa_backend.renderer.','vjepa_backend.feature_head.'))}
        metrics,queries=self.run_model(world,batch)
        self.assertEqual(len(queries),4)
        self.assertTrue(all(memory.shape==(2,72)for _,memory in queries))
        self.assertGreater(metrics['rssm_aux_enemy_posterior_kl'],0)
        self.assertAlmostEqual(metrics['rssm_kl_minimum_floor'],2.2)
        self.assertAlmostEqual(metrics['train_loss_without_kl_floor'],metrics['train_loss']-2.2)
        self.assertGreater(sum(float(p.grad.abs().sum())for p in world.core.parameters()if p.grad is not None),0)
        enemy=world.vjepa_backend.enemy_rssm
        self.assertGreater(sum(float(p.grad.abs().sum())for p in enemy.parameters()if p.grad is not None),0)
        self.assertTrue(all(torch.equal(value,world.state_dict()[name])for name,value in frozen.items()))
        self.assertFalse(metrics['rssm_posterior_feedback_to_prior'])
        self.assertEqual(metrics['rssm_prior_future_rgb_inputs'],0)

    def test_changing_future_rgb_does_not_change_either_causal_memory_trajectory(self):
        initial=model(True);first=source_batch();second={key:value.clone()for key,value in first.items()}
        second['frames'][:,1:]=1-second['frames'][:,1:]
        a,qa=self.run_model(deepcopy(initial),first);b,qb=self.run_model(deepcopy(initial),second)
        self.assertEqual(a['rssm_main_loss'],b['rssm_main_loss'])
        for old,new in zip(qa,qb):
            for x,y in zip(old,new):torch.testing.assert_close(x,y,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
