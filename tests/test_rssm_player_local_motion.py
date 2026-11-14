"""Exact player-only Source labels, free-axis gradients and no NPC repair."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.engine import Game
from turkeyworld.rssm_player_local_motion import (
    PLAYER_LOCAL_MOTION_PROTOCOL,normalize_player_local_motion_protocol,
    source_player_motion_step,player_motion_supervised_terms,
)
from test_rssm_player_contact_replay import source_tapes


class PlayerLocalMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_first_step_matches_full_Source_with_turn_fire_and_combat_death(self):
        actions=([1,0,.4,1],[-1,1,-.7,0],[0,1,.2,1],[1,1,0,0])
        for action in actions:
            game=Game(seed=0,endless=True)
            game.map=np.zeros((16,16),np.uint8);game.map[[0,-1],:]=1;game.map[:,[0,-1]]=1
            state=game.get_state()
            state['player'].update(x=3.5,y=3.5,angle=.6,hp=4.)
            state['enemies'][0].update(x=3.8,y=3.8,alive=True,hp=100.,attack_cooldown=0.)
            game.set_state(state);before=game.state_vector().copy();game.set_state_vector(before);grid=game.map.copy()
            label=source_player_motion_step(before,np.asarray(action,np.float32),grid)
            game.step(np.asarray(action,np.float32),dt=1/30)
            np.testing.assert_array_equal(label['target'][:2],game.state_vector()[:2])
            np.testing.assert_array_equal(label['previous'],before)
            np.testing.assert_array_equal(label['target'][2:],before[2:])
            self.assertEqual(label['supervised_state_indices'],[0,1])
            self.assertFalse(label['own_state_corrected'])

    def test_legal_player_label_survives_invalid_living_NPC_without_mutating_input(self):
        states,actions,maps=source_tapes();state=states[0,0].copy()
        state[8:10]=np.asarray([6.5,5.5])/16;state[12:14]=1
        before=state.copy();steps=[];real=Game.step
        def record(game,*args,**kwargs):
            steps.append(game.enemies_enabled);return real(game,*args,**kwargs)
        with patch.object(Game,'step',new=record):
            label=source_player_motion_step(state,actions[0,0],maps[0])
        self.assertEqual(steps,[False]);self.assertTrue(label['own_NPC_body_invalid'])
        np.testing.assert_array_equal(state,before)
        np.testing.assert_array_equal(label['target'][8:18],before[8:18])
        self.assertGreater(float(label['delta_xy'].sum()),0)

    def test_first_step_matches_exact_Source_at_both_sliding_orientations_and_corner(self):
        states,actions,maps=source_tapes()
        for actor in range(4):
            game=Game(seed=0,endless=True,enemies_enabled=False);game.map=maps[actor].copy()
            game.set_state_vector(states[actor,0]);before=game.state_vector().copy()
            label=source_player_motion_step(before,actions[actor,0],maps[actor])
            game.step(actions[actor,0],dt=1/30)
            np.testing.assert_array_equal(label['target'][:2],game.state_vector()[:2])

    def test_dead_invalid_or_noncanonical_player_never_obtains_label(self):
        states,actions,maps=source_tapes();base=states[0,0]
        for kind in('dead','wall','heading','ammo'):
            state=base.copy()
            if kind=='dead':state[4]=0
            elif kind=='wall':state[:2]=np.asarray([6.5,5.5])/16
            elif kind=='heading':state[2:4]=.1
            else:state[5]=.123456
            before=state.copy()
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                source_player_motion_step(state,actions[0,0],maps[0])
            np.testing.assert_array_equal(state,before)

    def test_class_mass_and_sliding_axes_have_correct_separate_gradients(self):
        states,actions,maps=source_tapes();old=torch.tensor(states[:,0],requires_grad=True)
        target=torch.tensor(np.stack([source_player_motion_step(states[i,0],actions[i,0],maps[i])['target']for i in range(4)]),requires_grad=True)
        predicted=old.detach().clone().requires_grad_();logits=torch.zeros(4,2,requires_grad=True)
        controls=torch.tensor(actions[:,0],requires_grad=True)
        terms,metrics=player_motion_supervised_terms(old,predicted,controls,target,logits,
            player_local_motion_protocol=PLAYER_LOCAL_MOTION_PROTOCOL)
        for axis in('x','y'):
            self.assertAlmostEqual(metrics['axis_components'][axis]['free']['weight_mass'],1/3,places=6)
            self.assertAlmostEqual(metrics['axis_components'][axis]['blocked']['weight_mass'],1/6,places=6)
        self.assertEqual(metrics['x_free_y_blocked']['count'],1)
        self.assertEqual(metrics['x_blocked_y_free']['count'],1)
        self.assertEqual(metrics['false_stop_axes'],4)
        self.assertGreater(float(terms['free_progress'].detach()),0)
        sum(terms.values()).backward()
        # Actor1 slides along X; actor2 slides along Y. Closing the other
        # axis must not eliminate the gradient that keeps the free one moving.
        self.assertLess(float(predicted.grad[1,0]),0)
        self.assertEqual(float(predicted.grad[1,1]),0)
        self.assertEqual(float(predicted.grad[2,0]),0)
        self.assertLess(float(predicted.grad[2,1]),0)
        self.assertLess(float(logits.grad[1,0]),0);self.assertGreater(float(logits.grad[1,1]),0)
        self.assertGreater(float(logits.grad[2,0]),0);self.assertLess(float(logits.grad[2,1]),0)
        self.assertIsNone(old.grad);self.assertIsNone(target.grad);self.assertIsNone(controls.grad)

    def test_opt_in_is_exact_and_has_no_global_RNG_side_effect(self):
        states,actions,maps=source_tapes();np_before=deepcopy(np.random.get_state());torch_before=torch.get_rng_state().clone()
        source_player_motion_step(states[0,0],actions[0,0],maps[0])
        np.testing.assert_array_equal(np.random.get_state()[1],np_before[1])
        self.assertTrue(torch.equal(torch.get_rng_state(),torch_before))
        self.assertIsNone(normalize_player_local_motion_protocol(None))
        bad=deepcopy(PLAYER_LOCAL_MOTION_PROTOCOL);bad['axis_class_mass']['free']=1.
        with self.assertRaises(ValueError):normalize_player_local_motion_protocol(bad)


if __name__=='__main__':unittest.main()
