import inspect
import unittest
from unittest.mock import patch

import torch

from turkeyworld import enemy_animation
from turkeyworld.enemy_animation import enemy_animation_index
from turkeyworld.engine import Game


class EnemyAnimationTests(unittest.TestCase):
    def test_seven_classes_priority_normalized_thresholds_and_shape(self):
        states=torch.zeros(7,20);states[:,13]=1
        states[1:3,10]=.1;states[2,17]=.5
        states[3,14]=1.;states[3,10]=.1
        states[4,15]=1.;states[4,14]=1.
        states[5:,13]=0;states[5:,15]=1.;states[6,16]=1.
        self.assertEqual(enemy_animation_index(states).tolist(),list(range(7)))
        self.assertEqual(enemy_animation_index(states.reshape(1,7,20)).shape,(1,7))
        self.assertEqual(enemy_animation_index(states).device,states.device)
        self.assertEqual(enemy_animation_index(states).dtype,torch.long)
        edge=torch.zeros(4,20);edge[:,13]=1
        edge[0,10]=.025;edge[1,14]=1/1.2
        edge[2:,13]=0;edge[2,16]=.35/.6;edge[3,16]=.35/.6-1e-4
        self.assertEqual(enemy_animation_index(edge).tolist(),[0,0,6,5])

    def test_matches_actual_source_sprite_selection_without_calling_source_in_utility(self):
        game=Game(seed=173,endless=True)
        cases=[{},dict(vx=.1,animation_phase=.2),dict(vx=.1,animation_phase=.5),
            dict(attack_cooldown=1.1,vx=.1),dict(hurt_timer=.1,attack_cooldown=1.1),
            dict(alive=False,hp=0.,death_timer=.1,hurt_timer=.1),
            dict(alive=False,hp=0.,death_timer=.35),dict(vx=.05),dict(attack_cooldown=1.)]
        for changes in cases:
            state=game.get_state();state['player'].update(x=2.5,y=2.5,angle=0.,hp=100.)
            state['enemies'][0].update(x=3.5,y=2.5,vx=0.,vy=0.,hp=60.,alive=True,
                attack_cooldown=0.,hurt_timer=0.,death_timer=0.,animation_phase=0.)
            state['enemies'][0].update(changes);game.set_state(state);seen=[]
            def blit(target,sprite,*args,**kwargs):
                seen.extend(i for i,value in enumerate(game._turkeys)if sprite is value)
            with patch.object(Game,'_blit_rgba',side_effect=blit):
                game.render(size=32,overlays=False)
            self.assertEqual(len(seen),1)
            self.assertEqual(enemy_animation_index(torch.from_numpy(game.state_vector())).item(),seen[0])
        self.assertNotIn('from .engine',inspect.getsource(enemy_animation))
        self.assertNotIn('Game',enemy_animation.__dict__)

    def test_invalid_or_nonfinite_states_fail_closed(self):
        for value in (torch.zeros(2,19),torch.zeros(2,20,dtype=torch.long),torch.full((2,20),float('nan'))):
            with self.assertRaises(ValueError):enemy_animation_index(value)


if __name__=='__main__':unittest.main()
