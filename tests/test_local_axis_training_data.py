import unittest
import numpy as np
import torch

from turkeyworld.engine import Game
from turkeyworld.structured_state import player_command_delta
from turkeyworld.train_local_axis_collision import teacher, walkable


class CounterfactualTeacherTest(unittest.TestCase):
    def test_x_before_y_for_slide_into_corner(self):
        g=Game(seed=9,enemies_enabled=False);g.map=np.zeros((16,16),dtype=np.uint8);g.map[4,4]=1
        grid=g.map[None]
        rng=np.random.default_rng(29)
        state=np.tile(g.state_vector(),(1024,1));state[:,:2]=rng.uniform([3.75,3.75],[4.05,4.05],(len(state),2))/16
        state[:,2:4]=[0.,1.]
        a=rng.uniform(-1,1,(len(state),4)).astype(np.float32);a[:,2:]=0
        use=walkable(grid,np.zeros(len(a),int),state[:,:2].astype(np.float64)*16)
        state=state[use];a=a[use]
        frac=teacher(grid,np.zeros(len(a),int),state,a)
        command=player_command_delta(torch.from_numpy(state),torch.from_numpy(a)).numpy()*16
        starts=state[:,:2].astype(np.float64)*16
        checked=0
        for i,xy in enumerate(starts):
            new=dict(x=xy[0],y=xy[1]);g._move(new,*command[i].astype(np.float64),Game.PLAYER_RADIUS)
            actual=np.array([new['x'],new['y']])-xy
            np.testing.assert_allclose(actual,frac[i]*command[i],rtol=0,atol=1e-7)
            checked+=int(frac[i,0]!=frac[i,1])
        self.assertGreater(checked,5)
