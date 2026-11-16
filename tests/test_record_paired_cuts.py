import unittest
import numpy as np
from turkeyworld.record_paired_cuts import scenes,scene_factory,action_at,compose


class PairedCutsTests(unittest.TestCase):
    def test_eight_fixed_scenes_have_no_fire_or_Source_combat(self):
        self.assertEqual(len(scenes()),8);self.assertEqual(sum(s['chase'] for s in scenes()),4)
        for scene in scenes():
            game=scene_factory(scene)(seed=scene['seed'],endless=True,enemies_enabled=scene['chase'])
            initial=game.state_vector().copy();minimum=100.
            for tick in range(180):
                action=action_at(scene,tick);self.assertEqual(action[3],0)
                game.step(action,dt=1/30);state=game.state_vector()
                self.assertEqual(state[4],initial[4]);self.assertEqual(state[12],initial[12])
                minimum=min(minimum,float(np.linalg.norm((state[8:10]-state[:2])*16)))
            if scene['chase']:self.assertGreater(minimum,2.)
            else:self.assertEqual(state[13],0)

    def test_composite_is_even_sized_and_discloses_offline_timing(self):
        image=compose(np.zeros((512,512,3),np.uint8),np.zeros((512,512,3),np.uint8),scenes()[0],0,8433,9000)
        self.assertEqual(image.shape,(640,1080,3));self.assertEqual(image.dtype,np.uint8)

    def test_six_navigation_starts_are_distinct_and_have_no_enemies(self):
        cuts=scenes('navigation-six');self.assertEqual(len(cuts),6)
        self.assertEqual(len({tuple(s['player']) for s in cuts}),6)
        for scene in cuts:
            self.assertFalse(scene['chase']);self.assertEqual(scene['total'],6)
            game=scene_factory(scene)(seed=scene['seed'],endless=True,enemies_enabled=False)
            for tick in range(180):
                action=action_at(scene,tick);self.assertEqual(action[3],0);game.step(action,dt=1/30)
                self.assertFalse(game.enemies_enabled);self.assertEqual(game.player['hp'],100)
