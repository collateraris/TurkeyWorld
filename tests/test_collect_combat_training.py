import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from turkeyworld.collect_combat_training import collect_job
from turkeyworld.engine import Game


class CombatCollectionTests(unittest.TestCase):
    def job(self, root, stage, case, seed=97000000, frames=32):
        return dict(stage=stage, case=case, split='train', index=0, seed=seed,
            frames=frames, relative_path=case+'.npz', path=str(Path(root)/(case+'.npz')),
            protocol='test_source_combat_v1')

    def test_combat_targets_are_exact_unmodified_source_transitions_and_masks(self):
        with tempfile.TemporaryDirectory() as root:
            job=self.job(root,'moving1','melee_then_shoot')
            result=collect_job(job)
            self.assertGreater(result['events']['player_hit'],0)
            with np.load(job['path'],allow_pickle=False) as saved:
                game=Game(seed=int(saved['seed']),endless=True,enemies_enabled=True)
                game.set_state(json.loads(str(saved['initial_state'])))
                for tick,action in enumerate(saved['actions']):
                    np.testing.assert_array_equal(game.state_vector(),saved['states'][tick])
                    np.testing.assert_array_equal(game.render(size=32,overlays=False),saved['frames'][tick])
                    event=game.step(action,dt=float(saved['dt']))
                    np.testing.assert_array_equal(Game.event_vector(event),saved['events'][tick])
                np.testing.assert_array_equal(game.state_vector(),saved['states'][-1])
                np.testing.assert_array_equal(np.any(saved['frames']!=saved['background_frames'],axis=-1),saved['object_masks'])
                self.assertTrue(bool(saved['enemies_enabled']))
            self.assertEqual(result,collect_job(job))

    def test_corner_has_two_close_walls_and_terminal_episode_is_not_fabricated_long_data(self):
        with tempfile.TemporaryDirectory() as root:
            job=self.job(root,'navigation','corner')
            result=collect_job(job)
            with np.load(job['path'],allow_pickle=False) as saved:
                self.assertFalse(bool(saved['enemies_enabled']))
                self.assertEqual(float(saved['attack_flags'].sum()),0.)
                game=Game(seed=int(saved['seed']),endless=True,enemies_enabled=False)
                game.set_state(json.loads(str(saved['initial_state'])))
                point=np.array([game.player['x'],game.player['y']])
                blocked=[]
                for direction in ((1,0),(0,1),(-1,0),(0,-1)):
                    if not game._walkable(*(point+.6*np.array(direction)),game.PLAYER_RADIUS):
                        blocked.append(np.array(direction))
                self.assertTrue(any(np.dot(a,b)==0 for a in blocked for b in blocked))
            terminal=collect_job(self.job(root,'moving1','terminal',97000001))
            self.assertTrue(terminal['terminal'])
            self.assertFalse(terminal['eligible_450'])
            self.assertGreaterEqual(terminal['transitions'],17)
            self.assertLess(terminal['transitions'],32)


if __name__=='__main__': unittest.main()
