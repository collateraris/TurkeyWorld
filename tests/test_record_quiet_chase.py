import unittest
from turkeyworld.record_quiet_chase import candidates,action_at,quality_score
from turkeyworld.record_paired_cuts import scene_factory

class QuietChaseTests(unittest.TestCase):
    def test_candidates_have_distinct_starts_and_stay_quiet_for_six_seconds(self):
        scenes=candidates()+candidates(True);self.assertEqual(len(scenes),32);self.assertEqual(len({tuple(s['player']) for s in scenes}),32)
        for index,scene in enumerate(scenes):
            game=scene_factory(scene)(seed=scene['seed'],endless=True,enemies_enabled=True)
            hp=game.player['hp'];npc_hp=game.enemies[0]['hp']
            for tick in range(180):
                action=action_at(tick,index>=16);self.assertEqual(action[3],0);events=game.step(action,dt=1/30)
                self.assertEqual(game.player['hp'],hp);self.assertEqual(game.enemies[0]['hp'],npc_hp);self.assertFalse(any(events[k] for k in ('shot','hit','kill','player_hit')))
                self.assertTrue(game._line_clear(game.player['x'],game.player['y'],game.enemies[0]['x'],game.enemies[0]['y']))
    def test_declared_score_prefers_less_drift_and_visible_sprites(self):
        good=dict(enemy_xy_mean=.1,player_xy_mean=.1,sprite_frames=180,wall_fault_frames=0,sprite_switches=0)
        self.assertLess(quality_score(good),quality_score({**good,'enemy_xy_mean':2}))
        self.assertLess(quality_score(good),quality_score({**good,'sprite_frames':0}))
    def test_tracking_uses_only_predicted_state_and_never_fires(self):
        from turkeyworld.record_quiet_chase import tracking_action
        import numpy as np
        state=np.zeros(20,np.float32);state[:2]=[.5,.5];state[3]=1;state[8:10]=[.5,.65]
        action=tracking_action(state);self.assertEqual(action[3],0);self.assertGreater(action[2],0)
        self.assertLessEqual(action[0],0);self.assertGreaterEqual(action[0],-.65)
    def test_walking_start_and_tracking_stay_quiet_in_reference(self):
        from turkeyworld.record_quiet_chase import tracking_action
        for scene in candidates(True):
            scene['walking_start']=True;game=scene_factory(scene)(seed=scene['seed'],endless=True,enemies_enabled=True);npc_hp=game.enemies[0]['hp']
            for tick in range(180):
                events=game.step(tracking_action(game.state_vector()),dt=1/30)
                self.assertEqual(game.player['hp'],100);self.assertEqual(game.enemies[0]['hp'],npc_hp)
                self.assertFalse(any(events[k] for k in ('shot','hit','kill','player_hit')))
