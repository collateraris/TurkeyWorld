import copy
import unittest
from turkeyworld.run_wall_ablation import decide
class WallSelectionTests(unittest.TestCase):
 def report(self,wall=10,drift=1,loss=1):
  return {s:dict(player_XY_H180_mean_tiles=drift,enemy_XY_H180_mean_tiles=drift,player_wall_overlap_ticks_mean=wall,objective_mean={'weighted_motion_loss':loss}) for s in ('navigation','enemy_motion')}
 def test_candidate_must_improve_walls_without_drift_loss_regression(self):
  self.assertEqual(decide({'025':self.report(),'050':self.report(wall=8)})['selected'],'050')
  self.assertEqual(decide({'025':self.report(),'050':self.report(wall=8,drift=1.2)})['selected'],'025')
  self.assertEqual(decide({'025':self.report(),'050':self.report(wall=8,loss=1.2)})['selected'],'025')
  self.assertEqual(decide({'025':self.report(),'050':self.report(wall=10)})['selected'],'025')
