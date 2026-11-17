import unittest
import numpy as np
from turkeyworld.collect_targeted_motion import SCENARIOS,control,initialize,episode
from turkeyworld.collect_policy_motion import _cohort_valid
class TargetedTests(unittest.TestCase):
 def test_holds_and_variations(self):
  for name in SCENARIOS:
   actions=np.stack([control(name,t) for t in range(360)])
   self.assertTrue(np.all(actions[:180]==actions[0]))
   self.assertTrue(np.all(actions[:,3]==0))
   self.assertTrue(np.isfinite(actions).all())
   self.assertFalse(np.array_equal(actions[0],actions[180]))
  np.testing.assert_array_equal(control('forward',0),[1,0,0,0])
  np.testing.assert_array_equal(control('left',0),[0,-1,0,0])
  np.testing.assert_array_equal(control('right',0),[0,1,0,0])
 def test_initialization_legal_and_reproducible(self):
  for name in SCENARIOS:
   for variant in range(4):
    a=initialize(100+variant,'enemy_motion',name,variant);b=initialize(100+variant,'enemy_motion',name,variant)
    np.testing.assert_array_equal(a.state_vector(),b.state_vector())
    s=a.state_vector();x,y=(s[:2]*16).astype(int);self.assertEqual(a.map[y,x],0)
    x,y=(s[8:10]*16).astype(int);self.assertEqual(a.map[y,x],0)
 def test_source_episode_alignment(self):
  p=episode(2026,'navigation','left',0)
  self.assertEqual(p['states'].shape,(361,20));self.assertEqual(p['frames'].shape,(361,32,32,3))
  self.assertEqual(p['actions'].shape,(360,4));self.assertTrue(_cohort_valid(p['states'],p['actions'],'navigation')[0])
  from tempfile import TemporaryDirectory
  from pathlib import Path
  from turkeyworld.jepa_training_utils import SequenceData
  with TemporaryDirectory() as temp:
   root=Path(temp)/'train';root.mkdir();np.savez_compressed(root/'episode.npz',**p)
   data=SequenceData(root,device='cpu',split='train',image_size=32)
   batch=data.batch(1,context=1,horizon=180,indices=np.asarray([[0,0]]))
   np.testing.assert_array_equal(batch['target_states'][0].numpy(),p['states'][1:181])
