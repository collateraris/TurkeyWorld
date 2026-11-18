import unittest
from dataclasses import replace
from turkeyworld.models import WorldModelConfig
from turkeyworld.train_dreamer_policy import compatible_vjepa_depth_growth
class CapacityTests(unittest.TestCase):
 def test_only_documented_depth_growth(self):
  old=WorldModelConfig(architecture='vjepa2_ac',image_size=32,vjepa_width=512,vjepa_encoder_depth=18,vjepa_predictor_depth=12,vjepa_patch_size=4)
  new=replace(old,vjepa_encoder_depth=34,vjepa_predictor_depth=59)
  proof=dict(protocol='vjepa_dynamics300_depth_growth_v1',old_weights_exact=True,new_blocks_identity=True,encoder_output_exact=True,predictor_output_exact=True,old_depths={'encoder':18,'predictor':12},encoder_depth=34,predictor_depth=59,width=512)
  self.assertTrue(compatible_vjepa_depth_growth(old,new,proof))
  self.assertFalse(compatible_vjepa_depth_growth(old,new,None))
  self.assertFalse(compatible_vjepa_depth_growth(old,replace(new,vjepa_width=768),proof))
  self.assertFalse(compatible_vjepa_depth_growth(old,replace(new,vjepa_context_frames=8),proof))
  self.assertFalse(compatible_vjepa_depth_growth(old,replace(new,vjepa_predictor_depth=60),proof))
  self.assertFalse(compatible_vjepa_depth_growth(old,new,{**proof,'encoder_output_exact':False}))
