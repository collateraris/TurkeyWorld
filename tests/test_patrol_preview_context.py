import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from turkeyworld.vjepa_validation import save_autoregressive_preview


class PatrolPreviewTests(unittest.TestCase):
    def test_own_states_and_navigation_flag_are_scoped_to_preview_call(self):
        class SpriteStub(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
                self.config={'architecture':'patrol_sprite_detail_sr'}
                self.enemies_enabled=True;self.predicted_sprite_state=None
                self.seen=None
            def forward(self,image):
                self.seen=(self.enemies_enabled,self.predicted_sprite_state.clone())
                return torch.nn.functional.interpolate(image,scale_factor=4)
        model=SpriteStub();states=torch.zeros(4,20);states[:,14]=1
        preview=dict(context=np.zeros((1,32,32,3),np.uint8),source=np.zeros((4,32,32,3),np.uint8),
            generated=np.zeros((4,32,32,3),np.uint8),sr_predicted_states=states,sr_enemies_enabled=False,
            metadata=dict(step=1,generated_finite=True,caption='Own rollout patrol preview'))
        with tempfile.TemporaryDirectory() as directory:
            result=save_autoregressive_preview(preview,Path(directory)/'preview.png',sr_model=model,sr_metadata={'training_steps':6000})
        self.assertFalse(model.seen[0]);self.assertTrue(torch.equal(model.seen[1],states))
        self.assertTrue(model.enemies_enabled);self.assertIsNone(model.predicted_sprite_state)
        self.assertEqual(result['sr_training_steps'],6000)
        self.assertFalse(result['sr_combat_animation_sprites'])
