import unittest
from types import SimpleNamespace
import torch
from torch import nn
from turkeyworld.base_only_renderer import enable_base_only
from turkeyworld.models import NeuralWorldModel

class Model(nn.Module):
    render=NeuralWorldModel.render
    render_with_aux=NeuralWorldModel.render_with_aux
    def __init__(self):
        super().__init__();self.base_student=nn.Linear(1,1)
        self.base_student_metadata={'training_steps':31709}
        self.config=SimpleNamespace(renderer_mode='rectified_flow',architecture='dreamer_rssm',spatial_combat=True)
        self.flow_renderer=nn.Module();self.flow_renderer.codec=nn.Linear(4,4)
        self.flow_renderer.network=nn.Linear(4,4);self.flow_renderer.geometry_opacity=nn.Linear(2,1)
    def render_base_with_residual(self,state,features,grid):return state,None

class BaseOnlyTests(unittest.TestCase):
    def test_flow_and_codec_removed_both_render_paths_equal_base(self):
        model=Model();base=model.base_student;opacity=model.flow_renderer.geometry_opacity
        metadata=enable_base_only(model);rgb=torch.rand(2,3,32,32)
        self.assertFalse(hasattr(model.flow_renderer,'codec'))
        self.assertFalse(hasattr(model.flow_renderer,'network'))
        self.assertIs(model.base_student,base);self.assertIs(model.flow_renderer.geometry_opacity,opacity)
        self.assertTrue(torch.equal(model.render(rgb,None,None),rgb))
        self.assertTrue(torch.equal(model.render_with_aux(rgb,None,None)['rgb'],rgb))
        self.assertEqual(metadata['removed_parameters'],40)
    def test_reject_missing_compact_base(self):
        with self.assertRaises(ValueError):enable_base_only(nn.Module())
