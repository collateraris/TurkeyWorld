"""Joint450 images must use matching GT pairs and reach every scheduled offset."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from turkeyworld.train_staged_world import StagedWorldExperiment


class JointLongImageAlignmentTests(unittest.TestCase):
    def test_full_rollout_precedes_aligned_image_query_and_all_offsets_are_reachable(self):
        experiment=object.__new__(StagedWorldExperiment)
        experiment.args=SimpleNamespace(batch=2,context=2,cache_mb=1)
        experiment.curriculum=SimpleNamespace(stage={'id':'full_game'})
        experiment.rng=np.random.default_rng(1)
        experiment.ema=object()
        experiment.amp=lambda **kwargs:nullcontext()
        parameter=torch.nn.Parameter(torch.tensor(1.))
        experiment.image_refinement=lambda model,endpoint:(parameter,{'adversarial_weight':.005})
        markers=torch.arange(450.).view(1,450,1)
        batch={'target_states':markers.expand(2,450,20),
            'target_frames':markers.reshape(1,450,1,1,1).expand(2,450,3,4,4),
            'target_object_masks':torch.zeros(2,450,4,4),
            'target_background_frames':torch.zeros(2,450,3,4,4),'map':torch.zeros(2,16,16)}
        pool=SimpleNamespace(batch=lambda *args,**kwargs:batch)
        pools=SimpleNamespace(get=lambda name:pool)
        order=[]
        def backward(*args,**kwargs):
            order.append('450-backward')
            return {'train_loss':2.,'long_horizon':450,'long_forward_transitions':450}
        def image(model,ema,query,**kwargs):
            order.append('image')
            self.assertEqual(query['target_states'].shape[1],1)
            self.assertTrue(torch.equal(query['target_states'][:,0,0],query['target_frames'][:,0,0,0,0]))
            return parameter,{'base_image':1.,'flow':0.},{'target':query['target_frames'][:,0]}
        for index,phase_update in enumerate((0,1,2,4)):
            experiment.long_rollout={'joint_long_update_count':index}
            experiment.components=SimpleNamespace(phase={'updates':phase_update})
            parameter.grad=None
            with patch('turkeyworld.long_dynamics_training.long_dynamics_backward',side_effect=backward),\
                    patch('turkeyworld.train_staged_world.render_only_loss',side_effect=image):
                metrics=experiment.long_component_backward(None,pools,joint=True)
            self.assertEqual(order[-2:],['450-backward','image'])
            self.assertEqual(metrics['joint_render_future_offset'],(450,225,64,16)[index])
            self.assertEqual(metrics['train_loss'],4.)
            self.assertEqual(parameter.grad.item(),2.)


if __name__=='__main__':unittest.main()
