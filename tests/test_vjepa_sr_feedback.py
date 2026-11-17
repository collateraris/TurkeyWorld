import unittest
from dataclasses import replace
import torch
from test_base_student import tiny_teacher,states
from turkeyworld.models import NeuralWorldModel
from turkeyworld.vjepa_sr_feedback import blend_memory,feedback_active

class PooledFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_pooling_strictly_reuses_weights_and_feedback_preserves_past(self):
        old=tiny_teacher();model=NeuralWorldModel(replace(old.config,vjepa_pooled_memory=True)).eval()
        model.load_state_dict(old.state_dict(),strict=True)
        state=states();grid=torch.zeros(2,16,16,dtype=torch.long);rgb=torch.rand(2,3,32,32)
        fmap=model.encode_map(grid)
        memory=model.initialize_from_image(rgb,state,fmap)
        core=model.vjepa_backend.core
        self.assertEqual(core.visual_tokens,1)
        self.assertEqual(core.memory_tokens,core.context_frames*3)
        self.assertEqual(memory.shape,(2,core.memory_tokens,core.width))
        memory=memory.detach().requires_grad_();before=memory.reshape(2,core.context_frames,3,core.width)
        following=blend_memory(model,memory,rgb,{'blend':.1},torch.tensor([True,False]))
        after=following.reshape_as(before)
        self.assertTrue(torch.equal(before[:,:-1],after[:,:-1]))
        self.assertTrue(torch.equal(before[:,-1,:2],after[:,-1,:2]))
        self.assertTrue(torch.equal(memory[1],following[1]))
        following.sum().backward()
        gradient=memory.grad.reshape_as(before)
        self.assertTrue(torch.allclose(gradient[0,-1,2],torch.full_like(gradient[0,-1,2],.9)))
        self.assertEqual(feedback_active(torch.tensor([[0.]*19+[8/9000],[0.]*19+[9/9000]]),{'every':8}).tolist(),[True,False])

    def test_validation_memory_ablation_changes_tokens_without_touching_weights(self):
        from turkeyworld.validate_vjepa_sr import configure_memory_mode
        model=tiny_teacher();before={k:v.clone() for k,v in model.state_dict().items()}
        full=model.core.memory_tokens
        configure_memory_mode(model,'pooled')
        self.assertEqual(model.core.memory_tokens,model.core.context_frames*3)
        self.assertTrue(model.config.vjepa_pooled_memory)
        configure_memory_mode(model,'spatial')
        self.assertEqual(model.core.memory_tokens,full)
        self.assertFalse(model.config.vjepa_pooled_memory)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]),k)
