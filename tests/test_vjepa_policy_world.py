import unittest
import torch
from test_base_student import tiny_teacher,states
from turkeyworld.vjepa_policy_world import imagination_adapter,VJEPAImaginationAdapter

class VJEPAPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_prior_has_full_memory_compact_features_and_no_future_encoder(self):
        m=tiny_teacher();adapter=imagination_adapter(m);self.assertIsInstance(adapter,VJEPAImaginationAdapter)
        state=states();grid=torch.zeros(2,16,16,dtype=torch.long);rgb=torch.rand(2,3,32,32)
        current=adapter.initialize(state,grid,rgb)
        self.assertEqual(adapter.features(current).shape,(2,148))
        calls=[];hook=m.core.encoder.register_forward_hook(lambda *args:calls.append(1))
        next_state=adapter.transition_prior(current,torch.zeros(2,4));self.assertEqual(calls,[])
        observed=adapter.observe(current,torch.zeros(2,4),state,rgb)
        self.assertEqual(len(calls),1);hook.remove()
        n=m.core.block_tokens*m.core.width
        self.assertTrue(torch.equal(current.memory[:,n:],observed.memory[:,:-n]))
        self.assertEqual(next_state.memory.shape,current.memory.shape)
        self.assertTrue(all(not p.requires_grad for p in m.parameters()))

    def test_observe_retains_the_actual_preceding_action_in_history(self):
        m=tiny_teacher();adapter=imagination_adapter(m);state=states();grid=torch.zeros(2,16,16,dtype=torch.long);rgb=torch.rand(2,3,32,32)
        current=adapter.initialize(state,grid,rgb);action=torch.full((2,4),.4)
        observed=adapter.observe(current,action,state,rgb);core=m.vjepa_backend.core
        blocks=observed.memory.reshape(2,core.context_frames,core.block_tokens,core.width)
        self.assertTrue(torch.allclose(blocks[:,-2,0],core.predictor.action_encoder(action)))
        self.assertTrue(torch.allclose(blocks[:,-1,0],core.predictor.action_encoder(torch.zeros_like(action))))
