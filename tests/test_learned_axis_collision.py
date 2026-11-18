import unittest
from dataclasses import asdict
from unittest.mock import patch
import torch

from test_player_axis_fraction import axis_config
from turkeyworld.engine import Game
from turkeyworld.models import NeuralWorldModel,WorldModelConfig
from turkeyworld.learned_axis_collision import LocalAxisCollisionHead, with_local_axis_head, MODE, PREFIX


class LocalCollisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.h=LocalAxisCollisionHead()
        self.g=Game(seed=10,endless=True,enemies_enabled=False)
        self.s=torch.tensor(self.g.state_vector())[None].repeat(2,1)
        self.s[:,2:4]=torch.tensor([0.,1.])
        self.s[:,:2]=torch.tensor([4.78,4.78])/16
        self.grid=torch.zeros(2,16,16,dtype=torch.long)
        self.grid[0,5,5]=1
        self.a=torch.tensor([[1.,1.,0.,0.]]).repeat(2,1)

    def test_moves_with_own_x_then_y_and_no_future_inputs(self):
        # Control the generic learned basis only for this directional wiring check.
        # Corner contact at proposed XY, no contact at the single-axis probes.
        with torch.no_grad():
            self.h.score.weight.zero_();self.h.score.bias.fill_(-.39)
            self.h.score.weight[0,4:6].fill_(.4)
        out=self.h(self.s,self.a,self.grid)
        self.assertEqual(out['player_pass_fraction'].tolist(),[[1,0],[1,1]])
        self.assertTrue(torch.equal(out['raw_player_delta'],out['player_command_delta']*out['player_pass_fraction']))
        y=self.h(self.s,torch.tensor([[0.,.7071068,0.,0.]]).repeat(2,1),self.grid)
        self.assertEqual(y['player_pass_fraction'][0,1].item(),1.)
        idle=self.h(self.s,torch.zeros_like(self.a),self.grid[0])
        self.assertTrue(torch.equal(idle['raw_player_delta'],torch.zeros_like(idle['raw_player_delta'])))
        # Values used by learned supervision have gradients to its own parameters only.
        torch.nn.functional.binary_cross_entropy_with_logits(out['player_pass_logits'], torch.ones_like(self.a[:,:2])).backward()
        self.assertGreater(self.h.score.weight.grad.abs().sum().item(),0.)

    def test_checkpoint_appends_only_and_model_uses_current_map(self):
        world=NeuralWorldModel(axis_config()).eval()
        payload={'config':asdict(world.config),'model':world.state_dict(),'training_steps':17,
                 'optimizer':{},'optimizer_parameter_names':[n for n,p in world.named_parameters()]}
        with torch.no_grad():
            self.h.score.weight.zero_();self.h.score.bias.fill_(-.5)
        moved=with_local_axis_head(payload,self.h)
        self.assertIs(moved['optimizer'],payload['optimizer'])
        self.assertEqual(set(moved['model'])-set(payload['model']),{PREFIX+'score.weight',PREFIX+'score.bias'})
        for k in payload['model']:self.assertIs(moved['model'][k],payload['model'][k])
        self.assertEqual(moved['config']['player_motion_mode'],MODE)
        new=NeuralWorldModel(WorldModelConfig(**moved['config'])).eval()
        new.load_state_dict(moved['model'],strict=True)
        with torch.inference_mode(),patch.object(Game,'step',side_effect=RuntimeError('Cannot read simulator')):
            for changed in (self.grid,self.grid.flip(0)):
                maps=new.encode_map(changed)
                memory=torch.full((2,new.core.memory_tokens,new.core.width),.01)
                out=new.transition(self.s,self.a,maps,memory)
                expected=self.h(self.s,self.a,changed)
                self.assertTrue(torch.equal(out['raw_player_delta'],expected['raw_player_delta']))
                torch.testing.assert_close(out['state'][:,:2],self.s[:,:2]+expected['raw_player_delta'],rtol=0,atol=1e-7)
        with self.assertRaises(ValueError):with_local_axis_head(moved,self.h)

    def test_nav_only_keeps_pursuit_exact_and_validates_migration_and_freeze(self):
        from dataclasses import replace
        from turkeyworld.learned_axis_collision import compatible_policy_refresh,verify_frozen_resume
        from turkeyworld.renderer_student import _hash_tensors
        from turkeyworld.rssm_combat_protocol import environment_profile
        # Miniature full world, including static-map and typed prior code paths.
        base=NeuralWorldModel(axis_config()).eval()
        legacy_frozen={'vjepa_backend.delta_head.0.bias':base.vjepa_backend.delta_head[0].bias}
        old_hash=_hash_tensors(legacy_frozen)
        payload={'config':asdict(base.config),'model':base.state_dict(),'training_steps':2,
                 'vjepa_sr_training':{'frozen_parameters_sha256':old_hash}}
        migrated=with_local_axis_head(payload,self.h,nav_only=True)
        model=NeuralWorldModel(WorldModelConfig(**migrated['config'])).eval()
        model.load_state_dict(migrated['model'],strict=True)
        self.assertTrue(compatible_policy_refresh(base.config,model,migrated))
        self.assertFalse(compatible_policy_refresh(replace(base.config,vjepa_predictor_depth=2),model,migrated))
        frozen={**legacy_frozen,**{PREFIX+k:v for k,v in model.vjepa_backend.local_axis_head.named_parameters()}}
        current_hash=_hash_tensors(frozen)
        self.assertTrue(verify_frozen_resume(model,migrated,frozen,current_hash))
        self.assertTrue(verify_frozen_resume(model,{**migrated,'vjepa_sr_training':{'frozen_parameters_sha256':current_hash},
                                                   'training_steps':3},frozen,current_hash))
        with self.assertRaises(ValueError):
            verify_frozen_resume(model,{**migrated,'training_steps':3},frozen,current_hash)
        def rollout(m):
            fmap=m.encode_map(self.grid);state=self.s
            mem=torch.full((2,m.core.memory_tokens,m.core.width),.01)
            with torch.inference_mode(),environment_profile(m,navigation_only=False):
                for _ in range(3):
                    result=m.transition(state,self.a,fmap,mem)
                    self.assertNotIn('player_pass_logits',result)
                    state=result['state'];mem=result['memory']
                return state,mem
        for x,y in zip(rollout(base),rollout(model)): self.assertTrue(torch.equal(x,y))
        with torch.no_grad():model.vjepa_backend.local_axis_head.score.bias.add_(.1)
        self.assertFalse(compatible_policy_refresh(base.config,model,migrated))
