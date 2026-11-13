"""Causal memory fusion, semantic sparse routing and exact zero activation."""
import copy
import inspect
import io
import unittest

import torch

from turkeyworld.rssm_fusion_moe import RSSMFusionMoE,RSSMFusionMoEConfig,ROLES,TOKEN_NAMES


OUTPUT_SHAPES={'player_pass_residual':(2,),'enemy_motion_residual':(2,),
    'interaction_target_residual':(2,),'interaction_attack_residual':(1,),
    'interaction_spawn_tile_residual':(1,256),'interaction_spawn_anchor_residual':(1,9)}


def module():
    return RSSMFusionMoE(RSSMFusionMoEConfig(width=32,depth=2,heads=4,expert_hidden=64),
        player_memory_dim=12,enemy_memory_dim=10,map_width=16,spatial_width=16)


def inputs(count=4,requires_grad=False):
    result=[torch.randn(count,width)for width in (12,10,20,4,16)]
    result.append(torch.randn(count,2,16))
    return tuple(value.requires_grad_(requires_grad)for value in result)


class RSSMFusionMoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_zero_heads_preserve_old_values_and_memories_exactly_in_both_modes_without_rng(self):
        net=module();values=inputs();original=[v.clone()for v in values]
        weights={n:v.clone()for n,v in net.state_dict().items()}
        for training in (True,False):
            net.train(training);rng=torch.get_rng_state().clone()
            result=net(*values)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))
            for name,shape in OUTPUT_SHAPES.items():
                self.assertEqual(tuple(result[name].shape),(4,*shape))
                self.assertTrue(torch.equal(result[name],torch.zeros_like(result[name])))
                old=torch.randn_like(result[name])
                self.assertTrue(torch.equal(old,old+result[name]))
            self.assertEqual(tuple(result['role_tokens'].shape),(4,3,32))
            self.assertTrue(all(torch.equal(before,after)for before,after in zip(original,values)))
            self.assertTrue(all(torch.equal(v,net.state_dict()[n])for n,v in weights.items()))
        self.assertEqual(len(TOKEN_NAMES),10);self.assertEqual(ROLES,('player_navigation','npc_navigation','interaction'))

    def test_top2_is_sparse_positive_role_biased_and_normalized_and_all_three_experts_execute(self):
        net=module();seen=[0,0,0];handles=[]
        # A zero content router isolates the documented role-prior schedule.
        with torch.no_grad():net.router.weight.zero_();net.router.bias.zero_()
        for index,expert in enumerate(net.experts):
            def save(module,args,index=index):seen[index]+=len(args[0])
            handles.append(expert.register_forward_pre_hook(save))
        try:result=net(*inputs())
        finally:
            for handle in handles:handle.remove()
        indices=result['router_top_indices'];weights=result['router_top_weights']
        self.assertEqual(tuple(indices.shape),(4,3,2))
        self.assertTrue(torch.equal(indices[:,:,0],torch.arange(3)[None].expand(4,-1)))
        self.assertTrue(torch.equal(indices[:,:,1],((torch.arange(3)+1)%3)[None].expand(4,-1)))
        self.assertTrue(bool((weights>0).all()))
        torch.testing.assert_close(weights.sum(-1),torch.ones(4,3),rtol=0,atol=1e-7)
        self.assertEqual(seen,[8,8,8]);self.assertEqual(sum(seen),4*3*2)

    def test_role_ce_trains_router_and_shared_transformer_even_while_task_heads_are_zero(self):
        net=module();values=inputs(requires_grad=True)
        result=net(*values);result['router_role_loss'].backward()
        self.assertGreater(float(net.router.weight.grad.abs().sum()),0)
        self.assertGreater(float(net.blocks[0].self_attn.in_proj_weight.grad.abs().sum()),0)
        self.assertGreater(float(values[0].grad.abs().sum()),0)
        self.assertGreater(float(values[1].grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in net.player_head.parameters()))

    def test_balance_loss_has_nonzero_router_gradient_under_collapsed_dispatch(self):
        net=module()
        with torch.no_grad():net.router.weight.zero_();net.router.bias.copy_(torch.tensor([8.,0.,-8.]))
        result=net(*inputs())
        self.assertTrue(result['router_balance_loss'].requires_grad)
        result['router_balance_loss'].backward()
        self.assertGreater(float(net.router.bias.grad.abs().sum()),0)
        self.assertGreater(float(result['router_load'][0]),float(result['router_load'][2]))

    def test_nonzero_player_head_reaches_both_previous_memories_without_detach_or_future_inputs(self):
        net=module();values=inputs(requires_grad=True)
        with torch.no_grad():
            net.player_head.weight[0,0]=.13;net.player_head.weight[1,3]=-.17
        result=net(*values)
        result['player_pass_residual'].square().sum().backward()
        for index in (0,1):
            self.assertIsNotNone(values[index].grad)
            self.assertTrue(bool(torch.isfinite(values[index].grad).all()))
            self.assertGreater(float(values[index].grad.abs().sum()),0)
        self.assertGreater(sum(float(p.grad.abs().sum())for expert in net.experts for p in expert.parameters()
                               if p.grad is not None),0)
        self.assertEqual(tuple(inspect.signature(net.forward).parameters),
            ('previous_player_memory','previous_enemy_memory','state','action','map_context','spatial_tokens',
             'combat_context'))
        with self.assertRaises(TypeError):net(*values,future_rgb=torch.zeros(4,3,32,32))

    def test_each_head_reads_its_role_and_strict_serialization_keeps_routes_and_predictions(self):
        net=module().eval();values=inputs()
        with torch.no_grad():
            net.player_head.bias.fill_(.1);net.enemy_head.bias.fill_(.2)
            net.combat_head.bias.copy_(torch.arange(268).float()/1000)
        expected=net(*values)
        torch.testing.assert_close(expected['interaction_target_residual'][0],torch.tensor([0.,.001]))
        self.assertAlmostEqual(float(expected['interaction_attack_residual'][0,0].detach()),.002,places=7)
        self.assertAlmostEqual(float(expected['interaction_spawn_tile_residual'][0,0,0].detach()),.003,places=7)
        self.assertAlmostEqual(float(expected['interaction_spawn_anchor_residual'][0,0,0].detach()),.259,places=7)
        buffer=io.BytesIO();torch.save(net.state_dict(),buffer);buffer.seek(0)
        restored=module().eval();restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
        actual=restored(*values)
        for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
        malformed=copy.deepcopy(net.state_dict());malformed.pop('role_router_bias')
        with self.assertRaises(RuntimeError):restored.load_state_dict(malformed,strict=True)

    def test_raises_on_invalid_shapes_nonfinite_inputs_and_invalid_role_counts(self):
        for kwargs in ({'experts':4},{'top_k':1},{'width':31},{'depth':True},{'role_bias':float('nan')}):
            with self.assertRaises(ValueError):RSSMFusionMoEConfig(**kwargs)
        net=module();values=list(inputs())
        values[1]=torch.zeros(4,11)
        with self.assertRaises(ValueError):net(*values)
        values=list(inputs());values[0][0,0]=float('inf')
        with self.assertRaises(FloatingPointError):net(*values)

    def test_optional_previous_combat_context_reuses_existing_token_and_zero_is_bitexact(self):
        net=module().eval();values=inputs();names=tuple(net.state_dict())
        old=net(*values);zero=net(*values,combat_context=torch.zeros(4,32))
        for key in old:self.assertTrue(torch.equal(old[key],zero[key]),key)
        self.assertEqual(names,tuple(net.state_dict()))
        combat=torch.randn(4,32,requires_grad=True)
        changed=net(*values,combat_context=combat)
        self.assertFalse(torch.equal(changed['role_tokens'],old['role_tokens']))
        changed['router_role_loss'].backward()
        self.assertGreater(float(combat.grad.abs().sum()),0)
        self.assertEqual(tuple(changed['role_tokens'].shape),(4,3,32))
        with self.assertRaises(ValueError):net(*values,combat_context=torch.zeros(4,31))
        with self.assertRaises(FloatingPointError):net(*values,combat_context=torch.full((4,32),float('nan')))


if __name__=='__main__':unittest.main()
