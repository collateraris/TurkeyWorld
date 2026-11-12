"""Migration checks must compare identical MHA paths and preserve training state."""
import random
import unittest

import numpy as np
import torch

from test_renderer_enemy_objectives import tiny_model,batch
from turkeyworld.base_student import enable_base_pixel_residual
from turkeyworld.renderer_enemy_objectives import enemy_conditions
from turkeyworld.renderer_refinement import configure_refinement
from turkeyworld.renderer_pixel_migration import verify_zero_pixel_residual


class RendererPixelMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_real_996844_head_mixed_train_eval_differs_but_exact_verifier_matches_and_restores(self):
        fastpath=torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(True)
            model=tiny_model();old=model.base_student
            model.base_student=enable_base_pixel_residual(old);configure_refinement(model)
            self.assertEqual(sum(p.numel()for p in old.parameters()),996844)
            source=batch((False,True,False,True));q=enemy_conditions(model,source)
            old.eval()
            with torch.no_grad():
                left=old(q['state'],q['features'],q['geometry'])
                right,residual=model.base_student.forward_with_residual(q['state'],q['features'],q['geometry'])
            self.assertFalse(torch.equal(left,right))  # Actual previous startup failure.
            self.assertGreater(float((left-right).abs().max()),0)
            self.assertFalse(bool(residual.any()))
            # Confirm the mode mismatch is the MHA fastpath switch, rather
            # than a real zero-residual or copied-weight change.
            torch.backends.mha.set_fastpath_enabled(False)
            with torch.no_grad():
                slow_left=old(q['state'],q['features'],q['geometry'])
                slow_right,_=model.base_student.forward_with_residual(q['state'],q['features'],q['geometry'])
            self.assertTrue(torch.equal(slow_left,slow_right))
            torch.backends.mha.set_fastpath_enabled(True)
            model.base_student.blocks[0].ff.eval()  # Mixed child modes must also survive.
            modes=[m.training for m in model.modules()];old_modes=[m.training for m in old.modules()]
            flags=[p.requires_grad for p in model.parameters()];old_flags=[p.requires_grad for p in old.parameters()]
            weights={key:value.clone()for key,value in model.state_dict().items()}
            rng=torch.get_rng_state().clone();py_rng=random.getstate();np_rng=np.random.get_state()
            proof=verify_zero_pixel_residual(model,old,source)
            self.assertTrue(proof['base_rgb_bitexact_at_initialization'])
            self.assertTrue(proof['final_rgb_bitexact_at_initialization'])
            self.assertEqual(proof['comparison_mode'],'eval')
            self.assertEqual([m.training for m in model.modules()],modes)
            self.assertEqual([m.training for m in old.modules()],old_modes)
            self.assertEqual([p.requires_grad for p in model.parameters()],flags)
            self.assertEqual([p.requires_grad for p in old.parameters()],old_flags)
            self.assertFalse(proof['comparison_parameter_gradients_enabled'])
            self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(random.getstate(),py_rng)
            np.testing.assert_equal(np.random.get_state(),np_rng)
            self.assertTrue(all(torch.equal(value,model.state_dict()[key])for key,value in weights.items()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        finally:torch.backends.mha.set_fastpath_enabled(fastpath)

    def test_original_half_head_is_compared_in_working_float32_then_dtype_modes_and_caches_restore(self):
        model=tiny_model();old=model.base_student.half().train()
        model.base_student=enable_base_pixel_residual(old).float();configure_refinement(model)
        source=batch((False,True))
        cache=torch.tensor([42.]);model.flow_renderer._map_condition=cache
        original={key:value.clone()for key,value in old.state_dict().items()}
        proof=verify_zero_pixel_residual(model,old,source)
        self.assertEqual(proof['comparison_dtype'],'torch.float32')
        self.assertEqual(next(old.parameters()).dtype,torch.float16)
        self.assertTrue(old.training);self.assertTrue(model.base_student.training)
        self.assertIs(model.flow_renderer._map_condition,cache)
        self.assertTrue(all(torch.equal(value,old.state_dict()[key])for key,value in original.items()))

    def test_strict_assert_remains_and_exception_restores_all_modes_dtype_and_rng(self):
        model=tiny_model();old=model.base_student.half().train()
        model.base_student=enable_base_pixel_residual(old).float();configure_refinement(model)
        source=batch((False,True))
        with torch.no_grad():model.base_student.pixel_residual.bias.fill_(.001)
        rng=torch.get_rng_state().clone();modes=[m.training for m in model.modules()]
        flags=[p.requires_grad for p in model.parameters()]
        with self.assertRaisesRegex(AssertionError,'residual_zero=False'):
            verify_zero_pixel_residual(model,old,source)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertEqual([m.training for m in model.modules()],modes)
        self.assertEqual([p.requires_grad for p in model.parameters()],flags)
        self.assertTrue(old.training);self.assertEqual(next(old.parameters()).dtype,torch.float16)
        with torch.no_grad():
            model.base_student.pixel_residual.bias.zero_();model.base_student.rgb.bias.add_(.001)
        with self.assertRaisesRegex(AssertionError,'base_equal=False'):
            verify_zero_pixel_residual(model,old,source)


if __name__=='__main__':unittest.main()
