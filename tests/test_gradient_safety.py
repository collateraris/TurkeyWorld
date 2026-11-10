"""Finite elements, FP32 norm overflow and fail-before-update semantics."""
import copy
import unittest

import torch
from torch import nn

from turkeyworld.gradient_safety import finite_gradient_norm, clip_finite_grad_norm_
from turkeyworld.jepa_training_utils import EMATargetEncoder


def assert_tree_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class BadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.square().sum()

    @staticmethod
    def backward(ctx, gradient):
        return gradient.new_full((2,), float('nan'))


class GradientSafetyTests(unittest.TestCase):
    def test_matches_standard_clipping_and_consumes_generator_once(self):
        actual = [nn.Parameter(torch.zeros(3)), nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(1))]
        expected = [nn.Parameter(value.detach().clone()) for value in actual]
        for left, right, gradient in zip(actual, expected, (torch.tensor([3., -4., 2.]), torch.tensor([.5, -1.]))):
            left.grad = gradient.clone()
            right.grad = gradient.clone()
        original = [value.grad.clone() if value.grad is not None else None for value in actual]
        norm = finite_gradient_norm((('weight'+str(i), value) for i, value in enumerate(actual)))
        self.assertEqual(norm.dtype, torch.float64)
        for before, value in zip(original, actual):
            self.assertTrue(before is None and value.grad is None or torch.equal(before, value.grad))
        visits = []
        def named():
            for index, value in enumerate(actual):
                visits.append(index)
                yield str(index), value
        measured = clip_finite_grad_norm_(named(), .7)
        standard = torch.nn.utils.clip_grad_norm_(expected, .7, error_if_nonfinite=True)
        self.assertEqual(visits, [0, 1, 2])
        self.assertAlmostEqual(float(measured), float(standard), places=6)
        for left, right in zip(actual[:2], expected[:2]):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-6, atol=1e-7)

    def test_finite_huge_fp32_gradients_do_not_overflow_fp64_reduction(self):
        parameter = nn.Parameter(torch.zeros(4))
        parameter.grad = torch.tensor([1e25, -1e25, 1e25, -1e25])
        with self.assertRaises(RuntimeError):
            torch.nn.utils.clip_grad_norm_([parameter], 1., error_if_nonfinite=True)
        before = parameter.grad.clone()
        norm = finite_gradient_norm([('large', parameter)])
        self.assertTrue(torch.isfinite(norm))
        self.assertAlmostEqual(float(norm)/2e25, 1., places=6)
        self.assertTrue(torch.equal(before, parameter.grad))
        result = clip_finite_grad_norm_([('large', parameter)])
        self.assertEqual(result.dtype, torch.float64)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(parameter.grad.norm()), .999999)
        self.assertLessEqual(float(parameter.grad.double().norm()), 1.+1e-6)

    def test_bad_backward_aborts_before_adam_ema_or_counter_changes(self):
        model = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        ema = EMATargetEncoder(model)
        model(torch.ones(1, 2)).square().mean().backward()
        optimizer.step()
        ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        weights, moments, target = (copy.deepcopy(value) for value in
            (model.state_dict(), optimizer.state_dict(), ema.state_dict()))
        clock = 1
        loss = BadBackward.apply(model.weight.flatten())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        with self.assertRaisesRegex(FloatingPointError, 'chunk16.*weight'):
            clip_finite_grad_norm_(model.named_parameters(), context='chunk16')
            optimizer.step()
            ema.update(model)
            clock += 1
        self.assertEqual(clock, 1)
        assert_tree_equal(self, weights, model.state_dict())
        assert_tree_equal(self, moments, optimizer.state_dict())
        assert_tree_equal(self, target, ema.state_dict())
        self.assertTrue(torch.isnan(model.weight.grad).all())

    def test_reports_all_bad_names_without_modifying_any_gradient(self):
        parameters = [nn.Parameter(torch.zeros(1)) for _ in range(3)]
        for parameter, value in zip(parameters, (3., float('inf'), float('nan'))):
            parameter.grad = torch.tensor([value])
        for function in (finite_gradient_norm, clip_finite_grad_norm_):
            with self.assertRaisesRegex(FloatingPointError, 'target1537.*bad_inf, bad_nan'):
                function(zip(('good', 'bad_inf', 'bad_nan'), parameters), context='target1537')
            self.assertEqual(float(parameters[0].grad), 3.)
            self.assertTrue(torch.isposinf(parameters[1].grad).all())
            self.assertTrue(torch.isnan(parameters[2].grad).all())

    def test_empty_missing_and_zero_gradients(self):
        for entries in ([], [('unused', nn.Parameter(torch.zeros(2)))],
                        [('empty', nn.Parameter(torch.zeros(0)))]):
            for function in (finite_gradient_norm, clip_finite_grad_norm_):
                norm = function(iter(entries))
                self.assertEqual(norm.dtype, torch.float64)
                self.assertEqual(float(norm), 0.)
        parameter = nn.Parameter(torch.ones(2))
        parameter.grad = torch.zeros_like(parameter)
        self.assertEqual(float(clip_finite_grad_norm_([('zero', parameter)])), 0.)
        self.assertTrue(torch.equal(parameter.grad, torch.zeros(2)))

    def test_sparse_rejected_and_bad_limit_does_not_mutate_gradients(self):
        parameter = nn.Parameter(torch.zeros(3))
        parameter.grad = torch.sparse_coo_tensor(torch.tensor([[1]]), torch.tensor([2.]), (3,))
        with self.assertRaisesRegex(ValueError, 'Dense gradient.*embedding'):
            finite_gradient_norm([('embedding', parameter)])
        parameter.grad = torch.ones(3)
        for limit in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                clip_finite_grad_norm_([('dense', parameter)], limit)
            self.assertTrue(torch.equal(parameter.grad, torch.ones(3)))


if __name__ == '__main__':
    unittest.main()
