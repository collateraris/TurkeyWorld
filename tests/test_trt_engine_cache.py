from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.trt_engine import _cache_key


class TensorRTCacheTests(TestCase):
    def test_weights_shape_and_runtime_identify_an_engine(self):
        model = nn.Linear(4, 2).eval()
        sample = torch.zeros(1, 4)
        with patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(name='test', major=8, minor=9)):
            def key(x=sample, version='10.13'):
                return _cache_key('linear', model, (x,), ('output',), version)
            before = key()
            self.assertEqual(before, key(sample + 1))
            self.assertNotEqual(before, key(sample.expand(2, 4)))
            self.assertNotEqual(before, key(version='10.14'))
            with torch.no_grad():
                model.weight.add_(1)
            self.assertNotEqual(before, key())
