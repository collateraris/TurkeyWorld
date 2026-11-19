import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from turkeyworld.trt_dynamics import GameplayPredictor, enable_dynamics_trt
from turkeyworld.vjepa2_ac import CompactVJEPA2AC


class TensorRTDynamicsTests(unittest.TestCase):
    def test_packed_prediction_matches_trained_blocks(self):
        # Both token layouts and an odd head width exercise unrotated tail channels.
        with torch.random.fork_rng(), torch.inference_mode():
            torch.manual_seed(17)
            for pooled, width in ((True, 36), (False, 32)):
                core = CompactVJEPA2AC(image_size=8, patch_size=4, width=width, heads=4,
                                       encoder_depth=1, predictor_depth=2, context_frames=4,
                                       pooled_memory=pooled).eval()
                x = torch.randn(1, core.memory_tokens, width)
                ref = core.predict_packed(x)
                torch.testing.assert_close(GameplayPredictor(core)(x), ref, atol=1e-6, rtol=1e-5)
                def factory(name, module, inputs, outputs):
                    self.assertEqual(name, 'vjepa_predictor')
                    self.assertEqual(inputs[0].shape, x.shape)
                    self.assertEqual(outputs, ('latent_prediction',))
                    return module

                runtime = SimpleNamespace(world_architecture='vjepa2_ac',
                                          model=SimpleNamespace(core=core), initial_memory=x)
                self.assertEqual(enable_dynamics_trt(runtime, factory), ['vjepa_predictor'])
                torch.testing.assert_close(core.predict_packed(x), ref, atol=1e-6, rtol=1e-5)

    @unittest.skipUnless(importlib.util.find_spec('onnx'), 'requires ONNX')
    def test_export_preserves_fp16_through_attention_and_layer_norm(self):
        import onnx

        core = CompactVJEPA2AC(image_size=8, patch_size=4, width=32, heads=4, encoder_depth=1,
                               predictor_depth=2, context_frames=4, pooled_memory=True).half().eval()
        memory = torch.zeros(1, core.memory_tokens, core.width, dtype=torch.float16)
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / 'predictor.onnx'
            torch.onnx.export(GameplayPredictor(core), (memory,), str(path), dynamo=False,
                              opset_version=18, input_names=['input_0'], output_names=['latent_prediction'])
            model = onnx.load(path)
            onnx.checker.check_model(model, full_check=True)
            self.assertEqual(model.graph.output[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT16)

    def test_non_vjepa_dynamics_are_untouched(self):
        self.assertEqual(enable_dynamics_trt(SimpleNamespace(world_architecture='dreamer_rssm'), None), [])
