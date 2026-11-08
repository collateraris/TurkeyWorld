"""The visible world must not depend on bookkeeping or hidden-history noise."""
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class RenderInvarianceTests(unittest.TestCase):
    def test_nonvisual_state_and_memory_do_not_change_rgb(self):
        from turkeyworld.engine import Game
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        torch.set_num_threads(2)
        game = Game(seed=3)
        original = torch.tensor(game.state_vector())[None]
        altered = original.clone()
        altered[:, 4] = .12  # Alive at both HP values: no death tint change.
        altered[:, 5] = 9 / 99
        altered[:, 6] = 1.
        altered[:, -2:] = torch.tensor([[.75, 20.]])
        grid = torch.tensor(game.map, dtype=torch.long)[None]
        before = altered.clone()
        for architecture in ("legacy_cnn", "camera_transformer"):
            with self.subTest(architecture=architecture):
                model = NeuralWorldModel(WorldModelConfig(
                    architecture=architecture, transition_mode="event_v1", image_size=128)).eval()
                with torch.inference_mode():
                    map_features = model.encode_map(grid)
                    first = model.render(original, torch.randn(1, 128), map_features)
                    second = model.render(altered, torch.randn(1, 128)*30, map_features)
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                torch.testing.assert_close(altered, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
