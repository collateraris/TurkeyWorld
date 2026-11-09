"""Partial sprite overlap remains available to neural appearance conditioning."""
import io
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


# Exact camera and NPC XY from preview340's first three predicted states.
# These compact coordinates reproduce the bug without checkpoints or datasets.
PREVIEW340 = [
    ([.7859240770339966, .5858250260353088, .4283638000488281, .9036063551902771],
     [[.8294886946678162, .5660632848739624], [.7578681111335754, .5468447804450989],
      [.8326023817062378, .5776045918464661], [.8023828268051147, .6300461292266846],
      [.46812519431114197, .4751967787742615], [.5931701064109802, .16257214546203613],
      [.40561020374298096, .8368108868598938], [.3431614637374878, .8364250659942627]]),
    ([.7859240770339966, .5858250260353088, .49341756105422974, .8697925806045532],
     [[.8292940258979797, .5658053159713745], [.757307767868042, .5469202995300293],
      [.832338273525238, .5773506164550781], [.8019165992736816, .6299218535423279],
      [.4675061106681824, .47521430253982544], [.5925934314727783, .16231510043144226],
      [.40497851371765137, .83661288022995], [.3425791561603546, .8359838724136353]]),
    ([.7859240770339966, .5858250260353088, .555819034576416, .8313032984733582],
     [[.8290995359420776, .5655472874641418], [.7567473649978638, .5469958782196045],
      [.8320742249488831, .5770965218544006], [.8014503121376038, .6297975778579712],
      [.4668872356414795, .4752318263053894], [.5920165181159973, .16205808520317078],
      [.404346764087677, .8364150524139404], [.341996967792511, .8355426788330078]]),
]


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class SpriteRenderVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def preview_states():
        states = torch.zeros(3, 90)
        states[:, 4:6] = 1.
        for index, (camera, xy) in enumerate(PREVIEW340):
            states[index, :4] = torch.tensor(camera)
            entities = states[index, 8:88].reshape(8, 10)
            entities[:, :2] = torch.tensor(xy)
            entities[:, 4:6] = torch.tensor([.6, 1.])
        return states

    def test_exact_preview340_center_hidden_sprites_receive_half_or_more_render_weight(self):
        from turkeyworld.camera import object_camera_features, project_sprite_bounds
        from turkeyworld.transformer_model import CameraGeometry
        states = self.preview_states()
        old = object_camera_features(states)
        self.assertEqual(old["features"].shape, (3, 8, 18))
        self.assertFalse(bool(old["frustum"].any()))
        projected = project_sprite_bounds(states)
        self.assertTrue(bool(projected["sprite_frustum"].any(-1).all()))
        geometry = CameraGeometry(32)
        with torch.no_grad():
            geometry.visibility_head[-1].weight.zero_()
            geometry.visibility_head[-1].bias.fill_(-100.)
        with patch("turkeyworld.engine.Game.render", side_effect=AssertionError("Runtime renderer oracle")), \
             patch("turkeyworld.geometry_targets.build_geometry_targets", side_effect=AssertionError("Visibility oracle")):
            result = geometry(states, torch.randn(3, 32, 16, 16))
        self.assertFalse(bool(result["frustum"].any()))
        torch.testing.assert_close(result["visibility_logits"], torch.full((3, 8), -20.), rtol=0, atol=0)
        torch.testing.assert_close(result["visibility"], torch.zeros(3, 8), rtol=0, atol=0)
        self.assertTrue(bool((result["render_gate"][result["sprite_frustum"]] >= .5).all()))
        torch.testing.assert_close(result["render_gate"][~result["sprite_frustum"]],
                                   torch.full_like(result["render_gate"][~result["sprite_frustum"]], .05))

    def test_clipped_edge_near_plane_behind_and_corpse_rectangles_are_finite(self):
        from turkeyworld.camera import object_camera_features, project_sprite_bounds
        state = torch.zeros(1, 90)
        state[:, :6] = torch.tensor([2.5/16, 2.5/16, 0., 1., 1., 1.])
        entities = state[:, 8:88].reshape(1, 8, 10)
        entities[0, :, :2] = torch.tensor([[2.63, 2.5], [2.5, 2.5], [2., 2.5], [3.5, 12.5],
                                          [3.5, 3.3], [3.5, 1.7], [5.5, 2.5], [5.5, 2.5]])/16
        entities[:, :, 4:6] = torch.tensor([.6, 1.])
        entities[:, 7, 4:6] = 0.  # Same square projection for the corpse atlas.
        result = project_sprite_bounds(state)
        self.assertEqual(result["sprite_frustum"].tolist(), [[True, False, False, False, True, True, True, True]])
        self.assertTrue(bool(torch.isfinite(result["unclipped_boxes"]).all()))
        self.assertTrue(bool(((result["boxes"] >= 0) & (result["boxes"] <= 1)).all()))
        self.assertFalse(bool(object_camera_features(state)["frustum"][0, 4:6].any()))
        self.assertEqual(float(result["boxes"][0, 4, 2]), 1.)
        self.assertEqual(float(result["boxes"][0, 5, 0]), 0.)
        torch.testing.assert_close(result["boxes"][0, 6], result["boxes"][0, 7])
        torch.testing.assert_close(result["boxes"][0, 1:4], torch.zeros(3, 4))
        self.assertGreater(float(result["area_fraction"][0, 0]), 0.)

    def test_projected_full_rectangles_match_actual_source_blit_for_live_and_corpse(self):
        from turkeyworld.camera import project_sprite_bounds
        from turkeyworld.engine import Game
        for alive in (True, False):
            with self.subTest(alive=alive):
                game = Game(endless=True)
                enemy = game.enemies[0]
                enemy.update(x=3.3, y=3.4, alive=alive, hp=60. if alive else 0., death_timer=.4)
                state = torch.from_numpy(game.state_vector())[None]
                projected = project_sprite_bounds(state)
                game.enemies = [enemy]  # Isolate this source sprite and its actual integer blit rectangle.
                with patch.object(Game, "_blit_rgba", wraps=Game._blit_rgba) as blit:
                    game.render(size=128, overlays=False)
                x, y, width, height = blit.call_args_list[0].args[2:6]
                expected = torch.tensor([x, y, x+width, y+height]).float()
                torch.testing.assert_close(projected["unclipped_boxes"][0, 0]*128, expected, rtol=0, atol=1.)
                self.assertTrue(bool(projected["sprite_frustum"][0, 0]))

    def test_both_appearance_and_attention_use_extent_gate_and_object_gradients_survive(self):
        from turkeyworld.models import WorldModelConfig
        from turkeyworld.transformer_model import CameraTransformerWorldModel
        torch.manual_seed(842)
        model = CameraTransformerWorldModel(WorldModelConfig(kind="student", architecture="vjepa2_ac",
            image_size=128, transition_mode="event_v1"), render_only=True)
        state = self.preview_states()[:1]
        map_features = torch.randn(1, 128, 16, 16, requires_grad=True)
        with torch.no_grad():
            model.geometry_reader.visibility_head[-1].weight.zero_()
            model.geometry_reader.visibility_head[-1].bias.fill_(-100.)
        geometry = model.geometry(state, map_features)
        first_block = model.render_blocks[0]
        with patch.object(model, "geometry", return_value=geometry), \
             patch.object(first_block, "forward", wraps=first_block.forward) as first:
            result = model.render_with_aux(state, torch.zeros(1, 128), map_features)
        context = first.call_args.args[1]
        gate = geometry["render_gate"]
        torch.testing.assert_close(first.call_args.kwargs["key_weights"][:, 32:40], gate)
        torch.testing.assert_close(context[:, 32:40], geometry["object_identity_tokens"]
                                   + gate[..., None]*geometry["object_appearance_tokens"])
        result["rgb"].square().mean().backward()
        gradient = model.geometry_reader.object_encoder[0].weight.grad
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_parameter_shapes_and_strict_checkpoint_load_are_unchanged(self):
        from turkeyworld.transformer_model import CameraGeometry
        model = CameraGeometry(32)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 7650)
        snapshot = io.BytesIO()
        torch.save(model.state_dict(), snapshot)
        snapshot.seek(0)
        restored = CameraGeometry(32, image_size=256)
        restored.load_state_dict(torch.load(snapshot, weights_only=True), strict=True)
        self.assertFalse(any("sprite" in name or "render_gate" in name for name in model.state_dict()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
