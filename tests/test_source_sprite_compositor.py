import math
import unittest
from pathlib import Path

import numpy as np
import torch

from turkeyworld.atlas_enemy_sr import ASSET_SHA256, _load_native_atlas, _pixel_hash
from turkeyworld.engine import Game
from turkeyworld.geometry_targets import _wall_depths
from turkeyworld.source_sprite_compositor import (
    SourceSpriteCompositor,
    composite_source_sprite,
    wall_column_depth,
)


class SevenPoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        path = Path(__file__).resolve().parents[1] / "assets" / "turkey_atlas.png"
        frames, sizes = _load_native_atlas(path, ASSET_SHA256)
        cls.layer = SourceSpriteCompositor(
            dict(
                atlas_buffer_shape=list(frames.shape),
                atlas_frame_sizes=sizes.tolist(),
                atlas_pixels_sha256=_pixel_hash(frames, sizes),
            )
        )
        cls.layer.load_state_dict(dict(atlas_frames_rgba=frames, atlas_native_sizes=sizes))

    def test_dda_matches_reference_on_player_map(self):
        game = Game(seed=911)
        s = game.state_vector()
        poses = np.repeat(s[None], 5, axis=0)
        angle = np.array([0, math.pi / 2, 0.3, -0.48, 2.3])
        poses[:, 2] = np.sin(angle)
        poses[:, 3] = np.cos(angle)
        grids = np.repeat(game.map[None], len(poses), axis=0)
        actual = wall_column_depth(torch.from_numpy(poses), torch.from_numpy(grids))
        screen = ((np.arange(128) + 0.5) / 64 - 1) * math.tan(math.radians(34))
        expected = []
        for row in poses:
            x, y = row[:2].astype(np.float64) * 16
            a = math.atan2(row[2], row[3])
            cs, sn = math.cos(a), math.sin(a)
            expected.append(_wall_depths(game.map, x, y, cs - sn * screen, sn + cs * screen, 100))
        np.testing.assert_allclose(actual.numpy(), np.stack(expected), atol=1e-8)

    def test_seven_poses_and_wall_occlusion_match_integer_source_render(self):
        game = Game(seed=0)
        # Explicitly remove training-time weapon art in this test to isolate
        # foreground alpha and the Source pixel shader.
        game._guns = [np.zeros_like(frame) for frame in game._guns]
        game.map[1:-1, 1:-1] = 0
        game.player.update(x=7.2, y=7.8, angle=0.12, hp=100.0)
        game.enemies[0].update(x=9.12, y=7.91, hp=70.0)
        for phase in range(7):
            with self.subTest(phase=phase):
                game.enemies[0].update(
                    alive=phase < 5,
                    vx=0.27 if phase in (1, 2) else 0.0,
                    vy=0.0,
                    attack_cooldown=1.12 if phase == 3 else 0.0,
                    hurt_timer=0.1 if phase == 4 else 0.0,
                    death_timer=0.45 if phase == 6 else 0.0,
                    animation_phase=0.75 if phase == 2 else 0.25,
                )
                # Make the Source comparison use the exact pose given to the compositor.
                state = game.state_vector()
                game.set_state_vector(state)
                game.enemies_enabled = False
                background = game.render(128, overlays=False)
                game.enemies_enabled = True
                target = game.render(128, overlays=False)
                image = torch.from_numpy(background).permute(2, 0, 1)[None].float() / 255
                s = torch.from_numpy(state)[None]
                m = torch.from_numpy(game.map)[None]
                layer = self.layer(image, s, m)
                self.assertEqual(int(layer["sprite_phase"]), phase)
                recon = composite_source_sprite(image, layer["sprite_rgb"], layer["sprite_alpha"])
                pixels = recon[0].permute(1, 2, 0).mul(255).round().byte().numpy()
                np.testing.assert_array_equal(pixels, target)
        game.map[1:-1, 8] = 1
        self.assertFalse(self.layer(image, s, m)["sprite_alpha"].any())

    def test_context_and_navigation(self):
        canvas = torch.rand(1, 3, 128, 128)
        off = self.layer(canvas, None, None, enabled=False)
        self.assertFalse(off["sprite_alpha"].any())
        self.assertEqual(int(off["sprite_phase"]), -1)
        self.assertTrue(
            torch.equal(composite_source_sprite(canvas, off["sprite_rgb"], off["sprite_alpha"]), canvas)
        )
        with self.assertRaisesRegex(ValueError, "state.*map"):
            self.layer(canvas, None, None)
