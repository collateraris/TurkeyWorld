import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from turkeyworld.collect_combat_training import collect_job
from turkeyworld.combat_visual_pairs import replay, weapon_labels
from turkeyworld.engine import Game
from turkeyworld.staged_data import _observe


class CombatVisualPairsTest(unittest.TestCase):
    def test_exact_episode_replay_and_snapshot_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.npz"
            collect_job(
                dict(
                    stage="full_game",
                    split="train",
                    index=0,
                    seed=43,
                    frames=10,
                    case="duel",
                    relative_path=path.name,
                    path=path,
                    protocol="test",
                )
            )
            ticks = {tick: "combat" for tick in (0, 1, 5, 9)}
            record, rows = replay((str(path), ticks))
            self.assertEqual(record["archive_native_rgb32_mismatches"], 0)
            self.assertEqual([row["frame"] for row in rows], list(ticks))
            game = Game()
            for row in rows:
                game.set_state(json.loads(row["state_json"]))
                np.testing.assert_array_equal(game.state_vector(), row["states"])
                for size in (32, 128):
                    rgb, enemy, _ = _observe(game, size)
                    np.testing.assert_array_equal(rgb, row[f"rgb{size}"])
                    np.testing.assert_array_equal(enemy, row[f"enemy{size}"])
            self.assertGreater(sum(row["flash32"].any() for row in rows), 0)

    def test_gun_masks_match_engine_blit_at_each_resolution(self):
        game = Game()
        for size in (32, 128):
            width = int(size * 0.67)
            for i, (weapon, flash) in enumerate(weapon_labels(game, size)):
                atlas = np.full_like(game._guns[i], 255)
                atlas[..., 3] = game._guns[i][..., 3]
                target = np.zeros((size, size, 3), dtype=np.uint8)
                game._blit_rgba(
                    target, atlas, size // 2 - width // 2, size - width, width, width
                )
                np.testing.assert_array_equal(weapon, target[..., 0] > 0)
                self.assertFalse((flash & ~weapon).any())
                self.assertEqual(bool(flash.any()), i == 1)
