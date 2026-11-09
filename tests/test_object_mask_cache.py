"""Offline masks match source pixels; cached supervision never runs the engine."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.engine import BEHAVIOR_VERSION, Game
from turkeyworld.object_mask_cache import (
    MASK_KIND, _generate_masks, main, prepare_object_masks, read_object_masks,
)
from turkeyworld.prepare_training_data import _generate_job
from turkeyworld.training_subset import create_training_subset


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def place_partial_edge_sprite(game, rng, policy):
    """A valid scene whose only forward NPC center lies outside the FOV."""
    game.player.update(x=2.5, y=2.5, angle=0., hp=100., cooldown=1., shot_flash=0.)
    game.enemies[0].update(x=3.5, y=3.3, alive=True, hp=60., attack_cooldown=.8)
    for enemy, y in zip(game.enemies[1:], (1.3, 3.7, 4.5, 5.5, 6.5, 7.5, 8.5)):
        enemy.update(x=1.5, y=y, alive=True, hp=60., attack_cooldown=.8)
    # Also ensure that the source state can be restored through its public API.
    game.set_state(game.get_state())


def make_episode(path, *, split="train", seed=73, size=64):
    job = {"path": str(path), "relative_path": f"rgb/{split}/{path.name}", "job_id": f"rgb/{split}/{seed}", "seed": seed,
           "split": split, "kind": "rgb", "policy": "attack", "frames": 3,
           "size": size, "hr_size": 0, "hz": 30, "endless": True,
           "long_geometry": False, "behavior_version": BEHAVIOR_VERSION}
    with patch("turkeyworld.prepare_training_data._random_initial_state", place_partial_edge_sprite):
        _generate_job(job)
    return {"source": str(path), "output": str(path.parent / "masks" / path.name), "split": split}


class ObjectMaskCacheTests(unittest.TestCase):
    def test_exact_ablation_captures_center_hidden_edge_sprite_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train" / "episode.npz"
            job = make_episode(source)
            before = (file_digest(source), source.stat().st_size, source.stat().st_mtime_ns)
            metadata = _generate_masks(job)
            masks, loaded = read_object_masks(job["output"], source, "train")
            self.assertEqual(metadata, loaded)
            self.assertEqual(masks.shape, (4, 64, 64))
            self.assertEqual(masks.dtype, np.bool_)
            self.assertEqual(loaded["mask_kind"], MASK_KIND)
            self.assertEqual(loaded["verified_source_frames"], 4)
            self.assertEqual(loaded["foreground_pixels"], int(masks.sum()))
            self.assertGreater(loaded["foreground_pixels"], 0)
            with np.load(source, allow_pickle=False) as saved:
                frames, actions = saved["frames"].copy(), saved["actions"].copy()
                game = Game(seed=int(saved["seed"]), endless=bool(saved["endless"]))
                game.set_state(json.loads(str(saved["initial_state"])))
                dt = float(saved["dt"])
                self.assertEqual(float(saved["visibility"][0, 0]), 0.)
            enemy = game.enemies[0]
            self.assertGreater(abs((enemy["y"]-game.player["y"])/(enemy["x"]-game.player["x"])),
                               math.tan(math.radians(34.)))
            self.assertTrue(bool(masks[0, :, -8:].any()))
            expected = []
            for index, source_frame in enumerate(frames):
                full = game.render(size=64, overlays=False)
                np.testing.assert_array_equal(full, source_frame)
                enemies = game.enemies
                try:
                    game.enemies = []
                    background = game.render(size=64, overlays=False)
                finally:
                    game.enemies = enemies
                # Weapon, walls, sky and ground remain in both renders: only
                # actual visible NPC pixels, including partial sprites, differ.
                expected.append(np.any(full != background, axis=-1))
                if index < len(actions):
                    game.step(actions[index], dt=dt)
            np.testing.assert_array_equal(masks, np.stack(expected))
            self.assertEqual(loaded["source_frames_sha256"], hashlib.sha256(frames.tobytes()).hexdigest())
            self.assertEqual(before, (file_digest(source), source.stat().st_size, source.stat().st_mtime_ns))

    def test_reader_and_existing_cache_validation_do_not_call_engine_or_decode_source_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "episode.npz"
            job = make_episode(source)
            original = _generate_masks(job)
            cache = Path(job["output"])
            before = (file_digest(cache), cache.stat().st_mtime_ns, file_digest(source))
            real_load = np.load
            with patch.object(Game, "__init__", side_effect=AssertionError("Runtime source initialization")), \
                 patch.object(Game, "render", side_effect=AssertionError("Runtime source renderer")), \
                 patch.object(Game, "step", side_effect=AssertionError("Runtime source transition")), \
                 patch("turkeyworld.object_mask_cache.np.load", wraps=real_load) as reads:
                masks, metadata = read_object_masks(cache, source, "train")
                self.assertEqual(_generate_masks(job), original)
            self.assertEqual(metadata, original)
            self.assertGreater(int(masks.sum()), 0)
            self.assertEqual([Path(call.args[0]) for call in reads.call_args_list], [cache, cache])
            self.assertEqual(before, (file_digest(cache), cache.stat().st_mtime_ns, file_digest(source)))

    def test_wrong_split_and_changed_source_identity_are_rejected_without_overwriting_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "episode.npz"
            job = make_episode(source)
            _generate_masks(job)
            cache = Path(job["output"])
            before = file_digest(cache)
            with self.assertRaisesRegex(ValueError, "split mismatch"):
                read_object_masks(cache, source, "val")
            with self.assertRaisesRegex(ValueError, "split mismatch"):
                _generate_masks({**job, "split": "val"})
            with self.assertRaisesRegex(ValueError, "source behavior/split mismatch"):
                _generate_masks({**job, "split": "val", "output": str(Path(directory) / "new-val.npz")})
            self.assertEqual(file_digest(cache), before)
            # Simulate an externally changed source file, only in this fixture.
            with source.open("ab") as stream:
                stream.write(b"changed-source-identity")
            with self.assertRaisesRegex(ValueError, "changed source episode"):
                read_object_masks(cache, source, "train")
            with self.assertRaisesRegex(ValueError, "changed source episode"):
                _generate_masks(job)
            self.assertEqual(file_digest(cache), before)

    def test_source_replay_corruption_fails_before_cache_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "episode.npz"
            job = make_episode(source)
            with np.load(source, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            arrays["frames"][1, 0, 0, 0] ^= np.uint8(1)
            np.savez_compressed(source, **arrays)
            with self.assertRaisesRegex(ValueError, "replay differs.*frame 1"):
                _generate_masks(job)
            self.assertFalse(Path(job["output"]).exists())

    def test_prepare_selects_manifest_train_and_separate_val_and_reuses_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            train = data / "rgb" / "train" / "episode.npz"
            val = data / "rgb" / "val" / "episode.npz"
            make_episode(train)
            make_episode(val, split="val", seed=1073)
            subset = root / "subset.json"
            create_training_subset(train.parent, subset, transitions=3, seed=173)
            before = {path: file_digest(path) for path in (train, val, subset)}
            output = root / "object_masks"
            # Real worker functions run in threads here: no Windows spawn or
            # additional Python processes are needed for this tiny integration.
            with patch("turkeyworld.object_mask_cache.ProcessPoolExecutor", ThreadPoolExecutor):
                report = prepare_object_masks(data, output, subset, workers=1)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["completed_episodes"], 2)
            self.assertEqual(report["verified_source_frames"], 8)
            self.assertFalse(report["runtime_engine_calls"])
            self.assertEqual({record["split"] for record in report["records"]}, {"train", "val"})
            sidecars = [output / split / "episode.npz" for split in ("train", "val")]
            sidecar_identity = [(file_digest(path), path.stat().st_mtime_ns) for path in sidecars]
            with patch("turkeyworld.object_mask_cache.ProcessPoolExecutor", ThreadPoolExecutor), \
                 patch.object(Game, "render", side_effect=AssertionError("Existing cache rendered")), \
                 patch.object(Game, "step", side_effect=AssertionError("Existing cache stepped")):
                resumed = prepare_object_masks(data, output, subset, workers=1)
            self.assertEqual(resumed["records"], report["records"])
            self.assertEqual(sidecar_identity, [(file_digest(path), path.stat().st_mtime_ns) for path in sidecars])
            self.assertEqual(before, {path: file_digest(path) for path in (train, val, subset)})

    def test_cli_passes_named_manifest_argument_to_preparation(self):
        fake = {"status": "complete", "completed_episodes": 2, "verified_source_frames": 8}
        output = io.StringIO()
        with patch("turkeyworld.object_mask_cache.prepare_object_masks", autospec=True, return_value=fake) as prepare, \
             contextlib.redirect_stdout(output):
            main(["--data-root", "tiny-data", "--output", "tiny-masks",
                  "--training-subset-manifest", "tiny-subset.json", "--workers", "1"])
        self.assertEqual(json.loads(output.getvalue()), fake)
        prepare.assert_called_once_with(data_root=Path("tiny-data"), output=Path("tiny-masks"),
                                        training_subset_manifest=Path("tiny-subset.json"), workers=1)


if __name__ == "__main__":
    unittest.main()
