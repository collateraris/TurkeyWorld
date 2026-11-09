"""Explicit codec acceptance changes the quality policy, never integrity or truth."""
import copy
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class VQUserAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def codec(self, path):
        from turkeyworld.vqgan import VQGAN, VQGANConfig
        from turkeyworld.vq_world import codec_fingerprint
        # Production image/code dimensions, small convolution widths for CPU.
        model = VQGAN(VQGANConfig(image_size=128, base_channels=8, latent_channels=16,
                                embedding_dim=16, n_embed=1488, data_init=False))
        checks = {name: True for name in ("all_frames_finite", "all_frames_in_range",
                  "quantization_finite", "valid_code_indices", "codebook_broad_usage")}
        payload = {"stage": "frozen_vqgan_codec", "codec_config": model.get_config(),
            "model": model.state_dict(), "codec_fingerprint": codec_fingerprint(model),
            "training_steps": 11586,
            "quality_gate": {"passed": False, "frozen": True, "consecutive_passes": 0,
                             "required_passes": 3, "checks": checks},
            "quality_acceptance": {"mode": "user_requested", "accepted": True,
                "reason": "User requested stopping codec training and continuing the teacher",
                "accepted_at": "2026-11-09T12:00:00+00:00", "quality_gate_passed": False},
            "latent_normalization": {"mean": [.1] * 16, "std": [.5] * 16}}
        torch.save(payload, path)
        return model, payload

    def test_default_policy_rejects_user_accepted_but_unqualified_codec(self):
        from turkeyworld.vq_world import validate_frozen_codec
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.pt"
            self.codec(path)
            with self.assertRaisesRegex(ValueError, "three consecutive"):
                validate_frozen_codec(path)

    def test_explicit_acceptance_allows_codec_and_preserves_failed_quality_truth(self):
        from turkeyworld.vq_world import validate_frozen_codec
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.pt"
            _, payload = self.codec(path)
            before = (path.stat().st_mtime_ns, path.read_bytes())
            info = validate_frozen_codec(path, accept_user_requested=True)
            self.assertFalse(info["qualified"])
            self.assertTrue(info["accepted_by_user"])
            self.assertFalse(info["quality_gate"]["passed"])
            self.assertEqual(info["quality_gate"]["consecutive_passes"], 0)
            self.assertEqual(info["quality_acceptance"], payload["quality_acceptance"])
            self.assertEqual(info["training_steps"], 11586)
            self.assertEqual(info["codec_fingerprint"], payload["codec_fingerprint"])
            self.assertEqual(before, (path.stat().st_mtime_ns, path.read_bytes()))

    def test_acceptance_never_bypasses_weight_hash_initialized_codebook_or_normalization(self):
        from turkeyworld.vq_world import validate_frozen_codec
        for failure in ("hash", "initialization", "normalization_zero", "normalization_nan", "normalization_shape"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.pt"
                _, payload = self.codec(path)
                if failure == "hash":
                    payload["model"]["decoder.conv_out.weight"].add_(1.)
                elif failure == "initialization":
                    payload["model"]["quantize.initialized"].fill_(False)
                elif failure == "normalization_zero":
                    payload["latent_normalization"]["std"][0] = 0.
                elif failure == "normalization_nan":
                    payload["latent_normalization"]["mean"][0] = float("nan")
                else:
                    payload["latent_normalization"]["std"].pop()
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    validate_frozen_codec(path, accept_user_requested=True)

    def test_acceptance_requires_all_integrity_checks_and_consistent_explicit_record(self):
        from turkeyworld.vq_world import validate_frozen_codec
        failures = ("all_frames_finite", "all_frames_in_range", "quantization_finite", "valid_code_indices",
                    "codebook_broad_usage", "missing_check", "acceptance_truth", "reason", "accepted_at", "mode")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.pt"
            _, original = self.codec(path)
            for failure in failures:
                with self.subTest(failure=failure):
                    payload = copy.deepcopy(original)
                    if failure in payload["quality_gate"]["checks"]:
                        payload["quality_gate"]["checks"][failure] = False
                    elif failure == "missing_check":
                        payload["quality_gate"]["checks"].pop("valid_code_indices")
                    elif failure == "acceptance_truth":
                        payload["quality_acceptance"]["quality_gate_passed"] = True
                    else:
                        payload["quality_acceptance"][failure] = "" if failure != "mode" else "automatic"
                    torch.save(payload, path)
                    with self.assertRaises(ValueError):
                        validate_frozen_codec(path, accept_user_requested=True)

    def test_pipeline_forwards_explicit_optin_to_validation_migration_child_and_honest_status(self):
        from turkeyworld.vq_world import validate_frozen_codec
        from turkeyworld.vqgan_pipeline import continue_world_after_codec
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control, source, world, data = [root / name for name in ("codec", "source", "world", "data")]
            control.mkdir()
            (source / "checkpoints").mkdir(parents=True)
            data.mkdir()
            codec = control / "accepted.pt"
            self.codec(codec)
            info = validate_frozen_codec(codec, accept_user_requested=True)
            (source / "checkpoints/teacher500_world.pt").write_bytes(b"mock source; never loaded")
            args = SimpleNamespace(run_dir=control, world_source_run=source, world_run_dir=world,
                                   data_dir=data, device="cpu", accept_current_codec=True)

            def migrate(source_path, codec_path, initial, resume_output, **options):
                initial.parent.mkdir(parents=True)
                initial.write_bytes(b"mock initial")
                resume_output.write_bytes(b"mock resume")
                return {"codec_fingerprint": info["codec_fingerprint"], "source_checkpoint": str(source_path),
                    "initial_checkpoint": str(initial), "codec_quality_verified": False,
                    "codec_quality_accepted_by_user": True, "codec_quality_acceptance": info["quality_acceptance"]}

            with patch("turkeyworld.vqgan_pipeline.validate_frozen_codec", return_value=info) as validate, \
                 patch("turkeyworld.vqgan_pipeline.migrate_teacher_to_vq", side_effect=migrate) as migration, \
                 patch("turkeyworld.vqgan_pipeline.link_data"), \
                 patch("turkeyworld.vqgan_pipeline.process_alive", return_value=False), \
                 patch("turkeyworld.vqgan_pipeline.subprocess.Popen", return_value=SimpleNamespace(pid=70001)) as launch:
                selected = continue_world_after_codec(codec, args)
            validate.assert_called_once_with(codec, accept_user_requested=True)
            self.assertIs(migration.call_args.kwargs["accept_user_requested"], True)
            self.assertIn("--accept-current-codec", launch.call_args.args[0])
            state = json.loads((selected / "status.json").read_text("utf-8"))
            self.assertFalse(state["codec_validation"]["passed"])
            self.assertFalse(state["codec_training"]["quality_gate_passed"])
            self.assertTrue(state["codec_training"]["accepted_by_user"])
            self.assertFalse(state["evaluation"]["passed"])
            lineage = json.loads((selected / "vqgan_lineage.json").read_text("utf-8"))
            self.assertFalse(lineage["codec_quality_verified"])
            self.assertTrue(lineage["codec_quality_accepted_by_user"])

    def test_world_constructor_preserves_accepted_metadata_without_claiming_quality(self):
        from turkeyworld.train_500m import build_parser
        from turkeyworld.train_vq_world import LatentWorldExperiment
        from turkeyworld.vq_world import validate_frozen_codec
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codec = root / "accepted.pt"
            self.codec(codec)
            info = validate_frozen_codec(codec, accept_user_requested=True)
            args = build_parser().parse_args(["--run-dir", str(root / "world"), "--device", "cpu"])
            args.codec_checkpoint, args.control_run_dir = codec, None
            args.accept_current_codec = True
            with patch("turkeyworld.train_vq_world.validate_frozen_codec", return_value=info) as validate:
                experiment = LatentWorldExperiment(args)
            validate.assert_called_once_with(codec, accept_user_requested=True)
            self.assertFalse(experiment.state["codec_validation"]["passed"])
            self.assertTrue(experiment.state["codec_training"]["accepted_by_user"])
            self.assertFalse(experiment.state["codec_training"]["quality_gate_passed"])
            self.assertEqual(experiment.state["codec_training"]["current"], 11586)


if __name__ == "__main__":
    unittest.main()
