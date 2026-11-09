"""Mocked codec-to-world handoff tests; no real model migration or worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class VQGANPipelineTests(unittest.TestCase):
    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def fixture(self, folder):
        root = Path(folder)
        control, source, world, data = [root / name for name in ("codec", "source500", "world_vq", "data")]
        control.mkdir()
        (source / "checkpoints").mkdir(parents=True)
        data.mkdir()
        codec = control / "qualified.pt"
        codec.write_bytes(b"mock codec checkpoint")
        (source / "checkpoints/teacher500_world.pt").write_bytes(b"mock 500M source")
        args = SimpleNamespace(run_dir=control, world_source_run=source, world_run_dir=world,
                               data_dir=data, device="cpu")
        info = {"codec_fingerprint": "qualified-codec-A", "checkpoint": str(codec.resolve()),
                "quality_gate": {"passed": True, "consecutive_passes": 3, "required_passes": 3}}
        report = {"codec_fingerprint": info["codec_fingerprint"], "source_fingerprint": "teacher-source-A",
                  "architecture": "vjepa2_ac_vq",
                  "source_checkpoint": str((source / "checkpoints/teacher500_world.pt").resolve()),
                  "initial_checkpoint": str((world / "checkpoints/teacher_vjepa500vq_initial.pt").resolve())}
        self.write(source / "validation_panel500.json", {"visual": [[0, 0]], "dynamics": [[1, 2]],
                                                        "context": 2, "horizon": 4})

        def migrate(source_path, codec_path, initial, resume_output):
            initial.parent.mkdir(parents=True, exist_ok=True)
            initial.write_bytes(b"mock latent initial")
            resume_output.write_bytes(b"mock latent resume")
            return dict(report)

        def link(destination, source_path):
            destination.mkdir(parents=True, exist_ok=True)

        return args, codec, info, report, migrate, link

    def test_unqualified_codec_rejected_before_any_worker_or_filesystem_side_effect(self):
        from turkeyworld.vqgan_pipeline import continue_world_after_codec
        with tempfile.TemporaryDirectory() as folder:
            args, codec, _, _, _, _ = self.fixture(folder)
            (args.world_source_run / "training.pid").write_text("12345", encoding="ascii")
            before = {str(path.relative_to(folder)): path.read_bytes()
                      for path in Path(folder).rglob("*") if path.is_file()}
            with patch("turkeyworld.vqgan_pipeline.validate_frozen_codec", side_effect=ValueError("not qualified")), \
                 patch("turkeyworld.vqgan_pipeline.process_alive") as alive, \
                 patch("turkeyworld.vqgan_pipeline.migrate_teacher_to_vq") as migrate, \
                 patch("turkeyworld.vqgan_pipeline.link_data") as link, \
                 patch("turkeyworld.vqgan_pipeline.subprocess.Popen") as launch, \
                 self.assertRaisesRegex(ValueError, "not qualified"):
                continue_world_after_codec(codec, args)
            for operation in (alive, migrate, link, launch):
                operation.assert_not_called()
            self.assertFalse(args.world_run_dir.exists())
            after = {str(path.relative_to(folder)): path.read_bytes()
                     for path in Path(folder).rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_link_data_rejects_an_existing_different_corpus_without_external_commands(self):
        from turkeyworld.vqgan_pipeline import link_data
        with tempfile.TemporaryDirectory() as folder:
            original, different = Path(folder) / "original", Path(folder) / "different"
            original.mkdir()
            different.mkdir()
            (different / "sentinel").write_bytes(b"preserve")
            with patch("turkeyworld.vqgan_pipeline.subprocess.run") as command, \
                 self.assertRaisesRegex(ValueError, "different corpus"):
                link_data(different, original)
            command.assert_not_called()
            self.assertEqual((different / "sentinel").read_bytes(), b"preserve")

    def test_link_data_accepts_existing_identical_directory_without_relinking(self):
        from turkeyworld.vqgan_pipeline import link_data
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "corpus"
            source.mkdir()
            with patch("turkeyworld.vqgan_pipeline.subprocess.run") as command, \
                 patch.object(Path, "symlink_to") as symlink:
                link_data(source, source)
            command.assert_not_called()
            symlink.assert_not_called()

    def test_launch_uses_hidden_detached_cli_pid_and_matching_codec_lineage(self):
        from turkeyworld.vqgan_pipeline import continue_world_after_codec
        with tempfile.TemporaryDirectory() as folder:
            args, codec, info, report, migrate, link = self.fixture(folder)
            with patch("turkeyworld.vqgan_pipeline.validate_frozen_codec", return_value=info) as validate, \
                 patch("turkeyworld.vqgan_pipeline.migrate_teacher_to_vq", side_effect=migrate) as migration, \
                 patch("turkeyworld.vqgan_pipeline.link_data", side_effect=link) as linking, \
                 patch("turkeyworld.vqgan_pipeline.process_alive", return_value=False), \
                 patch("turkeyworld.vqgan_pipeline.subprocess.Popen", return_value=SimpleNamespace(pid=63001)) as launch, \
                 patch.dict(os.environ, {"VQGAN_PIPELINE_TEST": "inherited"}):
                selected = continue_world_after_codec(codec, args)
            self.assertEqual(selected, args.world_run_dir.resolve())
            validate.assert_called_once_with(codec)
            linking.assert_called_once_with(args.world_run_dir / "data", args.data_dir)
            migration.assert_called_once_with(args.world_source_run / "checkpoints/teacher500_world.pt", codec,
                args.world_run_dir / "checkpoints/teacher_vjepa500vq_initial.pt",
                resume_output=args.world_run_dir / "checkpoints/teacher500vq_world.pt")
            command = launch.call_args.args[0]
            self.assertEqual(command[:6], [sys.executable, "-X", "utf8", "-u", "-m", "turkeyworld.train_vq_world"])
            options = dict(zip(command[6::2], command[7::2]))
            self.assertEqual(options, {"--run-dir": str(args.world_run_dir.resolve()),
                "--source": str(args.world_source_run / "checkpoints/teacher500_world.pt"),
                "--codec-checkpoint": str(codec.resolve()), "--control-run-dir": str(args.run_dir.resolve()),
                "--device": "cpu"})
            kwargs = launch.call_args.kwargs
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            self.assertTrue(kwargs["close_fds"])
            self.assertEqual(kwargs["creationflags"], subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.assertEqual(kwargs["start_new_session"], os.name != "nt")
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "4")
            self.assertEqual(kwargs["env"]["MKL_NUM_THREADS"], "4")
            self.assertEqual(kwargs["env"]["VQGAN_PIPELINE_TEST"], "inherited")
            self.assertEqual(kwargs["cwd"], Path(__file__).resolve().parents[1])
            self.assertEqual(Path(kwargs["stdout"].name), args.world_run_dir / "training.stdout.log")
            self.assertEqual(Path(kwargs["stderr"].name), args.world_run_dir / "training.stderr.log")
            self.assertEqual((args.world_run_dir / "training.pid").read_text().strip(), "63001")
            pointer = json.loads((args.run_dir / "active_run.json").read_text("utf-8"))
            self.assertEqual(pointer["run_dir"], str(selected))
            self.assertEqual(pointer["pid"], 63001)
            self.assertEqual(pointer["codec_fingerprint"], info["codec_fingerprint"])
            self.assertEqual(json.loads((selected / "vqgan_lineage.json").read_text("utf-8")), report)
            self.assertEqual(json.loads((selected / "codec_ready.json").read_text("utf-8")), info)
            handoff = json.loads((args.run_dir / "world_handoff.json").read_text("utf-8"))
            self.assertEqual(handoff["migration"], report)
            self.assertEqual(handoff["pid"], 63001)
            self.assertEqual((selected / "validation_panel500vq.json").read_bytes(),
                             (args.world_source_run / "validation_panel500.json").read_bytes())

    def test_repeated_handoff_reuses_live_world_without_remigration_or_duplicate_process(self):
        from turkeyworld.vqgan_pipeline import continue_world_after_codec
        with tempfile.TemporaryDirectory() as folder:
            args, codec, info, _, migrate, link = self.fixture(folder)
            with patch("turkeyworld.vqgan_pipeline.validate_frozen_codec", return_value=info), \
                 patch("turkeyworld.vqgan_pipeline.migrate_teacher_to_vq", side_effect=migrate) as migration, \
                 patch("turkeyworld.vqgan_pipeline.link_data", side_effect=link), \
                 patch("turkeyworld.vqgan_pipeline.process_alive", return_value=True), \
                 patch("turkeyworld.vqgan_pipeline.subprocess.Popen", return_value=SimpleNamespace(pid=63002)) as launch:
                first = continue_world_after_codec(codec, args)
                initial = first / "checkpoints/teacher_vjepa500vq_initial.pt"
                stamp = (initial.stat().st_mtime_ns, initial.read_bytes())
                (args.run_dir / "world_handoff.json").unlink()
                self.write(first / "launch_pending.json", {"codec_fingerprint": info["codec_fingerprint"]})
                second = continue_world_after_codec(codec, args)
            self.assertEqual(first, second)
            self.assertEqual(migration.call_count, 1)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(stamp, (initial.stat().st_mtime_ns, initial.read_bytes()))
            pointer = json.loads((args.run_dir / "active_run.json").read_text("utf-8"))
            self.assertEqual(pointer["run_dir"], str(first))
            self.assertEqual(pointer["codec_fingerprint"], info["codec_fingerprint"])
            self.assertEqual(pointer["pid"], 63002)
            self.assertEqual(json.loads((args.run_dir / "world_handoff.json").read_text("utf-8"))["pid"], 63002)
            self.assertFalse((first / "launch_pending.json").exists())

    def test_existing_wrong_codec_lineage_rejects_before_launch_or_checkpoint_replacement(self):
        from turkeyworld.vqgan_pipeline import continue_world_after_codec
        with tempfile.TemporaryDirectory() as folder:
            args, codec, info, _, _, link = self.fixture(folder)
            initial = args.world_run_dir / "checkpoints/teacher_vjepa500vq_initial.pt"
            initial.parent.mkdir(parents=True)
            initial.write_bytes(b"existing different-codec weights")
            self.write(args.world_run_dir / "vqgan_lineage.json", {"codec_fingerprint": "other-codec"})
            with patch("turkeyworld.vqgan_pipeline.validate_frozen_codec", return_value=info), \
                 patch("turkeyworld.vqgan_pipeline.link_data", side_effect=link), \
                 patch("turkeyworld.vqgan_pipeline.migrate_teacher_to_vq") as migrate, \
                 patch("turkeyworld.vqgan_pipeline.subprocess.Popen") as launch, \
                 self.assertRaisesRegex(ValueError, "different frozen codec"):
                continue_world_after_codec(codec, args)
            launch.assert_not_called()
            migrate.assert_not_called()
            self.assertEqual(initial.read_bytes(), b"existing different-codec weights")
            self.assertFalse((args.run_dir / "active_run.json").exists())

    def test_controller_stop_flag_returns_without_creating_world_or_process(self):
        from turkeyworld.vqgan_pipeline import continue_world_after_codec
        with tempfile.TemporaryDirectory() as folder:
            args, codec, info, _, _, _ = self.fixture(folder)
            (args.run_dir / "stop_training").touch()
            with patch("turkeyworld.vqgan_pipeline.validate_frozen_codec", return_value=info), \
                 patch("turkeyworld.vqgan_pipeline.subprocess.Popen") as launch:
                self.assertIsNone(continue_world_after_codec(codec, args))
            launch.assert_not_called()
            self.assertFalse(args.world_run_dir.exists())

    def test_launch_failure_clears_pending_and_retries_without_remigrating(self):
        from turkeyworld.vqgan_pipeline import continue_world_after_codec
        with tempfile.TemporaryDirectory() as folder:
            args, codec, info, _, migrate, link = self.fixture(folder)
            with patch("turkeyworld.vqgan_pipeline.validate_frozen_codec", return_value=info), \
                 patch("turkeyworld.vqgan_pipeline.migrate_teacher_to_vq", side_effect=migrate) as migration, \
                 patch("turkeyworld.vqgan_pipeline.link_data", side_effect=link), \
                 patch("turkeyworld.vqgan_pipeline.process_alive", return_value=False), \
                 patch("turkeyworld.vqgan_pipeline.subprocess.Popen",
                       side_effect=[OSError("cannot launch"), SimpleNamespace(pid=63003)]) as launch:
                with self.assertRaisesRegex(OSError, "cannot launch"):
                    continue_world_after_codec(codec, args)
                self.assertFalse((args.world_run_dir / "launch_pending.json").exists())
                self.assertFalse((args.world_run_dir / "training.pid").exists())
                self.assertFalse((args.run_dir / "active_run.json").exists())
                continue_world_after_codec(codec, args)
            self.assertEqual(migration.call_count, 1)
            self.assertEqual(launch.call_count, 2)

    def test_postlaunch_pid_write_failure_keeps_pending_lock_and_prevents_duplicate(self):
        from turkeyworld.vqgan_pipeline import atomic_json, continue_world_after_codec
        with tempfile.TemporaryDirectory() as folder:
            args, codec, info, _, migrate, link = self.fixture(folder)

            def fail_pid(path, value):
                if Path(path).name == "training.pid":
                    raise OSError("cannot publish PID")
                return atomic_json(path, value)

            with patch("turkeyworld.vqgan_pipeline.validate_frozen_codec", return_value=info), \
                 patch("turkeyworld.vqgan_pipeline.migrate_teacher_to_vq", side_effect=migrate), \
                 patch("turkeyworld.vqgan_pipeline.link_data", side_effect=link), \
                 patch("turkeyworld.vqgan_pipeline.process_alive", return_value=False), \
                 patch("turkeyworld.vqgan_pipeline.subprocess.Popen", return_value=SimpleNamespace(pid=63004)) as launch:
                with patch("turkeyworld.vqgan_pipeline.atomic_json", side_effect=fail_pid), \
                     self.assertRaisesRegex(OSError, "cannot publish PID"):
                    continue_world_after_codec(codec, args)
                self.assertTrue((args.world_run_dir / "launch_pending.json").is_file())
                with self.assertRaisesRegex(RuntimeError, "launch is pending"):
                    continue_world_after_codec(codec, args)
            self.assertEqual(launch.call_count, 1)

    def recovery_fixture(self, folder, optimizer_preserved=True):
        args, _, info, report, _, _ = self.fixture(folder)
        initial = args.world_run_dir / "checkpoints/teacher_vjepa500vq_initial.pt"
        resume = initial.with_name("teacher500vq_world.pt")
        lineage = args.world_run_dir / "vqgan_lineage.json"
        initial.parent.mkdir(parents=True)
        report["optimizer_preserved"] = optimizer_preserved
        saved = {"model": {"tiny_weight": torch.tensor([1., 2., 3.])}, "vq_migration": report,
            "pipeline_phase": "teacher500vq_world", "training_steps": 134, "global_training_step": 134,
            "optimizer": {"state": {0: {"exp_avg": torch.tensor([.25])}}},
            "ema_target_encoder": {"tiny": torch.tensor([.5])}, "rng_numpy": {},
            "rng_torch": torch.get_rng_state(), "rng_cuda": None, "phase_complete": False}
        torch.save(saved, resume)
        return initial, resume, lineage, Path(report["source_checkpoint"]), info["codec_fingerprint"], saved

    def test_orphan_resume_recovers_initial_and_lineage_without_rewriting_valid_resume(self):
        from turkeyworld.vqgan_pipeline import recover_migration
        with tempfile.TemporaryDirectory() as folder:
            initial, resume, lineage, source, fingerprint, saved = self.recovery_fixture(folder)
            before = (resume.stat().st_mtime_ns, resume.read_bytes())
            report = recover_migration(initial, resume, lineage, source, fingerprint)
            self.assertEqual(report, saved["vq_migration"])
            self.assertEqual(json.loads(lineage.read_text("utf-8")), report)
            self.assertEqual(before, (resume.stat().st_mtime_ns, resume.read_bytes()))
            restored = torch.load(initial, weights_only=False)
            self.assertEqual(restored["training_steps"], 0)
            self.assertEqual(restored["pipeline_phase"], "teacher500vq_initial")
            torch.testing.assert_close(restored["model"]["tiny_weight"], saved["model"]["tiny_weight"], rtol=0, atol=0)
            for key in ("optimizer", "ema_target_encoder", "rng_numpy", "rng_torch", "rng_cuda", "phase_complete"):
                self.assertNotIn(key, restored)

    def test_missing_lineage_recovers_without_overwriting_existing_initial(self):
        from turkeyworld.vqgan_pipeline import recover_migration
        with tempfile.TemporaryDirectory() as folder:
            initial, resume, lineage, source, fingerprint, saved = self.recovery_fixture(folder)
            initial_payload = {"model": saved["model"], "vq_migration": saved["vq_migration"],
                               "pipeline_phase": "teacher500vq_initial", "training_steps": 0}
            torch.save(initial_payload, initial)
            before = (initial.stat().st_mtime_ns, initial.read_bytes())
            recover_migration(initial, resume, lineage, source, fingerprint)
            self.assertEqual(before, (initial.stat().st_mtime_ns, initial.read_bytes()))
            self.assertEqual(json.loads(lineage.read_text("utf-8")), saved["vq_migration"])

    def test_recovery_rejects_wrong_codec_source_or_promised_missing_optimizer_resume(self):
        from turkeyworld.vqgan_pipeline import recover_migration
        for wrong in ("codec", "source", "missing_resume", "mismatched_resume"):
            with self.subTest(wrong=wrong), tempfile.TemporaryDirectory() as folder:
                initial, resume, lineage, source, fingerprint, saved = self.recovery_fixture(folder)
                if wrong == "codec":
                    fingerprint = "wrong-codec"
                elif wrong == "source":
                    source = source.with_name("different-source.pt")
                else:
                    torch.save({"model": saved["model"], "vq_migration": dict(saved["vq_migration"]),
                                "pipeline_phase": "teacher500vq_initial"}, initial)
                    if wrong == "missing_resume":
                        resume.unlink()
                    else:
                        saved["pipeline_phase"] = "different_phase"
                        torch.save(saved, resume)
                expected = RuntimeError if wrong == "missing_resume" else ValueError
                with self.assertRaises(expected):
                    recover_migration(initial, resume, lineage, source, fingerprint)
                self.assertFalse(lineage.exists())


if __name__ == "__main__":
    unittest.main()
