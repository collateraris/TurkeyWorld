from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from filelock import Timeout

from turkeyworld import distill_vjepa


class DistillationOutputTest(TestCase):
    def test_log_allowed_and_concurrent_writer_refused(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            args = SimpleNamespace(output=output)
            (output / 'training.log').write_text('log')

            def train(a):
                with self.assertRaises(Timeout):
                    distill_vjepa.run_in_output(a)

            with patch.object(distill_vjepa, '_train', side_effect=train) as trainer:
                distill_vjepa.run_in_output(args)
            trainer.assert_called_once_with(args)

    def test_prior_checkpoints_and_partial_writes_are_preserved(self):
        with TemporaryDirectory() as tmp, patch.object(distill_vjepa, '_train') as trainer:
            args = SimpleNamespace(output=Path(tmp))
            for name in ('final.pt', 'init.pt', 'student.pt.tmp', 'metrics.json.123.tmp'):
                with self.subTest(name=name):
                    path = args.output / name
                    path.write_bytes(b'original')
                    with self.assertRaisesRegex(FileExistsError, 'already contains'):
                        distill_vjepa.run_in_output(args)
                    self.assertEqual(path.read_bytes(), b'original')
                    path.unlink()
            trainer.assert_not_called()
