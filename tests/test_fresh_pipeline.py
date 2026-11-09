from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from turkeyworld.fresh_pipeline import FreshPipeline, PipelineStopped, build_commands, build_parser


class FreshPipelineTests(unittest.TestCase):
    def arguments(self, root):
        root = Path(root)
        return build_parser().parse_args(['--run-dir', str(root/'pipeline'), '--data-dir', str(root/'data'),
            '--codec-run-dir', str(root/'codec'), '--sr-run-dir', str(root/'sr'),
            '--world-run-dir', str(root/'world'), '--monitor-control-dir', str(root/'monitor'),
            '--image-size', '32', '--batch', '48', '--gradient-accumulation-steps', '21'])

    def test_all_stages_share_native32_and_world_uses_new_codec_sr_without_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            commands = build_commands(args, Path(directory)/'subset.json', Path(directory)/'masks')
            for command in commands.values():
                self.assertEqual(command[command.index('--image-size')+1], '32')
                self.assertEqual(command[command.index('--device')+1], args.device)
            self.assertIn('--world-from-scratch', commands['codec'])
            self.assertNotIn('--continue-world', commands['codec'])
            world = commands['world']
            self.assertIn('--from-scratch', world)
            self.assertNotIn('--source', world)
            self.assertNotIn('--accept-current-codec', world)
            self.assertEqual(world[world.index('--batch')+1], '48')
            self.assertEqual(world[world.index('--gradient-accumulation-steps')+1], '21')
            self.assertEqual(world[world.index('--sr-checkpoint')+1], str(args.sr_run_dir/'checkpoints/sr_for_game.pt'))
            self.assertEqual(world[world.index('--codec-checkpoint')+1], str(args.codec_run_dir/'checkpoints/codec_for_renderer.pt'))

    def test_first_launch_rejects_existing_worker_weights_but_normal_resume_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            checkpoint = args.world_run_dir/'checkpoints/old.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'Preserve old weights')
            with self.assertRaisesRegex(ValueError, 'new codec, SR and world'):
                FreshPipeline(args)
            self.assertEqual(checkpoint.read_bytes(), b'Preserve old weights')
        with tempfile.TemporaryDirectory() as directory:
            FreshPipeline(self.arguments(directory))
            FreshPipeline(self.arguments(directory))

    def test_stop_marker_prevents_worker_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = FreshPipeline(self.arguments(directory))
            (pipeline.root/'stop_training').touch()
            with patch('turkeyworld.fresh_pipeline.subprocess.Popen', side_effect=AssertionError('No launch')):
                with self.assertRaises(PipelineStopped):
                    pipeline.worker('codec', pipeline.args.codec_run_dir, ['never'])


if __name__ == '__main__':
    unittest.main()
