from pathlib import Path
import json
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

    def test_default_fifteen_second_target_reaches_worker_commands_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            self.assertEqual(args.target_seconds, 15)
            pipeline = FreshPipeline(args)
            self.assertEqual(pipeline.state['evaluation']['target_seconds'], 15)
            commands = build_commands(args, Path(directory)/'subset.json', Path(directory)/'masks')
            for name in ('codec', 'world'):
                command = commands[name]
                self.assertEqual(command[command.index('--target-seconds')+1], '15')

    def test_explicit_target_overrides_historical_migration_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            args.world_run_dir.mkdir()
            records = {'spatial_entities_migration.json': {'status': 'complete',
                'spatial_entities': True, 'target_seconds': 30},
                'combat_conditioning_migration.json': {'status': 'complete',
                'spatial_combat': True, 'target_seconds': 30},
                'training_v5_migration.json': {'status': 'complete', 'game_training_v5': True}}
            for name, value in records.items():
                (args.world_run_dir/name).write_text(json.dumps(value), encoding='utf-8')
            commands = build_commands(args, Path(directory)/'subset.json', Path(directory)/'masks')
            world = commands['world']
            self.assertEqual(world[world.index('--target-seconds')+1], '15')
            self.assertIn('--training-improvements-v5', world)
            for name, value in records.items():
                self.assertEqual(json.loads((args.world_run_dir/name).read_text(encoding='utf-8')), value)


if __name__ == '__main__':
    unittest.main()
