"""Durable accepted-quality handoff validates actual frozen codec provenance."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from turkeyworld.fresh_pipeline import (FreshPipeline, build_commands, build_parser,
                                         validate_pipeline_codec)
from turkeyworld.vq_world import codec_fingerprint
from turkeyworld.vqgan import VQGAN, VQGANConfig


class FreshPipelineCodecAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def args(root):
        return build_parser().parse_args(['--run-dir', str(root/'pipeline'), '--data-dir', str(root/'data'),
            '--codec-run-dir', str(root/'codec'), '--sr-run-dir', str(root/'sr'),
            '--world-run-dir', str(root/'world'), '--monitor-control-dir', str(root/'monitor'),
            '--device', 'cpu', '--image-size', '32'])

    @staticmethod
    def codec(root, *, accepted=True, qualified=False):
        root.mkdir(parents=True, exist_ok=True)
        model = VQGAN(VQGANConfig(image_size=32, base_channels=8, latent_channels=16,
                                 embedding_dim=16, n_embed=1488, data_init=False)).eval()
        path = root/'checkpoints/codec_for_renderer.pt'
        path.parent.mkdir(exist_ok=True)
        gate = {'passed': qualified, 'frozen': True, 'required_passes': 3,
            'consecutive_passes': 3 if qualified else 0,
            'checks': {key: True for key in ('all_frames_finite', 'all_frames_in_range',
                'quantization_finite', 'valid_code_indices', 'codebook_broad_usage')}}
        payload = {'stage': 'frozen_vqgan_codec', 'codec_config': model.get_config(),
            'model': model.state_dict(), 'codec_fingerprint': codec_fingerprint(model),
            'quality_gate': gate, 'latent_normalization': {'mean': [0.]*16, 'std': [1.]*16},
            'training_steps': 19000}
        ready = {'checkpoint': str(path.resolve()), 'codec_fingerprint': payload['codec_fingerprint'],
            'quality_gate_passed': qualified, 'training_steps': 19000}
        if accepted:
            acceptance = {'mode': 'user_requested', 'accepted': True,
                'reason': 'Explicit user request to continue SR at current quality',
                'accepted_at': '2026-11-09T18:00:00+00:00', 'quality_gate_passed': qualified}
            payload['quality_acceptance'] = acceptance
            ready.update(accepted_by_user=True, quality_acceptance=copy.deepcopy(acceptance))
        torch.save(payload, path)
        ready_path = root/'codec_ready.json'
        ready_path.write_text(json.dumps(ready), encoding='utf-8')
        return path, ready_path, payload, ready

    def test_actual_user_accepted_unqualified_codec_adds_only_world_flag_and_keeps_fingerprint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path, ready_path, payload, _ = self.codec(root/'codec')
            info = validate_pipeline_codec(path, ready_path, 32)
            self.assertTrue(info['accepted_by_user'])
            self.assertFalse(info['qualified'])
            self.assertFalse(info['quality_gate']['passed'])
            self.assertEqual(info['codec_fingerprint'], payload['codec_fingerprint'])
            commands = build_commands(self.args(root), root/'subset.json', root/'masks', codec_info=info)
            self.assertIn('--accept-current-codec', commands['world'])
            self.assertNotIn('--accept-current-codec', commands['codec'])
            self.assertEqual(commands['world'][commands['world'].index('--codec-checkpoint')+1], str(path))
            self.assertEqual(commands['world'][commands['world'].index('--sr-checkpoint')+1],
                             str(root/'sr/checkpoints/sr_for_game.pt'))

    def test_qualified_default_remains_strict_without_accept_flag(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path, ready, _, _ = self.codec(root/'codec', accepted=False, qualified=True)
            info = validate_pipeline_codec(path, ready, 32)
            self.assertTrue(info['qualified'])
            self.assertFalse(info['accepted_by_user'])
            self.assertNotIn('--accept-current-codec', build_commands(
                self.args(root), root/'subset.json', root/'masks', codec_info=info)['world'])

    def test_ready_sidecar_alone_never_authorizes_unaccepted_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path, ready_path, payload, _ = self.codec(root, accepted=True)
            payload.pop('quality_acceptance')
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'three consecutive'):
                validate_pipeline_codec(path, ready_path, 32)

    def test_false_mismatched_or_missing_sidecar_cannot_bypass_gate(self):
        for problem in ('false', 'wrong_mode', 'fingerprint', 'wrong_file', 'truthful_gate', 'acceptance', 'missing'):
            with self.subTest(problem=problem), TemporaryDirectory() as directory:
                root = Path(directory)
                path, ready_path, _, ready = self.codec(root)
                if problem == 'false':
                    ready['accepted_by_user'] = False
                elif problem == 'wrong_mode':
                    ready['quality_acceptance']['mode'] = 'automatic'
                elif problem == 'fingerprint':
                    ready['codec_fingerprint'] = 'forged'
                elif problem == 'wrong_file':
                    ready['checkpoint'] = str(root/'other.pt')
                elif problem == 'truthful_gate':
                    ready['quality_gate_passed'] = True
                elif problem == 'acceptance':
                    ready['quality_acceptance']['reason'] = 'different unrecorded acceptance'
                if problem == 'missing':
                    ready_path.unlink()
                else:
                    ready_path.write_text(json.dumps(ready), encoding='utf-8')
                with self.assertRaises(ValueError):
                    validate_pipeline_codec(path, ready_path, 32)

    def test_checkpoint_integrity_still_required_with_user_acceptance(self):
        for problem in ('weights', 'integrity', 'resolution'):
            with self.subTest(problem=problem), TemporaryDirectory() as directory:
                path, ready, payload, _ = self.codec(Path(directory))
                if problem == 'weights':
                    payload['model']['decoder.conv_out.weight'].add_(1.)
                elif problem == 'integrity':
                    payload['quality_gate']['checks']['all_frames_finite'] = False
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    validate_pipeline_codec(path, ready, 64 if problem == 'resolution' else 32)

    def test_resume_skips_codec_training_persists_actual_handoff_then_sr_and_same_world_codec(self):
        class WorldReached(Exception):
            pass

        with TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = FreshPipeline(self.args(root))
            path, ready, payload, _ = self.codec(pipeline.args.codec_run_dir)
            original = path.read_bytes()
            pipeline.args.data_dir.mkdir()
            (pipeline.args.data_dir/'status.json').write_text(json.dumps({'status': 'complete',
                'groups': {'rgb/train': {'transitions': 128000}}}), encoding='utf-8')
            fixed = {'fingerprint': 'fixed128k'}
            calls = []

            def worker(name, stage_root, command):
                self.assertNotEqual(name, 'codec', 'Accepted codec must never retrain')
                calls.append((name, command))
                if name == 'sr':
                    output = stage_root/'checkpoints/sr_for_game.pt'
                    output.parent.mkdir(parents=True)
                    torch.save({'stage': 'sr', 'training_steps': 2000, 'input_size': 32, 'output_size': 128}, output)
                else:
                    raise WorldReached()

            with patch('turkeyworld.fresh_pipeline.create_training_subset', return_value=fixed), \
                 patch('turkeyworld.object_mask_cache.prepare_object_masks'), \
                 patch('turkeyworld.fresh_pipeline.link_data'), \
                 patch.object(FreshPipeline, 'live_worker', return_value=False), \
                 patch.object(pipeline, 'worker', side_effect=worker):
                with self.assertRaises(WorldReached):
                    pipeline.run()
            self.assertEqual([name for name, _ in calls], ['sr', 'world'])
            self.assertIn('--accept-current-codec', calls[-1][1])
            handoff = json.loads((pipeline.root/'codec_handoff.json').read_text(encoding='utf-8'))
            self.assertEqual(handoff['codec_fingerprint'], payload['codec_fingerprint'])
            self.assertFalse(handoff['qualified'])
            self.assertTrue(handoff['accepted_by_user'])
            self.assertEqual(pipeline.state['pipeline']['codec_fingerprint'], payload['codec_fingerprint'])
            self.assertFalse(pipeline.state['pipeline']['codec_qualified'])
            self.assertEqual(path.read_bytes(), original)
            # A second process reconstructs acceptance from durable artifacts,
            # without adding any new option to immutable controller config.
            resumed = FreshPipeline(self.args(root))
            calls.clear()
            with patch('turkeyworld.fresh_pipeline.create_training_subset', return_value=fixed), \
                 patch('turkeyworld.object_mask_cache.prepare_object_masks'), \
                 patch('turkeyworld.fresh_pipeline.link_data'), \
                 patch.object(FreshPipeline, 'live_worker', return_value=False), \
                 patch.object(resumed, 'worker', side_effect=worker):
                with self.assertRaises(WorldReached):
                    resumed.run()
            self.assertEqual([name for name, _ in calls], ['world'])
            self.assertIn('--accept-current-codec', calls[0][1])
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
