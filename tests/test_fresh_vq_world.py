"""Fresh world/frame-generator lineage and deterministic-corpus readiness."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class FreshVQWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def codec_info(self):
        from turkeyworld.vqgan import VQGANConfig
        return {'codec_config': asdict(VQGANConfig(n_embed=1488)),
            'generator_parameters': 4_763_731, 'codec_fingerprint': 'frozen-local-codec',
            'training_steps': 11586, 'quality_gate': {'passed': False, 'frozen': True,
                'consecutive_passes': 0, 'required_passes': 3}, 'accepted_by_user': True}

    def args(self, root):
        from turkeyworld.train_500m import build_parser
        args = build_parser().parse_args(['--device', 'cpu', '--run-dir', str(root)])
        args.from_scratch, args.accept_current_codec = True, True
        args.codec_checkpoint = Path(root)/'frozen-codec.pt'
        args.control_run_dir = None
        return args

    def test_full_fresh_teacher_profile_keeps_selected_500m_budget_and_frozen_codec(self):
        from turkeyworld.models import NeuralWorldModel
        from turkeyworld.train_vq_world import teacher_vq_config
        from turkeyworld.vq_world import TEACHER_VQ_PARAMETERS
        with torch.device('meta'):
            model = NeuralWorldModel(teacher_vq_config(self.codec_info()['codec_config']))
        self.assertEqual(sum(p.numel() for p in model.parameters()), TEACHER_VQ_PARAMETERS)
        self.assertEqual(TEACHER_VQ_PARAMETERS, 503_825_084)
        self.assertEqual((model.core.width, len(model.core.encoder.blocks), len(model.core.predictor.blocks)),
                         (512, 76, 58))
        self.assertEqual(model.flow_renderer.steps, 12)
        self.assertTrue(all(not p.requires_grad for p in model.flow_renderer.codec.parameters()))

    def test_fresh_constructor_forces_joint_training_and_records_no_pretrained_world(self):
        from turkeyworld.engine import BEHAVIOR_VERSION
        from turkeyworld.train_vq_world import LatentWorldExperiment
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            with patch('turkeyworld.train_vq_world.validate_frozen_codec', return_value=self.codec_info()):
                experiment = LatentWorldExperiment(args)
        self.assertTrue(args.train_renderer_during_pretrain)
        initialization = experiment.state['initialization']
        self.assertEqual(initialization['mode'], 'from_scratch_world_and_renderer')
        self.assertFalse(initialization['pretrained_world_weights_loaded'])
        self.assertFalse(initialization['pretrained_renderer_weights_loaded'])
        self.assertTrue(initialization['trained_codec_reused'])
        self.assertEqual(experiment.state['enemy_policy']['version'], BEHAVIOR_VERSION)
        self.assertEqual(experiment.state['enemy_policy']['neural_spawn'], 'learned_argmax')
        self.assertEqual(experiment.state['data_growth']['additional_model_visited_transitions'], 0)

    def test_existing_checkpoint_without_fresh_lineage_is_rejected_before_parent_startup(self):
        from turkeyworld.train_vq_world import LatentWorldExperiment
        from turkeyworld.train_500m import MassiveExperiment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'checkpoints').mkdir()
            sentinel = root/'checkpoints'/'teacher500vq_world.pt'
            sentinel.write_bytes(b'preserve old trained world')
            with patch('turkeyworld.train_vq_world.validate_frozen_codec', return_value=self.codec_info()), \
                 patch.object(MassiveExperiment, '__init__', side_effect=AssertionError('No parent startup')):
                with self.assertRaisesRegex(ValueError, 'new run directory'):
                    LatentWorldExperiment(self.args(root))
            self.assertEqual(sentinel.read_bytes(), b'preserve old trained world')

    def test_fresh_factory_loads_only_codec_and_saves_random_world_at_zero_updates(self):
        from turkeyworld.engine import BEHAVIOR_VERSION
        from turkeyworld.models import NeuralWorldModel
        from turkeyworld.train_vq_world import LatentWorldExperiment, teacher_vq_config
        from turkeyworld.vqgan import VQGAN

        codec_config = dict(image_size=128, base_channels=8, latent_channels=16,
                            embedding_dim=4, n_embed=32, data_init=False, kmeans_iterations=0)
        tiny = replace(teacher_vq_config(codec_config), vjepa_width=32,
            vjepa_encoder_depth=1, vjepa_predictor_depth=1, vjepa_heads=4,
            vjepa_readout_mlp_ratio=1, vjepa_gradient_checkpointing=False,
            flow_width=32, flow_depth=1, flow_heads=4,
            flow_adapter_width=0, flow_adapter_depth=0, flow_adapter_heads=0)
        with torch.device('meta'):
            expected_count = sum(p.numel() for p in NeuralWorldModel(tiny).parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codec = VQGAN(codec_config).eval()
            codec_path = root/'trained-codec.pt'
            torch.save({'codec_config': codec.get_config(), 'model': codec.state_dict(),
                        'latent_normalization': {'mean': [.1]*4, 'std': [.7]*4}}, codec_path)
            old_source = root/'old-trained-world.pt'
            old_source.write_bytes(b'This world must never be opened')
            initial = root/'checkpoints'/'teacher_vjepa500vq_initial.pt'
            experiment = object.__new__(LatentWorldExperiment)
            experiment.root = root
            experiment.args = SimpleNamespace(from_scratch=True, source=old_source,
                                              codec_checkpoint=codec_path)
            experiment.codec_info = {'codec_config': codec.get_config()}
            experiment.state = {'initialization': {'mode': 'from_scratch_world_and_renderer',
                'pretrained_world_weights_loaded': False, 'pretrained_renderer_weights_loaded': False,
                'trained_codec_reused': True}, 'enemy_policy': {'version': BEHAVIOR_VERSION}}
            loaded_paths, real_load = [], torch.load

            def load_only_codec(path, *args, **kwargs):
                self.assertEqual(Path(path), codec_path)
                loaded_paths.append(Path(path))
                return real_load(path, *args, **kwargs)

            with patch('turkeyworld.train_vq_world.teacher_vq_config', return_value=tiny), \
                 patch('turkeyworld.train_vq_world.TEACHER_VQ_PARAMETERS', expected_count), \
                 patch('turkeyworld.train_vq_world.migrate_teacher_to_vq',
                       side_effect=AssertionError('Fresh training must not migrate world weights')), \
                 patch('torch.load', side_effect=load_only_codec):
                report = experiment.create_initial_teacher(initial)
            self.assertEqual(loaded_paths, [codec_path])
            saved = real_load(initial, weights_only=False, map_location='cpu')
            self.assertEqual(saved['training_steps'], 0)
            self.assertEqual(saved['global_training_step'], 0)
            self.assertEqual(saved['behavior_version'], BEHAVIOR_VERSION)
            self.assertNotIn('optimizer', saved)
            self.assertNotIn('ema', saved)
            self.assertTrue(report['from_scratch'])
            self.assertFalse(report['pretrained_world_weights_loaded'])
            self.assertFalse(report['pretrained_renderer_weights_loaded'])
            self.assertTrue(report['trained_codec_reused'])
            self.assertTrue(bool(saved['model']['flow_renderer.codec_loaded']))
            for name, value in codec.state_dict().items():
                self.assertTrue(torch.equal(saved['model']['flow_renderer.codec.'+name], value), name)
            self.assertGreater(float(saved['model']['vjepa_backend.core.encoder.patch_embed.weight'].std()), 0.)
            self.assertEqual(float(saved['model']['flow_renderer.output.weight'].abs().sum()), 0.)
            self.assertEqual(old_source.read_bytes(), b'This world must never be opened')
            marker = json.loads((root/'initialization.json').read_text('utf-8'))
            self.assertEqual(marker['mode'], 'from_scratch_world_and_renderer')

    def test_existing_marker_must_match_fresh_mode_seed_codec_behavior_and_weight_lineage(self):
        from turkeyworld.engine import BEHAVIOR_VERSION
        from turkeyworld.train_vq_world import LatentWorldExperiment
        from turkeyworld.train_500m import MassiveExperiment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root)
            expected = {'mode': 'from_scratch_world_and_renderer', 'seed': args.seed,
                'codec_fingerprint': self.codec_info()['codec_fingerprint'], 'behavior_version': BEHAVIOR_VERSION,
                'pretrained_world_weights_loaded': False, 'pretrained_renderer_weights_loaded': False,
                'trained_codec_reused': True}
            for key, wrong in (('mode', 'migration'), ('seed', args.seed + 1),
                               ('codec_fingerprint', 'another-codec'), ('behavior_version', 'old-random-enemies'),
                               ('pretrained_world_weights_loaded', True), ('pretrained_renderer_weights_loaded', True)):
                with self.subTest(field=key):
                    (root/'initialization.json').write_text(json.dumps({**expected, key: wrong}), encoding='utf-8')
                    with patch('turkeyworld.train_vq_world.validate_frozen_codec', return_value=self.codec_info()), \
                         patch.object(MassiveExperiment, '__init__', side_effect=AssertionError('No parent startup')):
                        with self.assertRaisesRegex(ValueError, 'different scratch experiment'):
                            LatentWorldExperiment(args)

    def test_matching_fresh_marker_allows_resume_without_touching_existing_checkpoint(self):
        from turkeyworld.engine import BEHAVIOR_VERSION
        from turkeyworld.train_vq_world import LatentWorldExperiment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root)
            (root/'checkpoints').mkdir()
            sentinel = root/'checkpoints'/'teacher500vq_world.pt'
            sentinel.write_bytes(b'preserve fresh trained checkpoint')
            expected = {'mode': 'from_scratch_world_and_renderer', 'seed': args.seed,
                'codec_fingerprint': self.codec_info()['codec_fingerprint'], 'behavior_version': BEHAVIOR_VERSION,
                'pretrained_world_weights_loaded': False, 'pretrained_renderer_weights_loaded': False,
                'trained_codec_reused': True}
            (root/'initialization.json').write_text(json.dumps(expected), encoding='utf-8')
            with patch('turkeyworld.train_vq_world.validate_frozen_codec', return_value=self.codec_info()):
                experiment = LatentWorldExperiment(args)
            self.assertEqual(sentinel.read_bytes(), b'preserve fresh trained checkpoint')
            self.assertEqual(experiment.state['initialization'], expected)

    def test_data_readiness_requires_new_behavior_and_full_validation_despite_old_visited_manifest(self):
        from turkeyworld.engine import BEHAVIOR_VERSION
        from turkeyworld.train_vq_world import LatentWorldExperiment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data').mkdir()
            experiment = object.__new__(LatentWorldExperiment)
            experiment.root, experiment.state = root, {}
            experiment.args = SimpleNamespace(from_scratch=True, minimum_train_clips=8)
            updates = []
            experiment.update = lambda *args, **kwargs: updates.append((args, kwargs))
            status = root/'data'/'status.json'
            (root/'data'/'model_visited_manifest.json').write_text(
                json.dumps({'status': 'completed', 'clips': 256}), encoding='utf-8')
            wrong = {'behavior_version': 'old_random_behavior', 'status': 'running',
                     'groups': {'rgb/train': {'files': 100000}, 'rgb/val': {'files': 512}}}
            status.write_text(json.dumps(wrong), encoding='utf-8')
            with patch('turkeyworld.train_vq_world.time.sleep', side_effect=AssertionError('Reject before waiting')):
                with self.assertRaisesRegex(ValueError, 'behavior version'):
                    experiment.wait_for_data()
            failed = {**wrong, 'behavior_version': BEHAVIOR_VERSION, 'status': 'failed'}
            status.write_text(json.dumps(failed), encoding='utf-8')
            with patch('turkeyworld.train_vq_world.time.sleep', side_effect=AssertionError('Reject before waiting')):
                with self.assertRaisesRegex(RuntimeError, 'collection failed'):
                    experiment.wait_for_data()
            pending = {'behavior_version': BEHAVIOR_VERSION, 'status': 'running',
                       'groups': {'rgb/train': {'files': 7}, 'rgb/val': {'files': 512}}}
            status.write_text(json.dumps(pending), encoding='utf-8')

            def finish_validation(seconds):
                self.assertEqual(seconds, 5)
                if pending['groups']['rgb/train']['files'] < 8:
                    pending['groups']['rgb/train']['files'] = 8
                    pending['groups']['rgb/val']['files'] = 511
                else:
                    pending['groups']['rgb/val']['files'] = 512
                status.write_text(json.dumps(pending), encoding='utf-8')

            with patch('turkeyworld.train_vq_world.time.sleep', side_effect=finish_validation) as sleep:
                experiment.wait_for_data()
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(len(updates), 2)
            self.assertEqual(experiment.state['data_preparation']['groups']['rgb/val']['files'], 512)


if __name__ == '__main__':
    unittest.main()
