from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.base_student import build_base_student, export_base_student, load_base_student
from turkeyworld.models import NeuralWorldModel, STATE_SCHEMA, WorldModelConfig, load_world_checkpoint
from turkeyworld.navigation_recovery_data import collect
from turkeyworld.renderer_refinement import (causal_conditions, configure_refinement,
    frozen_fingerprint, refinement_forward, refinement_loss)
from turkeyworld.renderer_student import build_renderer_student, export_renderer_student, load_renderer_student
from turkeyworld.train_renderer_refine import RendererRefinement, SourcePools


def source_model():
    config = WorldModelConfig(kind='teacher', architecture='dreamer_rssm', image_size=32,
        endless=True, navigation_only=True, spatial_entities=True, spatial_combat=True,
        player_motion_mode='axis_fraction', renderer_mode='flow_matching', renderer_codec='vqgan',
        renderer_codec_config=dict(image_size=32, base_channels=4, latent_channels=8,
            embedding_dim=4, n_embed=8, data_init=False), flow_steps=1,
        flow_width=16, flow_depth=1, flow_heads=2, vjepa_width=16,
        vjepa_readout_mlp_ratio=2, rssm_deter_dim=16, rssm_hidden_dim=16,
        rssm_stoch=2, rssm_classes=4, rssm_embed_dim=16, rssm_action_dim=20, rssm_blocks=2)
    model = NeuralWorldModel(config).eval()
    model.flow_renderer.codec_loaded.fill_(True)
    with torch.no_grad():
        model.vjepa_backend.feature_head.weight.fill_(.005)
        model.vjepa_backend.feature_head.bias.fill_(.02)
    return model


class RendererRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        collect(self.root/'data', short_train=16, short_val=16, long_train=0, long_val=0)
        source = source_model()
        self.world, self.bundle = self.root/'world.pt', self.root/'old_rf1.pt'
        torch.save({'stage': 'world', 'config': asdict(source.config), 'state_schema': STATE_SCHEMA,
                    'model': source.state_dict(), 'training_steps': 339}, self.world)
        base = self.root/'old_base.pt'
        export_base_student(source, build_base_student(source), base, training_steps=1100, metrics={'passed': True})
        load_base_student(source, base)
        flow = build_renderer_student(source, steps=1, width=16, depth=1, heads=2, mlp_hidden=32)
        export_renderer_student(source, flow, self.bundle, training_steps=1500)

    def working(self):
        model, _ = load_world_checkpoint(self.world)
        load_renderer_student(model, self.bundle)
        configure_refinement(model)
        return model

    def test_actual_gradient_step_updates_both_heads_but_all_codec_state_and_dynamics_stay_bitexact(self):
        model = self.working()
        data = SourcePools([self.root/'data/short/rgb/train'], 'train')
        batch = data.batch(2, indices=[(9, 0), (10, 0)])
        before = frozen_fingerprint(model)
        codec = {key: value.clone() for key, value in model.flow_renderer.codec.state_dict().items()}
        head_before = model.base_student.rgb.weight.clone()
        flow_before = model.flow_renderer.output.weight.clone()
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4)
        parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
        self.assertFalse(parameter_ids & {id(parameter) for parameter in model.flow_renderer.codec.parameters()})
        self.assertFalse(model.vjepa_backend.training)
        self.assertFalse(model.flow_renderer.codec.training)
        self.assertFalse(model.flow_renderer.codec.quantize.training)
        with (patch('turkeyworld.engine.Game.step', side_effect=AssertionError('No Source future')),
              patch('turkeyworld.engine.Game.render', side_effect=AssertionError('Recorded data only')),
              patch.object(model.flow_renderer, 'velocity', wraps=model.flow_renderer.velocity) as velocity):
            loss, metrics = refinement_loss(model, batch, latent_weight=.01)
            self.assertEqual(velocity.call_count, 1)
            loss.backward()
        self.assertGreater(float(model.base_student.rgb.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.flow_renderer.output.weight.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters() if not parameter.requires_grad))
        optimizer.step()
        self.assertFalse(torch.equal(head_before, model.base_student.rgb.weight))
        self.assertFalse(torch.equal(flow_before, model.flow_renderer.output.weight))
        self.assertEqual(before, frozen_fingerprint(model))
        self.assertTrue(all(torch.equal(value, model.flow_renderer.codec.state_dict()[key]) for key, value in codec.items()))
        self.assertFalse(metrics['future_source_rgb_in_prior'])

    def test_future_target_rgb_changes_loss_but_never_causal_prior_features_or_generated_rgb(self):
        model = self.working()
        data = SourcePools([self.root/'data/short/rgb/train'], 'train')
        batch = data.batch(2, indices=[(9, 0), (10, 0)])
        changed = {key: value.clone() for key, value in batch.items()}
        changed['frames'][:, 1:] = 1-changed['frames'][:, 1:]
        changed['target_frames'] = 1-changed['target_frames']
        first, second = causal_conditions(model, batch), causal_conditions(model, changed)
        self.assertTrue(torch.equal(first['features'], second['features']))
        self.assertTrue(torch.equal(first['prior_memory'], second['prior_memory']))
        self.assertTrue(torch.equal(first['state'], second['state']))
        self.assertFalse(torch.equal(first['target'], second['target']))
        with torch.no_grad():
            left = refinement_forward(model, first)
            right = refinement_forward(model, second)
        self.assertTrue(torch.equal(left[1], right[1]))
        self.assertTrue(torch.equal(first['state'], batch['target_states'][:, 0]))

    def args(self, run, steps):
        return Namespace(world=self.world, renderer_checkpoint=self.bundle, run_dir=run,
            data_root=self.root/'data', train_dir=[], val_dir=[], sr=None, device='cpu',
            allow_unqualified_codec=False, flow_to_base=False, steps=steps, batch=2, validation_frames=2,
            validation_batch=2, validate_every=1, preview_every=1, save_every=1, required_passes=2,
            seed=173, threads=2, lr=1e-4, base_lr=1e-4, ssim_weight=.2, edge_weight=.1,
            latent_weight=0., target_ssim=1., target_mae=0.)

    def test_real_full_resume_matches_optimizer_rng_and_exports_fresh_source_gate(self):
        direct, resumed = self.root/'direct', self.root/'resumed'
        RendererRefinement(self.args(direct, 2)).run()
        RendererRefinement(self.args(resumed, 1)).run()
        RendererRefinement(self.args(resumed, 2)).run()
        left = torch.load(direct/'checkpoints/renderer_refine_last.pt', weights_only=False)
        right = torch.load(resumed/'checkpoints/renderer_refine_last.pt', weights_only=False)
        for key in ('base', 'flow'):
            self.assertTrue(all(torch.equal(value, right[key][name]) for name, value in left[key].items()), key)
        for index, states in left['optimizer']['state'].items():
            self.assertTrue(all(torch.equal(value, right['optimizer']['state'][index][name]) for name, value in states.items()))
        self.assertEqual(left['sampler_rng'], right['sampler_rng'])
        self.assertTrue(torch.equal(left['rng_torch'], right['rng_torch']))
        bundle = torch.load(resumed/'checkpoints/renderer_student.pt', weights_only=False)
        self.assertEqual(bundle['solver_steps'], 1)
        self.assertFalse(bundle['metrics']['source_quality_passed'])
        self.assertFalse(bundle['base_student_bundle']['metrics']['passed'])
        self.assertIn('Source RGB', bundle['metrics']['criterion'])
        self.assertGreater(bundle['training_steps'], 1500)
        fresh, _ = load_world_checkpoint(self.world)
        load_renderer_student(fresh, resumed/'checkpoints/renderer_student.pt')
        self.assertEqual(fresh.flow_renderer.steps, 1)
        self.assertTrue(self.bundle.is_file())


if __name__ == '__main__':
    unittest.main()
