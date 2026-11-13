import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.rssm_stage_preview import stage_preview_spec, stage_preview_batch, generate_stage_preview
from turkeyworld.train_rssm_dynamics import RSSMTraining


class PreviewData:
    def __init__(self):
        self.episodes = [{'frames': np.zeros((10, 1))}, {'frames': np.zeros((500, 1))}]
        self.calls = []

    def batch(self, count, *, context, horizon, indices):
        self.calls.append((count, context, horizon, indices.copy()))
        states = torch.full((1, context+horizon, 20), .99)
        states[:, :context] = .25
        states[:, context-1, 0] = .26
        images = torch.zeros(1, context, 3, 32, 32)
        source = torch.arange(1, horizon+1).float()[None, :, None, None, None].expand(1, horizon, 3, 32, 32)/500
        return {'states': states, 'context_frames': images, 'target_frames': source,
                'actions': torch.zeros(1, horizon, 4), 'map': torch.zeros(1, 16, 16)}


class PreviewClient(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls, self.render_offsets = [], []
        self.initializations = 0

    def encode_map(self, grid):
        return grid.new_zeros(1, 1)

    def initialize_from_image(self, images, states, features):
        self.initializations += 1
        self.observed_context = images.shape[1]
        self.asserted_source_context = states.detach().clone()
        return images.new_zeros(1, 1)

    def transition(self, state, action, features, memory):
        self.calls.append((state.clone(), memory.clone()))
        torch.rand(())  # The trainer's preview wrapper must restore this RNG.
        result = state.clone(); result[:, 0] += .001
        return {'state': result, 'memory': memory+1, 'features': memory+1}

    def render(self, state, features, fmap):
        self.render_offsets.append(int(features[0, 0]))
        return (features/500)[:, :, None, None].expand(1, 3, 32, 32)


class StagePreviewTests(unittest.TestCase):
    def test_moe_active_31_stage_gate_overrides_historical_xy_gate_and_old_combat_clock(self):
        from turkeyworld.rssm_moe_protocol import MOE_HORIZONS
        self.assertEqual(len(MOE_HORIZONS),31)
        state={'rssm_training':{'combat_completed':999,
            'loss_gated_curriculum':{'current_horizon':128,'sha256':'historical'},
            'moe_training':{'current_horizon':4,'stage_index':0,'updates':0,'sha256':'moe'}}}
        for index,horizon in enumerate(MOE_HORIZONS):
            state['rssm_training']['moe_training'].update(current_horizon=horizon,stage_index=index)
            spec=stage_preview_spec(state)
            self.assertEqual(spec['horizon'],horizon);self.assertEqual(spec['plan_sha256'],'moe')
            self.assertEqual(spec['context'],1)
            self.assertEqual(spec['display_offsets'][0],1);self.assertEqual(spec['display_offsets'][-1],horizon)
            self.assertEqual(len(set(spec['display_offsets'])),4)
        state['rssm_training']['moe_training'].update(current_horizon=8,updates=16)
        self.assertEqual(stage_preview_spec(state)['context'],2)
        for horizon in (True,12,24,451):
            state['rssm_training']['moe_training']['current_horizon']=horizon
            with self.assertRaises(ValueError):stage_preview_spec(state)

    def test_moe_h4_h8_h48_full_own_rollout_and_actual_four_offsets(self):
        for horizon in (4,8,48):
            spec=stage_preview_spec({'rssm_training':{'moe_training':
                {'current_horizon':horizon,'updates':0,'sha256':'moe','stage_index':0}}})
            data,client=PreviewData(),PreviewClient()
            generated,source=generate_stage_preview(client,stage_preview_batch(data,spec),spec)
            self.assertEqual(len(client.calls),horizon)
            self.assertEqual(client.render_offsets,spec['display_offsets'])
            self.assertEqual(generated.shape,(4,3,32,32));self.assertEqual(source.shape,generated.shape)

    def test_trainer_moe_metadata_uses_current_plan_without_reusing_historical_xy_gate(self):
        with tempfile.TemporaryDirectory()as folder:
            trainer=RSSMTraining.__new__(RSSMTraining)
            trainer.root=Path(folder);trainer.device=torch.device('cpu');trainer.rng=np.random.default_rng(17)
            trainer.args=SimpleNamespace(preview_assets_plan='pinned',renderer_checkpoint='old.pt',sr='old_sr.pt')
            trainer.preview_model=PreviewClient();trainer.long_val=PreviewData()
            trainer.sr=nn.Identity();trainer.sr_metadata={'training_steps':38100}
            trainer._preview_renderer_checkpoint='latest6000.pt';trainer._preview_sr_checkpoint='sr38100.pt'
            trainer.state={'global_training_step':5350,'rssm_training':{'phase':'moe','combat_completed':999,
                'loss_gated_curriculum':{'current_horizon':128,'sha256':'historical'},
                'moe_training':{'current_horizon':4,'updates':0,'stage_index':0,'sha256':'moe'}}}
            trainer.publish=lambda:None
            pinned={'renderer_refinement_updates':6000,'renderer_training_steps':22200,'sr_training_steps':38100}
            saved=[]
            def save(preview,*args,**kwargs):saved.append(preview);return preview['metadata']
            source=nn.Linear(2,2);rng=torch.get_rng_state().clone()
            with patch.object(trainer,'initialize_preview_assets',return_value=pinned),\
                 patch('turkeyworld.vjepa_validation.save_autoregressive_preview',side_effect=save):
                trainer.preview(source)
            metadata=saved[0]['metadata']
            self.assertEqual(metadata['horizon'],4);self.assertEqual(metadata['generated_rollout_transitions'],4)
            self.assertEqual(metadata['display_offsets'],[1,2,3,4]);self.assertEqual(metadata['context'],1)
            self.assertEqual(metadata['moe_training_plan_sha256'],'moe')
            self.assertIsNone(metadata['loss_gated_curriculum_plan_sha256'])
            self.assertEqual(metadata['preview_assets'],pinned)
            self.assertIn('NPR 22200 (phase 6000)',metadata['caption'])
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))

    def test_active_stage_overrides_old_replay_next_horizon_and_preserves_context_policy(self):
        self.assertEqual(stage_preview_spec({'rssm_training': {}})['horizon'], 4)
        state = {'rssm_training': {'combat_completed': 16,
            'coordinate_curriculum': {'next_horizon': 450}, 'replay_curriculum': {'last_horizon': 450},
            'loss_gated_curriculum': {'current_horizon': 96, 'next_horizon': 128, 'stage_index': 3}}}
        spec = stage_preview_spec(state)
        self.assertEqual(spec['horizon'], 96)
        self.assertEqual(spec['context'], 2)
        self.assertEqual(spec['display_offsets'], [1, 33, 64, 96])
        for value in [True, 4, 80, 451]:
            changed = copy.deepcopy(state); changed['rssm_training']['loss_gated_curriculum']['current_horizon'] = value
            with self.assertRaises(ValueError): stage_preview_spec(changed)

    def test_all96_transitions_use_own_state_memory_and_only_four_labelled_frames_are_rendered(self):
        spec = {'horizon': 96, 'context': 2, 'display_offsets': [1, 33, 64, 96]}
        data, client = PreviewData(), PreviewClient()
        sampled = stage_preview_batch(data, spec)
        generated, source = generate_stage_preview(client, sampled, spec)
        self.assertEqual(data.calls[0][3].tolist(), [[1, 0]])
        self.assertEqual(len(client.calls), 96)
        self.assertEqual(client.initializations, 1)
        self.assertEqual(client.observed_context, 2)
        self.assertEqual(client.render_offsets, spec['display_offsets'])
        self.assertEqual(generated.shape, (4, 3, 32, 32))
        for index, (state, memory) in enumerate(client.calls):
            self.assertAlmostEqual(float(state[0, 0]), .26+.001*index, places=5)
            self.assertEqual(int(memory[0, 0]), index)
        torch.testing.assert_close(source[:, 0, 0, 0], torch.tensor(spec['display_offsets'])/500)
        self.assertFalse(generated.requires_grad)

    def test_trainer_refresh_uses_new_current_stage_and_keeps_rng_training_caches_and_asset_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = RSSMTraining.__new__(RSSMTraining)
            trainer.root = Path(folder); trainer.device = torch.device('cpu')
            trainer.rng = np.random.default_rng(17)
            trainer.args = SimpleNamespace(preview_assets_plan='pinned', renderer_checkpoint='old.pt', sr='old_sr.pt')
            trainer.preview_model = PreviewClient(); trainer.long_val = PreviewData()
            trainer.short_val = SimpleNamespace(batch=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Wrong short data')))
            trainer._preview_renderer_checkpoint = 'updated_renderer6000.pt'; trainer._preview_sr_checkpoint = 'updated_sr.pt'
            trainer.sr = nn.Identity(); trainer.sr_metadata = {'training_steps': 38100}
            trainer.state = {'global_training_step': 5089, 'rssm_training': {'phase': 'combat', 'combat_completed': 16,
                'loss_gated_curriculum': {'current_horizon': 16, 'stage_index': 0, 'sha256': 'gate', 'last_training_horizon': 16}}}
            published, saved = [], []
            trainer.publish = lambda: published.append(True)
            pinned = {'renderer_refinement_updates': 6000, 'renderer_training_steps':22200,'sr_training_steps': 38100}
            def save(preview, *args, **kwargs):
                saved.append(preview)
                return preview['metadata']
            source = nn.Linear(2, 2)
            source.register_buffer('_spawn_walkable', torch.ones(64, 16, 16, dtype=torch.bool), persistent=False)
            weights = {key: value.clone() for key, value in source.state_dict().items()}
            cache = source._spawn_walkable
            rng, private = torch.get_rng_state().clone(), copy.deepcopy(trainer.rng.bit_generator.state)
            with patch.object(trainer, 'initialize_preview_assets', return_value=pinned), \
                 patch('turkeyworld.vjepa_validation.save_autoregressive_preview', side_effect=save):
                trainer.preview(source)
                trainer.state['rssm_training']['loss_gated_curriculum'].update(current_horizon=32, stage_index=1)
                trainer.state['global_training_step'] += 1
                trainer.preview(source)
            self.assertEqual([item['metadata']['horizon'] for item in saved], [16, 32])
            self.assertEqual(saved[1]['metadata']['generated_rollout_transitions'], 32)
            self.assertEqual(saved[1]['metadata']['display_offsets'], [1, 11, 22, 32])
            self.assertEqual(saved[1]['metadata']['context'], 2)
            self.assertEqual(saved[1]['metadata']['preview_assets'], pinned)
            self.assertEqual(saved[1]['metadata']['renderer_checkpoint'], 'updated_renderer6000.pt')
            self.assertEqual(saved[1]['metadata']['sr_checkpoint'], 'updated_sr.pt')
            self.assertEqual(saved[1]['generated'].shape[0], 4)
            self.assertEqual(saved[1]['source'].shape[0], 4)
            self.assertIn('H32', saved[1]['metadata']['caption'])
            self.assertIn('NPR 22200 (phase 6000)',saved[1]['metadata']['caption'])
            self.assertEqual(len(trainer.preview_model.calls), 48)
            self.assertEqual(len(published), 2)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(trainer.rng.bit_generator.state, private)
            self.assertIs(source._spawn_walkable, cache)
            self.assertTrue(all(torch.equal(value, source.state_dict()[key]) for key, value in weights.items()))


if __name__ == '__main__':
    unittest.main()
