from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from test_cache_sr_neural_pairs import FrozenRenderer, identity
from test_sr_neural_pairs_data import source_snapshots
from turkeyworld.cache_sr_moe_pairs import (build_sr_moe_cache, render_profiled_cache_batch,
    predict_profiled_hypotheses, source_navigation_only)
from turkeyworld.enemy_decoder_data import EnemyDecoderData
from turkeyworld.small_enemy_data import digest
from turkeyworld.sr_neural_pairs_data import collect_source_pairs


@dataclass(frozen=True)
class Profile:
    navigation_only: bool = False


class ProfiledWorld(FrozenRenderer):
    def __init__(self):
        super().__init__()
        self.config = Profile()
        self.vjepa_backend = nn.Module()
        self.vjepa_backend.config = Profile()
        self.profiles = []

    def transition(self, state, action, features, memory):
        self.profiles.append((self.config.navigation_only, self.vjepa_backend.config.navigation_only))
        result = super().transition(state, action, features, memory)
        predicted = state.clone()
        predicted[:, 0] += .01 * action[:, 0]
        result['state'] = predicted
        result['features'] = result['features'] + (.2 if self.config.navigation_only else .05)
        return result


def mixed_batch():
    batch = {'state': torch.ones(4, 20)*.5, 'map': torch.zeros(4, 16, 16, dtype=torch.long),
        'dynamic': torch.tensor([True, True, True, False]),
        'previous_state': torch.ones(4, 20)*.2, 'previous_rgb': torch.ones(4, 3, 32, 32)*.3,
        'action': torch.ones(4, 4)*.1, 'lr_source': torch.ones(4, 3, 32, 32)*.9,
        'hr_source': torch.ones(4, 3, 128, 128), 'mask_hr': torch.ones(4, 1, 128, 128)}
    # A dead combat NPC must not silently select the navigation environment.
    batch['state'][1, 12:14] = 0
    records = [{'stratum': 'hp0', 'navigation_only': True}, {'stratum': 'corpse', 'navigation_only': False},
               {'stratum': 'player_hurt', 'navigation_only': False}, {'stratum': '17_64', 'navigation_only': False}]
    return batch, records


class ProfiledSRCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_provenance_partitions_priors_static_zero_and_future_labels_never_condition(self):
        model = ProfiledWorld().eval()
        batch, records = mixed_batch()
        old_config, old_backend = model.config, model.vjepa_backend.config
        weights = deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
        result = render_profiled_cache_batch(model, batch, records=records, include_codec=False)
        self.assertEqual(model.profiles, [(False, False), (True, True)])
        self.assertEqual([len(x[0]) for x in model.observations], [2, 1])
        self.assertEqual(result['valid'].tolist(), [True, True, True, False])
        self.assertTrue(np.array_equal(result['pred_state'][3], np.zeros(20, np.float32)))
        self.assertEqual(float(model.render_queries[0][1][-1].abs().sum()), 0.)
        self.assertIs(model.config, old_config); self.assertIs(model.vjepa_backend.config, old_backend)
        poisoned = {**batch, 'lr_source': torch.zeros_like(batch['lr_source']),
            'hr_source': torch.full_like(batch['hr_source'], float('nan')),
            'mask_hr': torch.full_like(batch['mask_hr'], float('nan'))}
        second = render_profiled_cache_batch(model, poisoned, records=records, include_codec=False)
        self.assertTrue(np.array_equal(result['lr_npr'], second['lr_npr']))
        self.assertTrue(np.array_equal(result['pred_state'], second['pred_state']))
        # Future pose is a permitted image query but never a prior/hint input.
        poisoned['state'] = torch.full_like(batch['state'], 9.)
        hint = predict_profiled_hypotheses(model, poisoned, records=records)
        self.assertTrue(np.array_equal(result['pred_state'], hint['pred_state']))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in weights.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with patch.object(model, 'render', side_effect=RuntimeError('render failure')):
            with self.assertRaisesRegex(RuntimeError, 'render failure'):
                render_profiled_cache_batch(model, batch, records=records)
        self.assertIs(model.config, old_config); self.assertIs(model.vjepa_backend.config, old_backend)

    def test_profile_requires_provenance_and_rejects_conflicting_metadata(self):
        self.assertTrue(source_navigation_only({'stratum': 'hp0', 'navigation_only': True}))
        self.assertTrue(source_navigation_only({'source_path': r'E:\data\navigation\rgb\train\x.npz'}))
        self.assertFalse(source_navigation_only({'stratum': 'corpse', 'enemies_enabled': True}))
        self.assertTrue(source_navigation_only({'enemies_enabled': False}))
        with self.assertRaisesRegex(ValueError, 'Contradictory'):
            source_navigation_only({'navigation_only': True, 'enemies_enabled': True})
        with self.assertRaisesRegex(ValueError, 'stratum/life'):
            source_navigation_only({'stratum': 'navigation'})
        with self.assertRaisesRegex(ValueError, 'provenance'):
            render_profiled_cache_batch(ProfiledWorld(), mixed_batch()[0], records=[])
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'unknown_episode.npz'
            np.savez_compressed(path, navigation_only=np.bool_(True), enemies_enabled=np.bool_(False))
            self.assertTrue(source_navigation_only({'source_path': str(path), 'stratum': 'hp0'}))
            static = Path(temporary)/'static_unknown.npz'
            np.savez_compressed(static, stage='static_enemy',
                snapshot_states=json.dumps([{'player': {'hp': 0}}]))
            self.assertFalse(source_navigation_only({'source_path': str(static), 'frame': 0}))
            np.savez_compressed(static, stage='navigation',
                snapshot_states=json.dumps([{'player': {'hp': 100}}]))
            self.assertTrue(source_navigation_only({'source_path': str(static), 'frame': 0}))

    def test_fresh_pairs_and_sidecar_preserve_source_bytes_and_reader_resume_contract(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); source = root/'source'
            for split in ('train', 'val'):
                path = source_snapshots(source, split)
                with np.load(path, allow_pickle=False) as arrays:
                    values = {k: arrays[k] for k in arrays.files}
                # This fixture's Game.get_state intentionally has no env flag;
                # an explicit Source header pins its original combat-enabled world.
                np.savez_compressed(path, **values, navigation_only=np.bool_(False))
            pairs = root/'source_pairs'
            original = collect_source_pairs(pairs, [source], train_count=8, val_count=8,
                workers=1, shard_size=3)
            manifest_bytes = (pairs/'manifest.json').read_bytes()
            world = ProfiledWorld().train(); world.flow_renderer.codec.eval()
            modes = {n: m.training for n, m in world.named_modules()}
            flags = [p.requires_grad for p in world.parameters()]
            rng = torch.get_rng_state().clone()
            settings = {**identity(), 'world_step': 5947, 'renderer_step': 29609}
            kwargs = dict(cache_identity=settings, world_sha256=settings['world_sha256'],
                          world_step=5947, batch_size=2)
            result = build_sr_moe_cache(pairs, root/'paired', world,
                hypotheses_output=root/'hypotheses', **kwargs)
            self.assertEqual(result['paired']['records'], original['records'])
            self.assertEqual(manifest_bytes, (pairs/'manifest.json').read_bytes())
            self.assertEqual(result['hypotheses']['world_step'], 5947)
            self.assertEqual(result['paired']['cache_identity']['renderer_step'], 29609)
            for entry in original['shards']:
                self.assertEqual(digest(pairs/entry['path']), entry['sha256'])
                with np.load(pairs/entry['path']) as a, np.load(root/'paired'/entry['path']) as b:
                    self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a.files))
            data = EnemyDecoderData(root/'paired', 'train', hypotheses_root=root/'hypotheses')
            heldout = EnemyDecoderData(root/'paired', 'val', hypotheses_root=root/'hypotheses')
            self.assertFalse(data.scene_keys & heldout.scene_keys)
            batch = data.batch(4)
            self.assertEqual(batch['hypothesis'].shape, (4, 21))
            state = deepcopy(data.sampler_state()); expected = data.batch(4)['row_index']
            data.restore_sampler(state)
            self.assertTrue(torch.equal(expected, data.batch(4)['row_index']))
            calls = len(world.render_queries)
            build_sr_moe_cache(pairs, root/'paired', world,
                hypotheses_output=root/'hypotheses', **kwargs)
            self.assertEqual(calls, len(world.render_queries))
            # Existing paired cache can acquire a missing aligned sidecar.
            build_sr_moe_cache(pairs, root/'paired', world,
                hypotheses_output=root/'second_hypotheses', **kwargs)
            self.assertEqual(calls, len(world.render_queries))
            self.assertEqual(modes, {n: m.training for n, m in world.named_modules()})
            self.assertEqual(flags, [p.requires_grad for p in world.parameters()])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            json.dumps(result, allow_nan=False)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                build_sr_moe_cache(pairs, root/'paired', world,
                    **{**kwargs, 'world_sha256': 'different-world'})


if __name__ == '__main__':
    unittest.main()
