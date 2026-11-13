from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.engine import BEHAVIOR_VERSION, Game
from turkeyworld.small_enemy_data import _observe, digest
from turkeyworld.sr_neural_pairs_data import (
    CACHE_PROTOCOL, SOURCE_ARRAY_KEYS, SourcePairData, SemanticSRData,
    collect_source_pairs, replay_source_job, source_cohort, write_shard,
)


def source_snapshots(root, split, *, seed=None):
    seed = (173 if split == 'train' else 274) if seed is None else seed
    game = Game(seed=seed,endless=True)
    rng = np.random.default_rng(seed)
    snapshots = []; arrays = {k:[] for k in ('states','frames','object_masks','background_frames')}
    for cohort in ('near','mid','far','guard'):
        for i in range(20):
            state = game.get_state()
            direction = 1 if split == 'train' else -1
            px = 2.5 if split == 'train' else 13.5
            distance = {'near':1.,'mid':3.25,'far':7.,'guard':1.}[cohort]
            state['player'].update(x=px,y=2.5,angle=(0 if direction == 1 else np.pi)+float(rng.uniform(-.16,.16)),
                hp=100.,shot_flash=0.,cooldown=0.)
            state['enemies_enabled'] = cohort != 'guard'
            state['enemies'][0].update(x=px+direction*distance,y=2.5,
                hp=0. if cohort=='guard' else 60.,alive=cohort!='guard',
                death_timer=.6 if cohort=='guard' else 0.,hurt_timer=0.,vx=0.,vy=0.)
            state['time'] = float(i/30)
            game.set_state(state)
            low,_,background,_,mask,_ = _observe(game)
            if cohort != 'guard' and not mask.any():
                raise AssertionError('Fixture enemy must be genuinely visible')
            snapshots.append(game.get_state())
            for key,value in {'states':game.state_vector(),'frames':low,'object_masks':mask,
                              'background_frames':background}.items():
                arrays[key].append(value)
    target = Path(root)/'snapshots'/split/f'episode_{seed}.npz'
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**{k:np.asarray(v) for k,v in arrays.items()},map=game.map,seed=seed,
        split=split,behavior_version=BEHAVIOR_VERSION,snapshot_states=json.dumps(snapshots),
        world_transition_compatible=False)
    return target


class SemanticSRDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_true_source_pairs_shards_split_guard_and_no_engine_reader(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary)/'source'
            paths = [source_snapshots(source,s) for s in ('train','val')]
            before = [digest(p) for p in paths]
            output = Path(temporary)/'pairs'
            report = collect_source_pairs(output,[source],train_count=8,val_count=8,workers=1,shard_size=3)
            self.assertEqual(report['status'],'complete')
            self.assertEqual(report['groups']['train']['cohorts'],dict.fromkeys(('near','mid','far','guard'),2))
            self.assertEqual(report['groups']['val']['cohorts'],dict.fromkeys(('near','mid','far','guard'),2))
            train = SourcePairData(output,'train'); val = SourcePairData(output,'val',exclude_scene_keys=train.scene_keys)
            state = deepcopy(val.sampler_state()); panel = val.fixed_panel()
            with patch.object(Game,'render',side_effect=AssertionError('Reader must not render')), \
                    patch.object(Game,'step',side_effect=AssertionError('Reader must not simulate')):
                batch = val.batch(8,indices=panel)
            self.assertEqual(batch['lr_source'].shape,(8,3,32,32))
            self.assertEqual(batch['hr_source'].shape,(8,3,128,128))
            self.assertEqual(batch['mask_hr'].shape,(8,1,128,128))
            self.assertEqual(batch['map'].shape,(8,16,16))
            self.assertTrue(torch.equal(batch['mask_lr'].bool(),(batch['lr_source']!=batch['background']).any(1,keepdim=True)))
            self.assertTrue(torch.equal(batch['mask_hr'].bool(),(batch['hr_source']!=batch['hr_background']).any(1,keepdim=True)))
            self.assertEqual(val.sampler_state(),state)
            self.assertEqual(float(batch['previous_rgb'].abs().sum()),0.)
            self.assertFalse(batch['dynamic'].any())
            self.assertFalse(train.scene_keys & val.scene_keys)
            self.assertFalse({r['rgb_sha256'] for r in train.rows}&{r['rgb_sha256'] for r in val.rows})
            self.assertEqual(before,[digest(p) for p in paths])
            raw = (output/'manifest.json').read_bytes()
            collect_source_pairs(output,[source],train_count=8,val_count=8,workers=1,shard_size=3)
            self.assertEqual(raw,(output/'manifest.json').read_bytes())
            json.dumps(report,allow_nan=False)

    def test_balanced_private_sampler_exact_resume_and_cache_input_domain(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary)/'source'
            for split in ('train','val'):
                source_snapshots(source,split)
            output = Path(temporary)/'pairs'
            report = collect_source_pairs(output,[source],train_count=8,val_count=8,workers=1,shard_size=8)
            with self.assertRaisesRegex(ValueError,'actual frozen-neural'):
                SemanticSRData(output,'train')
            data = SourcePairData(output,'train')
            state = data.sampler_state()
            first = data.batch(13)['row_index'].tolist()
            self.assertEqual(data.draws,13)
            self.assertLessEqual(max(data.case_counts.values())-min(data.case_counts.values()),1)
            data.restore_sampler(state)
            self.assertEqual(first,data.batch(13)['row_index'].tolist())
            cache = Path(temporary)/'cache'; shards = []; records = []
            for split in ('train','val'):
                selected = [r for r in report['records'] if r['split']==split]
                samples = []
                for row in selected:
                    with np.load(output/row['shard']) as z:
                        samples.append((row,{k:z[k][row['offset']] for k in SOURCE_ARRAY_KEYS}))
                lr = np.stack([s['lr_source'] for _,s in samples])
                new,shard = write_shard(cache,split,0,samples,extra_arrays={'lr_npr':255-lr,'codec_lr':lr})
                records.extend(new); shards.append(shard)
            neural_manifest = {**report,'protocol':CACHE_PROTOCOL,'shards':shards,'records':records,
                'cache_identity':{'renderer_sha256':'test-frozen-renderer','future_rgb_features':False,
                                  'paired_same_pose_map_tick':True}}
            (cache/'manifest.json').write_text(json.dumps(neural_manifest),encoding='utf-8')
            cached = SemanticSRData(cache,'val').batch(4,indices=[0,1,2,3])
            self.assertTrue(torch.allclose(cached['lr_npr'],1-cached['lr_source']))
            self.assertTrue(torch.equal(cached['codec_lr'],cached['lr_source']))
            self.assertEqual(cached['hr_source'].shape,(4,3,128,128))

    def test_dynamic_target_uses_actual_previous_tick_and_source_actions(self):
        with TemporaryDirectory() as temporary:
            game = Game(seed=173,endless=True)
            initial = game.get_state(); values = []; actions = []; events = []
            values.append((*_observe(game),game.state_vector()))
            for action in [np.array([0.,0.,0.,0.],np.float32),np.array([.2,0.,0.,0.],np.float32)]:
                actions.append(action); events.append(Game.event_vector(game.step(action,dt=1/30)))
                values.append((*_observe(game),game.state_vector()))
            path = Path(temporary)/'episode.npz'
            np.savez_compressed(path,states=np.stack([v[6] for v in values]),frames=np.stack([v[0] for v in values]),
                background_frames=np.stack([v[2] for v in values]),object_masks=np.stack([v[4] for v in values]),
                map=game.map,actions=np.stack(actions),events=np.stack(events),dt=1/30,initial_state=json.dumps(initial))
            info = {'path':str(path),'sha256':digest(path),'seed':173,'dynamic':True}
            result = replay_source_job({'source':info,'rows':[{'frame':2}]})[0][1]
            self.assertTrue(np.array_equal(result['previous_rgb'],values[1][0]))
            self.assertTrue(np.array_equal(result['previous_state'],values[1][6]))
            self.assertTrue(np.array_equal(result['state'],values[2][6]))
            self.assertTrue(np.array_equal(result['action'],actions[1]))
            self.assertTrue(np.array_equal(result['hr_source'],values[2][1]))
            self.assertTrue(bool(result['dynamic']))

    def test_distance_boundaries_and_hidden_enemy_never_labeled_visible(self):
        state = np.zeros(20,np.float32); state[4]=1; state[12]=1; state[13]=1
        mask = np.zeros((32,32),bool); mask[15,15]=True
        for distance,expected in [(1.99,'near'),(2.,'mid'),(4.99,'mid'),(5.,'far')]:
            state[8]=distance/16
            self.assertEqual(source_cohort(state,mask)[0],expected)
        self.assertEqual(source_cohort(state,np.zeros_like(mask))[:2],('guard','occluded'))
        state[4]=0
        self.assertEqual(source_cohort(state,mask)[:2],('guard','hp0'))

    def test_source_seed_overlap_fails_closed(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary)/'source'
            for split in ('train','val'):
                source_snapshots(source,split,seed=173)
            with self.assertRaisesRegex(ValueError,'Source seeds'):
                collect_source_pairs(Path(temporary)/'pairs',[source],train_count=4,val_count=4,workers=1)


if __name__ == '__main__':
    unittest.main()
