import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_renderer_enemy_training import corpus, fake_bounds
from turkeyworld.renderer_combat_data import CombatRendererFrames


def mixture(root, appearance, split):
    oldempty=root/'oldempty'
    corpus(appearance,oldempty,split)
    for source,target in ((appearance/'moving1/rgb'/split,root/'moving1/rgb'/split),
                          (oldempty/'short/rgb'/split,root/'navigation/rgb'/split)):
        target.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/'episode.npz',target/'episode.npz')
    path=root/'full_game/rgb'/split/'episode.npz';path.parent.mkdir(parents=True)
    with np.load(root/'moving1/rgb'/split/'episode.npz') as data:
        values={key:data[key] for key in data.files}
    for key in ('states','frames','object_masks','background_frames'):
        values[key]=np.repeat(values[key][1:2],8,axis=0)
    states=values['states'];states[:,0]+=np.arange(8)*.001
    states[4,12:14]=0;states[5:,12]=.6;states[5:,13]=1
    states[3:,4]=.9
    events=np.zeros((7,4),np.float32)
    events[0,:2]=1;events[1,0]=1;events[2,3]=1;events[3,2]=1
    values.update(actions=np.zeros((7,4),np.float32),events=events,seed=np.asarray(31 if split=='train' else 41))
    np.savez_compressed(path,**values)


class CombatRendererDataTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_balanced_actual_source_event_membership_and_exact_sampler_resume(self):
        with TemporaryDirectory() as tmp,patch('turkeyworld.renderer_enemy_data.project_sprite_bounds',side_effect=fake_bounds):
            root,appearance=Path(tmp)/'combat',Path(tmp)/'appearance';mixture(root,appearance,'train')
            data=CombatRendererFrames(root,appearance,'train')
            self.assertTrue(all(data.mix_buckets['combat'][key] for key in data.COMBAT_CASES))
            for _ in range(10):data.batch(16)
            self.assertEqual(data.mix_counts,{'combat':80,'appearance':40,'navigation':40})
            self.assertEqual(data.draws,160)
            self.assertEqual(set(data.cohort_counts['combat'].values()),{16})
            checkpoint=copy.deepcopy(data.sampler_state())
            expected=data.batch(16)['row_index']
            data.restore_sampler(checkpoint)
            self.assertTrue(torch.equal(expected,data.batch(16)['row_index']))
            self.assertTrue(all(row['frame']>=1 for row in data.rows if row['feature_policy']=='frozen_past_prior'))
            self.assertTrue(all(data.rows[index]['source_events'][0]==1 and data.rows[index]['source_events'][1]==0
                                for index in data.mix_buckets['combat']['miss']))
            self.assertTrue(all(data.rows[index]['frame']==5 for index in data.mix_buckets['combat']['respawn']))
            fixed=data.fixed_panel(8,seed=9)
            self.assertEqual(len(set(fixed)),8)
            self.assertEqual(fixed,data.fixed_panel(8,seed=9))

    def test_missing_combat_class_is_declared_and_never_fabricated(self):
        with TemporaryDirectory() as tmp,patch('turkeyworld.renderer_enemy_data.project_sprite_bounds',side_effect=fake_bounds):
            root,appearance=Path(tmp)/'combat',Path(tmp)/'appearance';mixture(root,appearance,'train')
            path=root/'full_game/rgb/train/episode.npz'
            with np.load(path) as source:values={key:source[key] for key in source.files}
            values['events'][:,0]=0
            np.savez_compressed(path,**values)
            data=CombatRendererFrames(root,appearance,'train')
            self.assertIn('miss',data.identity['missing_combat_cohorts'])
            data.batch(16)
            self.assertEqual(data.cohort_counts['combat']['miss'],0)
            self.assertEqual(data.mix_counts['combat'],8)
            invalid=copy.deepcopy(data.sampler_state());invalid['mix_counts']['combat']+=1
            with self.assertRaises(ValueError):data.restore_sampler(invalid)

    def test_train_scene_exclusion_and_finite_exact_mask_guards_stay_active(self):
        with TemporaryDirectory() as tmp,patch('turkeyworld.renderer_enemy_data.project_sprite_bounds',side_effect=fake_bounds):
            root,appearance=Path(tmp)/'combat',Path(tmp)/'appearance'
            mixture(root,appearance,'train');mixture(root,appearance,'val')
            train=CombatRendererFrames(root,appearance,'train')
            with self.assertRaisesRegex(ValueError,'needs all'):
                CombatRendererFrames(root,appearance,'val',exclude_scene_keys=train.scene_keys)
            path=root/'navigation/rgb/train/episode.npz'
            with np.load(path) as source:values={key:source[key] for key in source.files}
            values['object_masks'][0,0,0]=True;np.savez_compressed(path,**values)
            with self.assertRaisesRegex(ValueError,'not bitexact'):CombatRendererFrames(root,appearance,'train')


if __name__=='__main__':unittest.main()
