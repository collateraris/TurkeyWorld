"""Enemy renderer data quotas, held-out guards and real codec-safe validation."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.engine import BEHAVIOR_VERSION
from turkeyworld.renderer_enemy_data import EnemyFrames
from turkeyworld.renderer_enemy_validation import candidate_assessment,evaluate_enemy_renderer,renderer_metrics
from test_renderer_enemy_objectives import tiny_model,batch as source_batch


def fake_bounds(state,**kwargs):
    boxes=torch.tensor([.1,.1,.8,.8])[None,None].expand(len(state),1,4).clone()
    partial=state[:,17]>.5
    boxes[partial,0,0]=-.2
    return {'unclipped_boxes':boxes}


def corpus(root,empty,split):
    for kind,path in [('static',root/'static_enemy/snapshots'/split),
                      ('moving',root/'moving1/rgb'/split),('navigation',empty/'short/rgb'/split)]:
        path.mkdir(parents=True)
        n=8 if kind=='static' else 3
        state=np.zeros((n,20),np.float32);state[:,0:2]=.2;state[:,3:6]=1;state[:,8:10]=.3
        state[:,12]=.6;state[:,13]=1
        masks=np.zeros((n,32,32),bool)
        if kind=='static':
            masks[0,:16,:16]=True
            masks[1,:4,:4]=True;state[1,17]=.9
            masks[2,:2,:2]=True;state[2,8]=.8
            masks[3,:4,:4]=True
            masks[4,:4,:4]=True;state[4,12:14]=0
            # Row5 is active but invisible; row6 is an invisible corpse.
            state[6,12:14]=0;state[7,12:14]=0
        elif kind=='moving':masks[1:,:4,:4]=True
        else:state[:,12:14]=0
        background=np.full((n,32,32,3),40,np.uint8)
        frames=background.copy();frames[masks]=200
        values=dict(states=state,frames=frames,object_masks=masks,background_frames=background,
            map=np.zeros((16,16),np.uint8),split=np.asarray(split),behavior_version=np.asarray(BEHAVIOR_VERSION),
            world_transition_compatible=np.asarray(kind!='static'),seed=np.asarray(10 if split=='train' else 20))
        if kind!='static':values['actions']=np.zeros((n-1,4),np.float32)
        np.savez_compressed(path/'episode.npz',**values)


class EnemyDataAndValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_data_quotas_active_invisible_negative_and_sampler_exact_resume(self):
        with TemporaryDirectory() as tmp,patch('turkeyworld.renderer_enemy_data.project_sprite_bounds',side_effect=fake_bounds):
            root,empty=Path(tmp)/'source',Path(tmp)/'empty'
            corpus(root,empty,'train')
            data=EnemyFrames(root,empty,'train')
            self.assertEqual(len(data.buckets['empty_hidden']),1)
            for index in data.buckets['empty_hidden']:
                values=data.batch(1,indices=[index])
                self.assertGreater(float(values['state'][0,13]),.5)
                self.assertGreater(float(values['state'][0,12]),0)
                self.assertFalse(values['object_mask'].any())
            for _ in range(100):data.batch(8)
            self.assertEqual(data.case_counts['empty_navigation'],100)
            self.assertEqual(data.case_counts['empty_hidden'],100)
            self.assertEqual(data.draws,800)
            saved=copy.deepcopy(data.sampler_state())
            a=data.batch(8)['row_index']
            data.restore_sampler(saved)
            b=data.batch(8)['row_index']
            self.assertTrue(torch.equal(a,b))
            dynamic=next(i for i,row in enumerate(data.rows) if data.files[row['file']]['kind']=='moving')
            row=data.batch(1,indices=[dynamic])
            self.assertTrue(row['dynamic'][0])
            self.assertFalse(torch.equal(row['previous_rgb'],row['target']))
            self.assertEqual(data.identity['empty_hidden_definition'].startswith('Exact Source mask0'),True)

    def test_scene_overlap_exclusion_and_corrupted_mask_fail_closed(self):
        with TemporaryDirectory() as tmp,patch('turkeyworld.renderer_enemy_data.project_sprite_bounds',side_effect=fake_bounds):
            root,empty=Path(tmp)/'source',Path(tmp)/'empty'
            corpus(root,empty,'train');corpus(root,empty,'val')
            train=EnemyFrames(root,empty,'train')
            # All exact fixture poses overlap: strict exclusion must fail
            # rather than silently reuse train scenes for a held-out claim.
            with self.assertRaisesRegex(ValueError,'needs all'):
                EnemyFrames(root,empty,'val',exclude_scene_keys=train.scene_keys)
            path=root/'static_enemy/snapshots/train/episode.npz'
            with np.load(path) as saved:values={key:saved[key] for key in saved.files}
            values['object_masks'][0,31,31]=True
            np.savez_compressed(path,**values)
            with self.assertRaisesRegex(ValueError,'not bitexact'):EnemyFrames(root,empty,'train')

    def test_npc_metric_improvement_cannot_hide_empty_background_regression(self):
        source=torch.zeros(4,3,32,32);mask=torch.zeros(4,1,32,32)
        mask[:2,:,8:24,8:24]=1;source[:2,:,8:24,8:24]=1
        background=torch.zeros_like(source)
        categories=['large','corpse','empty_navigation','empty_hidden']
        dynamic=torch.tensor([False,True,True,True])
        old=renderer_metrics(background,source,mask,background,categories,dynamic)
        improved=renderer_metrics(source,source,mask,background,categories,dynamic)
        self.assertEqual(old['visible_enemy']['closer_to_source_than_without_npc_rate'],0)
        self.assertEqual(improved['visible_enemy']['closer_to_source_than_without_npc_rate'],1)
        baseline={'base':old,'rf1':old};candidate={'base':improved,'rf1':improved}
        self.assertTrue(candidate_assessment(candidate,baseline)['validated_improvement'])
        bad=copy.deepcopy(candidate)
        bad['rf1']['categories']['empty_navigation']['mae_mean']=.01
        self.assertFalse(candidate_assessment(bad,baseline)['validated_improvement'])

    def test_actual_tiny_validation_reports_static_dynamic_and_codec_without_training(self):
        model=tiny_model();batch=source_batch((False,True,False,True))
        class Data:
            split='val'
            rows=[{'category':key} for key in ('large','partial','corpse','empty_navigation')]
            def batch(self,n,indices):return {key:value[indices] for key,value in batch.items()}
            def panel_fingerprint(self,panel):return 'immutable_test_panel'
        before={name:value.clone() for name,value in model.state_dict().items()}
        report,arrays=evaluate_enemy_renderer(model,Data(),[0,1,2,3],batch_size=3)
        self.assertEqual(report['base']['static_zero_features']['frames'],2)
        self.assertEqual(report['rf1']['dynamic_past_prior_features']['frames'],2)
        self.assertIn('foreground_mae',report['codec_reference']['visible_enemy'])
        self.assertFalse(report['gameplay_qualified'])
        for name,value in model.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(model.flow_renderer.codec.training)


if __name__=='__main__':unittest.main()

