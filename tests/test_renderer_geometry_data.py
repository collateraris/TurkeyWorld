"""Independent all-depth Source pools, explicit quotas and held-out panels."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.renderer_geometry_data import (
    GeometryWallFrames,build_geometry_panel,GeometryPoolError,STRATA)


class Parent:
    def __init__(self,split='train',count=32):
        self.split=split;self.identity={'fingerprint':'Source_'+split}
        self.files=[{'path':'Source_'+split+'.npz','sha256':'actual_'+split+'_bytes'}]
        self.rows=[dict(file=0,frame=i,category='empty_navigation' if i%2==0 else 'medium')for i in range(count)]
        self.rng=np.random.default_rng(9);self.draws=48;self.fail=False;self.mutate=False;self.reads=[]
    def sampler_state(self):return dict(draws=self.draws,rng=deepcopy(self.rng.bit_generator.state))
    def restore_sampler(self,state):self.draws=state['draws'];self.rng.bit_generator.state=deepcopy(state['rng'])
    def panel_fingerprint(self,indices):
        return hashlib.sha256(json.dumps([self.identity,indices],sort_keys=True).encode()).hexdigest()
    def batch(self,count,*,indices=None):
        if indices is None:raise AssertionError('The main sampler must never be drawn')
        self.reads.append(list(indices))
        if self.mutate:self.draws+=1;self.rng.random()
        if self.fail:raise RuntimeError('Source read failed')
        state=torch.zeros(count,20);state[:,19]=torch.tensor(indices)
        for row,index in enumerate(indices):
            angle=((index//4)%4)*np.pi/2
            state[row,2]=np.sin(angle);state[row,3]=np.cos(angle)
        mask=torch.zeros(count,1,32,32)
        for row,index in enumerate(indices):
            if index%2:mask[row,0,12:14,12:14]=1
        return dict(state=state,map=torch.zeros(count,16,16),object_mask=mask,
                    target=torch.zeros(count,3,32,32),row_index=torch.tensor(indices))


def labels(grid,states,masks):
    count=len(states);del grid,masks
    def mask():return torch.zeros(count,1,32,32)
    near={'corner':mask(),'boundary':mask()};mid={'corner':mask(),'boundary':mask()}
    far={'corner':mask(),'thin_face':mask(),'boundary':mask()}
    for row,s in enumerate(states):
        index=int(s[19]);family=index%4
        if family==0:near['corner'][row,0,10,10]=1
        if family==1:mid['boundary'][row,0,10,10:12]=1
        if family==2:far['thin_face'][row,0,10,10]=1;far['boundary'][row,0,9,10]=1
        if family==3:far['boundary'][row,0,10,10:13]=1
        # Some views have close geometry AND distant corners, preventing a
        # simplistic near-first partition from preserving every wall family.
        if index>=16:
            near['boundary'][row,0,11,11]=1
            mid['corner'][row,0,11,12]=1
    return dict(bands={'near_0_2':near,'mid_2_5':mid},aggregates={'depth_ge_5':far})


class RendererGeometryDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def pool(self,parent=None,**kwargs):
        parent=Parent()if parent is None else parent
        with patch('turkeyworld.renderer_geometry_data.source_wall_depth_labels',side_effect=labels):
            return GeometryWallFrames(parent,candidate_count=32,label_batch_size=7,**kwargs)

    def test_train_only_disjoint_supported_pools_navigation_candidates_and_no_main_or_global_RNG_change(self):
        parent=Parent();before=parent.sampler_state();torch_rng=torch.get_rng_state().clone()
        numpy_rng=deepcopy(np.random.get_state())
        pool=self.pool(parent)
        self.assertEqual(before,parent.sampler_state())
        self.assertEqual(pool.identity['candidate_selection']['actual_empty_navigation'],16)
        self.assertEqual(pool.identity['candidate_selection']['shortage_fill_frames'],0)
        flattened=[i for values in pool.pool_buckets.values()for i in values]
        self.assertEqual(len(set(flattened)),len(flattened))
        self.assertEqual(set(flattened),set(range(32)))
        for name,values in pool.pool_buckets.items():
            self.assertTrue(values,name)
            self.assertTrue(all(pool.facts[i]['support'][name]for i in values))
        self.assertTrue(torch.equal(torch_rng,torch.get_rng_state()))
        self.assertEqual(numpy_rng[0],np.random.get_state()[0])
        np.testing.assert_array_equal(numpy_rng[1],np.random.get_state()[1])
        self.assertFalse(pool.identity['validation_used']);self.assertFalse(pool.identity['gpu_used'])
        json.dumps(pool.identity,allow_nan=False)
        pool.batch(20)
        self.assertEqual(pool.stratum_counts,dict.fromkeys(STRATA,5))
        self.assertEqual(pool.draws,20);self.assertEqual(before,parent.sampler_state())

    def test_exact_private_resume_and_explicit_pool_reads_do_not_advance_any_clock(self):
        first=self.pool();first.batch(7);saved=first.sampler_state();second=self.pool()
        second.restore_sampler(deepcopy(saved))
        a=first.batch(19)['row_index'];b=second.batch(19)['row_index']
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(first.sampler_state(),second.sampler_state())
        private=first.sampler_state();main=first.parent.sampler_state()
        ids=[next(iter(first.pool_buckets[name]))for name in STRATA]
        first.batch(4,indices=ids)
        self.assertEqual(private,first.sampler_state());self.assertEqual(main,first.parent.sampler_state())
        for field,value in [('fingerprint','other'),('protocol','old'),('draws',saved['draws']+1)]:
            bad=deepcopy(saved);bad[field]=value
            with self.assertRaises(ValueError):second.restore_sampler(bad)
        bad=deepcopy(saved);bad['stratum_counts'][STRATA[0]]+=1
        with self.assertRaises(ValueError):second.restore_sampler(bad)
        changed=Parent();changed.files[0]['sha256']='new_Source_bytes'
        with self.assertRaises(ValueError):self.pool(changed).restore_sampler(saved)
        json.dumps(first.sampler_state(),allow_nan=False)

    def test_failed_reads_rollback_private_state_and_mutating_explicit_parent_is_restored_and_rejected(self):
        pool=self.pool();private=pool.sampler_state();main=pool.parent.sampler_state()
        pool.parent.fail=True
        with self.assertRaisesRegex(RuntimeError,'Source read failed'):pool.batch(8)
        self.assertEqual(pool.sampler_state(),private);self.assertEqual(pool.parent.sampler_state(),main)
        pool.parent.fail=False;pool.parent.mutate=True
        with self.assertRaisesRegex(ValueError,'changed the main'):pool.batch(8)
        self.assertEqual(pool.sampler_state(),private);self.assertEqual(pool.parent.sampler_state(),main)

    def test_missing_stratum_has_explicit_support_receipt_and_no_silent_fallback(self):
        def missing(*args):
            result=labels(*args);result['aggregates']['depth_ge_5']['thin_face'].zero_()
            return result
        with patch('turkeyworld.renderer_geometry_data.source_wall_depth_labels',side_effect=missing):
            with self.assertRaises(GeometryPoolError)as error:
                GeometryWallFrames(Parent(),candidate_count=32)
        self.assertEqual(error.exception.receipt['support_frame_counts']['far_thin'],0)
        self.assertFalse(error.exception.receipt['feasible_disjoint_minimum'])
        json.dumps(error.exception.receipt,allow_nan=False)
        with self.assertRaisesRegex(ValueError,'TRAIN'):self.pool(Parent('val'))
        with self.assertRaises(ValueError):GeometryWallFrames(Parent(),candidate_count=16385)

    def test_held_out_panel_all_four_quadrants_unique_previews_and_no_sampler_changes(self):
        parent=Parent('val');before=parent.sampler_state();rng=torch.get_rng_state().clone()
        with patch('turkeyworld.renderer_geometry_data.source_wall_depth_labels',side_effect=labels):
            indices,receipt=build_geometry_panel(parent,candidate_count=32,frames=16)
            repeated,again=build_geometry_panel(parent,candidate_count=32,frames=16)
        self.assertEqual(indices,repeated);self.assertEqual(receipt,again)
        self.assertEqual(len(indices),len(set(indices)));self.assertEqual(len(indices),16)
        self.assertEqual(receipt['stratum_counts'],dict.fromkeys(STRATA,4))
        self.assertEqual(len(set(receipt['preview_indices'])),4)
        for name,index in zip(STRATA,receipt['preview_indices']):self.assertIn(index,receipt['stratum_indices'][name])
        self.assertFalse(receipt['TRAIN_used']);self.assertFalse(receipt['GPU_used'])
        self.assertEqual(receipt['fingerprint'],parent.panel_fingerprint(indices))
        self.assertEqual(parent.sampler_state(),before);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(receipt['parent_source_fingerprint'],'Source_val')
        json.dumps(receipt,allow_nan=False)
        with self.assertRaisesRegex(ValueError,'VAL'):build_geometry_panel(Parent(),candidate_count=32,frames=16)
        with self.assertRaises(ValueError):build_geometry_panel(parent,candidate_count=32,frames=15)


if __name__=='__main__':unittest.main()
