import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from turkeyworld.renderer_native_panel import build_native_panel


class Validation:
    split='val'

    def __init__(self,folder,partials=True):
        self.rows=[];states=[];self.rng=np.random.default_rng(99);self.draws=123
        self.case_counts={'old':123}
        for area,distance in [(4,6.),(8,3.),(31,3.),(100,1.)]:
            for index in range(12):
                state=np.zeros(20,np.float32);state[[0,1,8,9]]=[.25,.25,.25+distance/16,.25]
                state[[4,12,13]]=1
                self.rows.append({'file':0,'frame':len(states),'mask_fraction':area/1024,
                    'category':'partial'if partials and index<4 else'medium'})
                states.append(state)
        for _ in range(12):
            self.rows.append({'file':0,'frame':len(states),'mask_fraction':0.,'category':'empty_hidden'})
            states.append(np.zeros(20,np.float32))
        path=Path(folder)/'source.npz'
        # No RGB member: the selector must read states only.
        np.savez(path,states=np.stack(states))
        self.files=[{'path':str(path)}]

    def panel_fingerprint(self,indices):
        return hashlib.sha256(json.dumps(list(indices)).encode()).hexdigest()


class NativePanelTests(unittest.TestCase):
    def test_exact_stratified_disjoint_selection_is_repeatable_without_sampler_or_rng_changes(self):
        with TemporaryDirectory()as folder:
            val=Validation(folder);before=copy.deepcopy((val.rows,val.rng.bit_generator.state,val.case_counts,val.draws))
            indices,receipt=build_native_panel(val,excluded_indices=[0,12,24,48],seed=173,per_cohort=4)
            self.assertEqual(len(indices),16);self.assertEqual(len(set(indices)),16)
            self.assertFalse(set(indices)&{0,12,24,48})
            self.assertEqual(receipt['cohort_counts'],dict.fromkeys(['far','tiny','small','empty'],4))
            self.assertEqual(receipt['actual_partial_counts'],{'far':1,'tiny':1,'small':1,'empty':0})
            self.assertEqual((indices,receipt),build_native_panel(val,excluded_indices=[0,12,24,48],seed=173,per_cohort=4))
            self.assertEqual(before,(val.rows,val.rng.bit_generator.state,val.case_counts,val.draws))
            json.dumps(receipt,allow_nan=False)

    def test_missing_partial_support_is_reported_and_insufficient_or_train_panels_fail(self):
        with TemporaryDirectory()as folder:
            val=Validation(folder,partials=False)
            _,receipt=build_native_panel(val,per_cohort=4)
            self.assertEqual(receipt['requested_partial_per_visible_cohort'],1)
            self.assertEqual(receipt['actual_partial_counts'],dict.fromkeys(['far','tiny','small','empty'],0))
            with self.assertRaisesRegex(ValueError,'distinct'):
                build_native_panel(val,per_cohort=20)
            val.split='train'
            with self.assertRaisesRegex(ValueError,'Source VAL'):
                build_native_panel(val,per_cohort=4)


if __name__=='__main__':unittest.main()
