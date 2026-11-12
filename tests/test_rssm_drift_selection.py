"""A predicted death cannot hide drift; selectors follow durable full weights."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.rssm_drift_selection import (
    source_masked_drift_score, heldout_drift_score, drift_candidate_improves, publish_best_drift,
)


def trajectories():
    source=np.zeros((2,451,20),np.float32)
    source[...,0:2]=2.5/16;source[...,4]=1
    source[...,8:10]=4.5/16;source[...,12]=.6;source[...,13]=1
    return source.copy(),source


class DriftSelectionTests(unittest.TestCase):
    def test_source_life_mask_measures_missing_model_actor_and_valid_later_life(self):
        predicted,source=trajectories()
        source[:,3:5,12:14]=0 # source spawn at offset5; offset6 later life is valid
        predicted[:]=source
        predicted[:,6:,8]+=1/16
        predicted[:,6:,12:14]=0 # MODEL missing must never suppress position error
        predicted[:,8:,0]+=.1/16
        score=source_masked_drift_score(predicted,source,panel_fingerprint='fixed')
        self.assertEqual(score['player']['count'],900)
        self.assertEqual(score['npc']['count'],894) # offsets3,4,5 excluded per sequence
        self.assertEqual(score['source_alive_npc_model_missing_steps'],890)
        self.assertGreater(score['npc']['mean'],.99)
        self.assertAlmostEqual(score['npc_final']['mean'],1.,places=6)
        self.assertAlmostEqual(score['player_final']['mean'],.1,places=5)
        self.assertEqual(score['normalization_tiles'],{'player':.1,'npc':.25})
        visible=predicted.copy();visible[:,6:,12]=.6;visible[:,6:,13]=1
        self.assertEqual(source_masked_drift_score(visible,source,panel_fingerprint='fixed')['rank'],score['rank'])

    def test_fixed_source_panel_and_finite_full450_are_mandatory(self):
        predicted,source=trajectories();score=source_masked_drift_score(predicted,source,panel_fingerprint='fixed')
        self.assertFalse(drift_candidate_improves(score,score))
        worse=predicted.copy();worse[...,0]+=.1/16
        old=source_masked_drift_score(worse,source,panel_fingerprint='fixed')
        self.assertTrue(drift_candidate_improves(score,old))
        changed=deepcopy(score);changed['panel_fingerprint']='different'
        with self.assertRaises(ValueError):drift_candidate_improves(changed,old)
        changed=source.copy();changed[:,50,0]+=.1
        with self.assertRaises(ValueError):drift_candidate_improves(source_masked_drift_score(predicted,changed,panel_fingerprint='fixed'),score)
        for bad in (predicted[:,:17],np.full_like(predicted,np.nan)):
            with self.assertRaises(ValueError):source_masked_drift_score(bad,source,panel_fingerprint='fixed')
        for report in ({},{'source_held_out':True,'horizon':16}):
            with self.assertRaises(ValueError):heldout_drift_score(report,'must_not_open.npz')

    def test_wall_geometry_also_retains_source_alive_but_model_missing_npc(self):
        predicted,source=trajectories()
        grid=np.zeros((2,16,16),np.uint8);grid[:,0,:]=1;grid[:,-1,:]=1;grid[:,:,0]=1;grid[:,:,-1]=1
        predicted[:,1:,8:10]=.5/16;predicted[:,1:,12:14]=0
        report={'source_held_out':True,'source_future_feedback':False,'future_rgb_encoded':False,
            'horizon':450,'count':2,'all_states_finite':True,'all_memory_finite_and_shape_valid':True,'fingerprint':'fixed'}
        with TemporaryDirectory() as folder:
            path=Path(folder)/'states.npz';np.savez(path,predicted=predicted,reference=source,maps=grid)
            score=heldout_drift_score(report,path)
        self.assertEqual(score['source_alive_model_wall_steps']['npc'],900)
        self.assertEqual(score['source_alive_model_wall_steps']['player'],0)

    def test_best_full_checkpoint_is_published_before_selector_and_previous_only_owned_candidate_removed(self):
        predicted,source=trajectories()
        worse=predicted.copy();worse[...,0]+=.2/16
        oldscore=source_masked_drift_score(worse,source,panel_fingerprint='fixed')
        newscore=source_masked_drift_score(predicted,source,panel_fingerprint='fixed')
        with TemporaryDirectory() as folder:
            root=Path(folder);saved=root/'latest.pt'
            payload={'model':{'w':torch.tensor([2.])},'optimizer':{'state':{0:{'step':torch.tensor(5014.),'exp_avg':torch.tensor([.3])}}},'training_steps':5014}
            torch.save(payload,saved)
            first=publish_best_drift(saved,root,step=5014,score=oldscore,validation_report=root/'report.json',combined_quality_passed=False)
            oldpath=Path(first['checkpoint']);self.assertTrue(oldpath.exists())
            self.assertEqual(json.loads((root/'best_drift.json').read_text())['step'],5014)
            payload['model']['w'].add_(1);payload['training_steps']=5114;torch.save(payload,saved)
            second=publish_best_drift(saved,root,step=5114,score=newscore,validation_report=root/'report2.json',combined_quality_passed=False)
            actual=torch.load(second['checkpoint_alias'],weights_only=False)
            self.assertTrue(torch.equal(actual['model']['w'],payload['model']['w']))
            self.assertTrue(torch.equal(actual['optimizer']['state'][0]['exp_avg'],payload['optimizer']['state'][0]['exp_avg']))
            self.assertFalse(oldpath.exists());self.assertTrue(saved.exists())
            self.assertFalse(second['automatic_runtime_deployment']);self.assertFalse(second['combined_quality_passed'])
            before=(root/'best_drift.json').read_bytes()
            self.assertEqual(publish_best_drift(saved,root,step=5214,score=oldscore,validation_report=root/'report3.json',combined_quality_passed=False),second)
            self.assertEqual((root/'best_drift.json').read_bytes(),before)

    def test_failed_candidate_copy_never_publishes_a_new_selector(self):
        predicted,source=trajectories();score=source_masked_drift_score(predicted,source,panel_fingerprint='fixed')
        with TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'world.pt';checkpoint.write_bytes(b'full model and Adam')
            with patch('turkeyworld.rssm_drift_selection.copy_drift_checkpoint',side_effect=OSError('disk')):
                with self.assertRaises(OSError):publish_best_drift(checkpoint,root,step=5014,score=score,
                    validation_report=root/'report.json',combined_quality_passed=False)
            self.assertFalse((root/'best_drift.json').exists())


if __name__=='__main__':unittest.main()
