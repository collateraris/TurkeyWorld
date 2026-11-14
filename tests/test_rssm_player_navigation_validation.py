"""Independent geometry, Source life masks and whole-boundary best selection."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from turkeyworld.rssm_player_navigation_validation import (PROTOCOL,PANELS,
    _clear,_path_clear,player_panel_metrics,annotate_player_navigation_validation,
    player_navigation_improves,publish_best_player_navigation)
from turkeyworld.runtime_checks import circle_is_clear,segment_is_clear


def arrays(*,actors=1,error=.1):
    grid=np.zeros((actors,16,16),np.uint8);grid[:,[0,-1],:]=1;grid[:,:,[0,-1]]=1
    reference=np.zeros((actors,451,20),np.float32)
    reference[:,:,:2]=np.array([3.5,3.5])/16;reference[:,:,2]=1;reference[:,:,4]=1
    predicted=reference.copy();predicted[:,1:,0]+=error/16
    return predicted,reference,grid


def write_panels(root,*,error=.1):
    predicted,reference,grid=arrays(error=error)
    for name,relative in PANELS.items():
        directory=root/relative;directory.mkdir(parents=True,exist_ok=True)
        context=2 if name=='navigation_context2'else 1
        panel=dict(version='heldout_free_state_v1',split='val',context=context,horizon=450,seed=173,
            windows=[dict(episode='episode_0000000000.npz',episode_index=0,start=0)],
            source_content_sha256='f'*64,episode_identity_sha256='e'*64)
        metadata=dict(panel=panel,fingerprint=hashlib.sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest(),
            source_held_out=True,source_future_feedback=False,future_rgb_encoded=False,
            horizon=450,context=context,count=1,all_states_finite=True,all_memory_finite_and_shape_valid=True)
        (directory/'heldout_trajectory_validation.json').write_text(json.dumps(metadata),encoding='utf-8')
        np.savez(directory/'heldout_trajectory_states.npz',predicted=predicted,reference=reference,maps=grid,
            indices=np.zeros((1,2),np.int64),actions=np.zeros((1,450,4),np.float32),dt=np.array([1/30]))


def full_checkpoint(path,step,*,invalid=None):
    model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters())
    model(torch.ones(1,2)).sum().backward();optimizer.step()
    value=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),training_steps=step,
        global_training_step=step,run_state=dict(global_training_step=step),
        rng_state={'proof':'preserved'},torch_rng_state=torch.get_rng_state())
    if invalid=='step':value['training_steps']=step+1
    if invalid=='optimizer':value['optimizer']['state']={}
    if invalid=='model':value['model']={}
    torch.save(value,path);return value


class PlayerNavigationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);self.root=Path(directory.name)

    def score(self,*,step=10,error=.1):
        directory=self.root/f'validation_{step}';write_panels(directory,error=error)
        report=dict(step=step,passed=False,navigation={'short':{'metrics':{'false_stop_rate':.1}},
            'contact_generalization':{'metrics':{'missed_block_rate':.2}}})
        result=annotate_player_navigation_validation(report,{'sha256':'a'*64,'source_optimizer_step':9},directory)
        saved=directory/'rssm_combat_validation.json';saved.write_text(json.dumps(report),encoding='utf-8')
        return result,report,saved

    def test_vectorized_circle_matches_runtime_on_random_boundary_and_nonfinite_points(self):
        rng=np.random.default_rng(812);grid=(rng.random((16,16))<.15).astype(np.uint8)
        grid[[0,-1],:]=1;grid[:,[0,-1]]=1
        points=np.concatenate((rng.uniform(-.3,16.3,(2000,2)),
            np.array([[np.nan,3],[3,np.inf],[.2,.2],[15.8,3],[5,5],[5.2,5.2]])))
        expected=np.array([circle_is_clear(grid,point,.2)for point in points])
        np.testing.assert_array_equal(_clear(points,grid),expected)
        np.testing.assert_array_equal(_clear(points[:2000].reshape(20,100,2),grid),expected[:2000].reshape(20,100))

    def test_path_uses_x_then_y_and_matches_runtime_without_straight_chord_false_positives(self):
        grid=np.zeros((16,16),np.uint8);grid[5,5]=1
        old=np.array([[4.5,4.5],[5.5,4.7],[4.7,5.5]])
        new=np.array([[6.5,6.5],[5.5,5.3],[5.3,5.5]])
        np.testing.assert_array_equal(_path_clear(old,new,grid),
            np.array([segment_is_clear(grid,a,b,.2)for a,b in zip(old,new)]))
        self.assertTrue(_path_clear(old[:1],new[:1],grid)[0])
        self.assertFalse(all(circle_is_clear(grid,point,.2)for point in np.linspace(old[0],new[0],101)))
        rng=np.random.default_rng(99);old=rng.uniform(1,15,(100,2));new=old+rng.uniform(-.08,.08,(100,2))
        np.testing.assert_array_equal(_path_clear(old,new,grid),
            np.array([segment_is_clear(grid,a,b,.2)for a,b in zip(old,new)]))

    def test_false_predicted_death_cannot_hide_drift_or_wall_errors(self):
        predicted,reference,grid=arrays(error=.1)
        predicted[:,1:,4]=0;grid[0,3,4]=1;predicted[:,10:,:2]=np.array([4.5,3.5])/16
        score=player_panel_metrics(predicted,reference,grid,panel_fingerprint='test')
        self.assertEqual(score['valid_source_player_transitions'],450)
        self.assertEqual(score['source_alive_model_dead_steps'],450)
        self.assertEqual(score['player_wall_steps'],441);self.assertEqual(score['first_player_wall_step'],[10])
        self.assertGreater(score['drift']['mean'],0)
        # A real Source death ends support, irrespective of a predicted revival.
        reference[:,101:,4]=0;predicted[:,:,4]=1
        score=player_panel_metrics(predicted,reference,grid,panel_fingerprint='test')
        self.assertEqual(score['valid_source_player_transitions'],100)
        self.assertEqual(score['offsets']['128']['count'],0);self.assertEqual(score['final']['count'],0)
        self.assertIsNone(score['final']['mean'])
        reference[:,:,4]=0
        with self.assertRaisesRegex(ValueError,'Source-alive'):
            player_panel_metrics(predicted,reference,grid,panel_fingerprint='test')

    def test_annotation_uses_three_actual_panels_and_retains_contact_diagnostics(self):
        score,report,saved=self.score()
        self.assertEqual(score['valid_player_transitions'],1350);self.assertEqual(set(score['panels']),set(PANELS))
        self.assertAlmostEqual(score['player_drift']['mean'],.1,places=5)
        self.assertEqual(score['contact_axis_errors']['short']['false_stop_rate'],.1)
        self.assertEqual(score['contact_axis_errors']['contact']['missed_block_rate'],.2)
        self.assertTrue(score['training_loss_is_not_selection_metric']);self.assertFalse(score['automatic_runtime_deployment'])
        self.assertEqual(json.loads((saved.parent/'player_navigation_validation.json').read_text()),score)

    def test_annotation_rejects_train_panel_tampered_fingerprint_indices_or_wrong_initial_state(self):
        for defect in ('train','fingerprint','indices','initial'):
            with self.subTest(defect=defect):
                directory=self.root/defect;write_panels(directory)
                panel=directory/next(iter(PANELS.values()));metadata_path=panel/'heldout_trajectory_validation.json'
                metadata=json.loads(metadata_path.read_text())
                if defect=='train':
                    metadata['panel']['split']='train'
                    metadata['fingerprint']=hashlib.sha256(json.dumps(metadata['panel'],sort_keys=True).encode()).hexdigest()
                if defect=='fingerprint':metadata['fingerprint']='0'*64
                metadata_path.write_text(json.dumps(metadata),encoding='utf-8')
                if defect in ('indices','initial'):
                    path=panel/'heldout_trajectory_states.npz'
                    with np.load(path)as saved:values={key:saved[key].copy()for key in saved.files}
                    if defect=='indices':values['indices'][0,1]=1
                    else:values['predicted'][0,0,0]+=.01
                    np.savez(path,**values)
                with self.assertRaises(ValueError):
                    annotate_player_navigation_validation({'step':10},{'sha256':'a'*64,'source_optimizer_step':9},directory)

    def test_selection_blocks_wall_path_mean_regressions_and_panel_mismatches(self):
        previous,_,_=self.score();candidate=copy.deepcopy(previous);candidate['score']-=.01
        self.assertTrue(player_navigation_improves(candidate,previous))
        for name in ('player_wall_steps','player_wall_path_steps'):
            changed=copy.deepcopy(candidate);changed[name]+=1
            self.assertFalse(player_navigation_improves(changed,previous))
        changed=copy.deepcopy(candidate);changed['player_drift']['mean']+=.01
        self.assertFalse(player_navigation_improves(changed,previous))
        for key in ('panel_fingerprint','source_reference_sha256','actors'):
            changed=copy.deepcopy(candidate);changed['panels']['navigation_context1'][key]='changed'
            with self.assertRaises(ValueError):player_navigation_improves(changed,previous)
        changed=copy.deepcopy(candidate);changed['revision_plan_sha256']='b'*64
        with self.assertRaises(ValueError):player_navigation_improves(changed,previous)
        changed=copy.deepcopy(candidate);changed['score']=float('nan')
        with self.assertRaises(ValueError):player_navigation_improves(changed,None)

    def test_best_publication_preserves_entire_full_checkpoint_and_exact_report_hash(self):
        _,report,validation=self.score();checkpoint=self.root/'current.pt';snapshot=full_checkpoint(checkpoint,10)
        result=publish_best_player_navigation(checkpoint,self.root,report=report,validation_report=validation)
        alias=self.root/'checkpoints/best_player_navigation.pt';candidate=Path(result['checkpoint'])
        self.assertEqual(alias.read_bytes(),checkpoint.read_bytes());self.assertEqual(candidate.read_bytes(),checkpoint.read_bytes())
        self.assertEqual(result['checkpoint_sha256'],hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        self.assertEqual(result['validation_report_sha256'],hashlib.sha256(validation.read_bytes()).hexdigest())
        saved=torch.load(candidate,map_location='cpu',weights_only=False)
        torch.testing.assert_close(saved['model']['weight'],snapshot['model']['weight'],rtol=0,atol=0)
        self.assertTrue(saved['optimizer']['state']);self.assertTrue(result['contains_full_model_and_Adam'])
        self.assertFalse(result['combined_quality_passed']);self.assertFalse(result['automatic_runtime_deployment'])
        original=alias.read_bytes()
        again=publish_best_player_navigation(checkpoint,self.root,report=report,validation_report=validation)
        self.assertEqual(again,result);self.assertEqual(alias.read_bytes(),original)

    def test_publication_rejects_incomplete_wrong_step_stale_report_and_existing_bytes_without_alias_change(self):
        for defect in ('model','optimizer','step','report','candidate'):
            with self.subTest(defect=defect):
                run=self.root/defect;run.mkdir();checkpoint=run/'current.pt'
                _,report,validation=self.score(step=20+len(defect))
                full_checkpoint(checkpoint,report['step'],invalid=defect)
                alias=run/'checkpoints/best_player_navigation.pt';alias.parent.mkdir();alias.write_bytes(b'previous-best')
                if defect=='report':
                    wrong=copy.deepcopy(report);wrong['step']+=1;validation.write_text(json.dumps(wrong),encoding='utf-8')
                if defect=='candidate':
                    target=run/'checkpoints/player_navigation_best'/f"world_g{report['step']:06d}.pt"
                    target.parent.mkdir();target.write_bytes(b'different-checkpoint')
                with self.assertRaises(ValueError):
                    publish_best_player_navigation(checkpoint,run,report=report,validation_report=validation)
                self.assertEqual(alias.read_bytes(),b'previous-best')
                self.assertFalse((run/'best_player_navigation.json').exists())


if __name__=='__main__':unittest.main()
