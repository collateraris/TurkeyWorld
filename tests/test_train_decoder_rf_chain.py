from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from turkeyworld.train_decoder_rf_chain import DecoderRFChainTraining
from turkeyworld.decoder_rf_chain_best import validate_saved_decoder_chain_best,decoder_chain_measurement
from turkeyworld.training_pipeline import atomic_json


PANEL,PLAN,WORLD='a'*64,'b'*64,'c'*64


def report(score,step,*,source_passed=False):
    return dict(kind='decoder_rf_adaptation_validation_v1',protocol='decoder_rf_source_paired_adaptation_v1',step=step,
        frames=256,native_resolution=32,solver_steps=3,source_held_out=True,renderer_only=True,world_rollout=False,
        Source_RGB_into_RF_conditioning=False,future_source_rgb_in_prior=False,future_source_pose_in_prior=False,
        reference_codec_tensors_unchanged=True,encoder_quantizer_tensors_unchanged=True,
        data_identity_sha256='d'*64,reference_codec_sha256='e'*64,panel_fingerprint=PANEL,
        models=dict(rf_new=dict(summary=dict(ssim_mean=score,frames=256))),
        source_preservation=dict(passed=source_passed),quality_passed=False,val_loss=.25,
        validation_objective=dict(loss=.25,protocol='decoder_rf_source_paired_adaptation_v1'))


class Loop(DecoderRFChainTraining):
    def __init__(self,root,*,start=2999,steps=3000,scores=None,initial=True):
        super().__init__(SimpleNamespace(run_dir=Path(root),device='cpu',probe=False,steps=steps))
        self.start,self.scores,self.initial=start,scores or {3000:.76},initial
        self.updates,self.saves,self.exports,self.previews,self.events=[],[],[],[],[]

    def prepare(self):
        self.step=self.start;self.phase=dict(panel_fingerprint=PANEL,inputs=dict(world=WORLD),decoder_parent_updates=1725)
        self.plan_sha=PLAN;self.training=dict(local_step=self.step)
        self.generator=dict(training_steps=33717,base_training_steps=32209)
        self.model,self.reference,self.source=object(),object(),object()
        self.validation_data=object();self.panel=list(range(256));self.geometry_panel=list(range(128))
        if self.initial and self.start>0:
            initial=report(.75,0);initial.update(plan_sha256=PLAN,feature_world_sha256=WORLD)
            path=self.root/'validation/step_000000.json';atomic_json(path,initial);atomic_json(self.root/'validation_initial.json',initial)
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            measured=decoder_chain_measurement(initial,panel_fingerprint=PANEL,plan_sha256=PLAN,feature_world_sha256=WORLD)
            self.best={k:v for k,v in measured.items()if k not in('data_identity_sha256','reference_codec_sha256')}
            self.best['report_sha256']=sha;self.validation_history=[dict(step=0,ssim=.75,report_sha256=sha)]
    def payload(self):return dict(step=self.step,best=self.best,validation_history=deepcopy(self.validation_history))
    def save(self):self.saves.append(self.step)
    def update(self):
        self.step+=1;self.updates.append(self.step);self.events.append(('update',self.step))
        return dict(train_loss=.2,step=self.step)
    def preview(self):self.previews.append(self.step)
    def export(self,report=None,path=None):self.exports.append((self.step,str(path)if path else None))


def evaluator(trainer):
    def evaluate(*args,step,**kwargs):
        trainer.events.append(('validate',step));return report(trainer.scores[step],step,source_passed=step==100),None
    return evaluate


class DecoderRFChainDriverTests(unittest.TestCase):
    def test_stops_exactly_at3000_and_writes_last_validation_without_quality_claim(self):
        with tempfile.TemporaryDirectory()as root:
            trainer=Loop(root)
            with patch('turkeyworld.train_decoder_rf_chain.evaluate_decoder_adaptation',side_effect=evaluator(trainer)):trainer.run()
            self.assertEqual(trainer.updates,[3000]);self.assertEqual(trainer.state['status'],'completed')
            self.assertFalse(trainer.state['gameplay_qualified']);self.assertFalse(trainer.state['validation']['quality_passed'])
            self.assertEqual(trainer.state['decoder_refinement_updates'],4725)
            self.assertTrue((Path(root)/'validation/step_003000.json').exists())
            self.assertTrue((Path(root)/'checkpoints/decoder_best.pt').exists())

    def test_user_stop_saves_completed_boundary_and_initial_validation_precedes_SGD(self):
        with tempfile.TemporaryDirectory()as root:
            (Path(root)/'stop_training').write_text('stop',encoding='ascii');trainer=Loop(root,start=99)
            trainer.run();self.assertFalse(trainer.updates);self.assertEqual(trainer.state['status'],'stopped')
            self.assertIn(99,trainer.saves)
        with tempfile.TemporaryDirectory()as root:
            trainer=Loop(root,start=0,steps=1,scores={0:.75,1:.76},initial=False)
            with patch('turkeyworld.train_decoder_rf_chain.evaluate_decoder_adaptation',side_effect=evaluator(trainer)):trainer.run()
            self.assertLess(trainer.events.index(('validate',0)),trainer.events.index(('update',1)))
            self.assertTrue((Path(root)/'validation/step_000001.json').exists())

    def test_best_RF_SSIM_earliest_tie_independent_of_Source_preservation(self):
        with tempfile.TemporaryDirectory()as root:
            trainer=Loop(root,start=0,scores={0:.76,100:.75,200:.76,300:.761});trainer.prepare()
            with patch('turkeyworld.train_decoder_rf_chain.evaluate_decoder_adaptation',side_effect=evaluator(trainer)):
                trainer.validate(None);path=Path(root)/'checkpoints/decoder_best.pt';before=path.read_bytes()
                for step in(100,200):trainer.step=step;trainer.validate(None);self.assertEqual(path.read_bytes(),before)
                trainer.step=300;trainer.validate(None)
            self.assertEqual(trainer.best['step'],300);self.assertFalse(trainer.state['validation']['source_preservation']['passed'])
            self.assertEqual(trainer.best['report_sha256'],hashlib.sha256((Path(root)/'validation/step_000300.json').read_bytes()).hexdigest())
            saved=trainer.payload();best,history=validate_saved_decoder_chain_best(saved,panel_fingerprint=PANEL,
                plan_sha256=PLAN,feature_world_sha256=WORLD,run_dir=root)
            self.assertEqual(best,trainer.best);self.assertEqual(len(history),4)
            broken=deepcopy(saved);broken['best']['step']=200
            with self.assertRaises(ValueError):validate_saved_decoder_chain_best(broken,panel_fingerprint=PANEL,
                plan_sha256=PLAN,feature_world_sha256=WORLD,run_dir=root)

    def test_nonfinite_or_stale_report_cannot_replace_best(self):
        with tempfile.TemporaryDirectory()as root:
            trainer=Loop(root,start=0,scores={0:.76,100:float('nan')});trainer.prepare()
            with patch('turkeyworld.train_decoder_rf_chain.evaluate_decoder_adaptation',side_effect=evaluator(trainer)):
                trainer.validate(None);old=(Path(root)/'best.json').read_bytes();trainer.step=100
                with self.assertRaises(ValueError):trainer.validate(None)
                self.assertEqual((Path(root)/'best.json').read_bytes(),old)
            saved=trainer.payload();saved['step']=0
            path=Path(root)/'validation/step_000000.json';value=json.loads(path.read_text());value['panel_fingerprint']='f'*64
            path.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaises(ValueError):validate_saved_decoder_chain_best(saved,panel_fingerprint=PANEL,
                plan_sha256=PLAN,feature_world_sha256=WORLD,run_dir=root)

    def test_validation0_saved_before_initial_alias_is_recovered_without_duplicate_measurement(self):
        with tempfile.TemporaryDirectory()as root:
            trainer=Loop(root,start=0,steps=0,scores={0:.75},initial=False)
            trainer.prepare()
            with patch('turkeyworld.train_decoder_rf_chain.evaluate_decoder_adaptation',side_effect=evaluator(trainer)):
                trainer.validate(None)
            self.assertFalse((Path(root)/'validation_initial.json').exists())
            with patch.object(trainer,'prepare'):
                trainer.run()
            self.assertTrue((Path(root)/'validation_initial.json').exists())
            self.assertEqual(len(trainer.validation_history),1);self.assertEqual(trainer.updates,[])


if __name__=='__main__':unittest.main()
