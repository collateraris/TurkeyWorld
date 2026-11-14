from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from torch import nn

from test_renderer_rf_codec_goal import PINS,validation
from turkeyworld.train_renderer_rf_chain import RFChainTraining
from turkeyworld.renderer_rf_chain_best import validate_saved_rf_chain_best


class Loop(RFChainTraining):
    def __init__(self,root,start=2999,scores=None):
        super().__init__(SimpleNamespace(run_dir=Path(root),device='cpu',probe=False,steps=3000))
        self.start=start;self.scores=scores or {3000:.84};self.updates=[];self.saves=[]
    def prepare(self):
        self.step=self.start;self.plan_sha=PINS['plan_sha256'];self.training={'local_step':self.step}
        self.parent={'base_training_steps':29509}
        self.phase=dict(panel_fingerprint=PINS['panel_fingerprint'],inputs={'world':PINS['feature_world_sha256']})
        self.model=nn.Module();self.model.base_student=nn.Linear(3,3);self.model.flow_renderer=nn.Linear(3,3)
        self.reference=deepcopy(self.model);self.profiled={'val':None};self.panel=list(range(256));self.geometry_panel=list(range(128))
        (self.root/'validation_initial.json').write_text(json.dumps(validation(.74,0)))
    def update(self):
        self.step+=1;self.updates.append(self.step)
        return dict(step=self.step,train_loss=.1,renderer_training_steps=30717+self.step,base_training_steps=29509)
    def save(self):self.saves.append(self.step)
    def export(self,*args,**kwargs):pass
    def preview(self):pass
    def payload(self):return dict(step=self.step,best=self.best,validation_history=deepcopy(self.validation_history))


def evaluator(trainer):
    def measure(*args,step,**kwargs):
        report=validation(trainer.scores[step],step);report['codec_sha256']='e'*64
        return report,None
    return measure


class RFChainDriverTests(unittest.TestCase):
    def test_actual_loop_ignores_early85_success_and_completes_exactly3000(self):
        with tempfile.TemporaryDirectory()as root:
            trainer=Loop(root,start=1999,scores={i:.9 for i in range(2000,3001,100)})
            with patch('turkeyworld.train_renderer_rf_chain.evaluate_rf_codec',side_effect=evaluator(trainer)):
                trainer.run()
            self.assertEqual(trainer.updates,list(range(2000,3001)))
            self.assertEqual(trainer.step,3000);self.assertEqual(trainer.state['status'],'completed')
            self.assertEqual(trainer.state['renderer_training_steps'],33717)
            self.assertEqual(trainer.state['base_training_steps'],29509)
            self.assertFalse((Path(root)/'goal_reached.json').exists())

    def test_user_stop_and_completed_resume_do_no_extra_updates(self):
        for start,stop,expected in ((2400,True,'stopped'),(3000,False,'completed')):
            with tempfile.TemporaryDirectory()as root:
                if stop:(Path(root)/'stop_training').write_text('explicit')
                trainer=Loop(root,start=start);trainer.run()
                self.assertEqual(trainer.updates,[]);self.assertEqual(trainer.state['status'],expected)

    def test_full_best_checkpoint_not_rewritten_on_tie_and_history_SHA_checked_without_goal_stop(self):
        with tempfile.TemporaryDirectory()as root:
            trainer=Loop(root,start=0,scores={0:.85,100:.84,200:.85,300:.851})
            trainer.prepare()
            with patch('turkeyworld.train_renderer_rf_chain.evaluate_rf_codec',side_effect=evaluator(trainer)):
                trainer.validate(None)
                path=Path(root)/'checkpoints/renderer_best.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest()
                for step in(100,200):
                    trainer.step=step;trainer.validate(None)
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
                trainer.step=300;trainer.validate(None)
            best,history=validate_saved_rf_chain_best(trainer.payload(),run_dir=root,**PINS)
            self.assertEqual(best['step'],300);self.assertEqual(len(history),4)
            report=Path(root)/'validation/step_000100.json';report.write_text('{}')
            with self.assertRaisesRegex(ValueError,'SHA changed'):
                validate_saved_rf_chain_best(trainer.payload(),run_dir=root,**PINS)


if __name__=='__main__':unittest.main()
