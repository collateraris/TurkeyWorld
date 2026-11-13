"""Light15-step previews do not become100-step validations or consume RNG."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_semantic_sr_training as support
from turkeyworld.models import build_upscaler
from turkeyworld.train_semantic_sr import _light_preview, run


class SemanticPreviewCadenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_real_run_previews15_30_with_only_initial_final_full_validation_and_no_training_change(self):
        helper=support.SemanticTrainingTests()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);initial,selected=helper.initial(root)
            full_calls=[];light_calls=[]
            def full(*args,**kwargs):
                full_calls.append(kwargs['step'])
                report,_=support.evaluate(*args,**kwargs)
                return report,{'metadata':{'step':kwargs['step']}}
            def light(model,data,panel,*,step,baseline_report,batch_size):
                light_calls.append(step)
                # Deliberate interference verifies the trainer's outer guard.
                model.eval();torch.rand(3);np.random.random();random.random()
                data.batch(4,rng=data.rng);data.rng.random()
                return {'metadata':{'step':step,'preview_only':True}}
            def save(preview,path,*,every):
                return {'step':preview.get('metadata',{}).get('step',0),'every':every}
            with patch('turkeyworld.sr_neural_pairs_data.SemanticSRData',support.TinyPairs), \
                    patch('turkeyworld.semantic_sr_validation.evaluate_semantic_sr',full), \
                    patch('turkeyworld.semantic_sr_validation.preview_semantic_sr',light), \
                    patch('turkeyworld.semantic_sr_validation.save_semantic_sr_preview',save):
                for name,cadence in (('with_preview',15),('without_preview',999)):
                    args=helper.args(root/name,initial,selected,31)
                    args.prediction_every=cadence;args.validation_every=100
                    run(args)
                self.assertEqual(full_calls,[0,31,0,31])
                self.assertEqual(light_calls,[15,30])
                before=(root/'with_preview/checkpoints/sr_last.pt').read_bytes()
                args=helper.args(root/'with_preview',initial,selected,32)
                args.validation_every=100;args.prediction_every=14
                with self.assertRaisesRegex(ValueError,'Immutable'):
                    run(args)
                self.assertEqual(before,(root/'with_preview/checkpoints/sr_last.pt').read_bytes())
            a=torch.load(root/'with_preview/checkpoints/sr_last.pt',weights_only=False)
            b=torch.load(root/'without_preview/checkpoints/sr_last.pt',weights_only=False)
            self.assertEqual(a['semantic_updates'],31)
            self.assertEqual(a['sampler'],b['sampler']);self.assertEqual(a['rng_numpy'],b['rng_numpy'])
            self.assertEqual(a['rng_python'],b['rng_python'])
            torch.testing.assert_close(a['rng_torch'],b['rng_torch'],rtol=0,atol=0)
            for key,value in a['model'].items():
                torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)
            for index,state in a['optimizer']['state'].items():
                for key,value in state.items():
                    torch.testing.assert_close(value,b['optimizer']['state'][index][key],rtol=0,atol=0)
            status=json.loads((root/'with_preview/status.json').read_text())
            self.assertEqual(status['prediction_every'],15)
            self.assertEqual(status['validation_every'],100)
            self.assertEqual(status['prediction_preview']['step'],31)
            self.assertEqual(status['semantic_sr_validation']['step'],31)

    def test_preview_failure_restores_child_modes_private_sampler_and_all_cpu_rng(self):
        model=build_upscaler({'architecture':'semantic_edsr','channels':4,'blocks':1,
            'semantic_channels':4,'semantic_blocks':1}).train()
        model.semantic_body.eval()
        data=support.TinyPairs(None,'val')
        modes=[module.training for module in model.modules()]
        sampler=data.sampler_state();cpu=torch.get_rng_state();python=random.getstate();numpy=np.random.get_state()
        def fail(model,data,*args,**kwargs):
            model.eval();data.batch(4,rng=data.rng)
            torch.rand(3);random.random();np.random.random()
            raise RuntimeError('deliberate preview failure')
        with patch('turkeyworld.semantic_sr_validation.preview_semantic_sr',fail):
            with self.assertRaisesRegex(RuntimeError,'deliberate'):
                _light_preview(model,data,list(range(8)),step=15,baseline_report={},path=Path('unused'),every=15,batch_size=4)
        self.assertEqual(modes,[module.training for module in model.modules()])
        self.assertEqual(sampler,data.sampler_state());self.assertEqual(python,random.getstate())
        torch.testing.assert_close(cpu,torch.get_rng_state(),rtol=0,atol=0)
        after=np.random.get_state();self.assertEqual(numpy[0],after[0]);self.assertEqual(numpy[2:],after[2:])
        np.testing.assert_array_equal(numpy[1],after[1])


if __name__=='__main__':unittest.main()
