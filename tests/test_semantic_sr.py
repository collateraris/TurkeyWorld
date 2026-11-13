import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from turkeyworld.models import build_upscaler
from turkeyworld.semantic_sr_objectives import semantic_sr_loss
from turkeyworld.sr_semantic_migration import prepare_semantic_sr_continuation, restore_semantic_optimizer


class SemanticSRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def parent(self):
        model = build_upscaler({'architecture': 'edsr', 'channels': 8, 'blocks': 2, 'scale': 4})
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.01)
        image = torch.rand(2, 3, 32, 32)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            (model(image)-.7).square().mean().backward()
            optimizer.step()
        return model, optimizer

    def payload(self, model, optimizer):
        return dict(stage='sr', config=model.config, model=copy.deepcopy(model.state_dict()),
            optimizer=copy.deepcopy(optimizer.state_dict()), training_steps=40100,
            completed_updates=2000, sampler={'draws':16000,'rng':{'old':1}},
            rng_numpy=np.random.default_rng(173).bit_generator.state,
            rng_torch=torch.get_rng_state(), rng_python=('recorded',), rng_cuda=[])

    def fixture(self, root):
        model, optimizer = self.parent()
        saved = self.payload(model, optimizer)
        parent, selected = root/'sr_last.pt', root/'sr_for_game.pt'
        torch.save(saved, parent)
        torch.save({k:saved[k] for k in ('stage','config','model','training_steps')}, selected)
        return model, optimizer, saved, parent, selected

    def test_zero_append_preserves_trained_parent_rgb_and_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, _, _, parent, selected = self.fixture(root)
            prepare_semantic_sr_continuation(parent, root/'initial.pt', selected_checkpoint=selected)
            saved = torch.load(root/'initial.pt', weights_only=False)
            new = build_upscaler(saved['config']);new.load_state_dict(saved['model'])
            image = torch.rand(2,3,32,32)
            for training in (True,False):
                old.train(training);new.train(training)
                torch.testing.assert_close(old(image), new(image), rtol=0, atol=0)
            for name, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
            self.assertTrue(all(n.startswith('semantic_') for n in saved['semantic_migration']['added_parameter_names']))

    def test_full_adam_groups_mixed_ages_and_rng_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, before, parent, selected = self.fixture(root)
            rng = torch.get_rng_state().clone()
            report = prepare_semantic_sr_continuation(parent, root/'initial.pt', selected_checkpoint=selected)
            self.assertTrue(report['old_optimizer_preserved'])
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            after = torch.load(root/'initial.pt', weights_only=False)
            self.assertEqual(after['rng_numpy'], before['rng_numpy'])
            torch.testing.assert_close(after['rng_torch'], before['rng_torch'], rtol=0, atol=0)
            self.assertEqual(after['parent_sampler'], before['sampler'])
            for index,state in before['optimizer']['state'].items():
                for key,value in state.items():
                    torch.testing.assert_close(value, after['optimizer']['state'][index][key], rtol=0, atol=0)
            self.assertEqual(after['optimizer']['param_groups'][0], before['optimizer']['param_groups'][0])
            model=build_upscaler(after['config']);model.load_state_dict(after['model'])
            opt=restore_semantic_optimizer(model,after)
            self.assertEqual(len(opt.state),len(before['optimizer']['state']))
            self.assertEqual(len(opt.param_groups),2)
            self.assertTrue(all(p not in opt.state for p in opt.param_groups[-1]['params']))

    def test_selected_mismatch_rejected_and_model_only_requires_explicit_fresh(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            _,_,_,parent,selected=self.fixture(root)
            selected_model=torch.load(selected,weights_only=False)
            selected_model['model']['output.bias'].add_(.1);torch.save(selected_model,selected)
            with self.assertRaises(ValueError):
                prepare_semantic_sr_continuation(parent,root/'bad.pt',selected_checkpoint=selected)
            with self.assertRaises(ValueError):
                prepare_semantic_sr_continuation(selected,root/'bad2.pt')
            report=prepare_semantic_sr_continuation(selected,root/'fresh.pt',fresh_adam=True)
            self.assertFalse(report['old_optimizer_preserved']);self.assertEqual(report['old_optimizer_state_count'],0)

    def test_one_pixel_and_empty_supervision_is_finite_and_label_only(self):
        model=build_upscaler({'architecture':'semantic_edsr','channels':8,'blocks':2,
                              'semantic_channels':8,'semantic_blocks':1})
        low=torch.rand(2,3,32,32)
        mask=torch.zeros(2,1,32,32);mask[0,0,12,15]=1
        high_mask=torch.zeros(2,1,128,128);high_mask[0,0,48:52,60:64]=1
        batch={'hr_source':torch.rand(2,3,128,128,requires_grad=True),
               'mask_lr':mask.requires_grad_(),'mask_hr':high_mask.requires_grad_(),
               'hr_background':torch.rand(2,3,128,128,requires_grad=True)}
        output=model.forward_with_aux(low);loss,metrics=semantic_sr_loss(output,batch);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(metrics['visible_images'],1)
        self.assertGreater(float(model.semantic_output.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.semantic_mask.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.semantic_presence.weight.grad.abs().sum()),0)
        self.assertTrue(all(value.grad is None for value in batch.values()))
        empty={k:v.detach().clone() for k,v in batch.items()};empty['mask_lr'].zero_();empty['mask_hr'].zero_()
        loss,metrics=semantic_sr_loss(model.forward_with_aux(low),empty)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(metrics['native_crop_images'],0)
        self.assertEqual(metrics['foreground_contrast'],0.)


if __name__=='__main__':unittest.main()
