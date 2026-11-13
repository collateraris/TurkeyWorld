"""V8 optimizer ownership, direct depth integration and unqualified exports."""
from copy import deepcopy
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from test_renderer_geometry_opacity import working_model
from test_renderer_enemy_objectives import batch
from turkeyworld.renderer_geometry_opacity import enable_geometry_opacity,shared_camera_geometry
from turkeyworld.renderer_refinement import configure_refinement,frozen_fingerprint
from turkeyworld.renderer_student import _hash_tensors,load_renderer_student
from turkeyworld.renderer_native_phase import _same
from turkeyworld.train_renderer_enemy import (_renderer_optimizer,_add_geometry_auxiliary_loss,
    _phase_plan_filename,_export_private_analysis,_private_geometry_depth_weight)


class PrivateGeometryTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_ordered_old_groups_unchanged_four_new_params_only_private_lr_no_codec_or_shared_geometry(self):
        model=working_model();args=SimpleNamespace(lr=1e-4,base_lr=1e-4,private_geometry_supervision=False)
        old=_renderer_optimizer(model,args)
        before=[[n for p in group['params']for n,q in model.named_parameters()if p is q]for group in old.param_groups]
        enable_geometry_opacity(model);configure_refinement(model);args.private_geometry_supervision=True
        optimizer=_renderer_optimizer(model,args)
        after=[[n for p in group['params']for n,q in model.named_parameters()if p is q]for group in optimizer.param_groups]
        self.assertEqual(before,after[:2]);self.assertEqual(len(after[2]),4)
        self.assertTrue(all(n.startswith('flow_renderer.geometry_opacity.')for n in after[2]))
        self.assertEqual([g['lr']for g in optimizer.param_groups],[1e-4,1e-4,5e-5])
        self.assertEqual(sum(p.numel()for p in optimizer.param_groups[2]['params']),6241)
        self.assertFalse(any('codec.'in n or 'geometry_reader.'in n for group in after for n in group))
        args.private_geometry_supervision=False
        with self.assertRaises(ValueError):_renderer_optimizer(model,args)
    def test_depth_warmup_uses_actual_completed_phase_clock_and_reaches_one_at500(self):
        for step,expected in ((8000,0.),(8001,.002),(8250,.5),(8500,1.),(10000,1.)):
            self.assertEqual(_private_geometry_depth_weight(step,8000),expected)
        with self.assertRaises(ValueError):_private_geometry_depth_weight(7999,8000)
    def test_actual_geometry_aux_depth1_backward_reaches_private_Base_and_RF_and_freezes_Source(self):
        model=working_model();enable_geometry_opacity(model);configure_refinement(model)
        src=batch((False,)*8);src['row_index']=torch.arange(8)
        for key,value in src.items():
            if value.is_floating_point():value.requires_grad_()
        class Data:
            draws=0
            def batch(self,n):self.draws+=n;return src
        data=Data();frozen=frozen_fingerprint(model)
        old_shared=_hash_tensors(shared_camera_geometry(model).state_dict())
        loss,metrics=_add_geometry_auxiliary_loss(model,next(model.base_student.parameters()).sum()*0,{},data,opacity_depth_weight=1.)
        loss.backward()
        self.assertEqual(metrics['opacity_depth_weight'],1.)
        self.assertTrue(metrics['opacity_depth_source_labels_only'])
        self.assertEqual(metrics['geometry_wall_aux_examples'],8)
        for branch in(model.base_student,model.flow_renderer,model.flow_renderer.geometry_opacity):
            self.assertGreater(sum(float(p.grad.abs().sum())for p in branch.parameters()if p.grad is not None),0)
        self.assertTrue(all(v.grad is None for v in src.values()if v.is_floating_point()))
        self.assertTrue(all(p.grad is None for p in model.parameters()if not p.requires_grad))
        self.assertEqual(frozen_fingerprint(model),frozen)
        self.assertEqual(_hash_tensors(shared_camera_geometry(model).state_dict()),old_shared)
    def test_v8_phase_plan_priority_and_unqualified_export_keeps_rng_weights_and_strict_reload(self):
        source=working_model();model=deepcopy(source);enable_geometry_opacity(model);configure_refinement(model)
        args=SimpleNamespace(private_geometry_supervision=True,geometry_wall_supervision=True,far_wall_supervision=True)
        self.assertEqual(_phase_plan_filename(args),'private_geometry_phase_plan.json')
        args.private_geometry_supervision=False;self.assertEqual(_phase_plan_filename(args),'geometry_wall_phase_plan.json')
        before=_hash_tensors(model.state_dict());cpu=torch.get_rng_state().clone();rng=np.random.get_state();py=random.getstate()
        with TemporaryDirectory()as directory:
            receipt=_export_private_analysis(source,model,directory,step=8100,report={'passed':False},
                initial_renderer_steps=16200,initial_base_steps=15800,identity={'fixed_source':True})
            self.assertEqual(receipt['training_steps'],24300);self.assertFalse(receipt['release_qualified'])
            self.assertFalse(receipt['automatic_active_promotion'])
            saved=torch.load(Path(directory)/'renderer_student.pt',map_location='cpu',weights_only=False)
            self.assertFalse(saved['metrics']['source_quality_passed'])
            self.assertEqual(saved['geometry_opacity_override'],saved['base_student_bundle']['geometry_opacity_binding'])
            target=deepcopy(source);load_renderer_student(target,Path(directory)/'renderer_student.pt')
            self.assertEqual(_hash_tensors(target.flow_renderer.state_dict()),_hash_tensors(model.flow_renderer.state_dict()))
        self.assertEqual(before,_hash_tensors(model.state_dict()))
        # Strict reload constructed an independent model and may consume RNG;
        # check exporter RNG immediately separately.
        with TemporaryDirectory()as directory:
            cpu=torch.get_rng_state().clone();rng=np.random.get_state();py=random.getstate()
            _export_private_analysis(source,model,directory,step=8100,report={'passed':False},
                initial_renderer_steps=16200,initial_base_steps=15800,identity={})
            self.assertTrue(torch.equal(cpu,torch.get_rng_state()));self.assertTrue(_same(rng,np.random.get_state()));self.assertEqual(py,random.getstate())


if __name__=='__main__':unittest.main()
