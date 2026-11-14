"""NPC-specific Source masks, normalized footprint, and teleport-safe sweeps."""
from copy import deepcopy
import unittest
import torch
from turkeyworld.rssm_npc_geometry_objectives import(NPC_NAVIGATION_PROTOCOL,
    prepare_npc_geometry,npc_geometry_terms,add_npc_geometry_metrics)


def fixture(horizon=4):
    old=torch.zeros((1,horizon,20));old[...,4]=1;old[...,8:10]=torch.tensor([4.5,4.5])/16
    old[...,12:14]=1;target=old.clone();grid=torch.zeros((1,16,16),dtype=torch.long);grid[:,5,5]=1
    return old,target,grid


class NPCGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_opt_in_and_exact_protocol_do_not_change_legacy(self):
        old,target,grid=fixture()
        self.assertIsNone(prepare_npc_geometry(old,target,grid,npc_navigation_protocol=None))
        for key,value in(('version',True),('recorded_velocity_direction',0)):
            broken=deepcopy(NPC_NAVIGATION_PROTOCOL);broken[key]=value
            with self.assertRaises(ValueError):prepare_npc_geometry(old,target,grid,npc_navigation_protocol=broken)

    def test_false_model_death_cannot_hide_source_alive_penetration_and_gradient(self):
        old,target,grid=fixture();supervision=prepare_npc_geometry(old,target,grid,npc_navigation_protocol=NPC_NAVIGATION_PROTOCOL)
        prediction=target.clone();prediction[...,8:10]=torch.tensor([5.1,5.5])/16;prediction[...,12:14]=0;prediction.requires_grad_()
        loss,metrics=npc_geometry_terms(prediction,old[:,0],supervision,predicted_respawns=torch.zeros((1,4,1),dtype=torch.bool))
        self.assertGreater(float(loss.detach()),0);self.assertEqual(metrics['npc_wall_source_alive_model_missing_transitions'],4)
        self.assertEqual(metrics['npc_wall_source_alive_transitions'],4);loss.backward()
        self.assertGreater(float(prediction.grad[...,8:10].abs().sum()),0)
        self.assertEqual(float(prediction.grad[...,:8].abs().sum()),0)

    def test_legal_x_then_y_path_and_predicted_spawn_skip_only_sweep(self):
        old,target,grid=fixture(1);supervision=prepare_npc_geometry(old,target,grid,npc_navigation_protocol=NPC_NAVIGATION_PROTOCOL)
        prediction=target.clone();prediction[...,8:10]=torch.tensor([6.5,6.5])/16
        _,legal=npc_geometry_terms(prediction,old[:,0],supervision)
        self.assertEqual(legal['npc_wall_path_overlap_transitions'],0)
        previous=old[:,0].clone();previous[:,8:10]=torch.tensor([5.5,4.5])/16
        prediction[...,8:10]=torch.tensor([5.5,6.5])/16
        _,crossing=npc_geometry_terms(prediction,previous,supervision)
        _,teleport=npc_geometry_terms(prediction,previous,supervision,predicted_respawns=torch.ones((1,1,1),dtype=torch.bool))
        self.assertGreater(crossing['npc_wall_path_loss'],0);self.assertEqual(teleport['npc_wall_path_loss'],0)
        prediction[...,8:10]=torch.tensor([5.5,5.5])/16
        _,inside=npc_geometry_terms(prediction,previous,supervision,predicted_respawns=torch.ones((1,1,1),dtype=torch.bool))
        self.assertGreater(inside['npc_wall_endpoint_loss'],0)

    def test_full_tape_normalization_and_source_lifetime_mask_are_additive(self):
        old,target,grid=fixture(32);old[:,8:12,12:14]=0;target[:,8:12,12:14]=0
        supervision=prepare_npc_geometry(old,target,grid,npc_navigation_protocol=NPC_NAVIGATION_PROTOCOL)
        prediction=target.clone();prediction[...,8:10]=torch.tensor([5.2,5.5])/16
        respawn=torch.zeros((1,32,1),dtype=torch.bool);respawn[:,16]=True
        whole,summary=npc_geometry_terms(prediction,old[:,0],supervision,predicted_respawns=respawn)
        a,one=npc_geometry_terms(prediction[:,:16],old[:,0],supervision,predicted_respawns=respawn[:,:16])
        b,two=npc_geometry_terms(prediction[:,16:],prediction[:,15],supervision,first=16,predicted_respawns=respawn[:,16:])
        torch.testing.assert_close(a+b,whole)
        combined=add_npc_geometry_metrics(add_npc_geometry_metrics({},one),two)
        for key,value in summary.items():
            if isinstance(value,float):self.assertAlmostEqual(combined[key],value,places=6)
            else:self.assertEqual(combined[key],value)
        self.assertEqual(combined['npc_wall_source_alive_transitions'],28)


if __name__=='__main__':unittest.main()
