from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_rssm_combat_training import raw_model
from turkeyworld.engine import Game
from turkeyworld.rssm_combat_protocol import environment_profile
from turkeyworld.rssm_migration import configure_rssm_training
from turkeyworld.rssm_combat_branches import (capture_combat_candidate,
    normalize_combat_branch_protocol, select_combat_branches, combat_branches_backward)


def fixture():
    model=raw_model();configure_rssm_training(model);model.core.eval()
    images=[];states=[];maps=[]
    for index in range(2):
        game=Game(seed=0,endless=True,enemies_enabled=True)
        if index:
            state=game.get_state();state['enemies'][0].update(hp=0.,alive=False,death_timer=2.99)
            game.set_state(state)
        states.append(game.state_vector());maps.append(game.map.copy());images.append(game.render(size=32,overlays=False))
    states=torch.tensor(np.stack(states));grid=torch.tensor(np.stack(maps)).long()
    image=torch.tensor(np.stack(images)).permute(0,3,1,2).float()/255
    actions=torch.zeros(2,16,4);actions[:,:,0]=.5
    with environment_profile(model,navigation_only=False),torch.no_grad():
        fmap=model.encode_map(grid);memory=model.initialize_from_image(image,states,fmap)
        initial_memory=memory.clone();current=states;predictions=[]
        for index in range(16):
            result=model.transition(current,actions[:,index],fmap,memory)
            current,memory=result['state'],result['memory'];predictions.append(current)
    return model,capture_combat_candidate(0,states,initial_memory,actions,grid,torch.stack(predictions,1))


class CombatBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_protocol_bounds_and_capture_private_memory_no_graph(self):
        model,candidate=fixture()
        self.assertEqual(normalize_combat_branch_protocol({})['offsets'],[1,4,8,16])
        for bad in ({'max_branches':5},{'max_source_probes':65},{'numeric_state_feedback':'attached'}):
            with self.assertRaises(ValueError):normalize_combat_branch_protocol(bad)
        self.assertFalse(candidate['memory'].requires_grad)
        self.assertEqual(candidate['memory_provenance'],'actualownprefix')
        bad=deepcopy(candidate);bad['new_observed_rgb_frames']=1
        with self.assertRaises(ValueError):select_combat_branches([bad],combat_branch_protocol={})

    def test_real_source_counterfactuals_include_respawn_and_no_render_or_anchor_repair(self):
        model,candidate=fixture()
        with patch.object(Game,'render',side_effect=AssertionError('No Source RGB labels')):
            chosen,report=select_combat_branches([candidate],combat_branch_protocol={})
        self.assertGreater(len(chosen),0);self.assertLessEqual(report['source_probes'],64)
        self.assertTrue(any(row['cohort']=='spawn'for row in chosen))
        for row in chosen:
            np.testing.assert_array_equal(row['labels'][0],candidate['state'][row['actor']].numpy())
        invalid=deepcopy(candidate);invalid['state'][:,:2]=0
        selected,report=select_combat_branches([invalid],combat_branch_protocol={})
        self.assertFalse(selected)
        self.assertGreater(sum(report['skip_counts'].values()),0)

    def test_real_prior_backward_reuses_exact_own_hz_no_rgb_reinit_and_codec_frozen(self):
        model,candidate=fixture();before={n:v.clone()for n,v in model.state_dict().items()}
        chosen,_=select_combat_branches([candidate],combat_branch_protocol={})
        expected=torch.stack([r['item']['memory'][r['actor']]for r in chosen])
        original=model.transition;calls=[]
        def transition(state,action,fmap,memory):
            calls.append((state.detach().clone(),memory.detach().clone(),state.requires_grad,memory.requires_grad))
            return original(state,action,fmap,memory)
        model.transition=transition
        with environment_profile(model,navigation_only=False), \
                patch.object(model,'initialize_from_image',side_effect=AssertionError('No branch reinitialization')), \
                patch.object(model,'render',side_effect=AssertionError('No renderer')), \
                patch.object(Game,'render',side_effect=AssertionError('No Source frame')):
            metrics=combat_branches_backward(model,[candidate],combat_branch_protocol={})
        self.assertEqual(len(calls),16)
        self.assertTrue(torch.equal(calls[0][1],expected))
        self.assertFalse(any(row[2]for row in calls))
        self.assertTrue(all(row[3]for row in calls[1:]))
        self.assertEqual(metrics['new_observed_rgb_frames'],0)
        self.assertEqual(metrics['memory_resets'],0)
        self.assertEqual(metrics['source_future_inputs'],0)
        self.assertGreater(metrics['source_respawn_count'],0)
        self.assertGreater(metrics['loss'],0)
        self.assertGreater(float(model.vjepa_backend.spawn_head[-1].weight.grad.abs().sum()),0)
        self.assertGreater(sum(float(p.grad.abs().sum())for p in model.core.parameters()if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in model.flow_renderer.parameters()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[n])for n,v in before.items()))


if __name__=='__main__':unittest.main()
