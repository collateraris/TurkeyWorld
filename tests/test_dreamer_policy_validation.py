from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from turkeyworld.engine import Game
from turkeyworld.dreamer_reward_continue import REWARD_SETTINGS
from turkeyworld.dreamer_policy_validation import(build_policy_validation_panel,evaluate_policy,
    validate_panel,file_sha)
from turkeyworld.serve_dreamer_policy import monitor_payload


@dataclass(frozen=True)
class TinyWorldState:
    state:torch.Tensor
    memory:torch.Tensor
    map_features:torch.Tensor


class TinyAdapter:
    def __init__(self):
        self.world=nn.Linear(1,1);self.heads=nn.Linear(1,1);self.sample_latents=True;self.sessions=[]
    def initialize(self,state,grid,rgb):
        self.sessions.append([])
        return TinyWorldState(state.clone(),torch.zeros(1,1),grid.float())
    def features(self,current):return torch.cat((current.state,current.memory),-1)
    def step(self,current,action):
        self.sessions[-1].append((float(current.state[0,0]),float(current.memory[0,0])))
        torch.rand(1);random.random();np.random.random()
        next_state=current.state.clone();next_state[:,0]+=.001+.002*action[:,0]
        following=TinyWorldState(next_state,current.memory+1,current.map_features)
        return dict(world_state=following,reward=torch.tensor([3.]),continue_prob=torch.tensor([.9]))


def write_source(root,profile,seed,*,terminal=False):
    game=Game(seed=seed,endless=True,enemies_enabled=profile!='navigation')
    if terminal:
        game.player['hp']=8.;game.enemies[0].update(x=game.player['x']+.65,y=game.player['y'],attack_cooldown=0.)
    directory=root/profile/'rgb/val';directory.mkdir(parents=True,exist_ok=True);path=directory/f'episode_{seed}.npz'
    initial=game.get_state();state=game.state_vector();image=game.render(size=32,overlays=False)
    states=state[None]
    if terminal:
        game.step(np.zeros(4),dt=1/30);states=np.stack((state,game.state_vector()))
        if states[-1,4]>0:raise AssertionError('Fixture must contain genuine Source player death')
    np.savez(path,initial_state=np.asarray(json.dumps(initial)),states=states,frames=image[None],map=game.map,
        split=np.asarray('val'),dt=np.asarray(1/30))
    return dict(relative_path=path.relative_to(root).as_posix(),sha256=file_sha(path))


class DreamerPolicyValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_policy_Source_reward_is_separate_from_imagined_return_and_free_tapes_are_not_corrected(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);records=[write_source(root,'navigation',100),write_source(root,'full_game',101)]
            (root/'manifest.json').write_text(json.dumps({'records':records}))
            panel=build_policy_validation_panel(root,count=2);self.assertEqual(panel,build_policy_validation_panel(root,count=2))
            adapter=TinyAdapter();before=torch.get_rng_state().clone();py=random.getstate();nr=np.random.get_state()
            flags=[p.requires_grad for obj in(adapter.world,adapter.heads)for p in obj.parameters()]
            action=lambda f:torch.tensor([[1.,0.,0.,0.]])
            report=evaluate_policy(action,adapter,panel,step=10,
                policy_identity={'checkpoint_sha256':'a'*64,'step':10,'plan_sha256':'b'*64},
                world_identity={'checkpoint_sha256':'c'*64,'step':6746},reward_settings=REWARD_SETTINGS,horizon=4)
            self.assertFalse(report['gameplay_qualified']);self.assertFalse(report['automatic_policy_deployment'])
            self.assertFalse(report['learned_world_Source_feedback']);self.assertTrue(report['frozen_models_bitexact'])
            learned=report['arms']['learned_policy'];self.assertEqual(learned['source_return']['mean'],0.)
            self.assertEqual(learned['learned_predicted_head_return']['mean'],12.)
            self.assertGreater(learned['matched_final_player_drift_tiles']['mean'],0)
            self.assertTrue(all(row['source']['time_limit_truncated']for row in learned['cases']))
            self.assertTrue(adapter.sample_latents);self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertEqual(py,random.getstate());self.assertTrue(all(np.array_equal(a,b)for a,b in zip(nr,np.random.get_state())))
            self.assertEqual(flags,[p.requires_grad for obj in(adapter.world,adapter.heads)for p in obj.parameters()])
            # Source-controller prior memory is continuous; independent free
            # replay follows its own biased coordinates, not Source corrections.
            for first in range(0,len(adapter.sessions),3):
                source,free,replay=adapter.sessions[first:first+3]
                self.assertEqual([row[1]for row in source],[0.,1.,2.,3.])
                self.assertEqual([row[1]for row in replay],[0.,1.,2.,3.])
            json.dumps(report,allow_nan=False)

    def test_invalid_panel_TRAIN_identity_future_policy_and_bad_actions_fail_closed(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);records=[write_source(root,'navigation',100)]
            (root/'manifest.json').write_text(json.dumps({'records':records}));panel=build_policy_validation_panel(root,count=1)
            for mutate in(lambda p:p.update(split='train'),lambda p:p['starts'][0].update(split='train'),
                lambda p:p.update(fingerprint='changed'),lambda p:p['starts'].append(p['starts'][0])):
                invalid=deepcopy(panel);mutate(invalid)
                with self.assertRaises(ValueError):validate_panel(invalid)
            kwargs=dict(step=0,policy_identity={'checkpoint_sha256':'a'*64,'step':0,'plan_sha256':'b'*64},
                world_identity={'checkpoint_sha256':'c'*64,'step':1},reward_settings=REWARD_SETTINGS,horizon=1)
            with self.assertRaisesRegex(ValueError,'action4'):
                evaluate_policy(lambda f:torch.tensor([[2.,0.,0.,0.]]),TinyAdapter(),panel,**kwargs)
            kwargs['horizon']=451
            with self.assertRaisesRegex(ValueError,'no longer than450'):
                evaluate_policy(lambda f:torch.zeros(1,4),TinyAdapter(),panel,**kwargs)

    def test_terminal_opt_in_preserves_default_and_pins_navigation_combat_and_real_death(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);records=[write_source(root,'navigation',100),write_source(root,'navigation',101),
                write_source(root,'full_game',102),write_source(root,'moving1',103,terminal=True)]
            (root/'manifest.json').write_text(json.dumps({'records':records}))
            original=build_policy_validation_panel(root,count=3,seed=174)
            self.assertNotIn('include_terminal',original)
            self.assertEqual(original,build_policy_validation_panel(root,count=3,seed=174,include_terminal=False))
            terminal=build_policy_validation_panel(root,count=4,seed=174,include_terminal=True)
            self.assertEqual(terminal['recorded_terminal_starts'],1);self.assertTrue(terminal['include_terminal'])
            self.assertEqual({r['environment']for r in terminal['starts']},{'navigation','combat'})
            validate_panel(terminal)

    def test_monitor_reads_only_own_run_history_and_verified_immutable_validation_files(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);self.assertEqual(monitor_payload(root)['status'],'waiting')
            policy=root/'policy.pt';world=root/'world.pt';policy.write_bytes(b'fixed saved policy');world.write_bytes(b'fixed world')
            report=dict(step=10,panel_fingerprint='panel',data_identity_sha256='VALdata',
                policy_identity={'checkpoint_sha256':file_sha(policy),'plan_sha256':'plan'},
                world_identity={'checkpoint_sha256':file_sha(world),'step':6746})
            report_path=root/'report.json';report_path.write_text(json.dumps(report))
            receipt=dict(report_file=str(report_path),report_sha256=file_sha(report_path),policy_checkpoint_file=str(policy),
                policy_checkpoint_sha256=file_sha(policy),world_checkpoint_file=str(world),world_sha256=file_sha(world),
                world_step=6746,policy_step=10,panel_fingerprint='panel',validation_data_identity_sha256='VALdata',plan_sha256='plan')
            (root/'status.json').write_text(json.dumps({'step':20,'validation_receipt':receipt}))
            (root/'history.jsonl').write_text('{"step":10}\npartial')
            payload=monitor_payload(root);self.assertTrue(payload['validation_receipt_verified']);self.assertEqual(payload['history'],[{'step':10}])
            policy.write_bytes(b'tampered saved policy')
            payload=monitor_payload(root);self.assertFalse(payload['validation_receipt_verified'])
            self.assertIn('SHA',payload['validation_receipt_error'])


if __name__=='__main__':unittest.main()
