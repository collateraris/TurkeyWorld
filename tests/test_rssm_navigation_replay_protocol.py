"""Three-way main actors, exact PCG/ledger resume, and undiluted combat heads."""
import copy,hashlib,json,random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_rssm_moe_training import TinyModel,data,oracle
from turkeyworld.engine import Game
from turkeyworld.rssm_moe_protocol import moe_training_plan,bind_moe_training,commit_moe_update
from turkeyworld.rssm_moe_training import moe_dynamics_backward
from turkeyworld.rssm_navigation_replay_protocol import (navigation_replay_plan,bind_navigation_replay,
    commit_navigation_replay,NavigationReplaySampler,EnemyMotionReplaySampler,SETTINGS)
from turkeyworld.renderer_native_phase import _same
from turkeyworld.train_rssm_moe import merge_main_replay_metrics
from turkeyworld.rssm_accumulation import accumulated_metrics


class Tape:
    def __init__(self,enemy,split='train'):
        self.split=split;game=Game(seed=73,endless=True,enemies_enabled=enemy)
        states=[game.state_vector()]
        for _ in range(451):game.step([0,0,0,0],dt=1/30);states.append(game.state_vector())
        self.states=torch.from_numpy(np.stack(states));self.episodes=[{'frames':range(452)}for _ in range(3)]
        self.indices=[]
    def batch(self,n,*,context,horizon,indices):
        self.indices.append(np.asarray(indices).copy())
        states=torch.stack([self.states[int(start):int(start)+context+horizon]for _,start in indices])
        return dict(states=states,actions=torch.zeros(n,horizon,4),target_states=states[:,context:])


class NavigationReplayProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.source=self.root/'source.pt';self.source.write_bytes(b'full-source-placeholder-for-metadata-test')
        self.navroot=self.root/'nav';self.enemyroot=self.root/'enemy'
        for root in(self.navroot,self.enemyroot):root.mkdir();(root/'manifest.json').write_text('{"status":"complete"}')
        self.state={'global_training_step':5322,'rssm_training':{'old_nav_ema':{'value':[3,7]}},
            '_model':torch.arange(4),'_optimizer':{'moments':torch.arange(8)},'_rng':np.random.default_rng(1).bit_generator.state}
        p=moe_training_plan(self.state,source_checkpoint=str(self.source),source_checkpoint_sha256='a'*64,
            architecture_migration_sha256='b'*64,data_roots={},source_identities={})
        path=self.root/'moe.json';path.write_text(json.dumps(p));h=hashlib.sha256(path.read_bytes()).hexdigest()
        bind_moe_training(self.state,p,path=path,sha256=h,loaded_checkpoint_sha256='a'*64)
        self.plan=navigation_replay_plan(self.state,source_checkpoint=self.source,navigation_root=self.navroot,enemy_motion_root=self.enemyroot)
        self.path=self.root/'replay.json';self.path.write_text(json.dumps(self.plan));self.sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
    def bind(self,state=None):
        return bind_navigation_replay(self.state if state is None else state,self.plan,path=self.path,sha256=self.sha,
            loaded_checkpoint_sha256=self.plan['source_checkpoint_sha256'])
    def test_init_preserves_all_old_gate_weights_moments_rng_and_private_seeds_do_not_touch_globals(self):
        original=copy.deepcopy(self.state);np_before=copy.deepcopy(np.random.get_state());torch_before=torch.get_rng_state().clone()
        record=self.bind()
        for key in('_model','_optimizer','_rng'):self.assertTrue(_same(original[key],self.state[key]))
        self.assertTrue(_same(original['rssm_training']['moe_training'],self.state['rssm_training']['moe_training']))
        self.assertEqual(self.state['rssm_training']['old_nav_ema'],original['rssm_training']['old_nav_ema'])
        self.assertTrue(_same(np_before,np.random.get_state()));self.assertTrue(torch.equal(torch_before,torch.get_rng_state()))
        self.assertEqual(record['updates'],0);self.assertEqual(SETTINGS['enemy_motion_main_actors_per_update'],64)
        self.assertEqual(SETTINGS['navigation_fraction'],.15625)
    def test_four_real_source_cohorts_commit40_64_152_then_exact_private_resume_at450(self):
        record=self.bind();nav=Tape(False);enemy=Tape(True)
        n=NavigationReplaySampler(nav,record);e=EnemyMotionReplaySampler(enemy,record)
        rows=[]
        for _ in range(4):
            a,_=n.batch(context=2,horizon=4);b,_=e.batch(context=2,horizon=4)
            self.assertEqual(len(a['states']),10);self.assertEqual(len(b['states']),16)
            rows.append(dict(main_navigation_actor_count=10,main_enemy_motion_actor_count=16,main_combat_actor_count=38,main_discarded_combat_rng_count=26))
        self.state['global_training_step']+=1
        commit_moe_update(self.state,step=5323,horizon=4,train_loss=.6)
        commit_navigation_replay(self.state,step=5323,sampler_state=n.state_dict(),enemy_sampler_state=e.state_dict(),micro_rows=rows)
        self.assertEqual((record['navigation_main_actors'],record['enemy_motion_main_actors'],record['combat_main_actors'],record['discarded_combat_rng_draws']),(40,64,152,104))
        copied=copy.deepcopy(self.state);restored=self.bind(copied)
        n2=NavigationReplaySampler(nav,restored);e2=EnemyMotionReplaySampler(enemy,restored)
        for first,second in((n,n2),(e,e2)):
            _,a=first.batch(context=2,horizon=450);_,b=second.batch(context=2,horizon=450)
            self.assertEqual(a,b);self.assertTrue(all(start==0 for _,start in a['pairs']))
        self.assertEqual(restored['updates'],1)
    def test_partial_or_double_commit_bad_source_identity_and_val_sampling_rejected(self):
        record=self.bind();n=NavigationReplaySampler(Tape(False),record);e=EnemyMotionReplaySampler(Tape(True),record)
        rows=[dict(main_navigation_actor_count=10,main_enemy_motion_actor_count=16,main_combat_actor_count=38,main_discarded_combat_rng_count=26)]*4
        for _ in range(4):n.batch(context=1,horizon=4);e.batch(context=1,horizon=4)
        before=copy.deepcopy(record);self.state['global_training_step']+=1
        with self.assertRaises(ValueError):commit_navigation_replay(self.state,step=5323,sampler_state=n.state_dict(),enemy_sampler_state=e.state_dict(),micro_rows=rows[:3])
        self.assertEqual(before,record)
        commit_moe_update(self.state,step=5323,horizon=4,train_loss=.6)
        commit_navigation_replay(self.state,step=5323,sampler_state=n.state_dict(),enemy_sampler_state=e.state_dict(),micro_rows=rows)
        with self.assertRaises(ValueError):commit_navigation_replay(self.state,step=5323,sampler_state=n.state_dict(),enemy_sampler_state=e.state_dict(),micro_rows=rows)
        bad=copy.deepcopy(self.state);bad['rssm_training']['navigation_replay_revision']['navigation_main_actors']+=1
        with self.assertRaises(ValueError):self.bind(bad)
        (self.enemyroot/'manifest.json').write_text('tampered')
        with self.assertRaises(ValueError):self.bind(self.state)
        with self.assertRaises(ValueError):NavigationReplaySampler(Tape(False,'val'),record)
    def test_sampler_wrong_role_or_fire_rolls_back_private_pcg_and_no_external_rng_consumption(self):
        record=self.bind();n=NavigationReplaySampler(Tape(True),record);before=n.state_dict()
        with self.assertRaises(ValueError):n.batch(context=1,horizon=16)
        self.assertEqual(n.state_dict(),before)
        good=NavigationReplaySampler(Tape(False),record);rng=np.random.default_rng(73);control=np.random.default_rng(73)
        for _ in range(4):
            retained=rng.integers(64,size=64)[:38]
            expected=control.integers(64,size=64)[:38]
            good.batch(context=1,horizon=16)
            np.testing.assert_array_equal(retained,expected)
        self.assertEqual(rng.bit_generator.state,control.bit_generator.state)


class MainWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def execute(self,model,**weights):
        with patch('turkeyworld.rssm_combat_branches._source_step_at_own_state',side_effect=oracle):
            return moe_dynamics_backward(model,data(4),data(4),expected_horizon=4,**weights)
    def test_default_ones_exact_and_weights_do_not_dilute_aux_late_or_combat_head_gradients(self):
        torch.manual_seed(7);model=TinyModel();a=copy.deepcopy(model);b=copy.deepcopy(model)
        original=self.execute(model);explicit=self.execute(a,main_player_weight=1.,main_router_weight=1.,main_enemy_weight=1.)
        self.assertEqual(original,explicit)
        for x,y in zip(model.parameters(),a.parameters()):
            if x.grad is None:self.assertIsNone(y.grad)
            else:self.assertTrue(torch.equal(x.grad,y.grad))
        weighted=self.execute(b,main_player_weight=38/64,main_router_weight=38/64,main_enemy_weight=38/54)
        for key in('moe_auxiliary','moe_late_own_prefix','moe_combat_loss'):self.assertEqual(original[key],weighted[key])
        self.assertAlmostEqual(weighted['moe_main_player_loss'],original['moe_main_player_loss']*38/64,places=7)
        self.assertAlmostEqual(weighted['moe_main_enemy_loss'],original['moe_main_enemy_loss']*38/54,places=7)
        for name in('target','attack','events','tile','anchor'):
            self.assertTrue(torch.equal(getattr(model,name).weight.grad,getattr(b,name).weight.grad))
    def test_merge_role_loss_conservation_and_count_sums_for_four_micros(self):
        c=dict(moe_player_loss=2.,moe_enemy_loss=3.,moe_combat_loss=4.,moe_router_role_loss=.1,moe_router_balance_loss=.01)
        n=dict(nav_player_loss=.4,nav_router_role_loss=.02,nav_router_balance_loss=.002)
        e=dict(enemy_motion_enemy_loss=.8,enemy_motion_router_role_loss=.03,enemy_motion_router_balance_loss=.003)
        row=merge_main_replay_metrics(c,n,e,plan_sha256='a'*64)
        self.assertAlmostEqual(row['train_loss'],2.4+3.8+4.+.165)
        self.assertEqual(row['moe_combat_loss'],4.)
        totals=accumulated_metrics([row]*4)
        self.assertEqual((totals['main_combat_actor_count'],totals['main_enemy_motion_actor_count'],totals['main_navigation_actor_count'],totals['main_actor_count']),(152,64,40,256))
        self.assertEqual(totals['combat_auxiliary_actor_count'],256)
        self.assertEqual(totals['main_discarded_combat_rng_count'],104)


if __name__=='__main__':unittest.main()
