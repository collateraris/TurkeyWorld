"""Only coordinate parameters move; every original Adam state remains saved."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from test_rssm_replay_curriculum import setup as replay_setup, advance
from turkeyworld.rssm_replay_curriculum import bind_replay_curriculum, replay_training_horizon, commit_replay_update
from turkeyworld.rssm_coordinate_curriculum import commit_coordinate_update
from turkeyworld.rssm_combat_protocol import commit_combat_update
from turkeyworld.rssm_training_schedule import commit_combat_only_update
from turkeyworld.rssm_batch256 import commit_batch256_update
from turkeyworld.rssm_drift_only_protocol import (drift_only_plan, bind_drift_only,
    configure_drift_only_training, advance_frozen_balance_clock, frozen_balance_values,
    summarize_drift_groups, commit_drift_only_update)


def setup(folder):
    state, rp, rk, cp, ck, parameter, optimizer, _ = replay_setup(folder)
    _, balance = bind_replay_curriculum(state, rp, **rk)
    for _ in range(19): advance(state, balance, parameter, optimizer)
    plan = drift_only_plan(state, source_checkpoint='source5079.pt', source_checkpoint_sha256='source5079',
        base_plan_sha256='base', optimizer_parameter_names=['weight'])
    path = Path(folder)/'drift_only.json'; path.write_text(json.dumps(plan,sort_keys=True),encoding='utf-8')
    kwargs = dict(path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_checkpoint_sha256='source5079', loaded_checkpoint_sha256='source5079',
        base_plan_sha256='base', optimizer_parameter_names=['weight'])
    return state,plan,kwargs,balance,parameter,optimizer


def raw_report(player, enemy):
    rows={name:{'raw':value,'normalized':value,'weighted':value,'scale':1.,'weight':1.,
        'effective_weight':1.,'kl_floor':0.}for name,value in [('player',player),('enemy',enemy),('interaction',0.)]}
    return {**rows,'total':player+enemy,'kl_floor':0.}


def update_drift(state,balance,parameter,optimizer):
    q=state['rssm_training'];step=state['global_training_step']+1
    choice=replay_training_horizon(state);balance.select_update(horizon=choice['horizon'])
    optimizer.zero_grad(set_to_none=True);reports=[]
    for index in range(4):
        target=torch.linspace(0,1,64)+index*.1
        player=(parameter[0]-target).square().mean();enemy=(parameter[1]-target).square().mean()
        (player+enemy).backward();reports.append(raw_report(float(player.detach()),float(enemy.detach())))
    parameter.grad.div_(4);summary=summarize_drift_groups(reports,step=step)
    optimizer.step();commit_combat_update(state,step=step,environment='combat',batch_size=256)
    for key in ('feedback_revision','combat_recovery_revision','batch_revision','loss_balance_revision'):
        q[key]['updates']+=1;q[key]['last_optimizer_step']=step
    advance_frozen_balance_clock(balance,step=step);q['loss_balance_state']=balance.state_dict()
    commit_combat_only_update(state,step=step,environment='combat')
    q['combat_v6_revision']['updates']+=1;q['combat_v6_revision']['last_optimizer_step']=step
    sampler=q['combat_recovery_revision']['sampler_state']
    sampler['draws']+=256;sampler['v2_draws']+=256;sampler['negative_draws']+=64;sampler['positive_draws']+=64
    for key in sampler['anchor_counts']:sampler['anchor_counts'][key]+=8
    commit_batch256_update(state,step=step,sampler_draws=sampler['draws'])
    commit_coordinate_update(state,step=step,horizon=choice['horizon'],sampler_draws=sampler['draws'])
    commit_replay_update(state,step=step,horizon=choice['horizon'],forced_long_replay=choice['forced_long_replay'])
    commit_drift_only_update(state,step=step,sampler_draws=sampler['draws'])
    state['model']['weight']=parameter.detach().clone();state['optimizer']=deepcopy(optimizer.state_dict())
    return summary,choice


class DriftOnlyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_bind_only_metadata_clears_old_loss_preserves_all_adam_rng_ema_and_clocks(self):
        with TemporaryDirectory() as folder:
            state,plan,kwargs,balance,parameter,optimizer=setup(folder)
            state.update(loss=9.,latest_metrics={'combat_target':8.},loss_groups={'old':True})
            before=deepcopy(state);rng=torch.get_rng_state().clone()
            revision=bind_drift_only(state,plan,**kwargs)
            self.assertEqual(revision['updates'],0);self.assertEqual(state['global_training_step'],5079)
            self.assertEqual(state['rssm_training']['coordinate_curriculum']['updates'],65)
            self.assertEqual(state['rssm_training']['replay_curriculum']['updates'],19)
            self.assertEqual(state['loss_protocol'],before['loss_protocol'])
            self.assertIsNone(state['loss']);self.assertIsNone(state['latest_metrics']);self.assertIsNone(state['loss_groups'])
            self.assertEqual(state['rssm_training']['loss_balance_state'],before['rssm_training']['loss_balance_state'])
            self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(state['rng_state'],before['rng_state'])
            for key,value in before['optimizer']['state'][0].items():self.assertTrue(torch.equal(value,state['optimizer']['state'][0][key]))
            self.assertTrue(torch.equal(state['model']['weight'],before['model']['weight']))
            self.assertEqual(replay_training_horizon(state)['horizon'],450)
            self.assertTrue(state['training_protocol']['drift_only']['index_reservation_only'])

    def test_real_four_micro_adam_all_ema_values_frozen_clock_once_long_then_short(self):
        with TemporaryDirectory() as folder:
            state,plan,kwargs,balance,parameter,optimizer=setup(folder);bind_drift_only(state,plan,**kwargs)
            values=frozen_balance_values(balance.state);clock=float(optimizer.state[parameter]['step'])
            actor_count=balance.state['committed_updates'];component_count=balance.state['components']['committed_updates']
            results=[update_drift(state,balance,parameter,optimizer)for _ in range(2)]
            self.assertEqual([choice['horizon']for _,choice in results],[450,16])
            self.assertEqual(frozen_balance_values(balance.state),values)
            self.assertEqual(balance.state['committed_updates'],actor_count+2)
            self.assertEqual(balance.state['components']['committed_updates'],component_count+2)
            self.assertEqual(float(optimizer.state[parameter]['step']),clock+2)
            self.assertEqual(state['rssm_training']['drift_only_revision']['updates'],2)
            self.assertEqual(state['rssm_training']['coordinate_curriculum']['updates'],67)
            for summary,_ in results:
                self.assertEqual(summary['kl_floor'],0.)
                self.assertEqual(summary['interaction']['raw'],0.)
                self.assertAlmostEqual(summary['total'],summary['player']['raw']+summary['enemy']['raw'])
            bind_drift_only(state,plan,**{**kwargs,'loaded_checkpoint_sha256':'continued'})

    def test_exact_resume_keeps_phase_policy_and_next_raw_update(self):
        with TemporaryDirectory() as folder:
            state,plan,kwargs,balance,parameter,optimizer=setup(folder);bind_drift_only(state,plan,**kwargs)
            update_drift(state,balance,parameter,optimizer)
            path=Path(folder)/'saved.pt';torch.save(state,path)
            restored=torch.load(path,map_location='cpu',weights_only=False)
            bind_drift_only(restored,plan,**{**kwargs,'loaded_checkpoint_sha256':'next'})
            from turkeyworld.rssm_replay_curriculum import ReplayComponentLossBalancer
            second=ReplayComponentLossBalancer(restored['rssm_training']['loss_balance_state'])
            parameter2=torch.nn.Parameter(restored['model']['weight'].clone())
            optimizer2=torch.optim.AdamW([parameter2],lr=1e-4);optimizer2.load_state_dict(restored['optimizer'])
            update_drift(state,balance,parameter,optimizer);update_drift(restored,second,parameter2,optimizer2)
            self.assertTrue(torch.equal(parameter,parameter2));self.assertEqual(state['rssm_training'],restored['rssm_training'])
            for key,value in optimizer.state[parameter].items():self.assertTrue(torch.equal(value,optimizer2.state[parameter2][key]))

    def test_actual_heads_and_their_nonzero_adam_momenta_stay_bitexact_while_both_cores_learn_xy(self):
        from test_dual_rssm_backend import model
        from turkeyworld.rssm_migration import configure_rssm_training
        from turkeyworld.engine import Game
        world=model(True);configure_rssm_training(world);world.core.eval()
        with torch.no_grad():world.vjepa_backend.player_pass_head[-1].weight.normal_(std=.01)
        canonical=[name for name,p in world.named_parameters()if p.requires_grad];named=dict(world.named_parameters())
        optimizer=torch.optim.AdamW([named[name]for name in canonical],lr=1e-4,weight_decay=.1)
        sum(named[name].square().mean()for name in canonical).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        policy=configure_drift_only_training(world,optimizer_parameter_names=canonical);world.core.eval()
        off=set(policy['frozen_optimizer_parameter_names'])
        old_weights={name:named[name].detach().clone()for name in canonical}
        old_adam={name:deepcopy(optimizer.state[named[name]])for name in canonical}
        game=Game(seed=1,endless=True,enemies_enabled=True)
        state=torch.tensor(game.state_vector())[None];image=torch.tensor(game.render(32,overlays=False)).permute(2,0,1)[None].float()/255
        with torch.no_grad():fmap=world.encode_map(torch.tensor(game.map)[None]).detach()
        memory=world.initialize_from_image(image,state,fmap)
        result=world.transition(state,state.new_tensor([[1.,.2,.1,0.]]),fmap,memory)
        target=state[:,[0,1,8,9]]+.01
        (result['state'][:,[0,1,8,9]]-target).square().sum().backward();optimizer.step()
        self.assertTrue(off)
        for name in off:
            self.assertIsNone(named[name].grad,name)
            self.assertTrue(torch.equal(named[name],old_weights[name]),name)
            for key,value in old_adam[name].items():self.assertTrue(torch.equal(value,optimizer.state[named[name]][key]),name)
        for prefix in ('vjepa_backend.core.','vjepa_backend.enemy_rssm.core.'):
            self.assertTrue(any(name.startswith(prefix)and not torch.equal(named[name],old_weights[name])for name in canonical),prefix)
        self.assertEqual(len(optimizer.state),len(canonical))
        # Reapplying the base policy must never leave these heads trainable.
        configure_drift_only_training(world,optimizer_parameter_names=canonical)
        self.assertTrue(all(not named[name].requires_grad for name in off))

    def test_saved_checkpoint_keeps_full_optimizer_names_even_after_head_freeze(self):
        from test_dual_rssm_backend import model
        from turkeyworld.rssm_migration import configure_rssm_training
        from turkeyworld.train_rssm_combat import RSSMCombatTraining
        world=model(True);world.config=replace(world.config,navigation_only=False);world.vjepa_backend.config=world.config
        configure_rssm_training(world);canonical=[name for name,p in world.named_parameters()if p.requires_grad]
        optimizer=torch.optim.AdamW([dict(world.named_parameters())[name]for name in canonical],lr=1e-4)
        configure_drift_only_training(world,optimizer_parameter_names=canonical)
        with TemporaryDirectory() as folder:
            trainer=object.__new__(RSSMCombatTraining);trainer._optimizer_parameter_names=canonical
            trainer.loss_balancer=None;trainer.recovery_sampler=None;trainer.rng=np.random.default_rng(173)
            trainer.device=torch.device('cpu');trainer.plan_sha='base';trainer.publish=lambda:None
            trainer.checkpoint=Path(folder)/'saved.pt';trainer.state={'global_training_step':5079,
                'rssm_training':{},'combat_migration':{},'quality_status':'not qualified'}
            trainer.save(world,optimizer,{})
            saved=torch.load(trainer.checkpoint,map_location='cpu',weights_only=False)
            self.assertEqual(saved['optimizer_parameter_names'],canonical)
            self.assertEqual(saved['optimizer']['param_groups'],optimizer.state_dict()['param_groups'])
            self.assertGreater(len(canonical),sum(p.requires_grad for p in world.parameters()))

    def test_changed_source_names_settings_values_partial_or_nan_fail_before_commit(self):
        with TemporaryDirectory() as folder:
            state,plan,kwargs,balance,parameter,optimizer=setup(folder)
            for key,value in [('source_checkpoint_sha256','wrong'),('settings',{}),('optimizer_parameter_names',[]),('frozen_ema_values_sha256','wrong')]:
                bad=deepcopy(plan);bad[key]=value
                with self.assertRaises(ValueError):bind_drift_only(deepcopy(state),bad,**kwargs)
            with self.assertRaises(ValueError):bind_drift_only(deepcopy(state),plan,**{**kwargs,'loaded_checkpoint_sha256':'wrong'})
            bind_drift_only(state,plan,**kwargs);old=deepcopy(balance.state_dict())
            with self.assertRaises(ValueError):advance_frozen_balance_clock(balance,step=state['global_training_step']+2)
            self.assertEqual(balance.state_dict(),old)
            with self.assertRaises(FloatingPointError):summarize_drift_groups([raw_report(float('nan'),1.)],step=5080)
            with self.assertRaises(ValueError):commit_drift_only_update(state,step=5080,sampler_draws=0)


if __name__=='__main__':unittest.main()
