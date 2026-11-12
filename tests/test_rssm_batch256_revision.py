"""Explicit64x4 continuation preserves128-example history and all Adam state."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from test_rssm_combat_v6_revision import fixture as v6_fixture
from turkeyworld.rssm_batch256 import (
    BATCH_PROTOCOL, bind_batch256_revision, commit_batch256_update,
    batch256_boundary_valid, scaled_branch_protocol,
)
from turkeyworld.rssm_batch_revision import batch_commit_size
from turkeyworld.rssm_combat_protocol import commit_combat_update
from turkeyworld.rssm_combat_v6 import bind_combat_v6
from turkeyworld.rssm_component_balance import LOSS_PROTOCOL
from turkeyworld.rssm_spawn_curriculum import upgrade_attack_balanced_sampler_state
from turkeyworld.rssm_training_schedule import bind_combat_only_revision, commit_combat_only_update


def update(state, balancer, parameter, optimizer, *, effective):
    training = state['rssm_training']; step = state['global_training_step']+1
    optimizer.zero_grad(set_to_none=True); reports = []
    for index in range(4):
        # A real64-example objective in the new branch, rather than four Adam calls.
        target = torch.linspace(0., 1., effective//4, dtype=parameter.dtype)+index*.1
        acc = balancer.new_microbatch('combat')
        loss = acc.combine({'player': (parameter[0]-target).square().mean()+1.1,
            'enemy': (parameter[1]-target).square().mean()+1.1,
            'interaction': parameter.square().sum()}, floors={'player':1.1,'enemy':1.1})
        loss += acc.combine({'enemy': (parameter[0]-target).square().mean()}, component='spawn')
        loss += acc.combine({'player': parameter[0].square(), 'enemy': parameter[1].square(),
            'interaction': parameter.sum().square()}, component='recovery')
        loss.backward(); reports.append(acc.report())
    parameter.grad.div_(4)
    report = balancer.summarize(reports, environment='combat', step=step)
    optimizer.step(); commit_combat_update(state, step=step, environment='combat', batch_size=effective)
    for name in ('feedback_revision','combat_recovery_revision','batch_revision','loss_balance_revision'):
        training[name]['updates'] += 1; training[name]['last_optimizer_step'] = step
    balancer.commit(report, step=step); training['loss_balance_state'] = balancer.state_dict()
    commit_combat_only_update(state, step=step, environment='combat')
    training['combat_v6_revision']['updates'] += 1
    training['combat_v6_revision']['last_optimizer_step'] = step
    # Actual Source sampling is tested separately; this fixture advances its
    # persisted quarter ledger by exactly the number of completed Source draws.
    sampler = training['combat_recovery_revision']['sampler_state']
    sampler['draws'] += effective; sampler['v2_draws'] += effective
    sampler['negative_draws'] += effective//4; sampler['positive_draws'] += effective//4
    for anchor in sampler['anchor_counts']: sampler['anchor_counts'][anchor] += effective//32
    if training.get('batch256_revision'):
        commit_batch256_update(state, step=step, sampler_draws=sampler['draws'])
    state['model']['weight'] = parameter.detach().clone()
    state['optimizer'] = deepcopy(optimizer.state_dict())
    return report


def fixture(folder):
    state, v6_plan, calibration, v6_kwargs, _ = v6_fixture(folder)
    _, balance, _ = bind_combat_v6(state, v6_plan, calibration, **v6_kwargs)
    q = state['rssm_training']
    q['combat_recovery_revision']['sampler_state'] = upgrade_attack_balanced_sampler_state(q['combat_recovery_revision']['sampler_state'])
    parameter = torch.nn.Parameter(state['model']['weight'].clone())
    optimizer = torch.optim.AdamW([parameter], lr=1e-4); optimizer.load_state_dict(deepcopy(state['optimizer']))
    update(state, balance, parameter, optimizer, effective=128)
    update(state, balance, parameter, optimizer, effective=128)
    step = state['global_training_step']
    probe = {'passed':True,'source_step':step,'source_checkpoint_sha256':'source256',
        'microbatch_size':64,'optimizer_updated':False,'model_weights_and_buffers_unchanged':True,
        'all_gradients_finite':True}
    probe_path = Path(folder)/'probe64.json'; probe_path.write_text(json.dumps(probe,sort_keys=True),encoding='utf-8')
    probe_sha = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    plan = {'version':7,'batch_protocol':BATCH_PROTOCOL,'loss_protocol':LOSS_PROTOCOL,
        'microbatch_size':64,'gradient_accumulation_steps':4,'effective_batch_size':256,
        'previous_effective_batch_size':128,'gradient_reduction':'mean_over_microbatches',
        'source_checkpoint_sha256':'source256','batch64_probe_sha256':probe_sha,
        'parent_v6_sha256':q['combat_v6_revision']['sha256'],'parent_batch_sha256':q['batch_revision']['sha256'],
        'optimizer_reset':False,'source_future_feedback':False,'start_optimizer_step':step,
        'combat_branch_protocol':scaled_branch_protocol(state['training_protocol']['combat_branch_protocol'])}
    path = Path(folder)/'batch256.json'; path.write_text(json.dumps(plan,sort_keys=True),encoding='utf-8')
    kwargs = {'path':path,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'probe_sha256':probe_sha,
        'source_checkpoint_sha256':'source256','loaded_checkpoint_sha256':'source256',
        'cli_microbatch':64,'cli_accumulation':4}
    return state,plan,probe,kwargs,balance,parameter,optimizer


def rebind_schedule(state):
    q = state['rssm_training']['combat_only_revision']; path = Path(q['plan'])
    plan = json.loads(path.read_text(encoding='utf-8'))
    return bind_combat_only_revision(state,plan,path=path,sha256=q['sha256'],
        source_checkpoint_sha256=q['initial_loaded_checkpoint_sha256'],loaded_checkpoint_sha256='later_checkpoint',
        base_plan_sha256=plan['base_plan_sha256'],loss_balance_plan_sha256=plan['loss_balance_plan_sha256'])


class Batch256RevisionTests(unittest.TestCase):
    def test_initial_bind_preserves_all_old_model_adam_rng_ema_clocks_and_immutable_history(self):
        with TemporaryDirectory() as folder:
            state,plan,probe,kwargs,balance,_,_=fixture(folder)
            old=deepcopy(state);rng=torch.get_rng_state().clone()
            original={p:p.read_bytes()for p in Path(folder).glob('*.json')}
            revision,branches=bind_batch256_revision(state,plan,probe,**kwargs)
            self.assertEqual(state['loss_protocol'],LOSS_PROTOCOL)
            self.assertEqual(state['batch_protocol'],BATCH_PROTOCOL)
            self.assertTrue(batch256_boundary_valid(state));self.assertEqual(batch_commit_size(state),256)
            self.assertEqual(revision['updates'],0)
            for key,value in old['rssm_training'].items():self.assertEqual(state['rssm_training'][key],value,key)
            for key in ('rng_state','global_training_step','current'):self.assertEqual(state[key],old[key])
            self.assertTrue(torch.equal(state['torch_rng_state'],old['torch_rng_state']))
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))
            self.assertTrue(torch.equal(state['model']['weight'],old['model']['weight']))
            for key,value in old['optimizer']['state'][0].items():self.assertTrue(torch.equal(value,state['optimizer']['state'][0][key]))
            self.assertEqual(state['optimizer']['param_groups'],old['optimizer']['param_groups'])
            self.assertEqual(balance.state_dict(),old['rssm_training']['loss_balance_state'])
            self.assertEqual((branches['max_branches'],branches['max_source_probes'],branches['max_direct_source_queries']),(16,128,256))
            self.assertEqual(branches['weight'],old['training_protocol']['combat_branch_protocol']['weight'])
            for path,data in original.items():self.assertEqual(path.read_bytes(),data)
            count=len(state['batch_history'])
            self.assertIs(bind_batch256_revision(state,plan,probe,**kwargs)[0],revision)
            self.assertEqual(len(state['batch_history']),count)

    def test_real_four64_microbatches_one_adam_ema_commit_and_piecewise_draw_history(self):
        with TemporaryDirectory() as folder:
            state,plan,probe,kwargs,balance,parameter,optimizer=fixture(folder)
            before=deepcopy(state['rssm_training']);start=state['global_training_step']
            old_clock=float(optimizer.state_dict()['state'][0]['step'])
            revision,_=bind_batch256_revision(state,plan,probe,**kwargs)
            report=update(state,balance,parameter,optimizer,effective=256)
            self.assertEqual(report['microbatches'],4)
            self.assertEqual(float(optimizer.state_dict()['state'][0]['step']),old_clock+1)
            self.assertTrue(batch256_boundary_valid(state));self.assertEqual(batch_commit_size(state),256)
            self.assertEqual(revision['updates'],1);self.assertEqual(revision['last_optimizer_step'],start+1)
            training=state['rssm_training'];schedule=training['combat_only_revision']
            self.assertEqual(training['combat_auxiliary_draws']-schedule['combat_auxiliary_draws_at_start'],2*128+256)
            self.assertEqual(training['auxiliary_draws']-before['auxiliary_draws'],256)
            self.assertEqual(training['source_completed'],before['source_completed']+1)
            self.assertEqual(training['loss_balance_state']['committed_updates'],before['loss_balance_state']['committed_updates']+1)
            self.assertEqual(training['loss_balance_state']['components']['committed_updates'],before['loss_balance_state']['components']['committed_updates']+1)
            self.assertEqual(training['loss_balance_state']['banks']['navigation'],before['loss_balance_state']['banks']['navigation'])
            self.assertEqual(training['navigation_auxiliary_draws'],before['navigation_auxiliary_draws'])
            # The old combat-only binder must accept 128+128+256, not3*256.
            self.assertEqual(rebind_schedule(state),schedule)
            wrong=deepcopy(state)
            wrong['rssm_training']['combat_auxiliary_draws']=schedule['combat_auxiliary_draws_at_start']+3*256
            with self.assertRaises(ValueError):rebind_schedule(wrong)
            path=Path(folder)/'after.pt';torch.save(state,path)
            restored=torch.load(path,map_location='cpu',weights_only=False)
            old=deepcopy(restored['rssm_training'])
            bind_batch256_revision(restored,plan,probe,**{**kwargs,'loaded_checkpoint_sha256':'later_saved_checkpoint'})
            self.assertEqual(restored['rssm_training'],old)
            self.assertEqual(batch_commit_size(restored),256)
            self.assertEqual(rebind_schedule(restored),schedule)

    def test_old128_commit_and_half_or_duplicate256_commit_are_rejected(self):
        with TemporaryDirectory() as folder:
            state,plan,probe,kwargs,balance,parameter,optimizer=fixture(folder)
            revision,_=bind_batch256_revision(state,plan,probe,**kwargs)
            before=deepcopy(revision)
            with self.assertRaises(ValueError):commit_combat_update(state,step=state['global_training_step']+1,environment='combat',batch_size=128)
            with self.assertRaises(ValueError):commit_batch256_update(state,step=state['global_training_step']+1,sampler_draws=before['sampler_draws_at_start']+64)
            self.assertEqual(revision,before)
            update(state,balance,parameter,optimizer,effective=256)
            committed=deepcopy(revision)
            with self.assertRaises(ValueError):commit_batch256_update(state,step=state['global_training_step'],sampler_draws=state['rssm_training']['combat_recovery_revision']['sampler_state']['draws'])
            self.assertEqual(revision,committed)

    def test_source_cli_probe_and_scaled_coverage_pins_fail_before_binding(self):
        with TemporaryDirectory() as folder:
            for key,value in [('version',True),('microbatch_size',32),('gradient_accumulation_steps',2),
                ('effective_batch_size',128),('previous_effective_batch_size',256),('source_checkpoint_sha256','wrong'),
                ('batch64_probe_sha256','wrong'),('parent_v6_sha256','wrong'),('parent_batch_sha256','wrong'),
                ('optimizer_reset',True),('source_future_feedback',True),('start_optimizer_step',1)]:
                state,plan,probe,kwargs,*_=fixture(folder);changed=deepcopy(plan);changed[key]=value
                with self.assertRaises(ValueError):bind_batch256_revision(state,changed,probe,**kwargs)
                self.assertNotIn('batch256_revision',state['rssm_training'])
            state,plan,probe,kwargs,*_=fixture(folder)
            for key,value in [('passed',False),('source_step',1),('source_checkpoint_sha256','wrong'),
                ('microbatch_size',32),('optimizer_updated',True),('model_weights_and_buffers_unchanged',False),('all_gradients_finite',False)]:
                changed=deepcopy(probe);changed[key]=value
                with self.assertRaises(ValueError):bind_batch256_revision(deepcopy(state),plan,changed,**kwargs)
            for change in ({'cli_microbatch':32},{'cli_accumulation':2},{'loaded_checkpoint_sha256':'wrong_source_weights'}):
                with self.assertRaises(ValueError):bind_batch256_revision(deepcopy(state),plan,probe,**{**kwargs,**change})
            reduced=deepcopy(plan);reduced['combat_branch_protocol']['max_branches']=8
            with self.assertRaises(ValueError):bind_batch256_revision(deepcopy(state),reduced,probe,**kwargs)

    def test_invalid_parent_or_missing_revision_cannot_silently_return128_and_resume_counters_reject(self):
        with TemporaryDirectory() as folder:
            state,plan,probe,kwargs,balance,parameter,optimizer=fixture(folder)
            for parent in ('combat_v6_revision','loss_balance_revision','combat_only_revision','batch_revision','combat_recovery_revision','feedback_revision'):
                broken=deepcopy(state);broken['rssm_training'][parent]['last_optimizer_step']-=1
                with self.assertRaises(ValueError):bind_batch256_revision(broken,plan,probe,**kwargs)
            bind_batch256_revision(state,plan,probe,**kwargs)
            missing=deepcopy(state);missing['rssm_training'].pop('batch256_revision')
            with self.assertRaises(ValueError):batch_commit_size(missing)
            update(state,balance,parameter,optimizer,effective=256)
            for mutate in ('sampler','draws','nav','revision'):
                broken=deepcopy(state);q=broken['rssm_training']
                if mutate=='sampler':q['combat_recovery_revision']['sampler_state']['draws']+=1
                elif mutate=='draws':q['combat_auxiliary_draws']+=1
                elif mutate=='nav':q['navigation_auxiliary_draws']+=1
                else:q['batch256_revision']['updates']+=1
                with self.assertRaises(ValueError):bind_batch256_revision(broken,plan,probe,**{**kwargs,'loaded_checkpoint_sha256':'later'})


if __name__=='__main__':unittest.main()
