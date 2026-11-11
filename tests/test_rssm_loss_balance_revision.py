"""Preserved32x4 boundaries, exact source pins and durable task-normalizer EMA."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from turkeyworld.rssm_batch_revision import bind_batch_revision,batch_commit_size
from turkeyworld.rssm_combat_protocol import commit_combat_update
from turkeyworld.rssm_loss_balance import (SETTINGS,LOSS_PROTOCOL,bind_loss_balance_revision,
    balanced_batch_boundary_valid)
from test_rssm_batch_revision import fixture as batch_fixture


def fixture(folder):
    state,batch,batch_kwargs,parent=batch_fixture(folder)
    ledger=bind_batch_revision(state,batch,**batch_kwargs)
    plan={'version':1,'loss_protocol':LOSS_PROTOCOL,'settings':deepcopy(SETTINGS),
        'source_future_feedback':False,'optimizer_reset':False,'base_plan_sha256':'base',
        'batch_revision_sha256':'v4-sha','source_checkpoint_sha256':'source',
        'start_optimizer_step':state['global_training_step']}
    path=Path(folder)/'v5.json';path.write_text(json.dumps(plan,sort_keys=True),encoding='utf-8')
    kwargs={'path':path,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_checkpoint_sha256':'source','base_plan_sha256':'base','batch_revision_sha256':'v4-sha',
        'loaded_checkpoint_sha256':'source'}
    return state,plan,kwargs,ledger,parent


class LossBalanceRevisionTests(unittest.TestCase):
    def test_exact_bind_keeps_all_weights_adam_rng_counters_history_and_plan_bytes(self):
        with TemporaryDirectory()as folder:
            state,plan,kwargs,batch,parent=fixture(folder)
            old=deepcopy(state);parent_bytes=parent.read_bytes();plan_bytes=kwargs['path'].read_bytes()
            rng=torch.get_rng_state().clone()
            revision,balancer=bind_loss_balance_revision(state,plan,**kwargs)
            self.assertEqual(state['loss_protocol'],LOSS_PROTOCOL)
            self.assertEqual(revision['initial_loaded_checkpoint_sha256'],'source')
            self.assertEqual(revision['updates'],0);self.assertEqual(batch_commit_size(state),128)
            self.assertTrue(balanced_batch_boundary_valid(state))
            for key in ('global_training_step','current','completed_fragments','rng_state'):
                self.assertEqual(state[key],old[key])
            for key in ('combat_completed','source_completed','auxiliary_draws','combat_auxiliary_draws',
                'navigation_auxiliary_draws','last_validation_report','last_validation_passed',
                'combat_recovery_revision','feedback_revision','batch_revision'):
                self.assertEqual(state['rssm_training'][key],old['rssm_training'][key])
            for key,value in old['model'].items():self.assertTrue(torch.equal(value,state['model'][key]))
            for key,value in old['optimizer']['state'][0].items():self.assertTrue(torch.equal(value,state['optimizer']['state'][0][key]))
            self.assertTrue(torch.equal(state['torch_rng_state'],old['torch_rng_state']))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(parent.read_bytes(),parent_bytes);self.assertEqual(kwargs['path'].read_bytes(),plan_bytes)
            history=len(state['objective_history'])
            again,reloaded=bind_loss_balance_revision(state,plan,**kwargs)
            self.assertIs(again,revision);self.assertEqual(reloaded.state_dict(),balancer.state_dict())
            self.assertEqual(len(state['objective_history']),history)

    def test_four_micro_gradients_mean_one_adam_commit_then_ema_and_exact_later_resume(self):
        with TemporaryDirectory()as folder:
            state,plan,kwargs,batch,_=fixture(folder)
            revision,balancer=bind_loss_balance_revision(state,plan,**kwargs)
            before=deepcopy(balancer.state_dict());start=state['global_training_step'];reports=[]
            parameter=torch.nn.Parameter(torch.tensor([.2,.4,.6]));reference=torch.nn.Parameter(parameter.detach().clone())
            optimizer=torch.optim.AdamW([parameter],lr=1e-4);reference_opt=torch.optim.AdamW([reference],lr=1e-4)
            for index in range(4):
                accumulator=balancer.new_microbatch('combat')
                values={'player':(parameter[0]-.1*(index+1)).square()+1.1,
                    'enemy':(parameter[1]-.2*(index+1)).square()+1.1,
                    'interaction':(parameter[2]-.3*(index+1)).square()}
                accumulator.combine(values,floors={'player':1.1,'enemy':1.1}).backward()
                reports.append(accumulator.report())
                self.assertEqual(balancer.state_dict(),before)
                self.assertEqual(state['global_training_step'],start);self.assertEqual(batch_commit_size(state),128)
            parameter.grad.div_(4)
            expected=sum((reference[0]-.1*(i+1)).square()+(reference[1]-.2*(i+1)).square()
                +(reference[2]-.3*(i+1)).square()+2.2 for i in range(4))/4
            expected.backward();torch.testing.assert_close(parameter.grad,reference.grad,rtol=1e-6,atol=1e-7)
            summary=balancer.summarize(reports,environment='combat',step=start+1)
            self.assertEqual(summary['microbatches'],4);self.assertAlmostEqual(summary['kl_floor'],2.2,places=6)
            optimizer.step();reference_opt.step();torch.testing.assert_close(parameter,reference,rtol=0,atol=0)
            commit_combat_update(state,step=start+1,environment='combat',batch_size=128)
            # A half-committed boundary must never be serialized as a validv5 update.
            with self.assertRaises(ValueError):batch_commit_size(state)
            for name in ('feedback_revision','combat_recovery_revision','batch_revision'):
                state['rssm_training'][name]['last_optimizer_step']=start+1
                state['rssm_training'][name]['updates']+=1
            balancer.commit(summary,step=start+1)
            revision.update(last_optimizer_step=start+1,updates=1)
            state['rssm_training']['loss_balance_state']=balancer.state_dict()
            self.assertTrue(balanced_batch_boundary_valid(state));self.assertEqual(batch_commit_size(state),128)
            self.assertEqual(state['rssm_training']['auxiliary_draws'],68160+128)
            self.assertEqual(balancer.state_dict()['banks']['combat']['updates'],1)
            self.assertEqual(balancer.state_dict()['banks']['navigation']['updates'],0)
            self.assertAlmostEqual(balancer.state_dict()['banks']['combat']['ema']['player'],summary['player']['raw'])
            path=Path(folder)/'after.pt';torch.save({'state':state,'model':parameter.detach(),'optimizer':optimizer.state_dict()},path)
            saved=torch.load(path,map_location='cpu',weights_only=False)
            restored=saved['state'];old_training=deepcopy(restored['rssm_training'])
            _,resumed=bind_loss_balance_revision(restored,plan,**{**kwargs,'loaded_checkpoint_sha256':'later_durable_checkpoint'})
            self.assertEqual(resumed.state_dict(),balancer.state_dict());self.assertEqual(restored['rssm_training'],old_training)
            self.assertEqual(batch_commit_size(restored),128)
            with self.assertRaises(ValueError):resumed.commit(summary,step=start+1)

    def test_wrong_initial_source_pins_settings_and_missing_or_inconsistent_v5_ema_fail_closed(self):
        with TemporaryDirectory()as folder:
            for key,value in [('version',True),('source_checkpoint_sha256','wrong'),('batch_revision_sha256','wrong'),
                ('base_plan_sha256','wrong'),('optimizer_reset',True),('source_future_feedback',True),('start_optimizer_step',4259)]:
                state,plan,kwargs,_,_=fixture(folder);plan[key]=value
                with self.assertRaises(ValueError):bind_loss_balance_revision(state,plan,**kwargs)
                self.assertNotIn('loss_balance_revision',state['rssm_training'])
            state,plan,kwargs,_,_=fixture(folder)
            with self.assertRaises(ValueError):bind_loss_balance_revision(state,plan,**{**kwargs,'loaded_checkpoint_sha256':'same_step_wrong_weights'})
            self.assertNotIn('loss_balance_revision',state['rssm_training'])
            bind_loss_balance_revision(state,plan,**kwargs)
            for mutate in ('missing_ema','wrong_settings','last_step','revision_counter','bank_counter_sum','initial_source_pin'):
                broken=deepcopy(state);training=broken['rssm_training']
                if mutate=='missing_ema':training.pop('loss_balance_state')
                elif mutate=='wrong_settings':training['loss_balance_state']['settings']['ema_decay']=.5
                elif mutate=='last_step':training['loss_balance_state']['last_optimizer_step']-=1
                elif mutate=='revision_counter':training['loss_balance_revision']['updates']=1
                elif mutate=='bank_counter_sum':training['loss_balance_state']['banks']['combat']['updates']=1
                else:training['loss_balance_revision']['initial_loaded_checkpoint_sha256']='wrong'
                with self.assertRaises(ValueError):bind_loss_balance_revision(broken,plan,**kwargs)
                if mutate!='initial_source_pin':
                    with self.assertRaises(ValueError):batch_commit_size(broken)


if __name__=='__main__':unittest.main()
