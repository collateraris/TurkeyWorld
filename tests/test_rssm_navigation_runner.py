"""Execute the real restored runner/sampler/schedule for one full16-update cycle."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.rssm_contact_branches import normalize_contact_branch_protocol
from turkeyworld.rssm_navigation_revision import navigation_revision_setting, validate_revision_ledger
from turkeyworld.train_rssm_dynamics import RSSMTraining
from turkeyworld.training_pipeline import atomic_json


def make_corpus(root):
    folder=root/'short/rgb/train';folder.mkdir(parents=True)
    cohorts=('free','blocked','slide','turn_idle')
    outcomes=('free_free','free_blocked','blocked_free','blocked_blocked')
    records=[]
    for index in range(8):
        path=folder/f'episode_{index:04d}.npz'
        np.savez_compressed(path,contact_first16_cohort=np.asarray(cohorts[index%4]))
        records.append({'relative_path':path.relative_to(root).as_posix(),'stage':'short','split':'train',
            'pool':'corner' if index<4 else 'singlewall',
            'axis_outcomes_first16':[outcomes[index]] if index<4 else [],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    atomic_json(root/'manifest.json',{'status':'complete','records':records})
    return sorted(folder.glob('*.npz'))


class RSSMNavigationRunnerTests(unittest.TestCase):
    def test_real_loop_preserves_adam_and_clocks_and_balances_all_pool_context_modes(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);data_root=root/'new_corner_data';paths=make_corpus(data_root)
            args=SimpleNamespace(run_dir=root,seed=173,device='cpu',batch=16,lr=.0001,
                teacher=root/'teacher.pt',initial=root/'initial.pt',data_root=root/'old',
                original_data=root/'original',renderer_checkpoint=root/'renderer.pt',sr=root/'sr.pt',
                prior_tape=root/'prior_tape.npz',updates=2000,distill_updates=200,
                continue_until_quality=True,continuation_contact_root=data_root,
                navigation_revision_root=data_root)
            experiment=RSSMTraining(args)
            model=nn.Module();model.weight=nn.Parameter(torch.tensor(.3));model.core=nn.Module()
            model.core.memory_size=2032
            model.config=SimpleNamespace(rssm_deter_dim=1008,rssm_stoch=32,rssm_classes=32)
            old_optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01)
            model.weight.square().backward();old_optimizer.step();old_optimizer.zero_grad(set_to_none=True)
            old_optimizer.state[model.weight]['step'].fill_(3151)
            original_moment=old_optimizer.state[model.weight]['exp_avg'].clone()
            original_second_moment=old_optimizer.state[model.weight]['exp_avg_sq'].clone()
            state=deepcopy(experiment.state)
            state.update(global_training_step=3151,current=3151,total=None,eta_seconds=None,
                training_until_quality=True,rssm_validation={
                    'passed':False,'long450':{'fingerprint':'fixed-two-frame450'},
                    'one_frame450':{'fingerprint':'fixed-one-frame450'}})
            state['rssm_training'].update(phase='source',source_completed=2951,distillation_completed=200,
                short_draws=3200,auxiliary_draws=3151*16,coverage_long_draws=1476*16,
                original_long_draws=1475*16,total_updates=None,source_updates=None)
            revision={'updates':2812,'last_optimizer_step':3151,'dataset_directory':str(data_root),
                'loss_settings':{'version':2},'fingerprint':'old-contact-immutable',
                'new_auxiliary_draws':2812*12,'old_auxiliary_draws':2812*4,
                'cohort_draws':dict.fromkeys(('free','blocked','slide','turn_idle'),2812*3),
                'latent_training_cycle':['argmax','argmax','sample','argmax']}
            continuation={'updates':1151,'last_optimizer_step':3151,'new_auxiliary_draws':1151*12,
                'old_auxiliary_draws':1151*4,'dataset_directory':str(data_root),'fingerprint':'old-continuation-immutable',
                'consecutive_passes':0,'status':'training','baseline_pending':False}
            state['rssm_training'].update(contact_revision=revision,continuation=continuation)
            for filename in ('contact_revision_plan.json','continuation_plan.json'):
                (root/filename).write_bytes(b'{"old_plan":"immutable"}')
            base_plan={'version':1,**{k:str(v.resolve()) if isinstance(v,Path) else v
                for k,v in vars(args).items() if k not in
                ('continue_until_quality','continuation_contact_root','navigation_revision_root')},
                'dataset_manifest_sha256':'data','original_manifest_sha256':'data',
                'distillation_teacher_weight':.1,'own_state_horizon':450,'tbptt_chunk':16,
                'source_future_feedback':False,'source_teacher_validation':str(
                    args.teacher.parent.parent/'rssm_migration_validation/teacher_g1633/motion_recovery_validation.json')}
            atomic_json(root/'training_plan.json',base_plan)
            immutable={p:p.read_bytes() for p in (root/'training_plan.json',root/'contact_revision_plan.json',
                root/'continuation_plan.json',data_root/'manifest.json',*paths)}
            experiment.checkpoint.parent.mkdir();experiment.checkpoint.write_bytes(b'full saved g3151 Adam/RNG checkpoint')
            metadata={'plan_sha256':hashlib.sha256(immutable[root/'training_plan.json']).hexdigest(),
                'model':deepcopy(model.state_dict()),'training_steps':3151,
                'optimizer_parameter_names':['weight'],'optimizer':deepcopy(old_optimizer.state_dict()),
                'run_state':state,'rng_state':deepcopy(experiment.rng.bit_generator.state),
                'torch_rng_state':torch.get_rng_state().clone(),'rssm_migration':{}}
            calls=[];auxiliary_calls=[];snapshots=[];validations=[]
            long_roles={str((args.data_root/'long/rgb/train').resolve()):'coverage',
                        str((args.original_data/'rgb/train').resolve()):'original'}
            class Data:
                def __init__(self,path,*unused,episode_paths=None,**kwargs):
                    self.path=Path(path).resolve();self.role=long_roles.get(str(self.path),'auxiliary')
                    self.episodes=[{'path':str(p)} for p in (episode_paths or paths)]
                def batch(self,count,*,context,horizon,rng=None,indices=None):
                    self_pairs=np.asarray(indices,dtype=np.int64)
                    self_role=self.role
                    if self_pairs.shape!=(count,2): raise AssertionError('Explicit pinned window indices required')
                    states=torch.zeros(count,context+horizon,20)
                    frames=torch.zeros(count,context+horizon,3,1,1)
                    actions=torch.zeros(count,horizon,4)
                    for row,(_,start) in enumerate(self_pairs):
                        states[row,:,0]=torch.arange(start,start+context+horizon)
                        frames[row,:,0,0,0]=torch.arange(start,start+context+horizon)
                        actions[row,:,0]=torch.arange(start+context-1,start+context-1+horizon)
                    if self_role=='auxiliary':
                        auxiliary_calls.append({'count':count,'context':context,'horizon':horizon,
                            'indices':self_pairs.copy(),'path':str(self.path)})
                    return {'states':states,'frames':frames,'context_frames':frames[:,:context],
                        'target_states':states[:,context:],'actions':actions,
                        'map':torch.zeros(count,16,16,dtype=torch.long),
                        'indices':torch.as_tensor(self_pairs),'pool_id':torch.full((count,),
                            {'original':1,'coverage':2,'auxiliary':0}[self_role])}
            def backward(actual,batch,**kwargs):
                context=kwargs['context'];observed=kwargs['observed_context']
                role='original' if int(batch['pool_id'][0])==1 else 'coverage'
                self.assertEqual(kwargs['expected_horizon'],450)
                self.assertEqual((context,observed),(context,context))
                self.assertTrue(torch.all(batch['indices'][:,1]==0))
                self.assertEqual(batch['states'].shape,(16,context+450,20))
                self.assertEqual(batch['actions'].shape,(16,450,4))
                self.assertTrue(torch.all(batch['states'][:,context-1,0]==context-1))
                self.assertTrue(torch.all(batch['actions'][:,0,0]==context-1))
                self.assertEqual(kwargs['contact_branch_protocol'],normalize_contact_branch_protocol({}))
                self.assertEqual((kwargs['auxiliary_context'],kwargs['auxiliary_observed_context']),(1,1))
                self.assertEqual(kwargs['auxiliary_batch']['states'].shape,(16,17,20))
                self.assertTrue(torch.all(kwargs['auxiliary_batch']['actions'][:,0,0]==0))
                calls.append((role,context,'sample' if actual.core.training else 'argmax'))
                kwargs['progress_callback'](450,450)
                loss=actual.weight.square();loss.backward()
                if len(calls)==16: (root/'stop_training').write_text('Stop after this successful update')
                return {'train_loss':float(loss.detach())}
            def validate(actual):
                validations.append(experiment.state['global_training_step'])
                experiment.navigation_revision.update(baseline_pending=False,last_validation_step=3151,
                    last_validation_passed=False)
            def save(actual,optimizer,meta):
                snapshots.append({'step':experiment.state['global_training_step'],
                    'adam_step':int(optimizer.state[model.weight]['step']),
                    'moment':optimizer.state[model.weight]['exp_avg'].clone(),
                    'second_moment':optimizer.state[model.weight]['exp_avg_sq'].clone()})
            with (patch('turkeyworld.train_rssm_dynamics.verify_data',return_value='data'),
                  patch('turkeyworld.train_rssm_dynamics.load_world_checkpoint',return_value=(model,metadata)),
                  patch('turkeyworld.train_rssm_dynamics.configure_rssm_training',return_value={'dynamics_trainable':1,'frozen':0}),
                  patch('turkeyworld.train_rssm_dynamics.parameter_counts',return_value={'world_total':1,'dynamics_including_map':1,'renderer':0}),
                  patch('turkeyworld.train_rssm_dynamics.StagedSequenceData',Data),
                  patch('turkeyworld.rssm_contact_revision.bind_contact_revision',side_effect=lambda s,*a,**kw:s['rssm_training']['contact_revision']),
                  patch('turkeyworld.rssm_continuation.bind_continuation',side_effect=lambda s,*a,**kw:(s['rssm_training']['continuation'],paths)),
                  patch('turkeyworld.rssm_objectives.rssm_dynamics_backward',side_effect=backward),
                  patch.object(experiment,'validate',side_effect=validate),patch.object(experiment,'save',side_effect=save),
                  patch.object(experiment,'publish'),patch.object(experiment,'preview')):
                experiment.run()
            self.assertEqual(len(calls),16)
            joint=Counter(calls)
            self.assertEqual(len(joint),8)
            for role in ('original','coverage'):
                for context in (1,2):
                    self.assertEqual(joint[role,context,'argmax'],3)
                    self.assertEqual(joint[role,context,'sample'],1)
            self.assertEqual(calls,[(navigation_revision_setting(i)['long_pool'],
                navigation_revision_setting(i)['context'],navigation_revision_setting(i)['latent_mode']) for i in range(16)])
            self.assertEqual(experiment.state['global_training_step'],3167)
            self.assertEqual(experiment.state['status'],'stopped')
            self.assertFalse(experiment.state['rssm_training']['gameplay_qualified'])
            self.assertEqual(experiment.continuation['status'],'training')
            self.assertEqual(experiment.continuation['consecutive_passes'],0)
            self.assertEqual(validations,[3151])
            self.assertEqual((snapshots[0]['step'],snapshots[0]['adam_step']),(3151,3151))
            self.assertTrue(torch.equal(snapshots[0]['moment'],original_moment))
            self.assertTrue(torch.equal(snapshots[0]['second_moment'],original_second_moment))
            self.assertEqual((snapshots[-1]['step'],snapshots[-1]['adam_step']),(3167,3167))
            self.assertEqual(experiment.state['rssm_training']['source_completed'],2967)
            self.assertEqual(experiment.contact_revision['updates'],2828)
            self.assertEqual(experiment.contact_revision['new_auxiliary_draws'],2828*12)
            self.assertEqual(experiment.contact_revision['old_auxiliary_draws'],2828*4)
            self.assertEqual(experiment.continuation['updates'],1167)
            self.assertEqual(experiment.continuation['last_optimizer_step'],3167)
            self.assertEqual(experiment.continuation['new_auxiliary_draws'],1167*12)
            self.assertEqual(experiment.continuation['old_auxiliary_draws'],1167*4)
            ledger=experiment.navigation_revision
            self.assertEqual(ledger['revision_updates'],16)
            self.assertEqual(ledger['auxiliary_draws'],{'corner':128,'single_wall':64,'legacy':64})
            validate_revision_ledger(ledger,step=3167,source_completed=2967)
            self.assertEqual(ledger['joint_long_draws'],{k:v*16 for k,v in ledger['joint_update_counts'].items()})
            # Real NavigationContactSampler calls two corner windows per outcome,
            # one per singlewall cohort, then four genuine old-pool entries.
            self.assertEqual(len(auxiliary_calls),32)
            for a,b in zip(auxiliary_calls[::2],auxiliary_calls[1::2]):
                self.assertEqual((a['count'],b['count']),(12,4))
                self.assertEqual(a['indices'][:8,0].tolist(),[0,0,1,1,2,2,3,3])
                self.assertEqual(set(a['indices'][8:,0]),{4,5,6,7})
                self.assertTrue((a['indices'][:,1]==0).all() and (b['indices'][:,1]==0).all())
            self.assertEqual(immutable,{p:p.read_bytes() for p in immutable})


if __name__=='__main__': unittest.main()
