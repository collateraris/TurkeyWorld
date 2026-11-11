"""Real continuation clocks, immutable data, numerical gates and unbounded loop."""
import copy
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

from turkeyworld.rssm_continuation import (bind_continuation, commit_continuation,
    verified_episode_paths, complete_quality_gate, record_continuation_validation,
    validation_due, training_should_continue, evaluate_heldout_idle)
from turkeyworld.motion_recovery_validation import THRESHOLDS
from turkeyworld.train_rssm_dynamics import RSSMTraining


def good_report():
    short = {'source_held_out': True, 'counts': {'samples': 256, 'free_axes': 80,
        'blocked_axes': 40, 'sliding_samples': 30}, 'cohorts': {name: {'samples': 32}
        for name in ('positive_turn', 'negative_turn', 'moving_turn', 'idle')},
        'metrics': {'velocity_error_tiles_s': {'mean': .1}, 'false_stop_rate': .02,
                    'missed_block_rate': .025}}
    long = {'all_states_finite': True, 'all_memory_finite': True, 'windows': 16,
        'horizon': 450, 'source_wall_steps': 0, 'npc_active_steps': 0,
        'model_wall_steps': 0, 'model_wall_path_steps': 0,
        'final_player_xy_error_tiles': {'mean': .05, 'p95': .1}}
    contact = copy.deepcopy(short); contact['cohorts']['idle']['samples'] = 0
    contact.update(passed=False, coverage_complete=False)
    return {'short': short, 'contact_generalization': contact, 'long450': long,
        'one_frame450': copy.deepcopy(long), 'heldout_idle': {'source_held_out': True,
        'samples': 32, 'all_states_memory_finite': True, 'velocity_error_tiles_s_max': 0.},
        'default_spawn': {'passed': True, 'source_held_out': False},
        'source_future_feedback': False, 'future_rgb_encoded': False}


def corpus(root):
    directory = root/'short/rgb/train'; directory.mkdir(parents=True)
    records = []
    for name in ('free', 'blocked', 'slide', 'turn_idle'):
        path = directory/f'{name}.npz'
        np.savez_compressed(path, contact_first16_cohort=np.asarray(name))
        records.append({'relative_path': path.relative_to(root).as_posix(),
            'stage': 'short', 'split': 'train', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (root/'manifest.json').write_text(json.dumps({'status': 'complete', 'records': records}), encoding='utf-8')
    return directory


class ContinuationTests(unittest.TestCase):
    def fixture(self, root):
        corpus(root/'fresh')
        cp = root/'checkpoint.pt'; cp.write_bytes(b'original complete g2000 checkpoint')
        contact = root/'contact.json'; contact.write_text('{"immutable": true}', encoding='utf-8')
        state = {'global_training_step': 2000, 'training_protocol': {}, 'rssm_training': {
            'phase': 'source', 'source_completed': 1800, 'distillation_completed': 200,
            'distillation_updates': 200, 'contact_revision': {'updates': 1661}}}
        kwargs = dict(enabled=True, dataset_root=root/'fresh', base_plan_sha='old-base',
            contact_plan_path=contact, checkpoint_path=cp, step=2000)
        return state, root/'continuation_plan.json', kwargs

    def test_bind_preserves_rng_original_files_and_resumes_without_resetting_clocks(self):
        with TemporaryDirectory() as tmp:
            state, path, kwargs = self.fixture(Path(tmp))
            numpy_rng = np.random.get_state(); torch_rng = torch.get_rng_state().clone()
            contact_before = kwargs['contact_plan_path'].read_bytes(); cp_before = kwargs['checkpoint_path'].read_bytes()
            continuation, paths = bind_continuation(state, path, **kwargs)
            first_plan = path.read_bytes()
            self.assertEqual(len(paths), 4)
            self.assertEqual(state['global_training_step'], 2000)
            self.assertEqual(state['rssm_training']['source_completed'], 1800)
            self.assertIsNone(state['rssm_training']['total_updates']); self.assertIsNone(state['total'])
            self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
            self.assertTrue(np.array_equal(numpy_rng[1], np.random.get_state()[1]))
            self.assertEqual(kwargs['contact_plan_path'].read_bytes(), contact_before)
            self.assertEqual(kwargs['checkpoint_path'].read_bytes(), cp_before)
            commit_continuation(continuation, step=2001)
            state['rssm_training']['source_completed'] += 1
            resumed, _ = bind_continuation(state, path, **{**kwargs, 'step': 2001})
            self.assertIs(resumed, continuation)
            self.assertEqual((resumed['updates'], resumed['new_auxiliary_draws'], resumed['old_auxiliary_draws']), (1, 12, 4))
            self.assertEqual(path.read_bytes(), first_plan)
            with self.assertRaisesRegex(ValueError, 'requires --continue'):
                bind_continuation(state, path, **{**kwargs, 'enabled': False})

    def test_changed_corpus_plan_checkpoint_or_clocks_fail_closed(self):
        with TemporaryDirectory() as tmp:
            state, path, kwargs = self.fixture(Path(tmp))
            continuation, _ = bind_continuation(state, path, **kwargs)
            bad = copy.deepcopy(state); bad['rssm_training']['source_completed'] += 1
            with self.assertRaisesRegex(ValueError, 'clocks'): bind_continuation(bad, path, **kwargs)
            with self.assertRaisesRegex(ValueError, 'identity/settings'): bind_continuation(state, path, **{**kwargs, 'base_plan_sha': 'changed'})
            extra = kwargs['dataset_root']/'short/rgb/train/unlisted.npz'; extra.write_bytes(b'new')
            with self.assertRaisesRegex(ValueError, 'unlisted'): verified_episode_paths(kwargs['dataset_root'])
            extra.unlink(); paths, _ = verified_episode_paths(kwargs['dataset_root']); paths[0].write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'member changed'): bind_continuation(state, path, **kwargs)

    def test_two_distinct_complete_quality_checks_unbounded_and_250_cadence(self):
        with TemporaryDirectory() as tmp:
            state, path, kwargs = self.fixture(Path(tmp))
            c, _ = bind_continuation(state, path, **kwargs)
            self.assertTrue(training_should_continue(state, 2000))
            self.assertFalse(validation_due(state, 2100)); self.assertTrue(validation_due(state, 2250))
            report = good_report()
            self.assertTrue(complete_quality_gate(report)['passed'])
            record_continuation_validation(c, report, step=2000)
            self.assertEqual(c['consecutive_passes'], 1)
            self.assertFalse(record_continuation_validation(c, report, step=2000))
            self.assertTrue(training_should_continue(state, 2000))
            c['last_optimizer_step'] = 2250
            broken = copy.deepcopy(report); broken['one_frame450']['model_wall_path_steps'] = 1
            record_continuation_validation(c, broken, step=2250)
            self.assertEqual(c['consecutive_passes'], 0)
            for step in (2500, 2750):
                c['last_optimizer_step'] = step
                record_continuation_validation(c, report, step=step)
            self.assertEqual(c['status'], 'qualified')
            self.assertFalse(training_should_continue(state, 2000))

    def test_false_passed_flags_missing_idle_nonfinite_or_contact_errors_cannot_qualify(self):
        report = good_report()
        for mutate in (
                lambda d: d.pop('heldout_idle'),
                lambda d: d['heldout_idle'].update(samples=31),
                lambda d: d['contact_generalization']['metrics'].update(missed_block_rate=.06),
                lambda d: d['short']['metrics']['velocity_error_tiles_s'].update(mean=float('nan')),
                lambda d: d['long450'].update(all_memory_finite=False),
                lambda d: d['short']['cohorts']['idle'].update(samples=0)):
            case = copy.deepcopy(report); mutate(case); case['passed'] = True
            self.assertFalse(complete_quality_gate(case)['passed'])

    def test_idle_validation_uses32_real_val_windows_and_restores_modes_rng(self):
        with TemporaryDirectory() as tmp:
            episodes = []
            for index in range(32):
                path = Path(tmp)/f'episode_{index}.npz'
                controls = np.ones((10,4), np.float32); controls[5] = 0
                np.savez_compressed(path, actions=controls)
                episodes.append({'path':str(path)})
            class Data:
                split = 'val'
                def __init__(self): self.episodes = episodes; self.windows = []
                def batch(self,n,*,context,horizon,indices):
                    self.windows.extend(indices.tolist())
                    assert context == horizon == 1 and np.all(indices[:,1] == 5)
                    state = torch.zeros(n,2,20); state[:,:,:2] = .2
                    return {'states':state,'target_states':state[:,1:], 'actions':torch.zeros(n,1,4),
                        'map':torch.zeros(n,16,16,dtype=torch.long),'context_frames':torch.zeros(n,1,3,32,32)}
            class Model(nn.Module):
                def __init__(self): super().__init__(); self.weight=nn.Parameter(torch.tensor(1.)); self.observed=[]
                def encode_map(self,grid): return grid.float()
                def initialize_from_image(self,frames,states,features):
                    self.observed.append(frames.shape[1]); return torch.zeros(len(states),4)
                def transition(self,current,action,features,memory):
                    assert not action.any()
                    return {'state':current+self.weight*0,'memory':memory}
            model=Model(); data=Data(); model.train(); rng=torch.get_rng_state().clone()
            report=evaluate_heldout_idle(model,data)
            self.assertTrue(report['passed']); self.assertEqual(report['samples'],32)
            self.assertEqual(len(data.windows),32); self.assertEqual(model.observed,[1,1])
            self.assertTrue(torch.equal(rng,torch.get_rng_state())); self.assertTrue(model.training)
            self.assertIsNone(model.weight.grad)
            data.split='train'
            with self.assertRaisesRegex(ValueError,'held-out val'): evaluate_heldout_idle(model,data)

    def test_actual_runner_goes_beyond_2000_restores_adam_and_stops_only_after_quality(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); fresh = root/'fresh'; directory = corpus(fresh)
            (root/'contact_revision_plan.json').write_text('{}', encoding='utf-8')
            args = SimpleNamespace(run_dir=root, seed=173, device='cpu', batch=16, lr=.0001,
                teacher=root/'teacher.pt', initial=root/'initial.pt', data_root=root/'old',
                original_data=root/'original', renderer_checkpoint=root/'renderer.pt', sr=root/'sr.pt',
                prior_tape=root/'tape.npz', updates=2000, distill_updates=200,
                continue_until_quality=True, continuation_contact_root=fresh)
            experiment = RSSMTraining(args)
            model = nn.Module(); model.weight = nn.Parameter(torch.tensor(.3)); model.core = nn.Module()
            model.core.memory_size = 2032
            model.config = SimpleNamespace(rssm_deter_dim=1008, rssm_stoch=32, rssm_classes=32)
            old_optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
            model.weight.square().backward(); old_optimizer.step(); old_optimizer.zero_grad(set_to_none=True)
            state = copy.deepcopy(experiment.state)
            state.update(global_training_step=2000, current=2000, rssm_validation={'passed': False})
            state['rssm_training'].update(phase='source', source_completed=1800, distillation_completed=200,
                short_draws=3200, auxiliary_draws=32000, coverage_long_draws=14400, original_long_draws=14400)
            revision = {'updates': 1661, 'last_optimizer_step': 2000, 'dataset_directory': str(fresh),
                'loss_settings': {'version': 2}, 'fingerprint': 'original-contact', 'new_auxiliary_draws': 19932,
                'old_auxiliary_draws': 6644, 'cohort_draws': {k:4983 for k in ('free','blocked','slide','turn_idle')},
                'latent_training_cycle': ['argmax', 'argmax', 'sample', 'argmax']}
            state['rssm_training']['contact_revision'] = revision
            experiment.checkpoint.parent.mkdir(); experiment.checkpoint.write_bytes(b'full saved g2000')
            from turkeyworld.training_pipeline import atomic_json
            baseplan = {'version': 1, **{k: str(v.resolve()) if isinstance(v, Path) else v
                for k,v in vars(args).items() if k not in ('continue_until_quality','continuation_contact_root')},
                'dataset_manifest_sha256':'data', 'original_manifest_sha256':'data',
                'distillation_teacher_weight':.1, 'own_state_horizon':450, 'tbptt_chunk':16,
                'source_future_feedback':False, 'source_teacher_validation':str(args.teacher.parent.parent/'rssm_migration_validation/teacher_g1633/motion_recovery_validation.json')}
            atomic_json(root/'training_plan.json', baseplan)
            basebytes=(root/'training_plan.json').read_bytes()
            metadata = {'plan_sha256': hashlib.sha256(basebytes).hexdigest(),
                'optimizer_parameter_names':['weight'], 'optimizer':old_optimizer.state_dict(),
                'run_state':state, 'rng_state':experiment.rng.bit_generator.state,
                'torch_rng_state':torch.get_rng_state(), 'rssm_migration':{}}
            optimizer_steps=[]; validations=[]; snapshots=[]
            class Data:
                def __init__(self, *a, **kw): self.episodes=[{'path':str(p)} for p in directory.glob('*.npz')]
                def batch(self,*a,rng=None,**kw):
                    return {'token':int(rng.integers(100)) if rng is not None else 1}
            class Sampler:
                def __init__(self,*a): pass
                def sample(self,rev,rng): return {'token':int(rng.integers(100))}, {k:3 for k in ('free','blocked','slide','turn_idle')}
            def backward(actual,batch,**kw):
                self.assertEqual(kw['expected_horizon'],450)
                kw['progress_callback'](450,450)
                loss=actual.weight.square(); loss.backward()
                return {'train_loss':float(loss.detach())}
            def validate(actual):
                validations.append(experiment.state['global_training_step'])
                record_continuation_validation(experiment.continuation,good_report(),step=validations[-1])
            def save(actual,opt,meta):
                snapshots.append(experiment.state['global_training_step'])
                optimizer_steps.append(int(opt.state[model.weight]['step']))
            with patch('turkeyworld.train_rssm_dynamics.verify_data',return_value='data'), \
                 patch('turkeyworld.train_rssm_dynamics.load_world_checkpoint',return_value=(model,metadata)), \
                 patch('turkeyworld.train_rssm_dynamics.configure_rssm_training',return_value={'dynamics_trainable':1,'frozen':0}), \
                 patch('turkeyworld.train_rssm_dynamics.parameter_counts',return_value={'world_total':1,'dynamics_including_map':1,'renderer':0}), \
                 patch('turkeyworld.train_rssm_dynamics.StagedSequenceData',Data), \
                 patch('turkeyworld.rssm_contact_revision.bind_contact_revision',side_effect=lambda s,*a,**kw:s['rssm_training']['contact_revision']), \
                 patch('turkeyworld.rssm_contact_revision.ContactAuxiliarySampler',Sampler), \
                 patch('turkeyworld.rssm_objectives.rssm_dynamics_backward',side_effect=backward), \
                 patch.object(experiment,'validate',side_effect=validate), \
                 patch.object(experiment,'publish'), patch.object(experiment,'preview'), \
                 patch.object(experiment,'save',side_effect=save):
                experiment.run()
            self.assertEqual(validations,[2000,2250])
            self.assertEqual(experiment.state['global_training_step'],2250)
            self.assertEqual(experiment.state['status'],'qualified')
            self.assertEqual(experiment.continuation['updates'],250)
            self.assertEqual(experiment.state['rssm_training']['source_completed'],2050)
            self.assertEqual(experiment.state['rssm_training']['auxiliary_draws'],36000)
            self.assertEqual(optimizer_steps[0],1)
            self.assertEqual(optimizer_steps[-1],251)
            self.assertEqual((root/'training_plan.json').read_bytes(),basebytes)


if __name__ == '__main__': unittest.main()
