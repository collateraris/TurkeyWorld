"""Teacher limits describe additional optimizer updates and survive resume."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from turkeyworld.teacher_budget import prepare_teacher_budget, teacher_budget_progress


class TeacherUpdateBudgetTests(unittest.TestCase):
    def prepare(self, path, **overrides):
        options = dict(phase='teacher500vq_world', current_optimizer_step=20,
                       current_phase_step=7, limit_updates=6000,
                       subset_fingerprint='fixed-128000-transitions')
        options.update(overrides)
        return prepare_teacher_budget(path, **options)

    def test_6000_additional_updates_start_at_20_and_stop_at_6020(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'teacher_budget.json'
            budget = self.prepare(path)
            self.assertEqual((budget['start_optimizer_step'], budget['stop_optimizer_step']), (20, 6020))
            self.assertEqual((budget['start_phase_step'], budget['stop_phase_step']), (7, 6007))
            self.assertEqual(budget['limit_updates'], 6000)
            self.assertEqual(budget['status'], 'running')
            self.assertTrue(budget['extended_data_requires_user'])
            self.assertEqual(json.loads(path.read_text('utf-8')), budget)

    def test_resume_at_25_keeps_original_baseline_boundary_and_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'teacher_budget.json'
            original = self.prepare(path)
            saved_bytes = path.read_bytes()
            resumed = self.prepare(path, current_optimizer_step=25, current_phase_step=12)
            self.assertEqual(resumed, original)
            self.assertEqual(path.read_bytes(), saved_bytes)
            progress = teacher_budget_progress(resumed, 25)
            self.assertEqual(progress['completed_updates'], 5)
            self.assertEqual(progress['remaining_updates'], 5995)
            self.assertEqual(progress['stop_optimizer_step'], 6020)
            self.assertEqual(progress['current_optimizer_step'], 25)

    def test_subset_limit_or_phase_change_cannot_silently_reset_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'teacher_budget.json'
            self.prepare(path)
            original = path.read_bytes()
            for change in ({'subset_fingerprint': 'expanded-data'}, {'limit_updates': 6001},
                           {'phase': 'teacher500vq_world_001'}):
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'different training segment'):
                    self.prepare(path, **change)
                self.assertEqual(path.read_bytes(), original)

    def test_checkpoint_outside_boundary_or_inconsistent_phase_counter_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'teacher_budget.json'
            self.prepare(path)
            for global_step, phase_step, message in [(19, 6, 'outside'), (6021, 6008, 'outside'),
                                                     (25, 13, 'counters disagree')]:
                with self.subTest(global_step=global_step, phase_step=phase_step), \
                     self.assertRaisesRegex(ValueError, message):
                    self.prepare(path, current_optimizer_step=global_step, current_phase_step=phase_step)

    def test_inconsistent_saved_boundaries_are_rejected(self):
        for field in ('stop_optimizer_step', 'stop_phase_step'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'teacher_budget.json'
                budget = self.prepare(path)
                budget[field] += 1
                path.write_text(json.dumps(budget), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Inconsistent saved'):
                    self.prepare(path)

    def test_completed_ledger_requires_matching_final_checkpoint_and_has_no_remaining_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'teacher_budget.json'
            budget = self.prepare(path)
            budget['status'] = 'awaiting_user'
            path.write_text(json.dumps(budget), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'matching final checkpoint'):
                self.prepare(path, current_optimizer_step=25, current_phase_step=12)
            finished = self.prepare(path, current_optimizer_step=6020, current_phase_step=6007)
            progress = teacher_budget_progress(finished, 6020)
            self.assertEqual(progress['completed_updates'], 6000)
            self.assertEqual(progress['remaining_updates'], 0)
            self.assertEqual(progress['status'], 'awaiting_user')
            self.assertEqual(progress['start_optimizer_step'], 20)

    def test_progress_and_initial_inputs_reject_invalid_counters_and_empty_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, change in enumerate(({'current_optimizer_step': -1}, {'current_phase_step': -1},
                                            {'limit_updates': 0}, {'subset_fingerprint': ''})):
                path = root/f'invalid-{index}.json'
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.prepare(path, **change)
                self.assertFalse(path.exists())
            budget = self.prepare(root/'valid.json')
            start = teacher_budget_progress(budget, 20)
            self.assertEqual((start['completed_updates'], start['remaining_updates']), (0, 6000))
            for current in (19, 6021):
                with self.subTest(current=current), self.assertRaisesRegex(ValueError, 'authorized teacher segment'):
                    teacher_budget_progress(budget, current)


class TeacherBudgetRunnerTests(unittest.TestCase):
    def harness(self, directory, *, resumed_global=20):
        from turkeyworld.train_vq_world import LatentWorldExperiment
        experiment = object.__new__(LatentWorldExperiment)
        experiment.root = Path(directory)
        (experiment.root/'checkpoints').mkdir(exist_ok=True)
        experiment.device = __import__('torch').device('cpu')
        experiment.started = time.time()
        experiment.args = SimpleNamespace(optimizer_update_budget=6000, minimum_train_clips=1024,
            from_scratch=True, seed=173, cache_mb=512, context=2, horizon=4,
            batch=1, gradient_accumulation_steps=4)
        experiment.global_step = resumed_global
        experiment.teacher_budget = None
        filenames = [f'episode_{index:06d}.npz' for index in reversed(range(1000))]
        experiment.training_subset = {'filenames': filenames, 'count_files': 1000,
            'total_transitions': 32000, 'fingerprint': 'selected-32000-fixed'}
        experiment.state = {'student_plan': {'status': 'pending'},
            'training_subset': {'fingerprint': 'selected-32000-fixed', 'train_clips': 1000,
                                'train_transitions': 32000},
            'renderer_training': {'active': True}, 'teacher_quality_qualified': False}
        experiment.fit = Mock()
        experiment.prediction_preview_report = Mock()
        return experiment

    def write_checkpoint(self, experiment, step=20):
        import torch
        checkpoint = experiment.root/'checkpoints'/'teacher500vq_world.pt'
        torch.save({'training_steps': step, 'global_training_step': step}, checkpoint)
        return checkpoint

    def parent_save(self, experiment, model, optimizer, phase, step, metrics, extra=None):
        import torch
        output = experiment.root/'checkpoints'/f'{phase}.pt'
        torch.save({'training_steps': step, 'global_training_step': experiment.global_step,
            'optimizer': {'moment_sentinel': 'kept-by-parent'}, **dict(extra or {})}, output)
        return output

    def fit_to_boundary(self, experiment):
        def fit(model, data, phase, stop_phase_step, *, pretrain):
            self.assertTrue(pretrain)
            self.assertEqual(phase, 'teacher500vq_world')
            self.assertEqual(stop_phase_step, 6020)
            experiment.global_step = 6020
            experiment.save(model, object(), phase, stop_phase_step, {'train_loss': .1})
        return fit

    def test_fixed_manifest_filenames_are_forwarded_to_parent_loader_without_reordering(self):
        from turkeyworld.train_vq_world import LatentWorldExperiment
        with tempfile.TemporaryDirectory() as directory:
            experiment = self.harness(directory)
            self.assertEqual(experiment.training_data_options(),
                             {'episode_paths': experiment.training_subset['filenames']})
            initial = experiment.root/'checkpoints'/'teacher_vjepa500vq_initial.pt'
            initial.write_bytes(b'pre-existing fresh initialization')
            experiment.wait_for_data = Mock()
            stop = RuntimeError('stop after capturing train loader options')
            with patch('turkeyworld.train_250m.SequenceData', side_effect=stop) as loader:
                with self.assertRaisesRegex(RuntimeError, 'capturing train loader'):
                    LatentWorldExperiment.run(experiment)
            self.assertEqual(loader.call_args.kwargs['episode_paths'], experiment.training_subset['filenames'])
            self.assertEqual(loader.call_args.kwargs['mode'], 'stream')
            self.assertEqual(experiment.training_subset['filenames'][0], 'episode_000999.npz')

    def test_real_parent_run_stops_at_budget_before_repairs_long_quality_or_student(self):
        import torch
        from turkeyworld.train_500m import MassiveExperiment
        from turkeyworld.train_vq_world import LatentWorldExperiment
        with tempfile.TemporaryDirectory() as directory:
            experiment = self.harness(directory)
            checkpoint = self.write_checkpoint(experiment)
            (experiment.root/'checkpoints'/'teacher_vjepa500vq_initial.pt').write_bytes(b'fresh initial')
            experiment.wait_for_data = Mock()
            experiment.fit.side_effect = self.fit_to_boundary(experiment)
            data = SimpleNamespace(files=1000, count=32000)
            validation = SimpleNamespace(files=512, count=16384, fixed_panel=lambda **options: [(0, 0)])
            experiment.long_curriculum = Mock(side_effect=AssertionError('No teacher long stage after budget'))
            experiment.make_student_model = Mock(side_effect=AssertionError('No student after budget'))
            experiment.evaluate_9000 = Mock(side_effect=AssertionError('No automatic handoff evaluation'))
            with patch('turkeyworld.train_250m.SequenceData', side_effect=[data, validation]), \
                 patch('turkeyworld.train_250m.load_world_checkpoint', return_value=(object(), {})), \
                 patch.object(MassiveExperiment, 'save', autospec=True, side_effect=self.parent_save):
                self.assertIsNone(LatentWorldExperiment.run(experiment))
            self.assertEqual(experiment.fit.call_count, 1)
            self.assertEqual(experiment.global_step, 6020)
            experiment.long_curriculum.assert_not_called()
            experiment.make_student_model.assert_not_called()
            experiment.evaluate_9000.assert_not_called()
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(saved['optimizer']['moment_sentinel'], 'kept-by-parent')
            self.assertEqual(saved['training_subset'], experiment.state['training_subset'])
            self.assertEqual(saved['optimizer_budget']['start_optimizer_step'], 20)
            self.assertEqual(saved['optimizer_budget']['completed_updates'], 6000)
            self.assertEqual(saved['optimizer_budget']['remaining_updates'], 0)
            self.assertEqual(experiment.state['optimizer_budget']['completed_updates'], 6000)
            self.assertEqual(experiment.state['optimizer_budget']['status'], 'awaiting_user')
            self.assertEqual(experiment.state['status'], 'awaiting_user')
            self.assertEqual(experiment.state['student_plan']['status'], 'pending_user_decision')
            self.assertFalse(experiment.state['teacher_quality_qualified'])
            self.assertFalse(experiment.state['renderer_training']['active'])
            ledger = json.loads((experiment.root/'teacher_budget.json').read_text('utf-8'))
            self.assertEqual(ledger['checkpoint_optimizer_step'], 6020)
            self.assertEqual((ledger['start_optimizer_step'], ledger['stop_optimizer_step']), (20, 6020))

    def test_completed_budget_resume_does_not_fit_preview_or_add_new_6000(self):
        from turkeyworld.teacher_budget import TeacherBudgetFinished
        from turkeyworld.train_500m import MassiveExperiment
        with tempfile.TemporaryDirectory() as directory:
            first = self.harness(directory)
            self.write_checkpoint(first)
            first.fit.side_effect = self.fit_to_boundary(first)
            with patch.object(MassiveExperiment, 'save', autospec=True, side_effect=self.parent_save):
                with self.assertRaises(TeacherBudgetFinished):
                    first.world_until_ready(object(), object())
            before = (first.root/'teacher_budget.json').read_bytes()
            resumed = self.harness(directory, resumed_global=0)
            with self.assertRaises(TeacherBudgetFinished):
                resumed.world_until_ready(object(), object())
            resumed.fit.assert_not_called()
            resumed.prediction_preview_report.assert_not_called()
            self.assertEqual(resumed.global_step, 6020)
            self.assertEqual((resumed.root/'teacher_budget.json').read_bytes(), before)
            self.assertEqual(resumed.state['optimizer_budget']['remaining_updates'], 0)
            self.assertEqual(resumed.teacher_budget['start_optimizer_step'], 20)
            self.assertEqual(resumed.teacher_budget['stop_optimizer_step'], 6020)

    def test_checkpoint_at_25_resumes_remaining_5995_updates_without_moving_boundary(self):
        from turkeyworld.teacher_budget import TeacherBudgetFinished
        from turkeyworld.train_500m import MassiveExperiment
        with tempfile.TemporaryDirectory() as directory:
            first = self.harness(directory)
            self.write_checkpoint(first)

            def stop_after_five(model, data, phase, stop_phase_step, *, pretrain):
                self.assertEqual(stop_phase_step, 6020)
                first.global_step = 25
                first.update(phase, 25, stop_phase_step, 'checkpoint test')
                first.save(model, object(), phase, 25, {'train_loss': .2})
                raise SystemExit(0)

            first.fit.side_effect = stop_after_five
            with patch.object(MassiveExperiment, 'save', autospec=True, side_effect=self.parent_save):
                with self.assertRaises(SystemExit):
                    first.world_until_ready(object(), object())
            self.assertEqual(first.state['optimizer_budget']['completed_updates'], 5)
            self.assertEqual(first.state['optimizer_budget']['remaining_updates'], 5995)
            resumed = self.harness(directory, resumed_global=0)

            def finish(model, data, phase, stop_phase_step, *, pretrain):
                self.assertEqual(resumed.global_step, 25)
                self.assertEqual(stop_phase_step-resumed.global_step, 5995)
                return self.fit_to_boundary(resumed)(model, data, phase, stop_phase_step, pretrain=pretrain)

            resumed.fit.side_effect = finish
            with patch.object(MassiveExperiment, 'save', autospec=True, side_effect=self.parent_save):
                with self.assertRaises(TeacherBudgetFinished):
                    resumed.world_until_ready(object(), object())
            self.assertEqual(resumed.teacher_budget['start_optimizer_step'], 20)
            self.assertEqual(resumed.teacher_budget['stop_optimizer_step'], 6020)
            self.assertEqual(resumed.state['optimizer_budget']['completed_updates'], 6000)

    def test_subset_of_1000_clips_is_ready_despite_regular_minimum_1024(self):
        from turkeyworld.engine import BEHAVIOR_VERSION
        with tempfile.TemporaryDirectory() as directory:
            experiment = self.harness(directory)
            (experiment.root/'data').mkdir()
            report = {'behavior_version': BEHAVIOR_VERSION, 'status': 'running',
                'groups': {'rgb/train': {'files': 1000}, 'rgb/val': {'files': 512}}}
            (experiment.root/'data'/'status.json').write_text(json.dumps(report), encoding='utf-8')
            with patch('turkeyworld.train_vq_world.time.sleep', side_effect=AssertionError('Already ready')):
                experiment.wait_for_data()
            self.assertEqual(experiment.state['data_preparation'], report)


if __name__ == '__main__':
    unittest.main()
