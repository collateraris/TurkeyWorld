"""120-second target wiring across the runner, evaluator and long curriculum."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class TwoMinuteTrainingTargetTests(unittest.TestCase):
    def evaluation_harness(self, root, seconds=120):
        from turkeyworld.vjepa_training import VJEPAExperiment
        experiment = object.__new__(VJEPAExperiment)
        experiment.root, experiment.device = Path(root), torch.device('cpu')
        experiment.args = SimpleNamespace(quality_psnr=35.)
        if seconds is not None:
            experiment.args.target_seconds = seconds
        experiment.state, experiment.updates = {}, []
        experiment.update = lambda *args, **kwargs: experiment.updates.append((args, kwargs))
        return experiment

    def evaluate(self, root, *, seconds=120, mutate=None):
        experiment = self.evaluation_harness(root, seconds)
        commands = []
        duration = 300 if seconds is None else seconds

        class CompletedProcess:
            returncode = 0

            def __init__(self):
                self.polls = 0

            def poll(self):
                self.polls += 1
                return None if self.polls == 1 else 0

        def launch(command, **options):
            commands.append(command)
            scenario = command[command.index('--scenario') + 1]
            report = {'episode_count': 1, 'target_seconds': duration, 'target_steps': duration*30,
                'requested_active_pass_count': 1, '300s_active_pass_count': int(duration >= 300),
                'episodes': [{'requested_seconds': duration, 'requested_steps': duration*30,
                    'checked_steps': duration*30, 'checked_simulated_seconds': float(duration),
                    'requested_active_pass': True, '300s_active_pass': duration >= 300}]}
            if mutate is not None:
                mutate(report, scenario)
            Path(command[command.index('--output') + 1]).write_text(json.dumps(report), encoding='utf-8')
            options['stdout'].write(json.dumps({'evaluation_step': duration*30}) + '\n')
            options['stdout'].flush()
            return CompletedProcess()

        with patch('turkeyworld.vjepa_training.subprocess.Popen', side_effect=launch), \
             patch('turkeyworld.vjepa_training.time.sleep'):
            result = experiment.evaluate_9000(Path(root)/'selected-world.pt', 'student')
        return experiment, result, commands

    def test_vq_cli_defaults_to_120_seconds_without_starting_training(self):
        from turkeyworld.train_vq_world import main
        with tempfile.TemporaryDirectory() as directory:
            fake = SimpleNamespace(root=Path(directory), codec_info={}, state={}, run=Mock())
            with patch('turkeyworld.train_vq_world.LatentWorldExperiment', return_value=fake) as constructor, \
                 patch('turkeyworld.train_vq_world.atomic_json'), \
                 patch('turkeyworld.train_vq_world.KeepAwake',
                       return_value=nullcontext(SimpleNamespace(active=False, error=None))):
                main(['--run-dir', directory, '--codec-checkpoint', 'frozen-codec.pt',
                      '--from-scratch', '--device', 'cpu'])
            self.assertEqual(constructor.call_args.args[0].target_seconds, 120)
            self.assertTrue(constructor.call_args.args[0].from_scratch)
            fake.run.assert_called_once_with()

    def test_runner_requests_two_3600_step_runs_and_publishes_actual_120_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment, result, commands = self.evaluate(directory)
            self.assertTrue(result['passed'])
            self.assertEqual(result['target_seconds'], 120)
            self.assertEqual(result['target_steps'], 3600)
            self.assertEqual(result['episodes'], 2)
            self.assertEqual(len(commands), 2)
            self.assertEqual([c[c.index('--scenario')+1] for c in commands], ['explore', 'mixed'])
            self.assertTrue(all(c[c.index('--seconds')+1] == '120' for c in commands))
            self.assertTrue(all(r['300s_active_pass_count'] == 0 for r in result['reports']))
            self.assertTrue(all(not r['episodes'][0]['300s_active_pass'] for r in result['reports']))
            progress = [values['evaluation_progress'] for _, values in experiment.updates
                        if 'evaluation_progress' in values]
            self.assertEqual([row['requested_steps'] for row in progress], [3600, 3600])
            self.assertEqual([row['simulated_seconds'] for row in progress], [120., 120.])
            self.assertTrue(all(row['engine_corrections'] == row['model_resets'] == 0 for row in progress))
            self.assertEqual(experiment.state['evaluation'], result)

    def test_120_second_runner_requires_generic_pass_for_both_matching_duration_reports(self):
        def legacy_only(report, scenario):
            report.pop('requested_active_pass_count')
            report['300s_active_pass_count'] = 1

        def wrong_duration(report, scenario):
            report['target_seconds'] = 300

        def one_failed_scenario(report, scenario):
            if scenario == 'mixed':
                report['requested_active_pass_count'] = 0

        for change in (legacy_only, wrong_duration, one_failed_scenario):
            with self.subTest(case=change.__name__), tempfile.TemporaryDirectory() as directory:
                _, result, _ = self.evaluate(directory, mutate=change)
                self.assertFalse(result['passed'])

    def test_missing_target_preserves_legacy_300_second_report_compatibility(self):
        def legacy_report(report, scenario):
            report.pop('requested_active_pass_count')

        with tempfile.TemporaryDirectory() as directory:
            _, result, commands = self.evaluate(directory, seconds=None, mutate=legacy_report)
            self.assertTrue(result['passed'])
            self.assertEqual((result['target_seconds'], result['target_steps']), (300, 9000))
            self.assertTrue(all(c[c.index('--seconds')+1] == '300' for c in commands))

    def test_long_runner_forwards_3600_final_horizon_with_correct_transition_total(self):
        from turkeyworld.vjepa_training import VJEPAExperiment
        with tempfile.TemporaryDirectory() as directory:
            experiment = object.__new__(VJEPAExperiment)
            experiment.root, experiment.device = Path(directory), torch.device('cpu')
            experiment.args = SimpleNamespace(target_seconds=120, ema=.996, lr=1e-4,
                gradient_accumulation_steps=8, batch=1, chunk=16, seed=73, log_every=1,
                save_every=10, aligned_supervision=True)
            experiment.extra, experiment.state = {}, {}
            experiment.global_step, experiment.fragments_seen = 0, 0
            experiment.restore = lambda *args: 0
            experiment.save = Mock()
            expected = (32,)*8 + (128,)*4 + (512,)*2 + (2048,)*2 + (3600,)
            result = {'metrics': {}, 'resume_state': {'safe': True},
                'completed_transitions': sum(expected), 'total_transitions': sum(expected)}
            model = torch.nn.Module()
            model.core = torch.nn.Module()
            model.core.encoder = torch.nn.Linear(2, 2)
            with patch('turkeyworld.vjepa_long_rollout.train_curriculum', return_value=result) as train:
                experiment.long_curriculum(model, 'teacher500vq')
            self.assertEqual(train.call_args.kwargs['schedule'], expected)
            self.assertEqual(sum(expected), 9488)
            self.assertEqual(max(expected), 3600)
            self.assertNotIn(9000, expected)
            self.assertEqual(train.call_args.kwargs['gradient_accumulation_steps'], 8)

    def test_curriculum_completion_declares_requested_3600_without_claiming_9000(self):
        from turkeyworld.vjepa_long_rollout import train_curriculum
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = (32, 128, 512, 2048, 3600)
            descriptor = {'path': str(root/'long-record.npz'), 'transitions': 3600}
            calls = []

            def episode(model, recording, optimizer, *, horizon, **options):
                calls.append(horizon)
                return {'metrics': {'loss': .1}, 'memory_metrics': {},
                    'declaration': {'optimizer_steps': 1, 'interrupted': False}}

            model = SimpleNamespace(config=SimpleNamespace(endless=True, simulation_hz=30., image_size=128))
            with patch('turkeyworld.vjepa_long_rollout.prepare_return_corpus', return_value={}), \
                 patch('turkeyworld.vjepa_long_rollout._corpus_files', side_effect=[[descriptor], []]), \
                 patch('turkeyworld.long_train.load_episode', return_value={}), \
                 patch('turkeyworld.vjepa_long_rollout.train_autoregressive_episode', side_effect=episode):
                report = train_curriculum(model, root/'data'/'long'/'train',
                    return_directory=root/'data'/'memory', schedule=schedule)
            self.assertEqual(calls, list(schedule))
            self.assertEqual(report['completed_transitions'], sum(schedule))
            self.assertTrue(report['declaration']['autonomous_target_rollout_completed'])
            self.assertEqual(report['declaration']['target_rollout_steps'], 3600)
            self.assertFalse(report['declaration']['autonomous_9000_step_task_completed'])
            self.assertEqual(report['declaration']['teacher_forced_future_inputs'], 0)


if __name__ == '__main__':
    unittest.main()
