"""Supervisor failure reporting without launching a process, loader or GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from turkeyworld import staged_pipeline


class StagedFailureReportingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/'run'
        self.data = Path(temporary.name)/'data'
        self.monitor = Path(temporary.name)/'monitor'
        self.root.mkdir()
        self.data.mkdir()
        self.source = self.root/'initial.pt'
        self.sr = self.root/'sr.pt'
        self.source.write_bytes(b'fixture: never loaded')
        self.sr.write_bytes(b'fixture: never loaded')
        self.status_path = self.root/'status.json'
        self.write_status({'status': 'failed', 'global_training_step': 1530,
            'staged_curriculum': {'stage': 'navigation'},
            'error': 'OLD configuration mismatch from a previous launch',
            'failure_cause': {'source': 'worker', 'message': 'OLD child failure'},
            'worker_exit_code': 77})
        (self.root/'training_ready.json').write_text('{}', encoding='utf-8')
        (self.data/'status.json').write_text('{"status":"complete"}', encoding='utf-8')
        (self.root/'training.stderr.log').write_text(
            'OLD appended log: configuration mismatch\n', encoding='utf-8')
        self.before_launch = None

    def write_status(self, value):
        self.status_path.write_text(json.dumps(value), encoding='utf-8')

    def read_status(self):
        return json.loads(self.status_path.read_text(encoding='utf-8'))

    def arguments(self):
        return ['--run-dir', str(self.root), '--data-dir', str(self.data),
            '--source', str(self.source), '--sr-checkpoint', str(self.sr),
            '--monitor-dir', str(self.monitor), '--device', 'cpu']

    def launch(self, exit_code, child_status, *, plan_error=None):
        owner = self

        class Guard:
            active = False
            error = None
            def __enter__(self):
                return self
            def __exit__(self, *unused):
                return False

        class Child:
            pid = 12345
            def wait(self):
                if child_status is not None:
                    value = owner.read_status()
                    value.update(child_status)
                    owner.write_status(value)
                return exit_code

        def popen(*unused, **options):
            owner.before_launch = owner.read_status()
            self.assertEqual(options['stdin'], staged_pipeline.subprocess.DEVNULL)
            self.assertTrue(options['close_fds'])
            return Child()

        original_read = Path.read_text
        def no_log_read(path, *args, **kwargs):
            if path.name.endswith(('.stdout.log', '.stderr.log')):
                raise AssertionError('Append logs cannot identify the current failure')
            return original_read(path, *args, **kwargs)

        with patch.object(staged_pipeline, 'KeepAwake', Guard), \
             patch.object(staged_pipeline, 'process_alive', return_value=False), \
             patch.object(staged_pipeline, 'build_plan', return_value={'version': 1}, side_effect=plan_error), \
             patch.object(staged_pipeline.subprocess, 'Popen', side_effect=popen), \
             patch.object(Path, 'read_text', no_log_read):
            staged_pipeline.main(self.arguments())

    def test_current_child_cause_survives_supervisor_wrapper_with_exit_code(self):
        message = 'Non-finite gradient norm before optimizer at target 1537'
        reported = '2026-11-10T19:14:40+00:00'
        with self.assertRaises(staged_pipeline.StagedWorkerError) as caught:
            self.launch(1, {'status': 'failed', 'error': message,
                'global_training_step': 1536, 'updated_at': reported,
                'failure_cause': {'exception_type': 'RuntimeError', 'gradient_check': 'before_step'}})
        status = self.read_status()
        self.assertEqual(str(caught.exception), message)
        self.assertEqual(status['error'], message)
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['worker_exit_code'], 1)
        self.assertEqual(status['global_training_step'], 1536)
        self.assertEqual(status['failure_cause']['message'], message)
        self.assertEqual(status['failure_cause']['source'], 'worker')
        self.assertEqual(status['failure_cause']['worker_pid'], 12345)
        self.assertEqual(status['failure_cause']['reported_at'], reported)
        self.assertEqual(status['failure_cause']['gradient_check'], 'before_step')
        self.assertEqual(status['staged_curriculum']['stage'], 'navigation')
        self.assertNotIn('OLD', json.dumps(status['failure_cause']))

    def test_resume_clears_old_error_before_child_launch_and_success_stays_healthy(self):
        self.launch(0, {'status': 'running', 'global_training_step': 1531})
        for key in ('error', 'failure_cause', 'worker_exit_code'):
            self.assertNotIn(key, self.before_launch)
            self.assertNotIn(key, self.read_status())
        self.assertEqual(self.before_launch['status'], 'running')
        self.assertEqual(self.before_launch['global_training_step'], 1530)
        self.assertEqual(self.read_status()['global_training_step'], 1531)
        self.assertEqual(self.read_status()['status'], 'running')
        record = json.loads((self.monitor/'active_run.json').read_text(encoding='utf-8'))
        self.assertEqual(record['pid'], 12345)

    def test_nonzero_exit_without_child_cause_uses_code_not_a_previous_log(self):
        with self.assertRaises(staged_pipeline.StagedWorkerError):
            self.launch(2, None)
        status = self.read_status()
        self.assertEqual(status['worker_exit_code'], 2)
        self.assertEqual(status['error'], 'Staged teacher exited with code 2')
        self.assertEqual(status['failure_cause']['source'], 'worker')
        self.assertNotIn('configuration', status['error'])

    def test_structured_child_message_is_used_when_plain_error_is_absent(self):
        with self.assertRaises(staged_pipeline.StagedWorkerError):
            self.launch(1, {'status': 'failed', 'failure_cause': {
                'message': 'Current worker cause <b>is plain text</b>', 'kind': 'gradient'}})
        status = self.read_status()
        self.assertEqual(status['error'], 'Current worker cause <b>is plain text</b>')
        self.assertEqual(status['failure_cause']['kind'], 'gradient')

    def test_preflight_supervisor_failure_is_separate_and_never_reuses_child_error(self):
        with self.assertRaisesRegex(ValueError, 'CURRENT immutable plan changed'):
            self.launch(0, None, plan_error=ValueError('CURRENT immutable plan changed'))
        status = self.read_status()
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['error'], 'CURRENT immutable plan changed')
        self.assertEqual(status['failure_cause']['source'], 'supervisor')
        self.assertEqual(status['failure_cause']['exception_type'], 'ValueError')
        self.assertNotIn('worker_exit_code', status)
        self.assertIsNone(self.before_launch)


if __name__ == '__main__':
    unittest.main()
