"""Checkpoint-free operator tests; subprocess creation is always mocked."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from turkeyworld.continue_rssm_preview_revision import (
    prepare_preview_revision, check_prepared_preview_revision,
    start_prepared_preview_revision, preview_revision_argv, _sha)
from turkeyworld.rssm_preview_assets import resolve_preview_assets_plan


class PreviewRevisionResumeOperatorTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)/'same_run'; self.root.mkdir()
        (self.root/'checkpoints').mkdir()
        self.checkpoint = self.root/'checkpoints/rssm_world.pt'
        # Intentionally not torch-readable: the operator must never load it.
        self.checkpoint.write_bytes(b'complete opaque model Adam RNG sampler EMA checkpoint')
        self.old_source = self.root/'source_original.pt'; self.old_source.write_bytes(b'original source pin')
        self.sr = self.root/'sr.pt'; self.sr.write_bytes(b'frozen SR38100')
        self.initial_renderer = self.root/'initial_renderer.pt'; self.initial_renderer.write_bytes(b'initial renderer')
        self.old_renderer = self.root/'renderer6600.pt'; self.old_renderer.write_bytes(b'old preview6600')
        self.new_renderer = self.root/'renderer6000_v6.pt'; self.new_renderer.write_bytes(b'latest v6 preview')
        self.initial = self.plan('initial_assets.json', self.initial_renderer)
        self.old_selected = self.plan('old_assets.json', self.old_renderer)
        self.new_selected = self.plan('new_assets.json', self.new_renderer)
        self.old_revision = self.write('old_revision.json', {
            'version': 1, 'kind': 'preview_assets_revision_v1',
            'initial_plan_sha256': _sha(self.initial), 'selected_plan': str(self.old_selected),
            'selected_plan_sha256': _sha(self.old_selected)})
        self.new_revision = self.write('new_revision.json', {
            'version': 2, 'kind': 'preview_assets_revision_v2', 'initial_plan_sha256': _sha(self.initial),
            'previous_revision_plan': str(self.old_revision), 'previous_revision_sha256': _sha(self.old_revision),
            'selected_plan': str(self.new_selected), 'selected_plan_sha256': _sha(self.new_selected)})
        _, _, old_record = resolve_preview_assets_plan(self.initial, self.old_revision)
        self.base = self.write('combat_training_plan.json', {'original': 'immutable'})
        self.gate = self.write('gate.json', {'required_consecutive_passes': 10})
        self.batch = self.write('batch.json', {'microbatch': 64, 'accumulation': 4})
        self.argv = ['python.exe', '-X', 'utf8', '-u', '-m', 'turkeyworld.train_rssm_combat',
                     '--run-dir', str(self.root), '--source-checkpoint', str(self.old_source),
                     '--renderer-checkpoint', str(self.initial_renderer), '--sr', str(self.sr),
                     '--updates', '0', '--batch', '64', '--accumulation-steps', '4',
                     '--batch-revision-plan', str(self.batch), '--loss-gated-curriculum-plan', str(self.gate),
                     '--preview-assets-plan', str(self.initial), '--preview-assets-revision-plan', str(self.old_revision)]
        self.write('launch_command.json', {'argv': self.argv, 'pid': 68032, 'history': ['all old flags kept']})
        self.state = {'status': 'stopped', 'global_training_step': 5340,
                      'preview_assets_revision': old_record,
                      'rssm_training': {'loss_gated_curriculum': {'last_optimizer_step': 5340,
                          'settings': {'required_consecutive_passes': 10}, 'updates': 228,
                          'stage_index': 4, 'current_horizon': 128, 'consecutive_below_threshold': 7},
                          'sampler_draws': 123456, 'loss_balance_state': {'combat': [1, 2, 3]}}}
        self.write('status.json', self.state)
        (self.root/'training.pid').write_text('68032')
        alive = patch('turkeyworld.continue_rssm_preview_revision.process_alive', return_value=False)
        self.addCleanup(alive.stop); alive.start()

    def write(self, name, value):
        path = self.root/name; path.write_text(json.dumps(value), encoding='utf-8'); return path

    def plan(self, name, renderer):
        return self.write(name, {'version': 1, 'kind': 'frozen_preview_assets_v1',
            'allow_unqualified_codec': True, 'renderer': {'checkpoint': str(renderer), 'file_sha256': _sha(renderer)},
            'sr': {'checkpoint': str(self.sr), 'file_sha256': _sha(self.sr)}})

    def prepare(self):
        return prepare_preview_revision(self.root, self.new_revision, expected_step=5340,
                                        expected_checkpoint_sha256=_sha(self.checkpoint))

    def test_prepare_archives_exact_opaque_checkpoint_and_changes_only_revision_argument(self):
        before = {name: (self.root/name).read_bytes() for name in
                  ('status.json', 'launch_command.json', 'training.pid', 'checkpoints/rssm_world.pt')}
        with patch('subprocess.Popen', side_effect=AssertionError('Prepare must not start anything')):
            path, receipt = self.prepare()
            checked = check_prepared_preview_revision(self.root, self.new_revision, path)
        for name, content in before.items():
            self.assertEqual((self.root/name).read_bytes(), content)
        self.assertEqual(Path(receipt['checkpoint_backup']).read_bytes(), before['checkpoints/rssm_world.pt'])
        changed = [i for i, (a, b) in enumerate(zip(self.argv, receipt['argv'])) if a != b]
        self.assertEqual(changed, [self.argv.index('--preview-assets-revision-plan')+1])
        self.assertEqual(len(self.argv), len(receipt['argv']))
        self.assertEqual(checked['loss_gated_curriculum_at_boundary']['consecutive_below_threshold'], 7)
        self.assertEqual(checked['loss_gated_curriculum_at_boundary']['updates'], 228)
        self.assertFalse(checked['checkpoint_step_deserialized_by_operator'])
        self.assertEqual(checked['original_preview_revision'], self.state['preview_assets_revision'])
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_running_worker_wrong_boundary_or_parent_chain_rejected_before_archive(self):
        with patch('turkeyworld.continue_rssm_preview_revision.process_alive', return_value=True):
            with self.assertRaisesRegex(ValueError, 'still alive'):
                self.prepare()
        with self.assertRaisesRegex(ValueError, 'step'):
            prepare_preview_revision(self.root, self.new_revision, expected_step=1)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            prepare_preview_revision(self.root, self.new_revision, expected_checkpoint_sha256='wrong')
        raw = json.loads(self.new_revision.read_text()); raw['previous_revision_sha256'] = '0'*64
        self.new_revision.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.root/'preview_revision_resumes').exists())

    def test_changed_checkpoint_gate_old_plan_or_command_cannot_start_prepared_resume(self):
        path, receipt = self.prepare()
        for target, change in (
            (self.checkpoint, lambda value: value+b'changed'),
            (self.batch, lambda value: value+b' '),
            (self.root/'status.json', lambda value: value.replace(b'"consecutive_below_threshold": 7', b'"consecutive_below_threshold": 0')),
            (self.root/'launch_command.json', lambda value: value.replace(b'"64"', b'"32"'))):
            original = target.read_bytes(); target.write_bytes(change(original))
            with patch('subprocess.Popen', side_effect=AssertionError('Invalid prepared inputs must not launch')):
                with self.assertRaises(ValueError):
                    start_prepared_preview_revision(self.root, self.new_revision, path)
            target.write_bytes(original)
        self.assertEqual(check_prepared_preview_revision(self.root, self.new_revision, path)['source_step'], 5340)

    def test_mocked_start_is_hidden_binary_logs_and_leaves_training_pid_checkpoint_status_untouched(self):
        path, receipt = self.prepare()
        before = {name: (self.root/name).read_bytes() for name in ('training.pid', 'status.json', 'checkpoints/rssm_world.pt')}
        with patch('turkeyworld.continue_rssm_preview_revision.subprocess.Popen', return_value=SimpleNamespace(pid=76543)) as child:
            started = start_prepared_preview_revision(self.root, self.new_revision, path)
            self.assertEqual(started['pid'], 76543)
            args, kwargs = child.call_args
            self.assertEqual(args[0], receipt['argv'])
            self.assertEqual(kwargs['stdout'].mode, 'ab')
            self.assertEqual(kwargs['stderr'].mode, 'ab')
            self.assertIn('creationflags', kwargs)
            for name, content in before.items():
                self.assertEqual((self.root/name).read_bytes(), content)
            with self.assertRaises((FileExistsError, ValueError)):
                start_prepared_preview_revision(self.root, self.new_revision, path)
            self.assertEqual(child.call_count, 1)
        launch = json.loads((self.root/'launch_command.json').read_text())
        self.assertEqual(launch['argv'], receipt['argv'])
        self.assertEqual(json.loads(Path(receipt['old_launch_archive']).read_text())['argv'], self.argv)

    def test_argv_validation_rejects_duplicate_revision_wrong_run_or_different_entrypoint(self):
        for bad in (self.argv+['--preview-assets-revision-plan', str(self.old_revision)],
                    ['python', '-m', 'other.module', *self.argv[6:]]):
            with self.assertRaises(ValueError):
                preview_revision_argv(bad, self.new_revision, self.root)
        with self.assertRaises(ValueError):
            preview_revision_argv(self.argv, self.new_revision, self.root/'new_phase_is_forbidden')


if __name__ == '__main__':
    unittest.main()
