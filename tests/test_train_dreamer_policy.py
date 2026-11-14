"""CPU-only completed-boundary and continuation guards for the policy worker."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld import train_dreamer_policy as runner


class DreamerPolicyRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def optimizer_fixture(self):
        module = nn.Linear(3, 2)
        optimizer = torch.optim.Adam(module.parameters(), lr=3e-5, eps=1e-5)
        module(torch.ones(2, 3)).square().mean().backward(); optimizer.step()
        return module, deepcopy(optimizer.state_dict()), [name for name, _ in module.named_parameters()]

    def test_valid_completed_adam_retains_names_moments_hyperparameters_and_clocks(self):
        module, payload, names = self.optimizer_fixture()
        optimizer = torch.optim.Adam(module.parameters(), lr=3e-5, eps=1e-5)
        runner.restore_optimizer(optimizer, payload, module, names, 1)
        self.assertEqual(optimizer.state_dict()['param_groups'], payload['param_groups'])
        for index, state in payload['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(optimizer.state_dict()['state'][index][name], value, rtol=0, atol=0)

    def test_changed_parameter_names_order_and_completed_clock_are_rejected(self):
        module, original, names = self.optimizer_fixture()
        for mutation in ('names', 'order', 'clock', 'incomplete', 'nonfinite'):
            with self.subTest(mutation=mutation):
                payload, declared = deepcopy(original), names.copy()
                if mutation == 'names': declared[0] = 'unrelated_world.weight'
                elif mutation == 'order': payload['param_groups'][0]['params'].reverse()
                elif mutation == 'clock': next(iter(payload['state'].values()))['step'] = torch.tensor(2.)
                elif mutation == 'incomplete': payload['state'].pop(next(iter(payload['state'])))
                else: next(iter(payload['state'].values()))['exp_avg'].flatten()[0] = float('nan')
                optimizer = torch.optim.Adam(module.parameters(), lr=3e-5, eps=1e-5)
                with self.assertRaises(ValueError): runner.restore_optimizer(optimizer, payload, module, declared, 1)

    def test_fractional_Adam_clocks_wrong_shapes_negative_variance_and_hyperparameter_drift_reject(self):
        module, original, names = self.optimizer_fixture()
        for mutation in ('fractional_step', 'moment_shape', 'negative_variance', 'lr', 'betas'):
            with self.subTest(mutation=mutation):
                payload = deepcopy(original)
                state = next(iter(payload['state'].values()))
                if mutation == 'fractional_step': state['step'] = torch.tensor(1.5)
                elif mutation == 'moment_shape': state['exp_avg'] = torch.zeros(1)
                elif mutation == 'negative_variance': state['exp_avg_sq'].flatten()[0] = -1
                elif mutation == 'lr': payload['param_groups'][0]['lr'] = 1.
                else: payload['param_groups'][0]['betas'] = (.1, .2)
                optimizer = torch.optim.Adam(module.parameters(), lr=3e-5, eps=1e-5)
                with self.assertRaises(ValueError): runner.restore_optimizer(optimizer, payload, module, names, 1)

    def test_fresh_branch_requires_empty_adam_history(self):
        module, trained, names = self.optimizer_fixture()
        fresh = torch.optim.Adam(module.parameters(), lr=3e-5, eps=1e-5)
        runner.restore_optimizer(fresh, deepcopy(fresh.state_dict()), module, names, 0)
        self.assertEqual(len(fresh.state), 0)
        with self.assertRaises(ValueError): runner.restore_optimizer(fresh, trained, module, names, 0)

    def world_payload(self, step=3):
        return dict(stage='world', training_steps=step, global_training_step=step,
            run_state={'global_training_step': step}, config={'world_architecture': 'test'}, model={'weight': torch.ones(1)})

    def test_pinned_world_rejects_disagreeing_complete_checkpoint_clocks(self):
        for field in ('training_steps', 'global_training_step', 'run_state', 'stage'):
            with self.subTest(field=field):
                payload = self.world_payload()
                if field == 'run_state': payload['run_state']['global_training_step'] = 2
                elif field == 'stage': payload['stage'] = 'partial'
                else: payload[field] = 2
                source = self.root/f'{field}.pt'; torch.save(payload, source)
                with self.assertRaises(ValueError): runner.pinned_world(source, self.root/f'pinned_{field}.pt')

    def test_pinned_boundary_survives_atomic_replacement_of_live_world_checkpoint(self):
        source = self.root/'live_world.pt'; torch.save(self.world_payload(3), source)
        destination = self.root/'snapshot/world.pt'
        pinned = runner.pinned_world(source, destination)
        self.assertEqual(pinned['step'], 3)
        before = runner.file_sha(destination)
        runner.atomic_torch(source, self.world_payload(4))
        self.assertNotEqual(runner.file_sha(source), before)
        self.assertEqual(runner.file_sha(destination), before)
        repeated = runner.pinned_world(source, destination)
        self.assertEqual(repeated, pinned)
        self.assertEqual(runner.pinned_world(source, self.root/'new_snapshot.pt')['step'], 4)

    def test_unsaved_history_tail_is_archived_and_cannot_create_duplicate_steps_on_resume(self):
        rows = [dict(step=step, metrics={'loss': step/10}) for step in range(1, 6)]
        path = self.root/'history.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
        runner.archive_unsaved_history(self.root, 3)
        self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], rows[:3])
        archives = list(self.root.glob('history_unsaved_tail_before_resume_*.jsonl'))
        self.assertEqual(len(archives), 1)
        self.assertEqual([json.loads(line) for line in archives[0].read_text().splitlines()], rows[3:])
        runner.archive_unsaved_history(self.root, 3)
        self.assertEqual(len(list(self.root.glob('history_unsaved_tail_before_resume_*.jsonl'))), 1)
        with path.open('a', encoding='utf-8') as stream: stream.write(json.dumps(rows[3])+'\n')
        self.assertEqual([json.loads(line)['step'] for line in path.read_text().splitlines()], [1, 2, 3, 4])

    def test_interrupted_json_append_is_archived_without_losing_committed_history(self):
        committed = json.dumps({'step': 1, 'metrics': {'loss': 1.}})+'\n'
        partial = '{"step":2,"metrics":{"loss":'
        path = self.root/'history.jsonl'; path.write_text(committed+partial, encoding='utf-8')
        runner.archive_unsaved_history(self.root, 1)
        self.assertEqual(path.read_text(encoding='utf-8'), committed)
        archives = list(self.root.glob('history_unsaved_tail_before_resume_*.jsonl'))
        self.assertEqual(len(archives), 1)
        self.assertIn(partial, archives[0].read_text(encoding='utf-8'))

    def test_failed_atomic_torch_write_does_not_replace_the_previous_complete_checkpoint(self):
        destination = self.root/'policy.pt'; torch.save({'step': 3}, destination)
        before = runner.file_sha(destination)
        with patch.object(runner.torch, 'save', side_effect=OSError('simulated interrupted write')):
            with self.assertRaises(OSError): runner.atomic_torch(destination, {'step': 4})
        self.assertEqual(runner.file_sha(destination), before)
        self.assertEqual(torch.load(destination, weights_only=False)['step'], 3)


if __name__ == '__main__': unittest.main()
