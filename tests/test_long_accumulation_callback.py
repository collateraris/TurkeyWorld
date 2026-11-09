"""Runner-side clocks/checkpoints advance only after a flushed TBPTT group."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class LongAccumulationCallbackTests(unittest.TestCase):
    def harness(self, folder, pause_at_flush):
        from turkeyworld.vjepa_training import VJEPAExperiment

        class BoundaryStop(Exception):
            pass

        class Experiment(VJEPAExperiment):
            def __init__(self):
                self.root, self.device = Path(folder), torch.device('cpu')
                self.args = SimpleNamespace(ema=.996, lr=1e-4, gradient_accumulation_steps=24,
                    batch=1, chunk=16, seed=73, log_every=1, save_every=1, aligned_supervision=True)
                self.global_step, self.fragments_seen = 1000, 2000
                self.original_resume = {'safe': 'previous-flush'}
                self.extra = {'long_curriculum_state': self.original_resume}
                self.state = {'adversarial': {'active': True}}
                self.updates, self.metrics, self.saves, self.stops, self.previews = [], [], [], [], []

            def restore(self, model, optimizer, phase):
                self.restored_phase = phase
                return 0

            def update(self, stage, current, total, label, **values):
                self.state.update(stage=stage, current=current, total=total, **values)
                self.updates.append({'current': current, **deepcopy(values),
                    'batch': deepcopy(self.state['training_batch']),
                    'resume': deepcopy(self.extra['long_curriculum_state'])})

            def metric(self, values):
                self.metrics.append(deepcopy(values))

            def save(self, model, optimizer, phase, step, metrics, extra=None):
                self.saves.append({'phase': phase, 'step': step, 'metrics': deepcopy(metrics),
                    'resume': deepcopy(self.extra['long_curriculum_state']),
                    'fragments_seen': self.fragments_seen,
                    'phase_complete': self.extra.get('phase_complete', False)})

            def preview_if_due(self, model):
                self.previews.append(self.global_step)

            def check_stop(self, model, optimizer, phase, step, metrics):
                self.stops.append((phase, step, self.global_step))
                if pause_at_flush:
                    raise BoundaryStop()

        experiment = Experiment()
        return experiment, BoundaryStop

    def model(self):
        model = torch.nn.Module()
        model.core = torch.nn.Module()
        model.core.encoder = torch.nn.Linear(2, 2)
        return model

    def ema(self):
        class EMA:
            def __init__(self, encoder, momentum):
                self.encoder, self.momentum, self.updates = encoder, momentum, []

            def to(self, device):
                return self

            def update(self, encoder):
                self.updates.append(encoder)

        return EMA

    def fake_curriculum(self, experiment, captured):
        final_resume = {'safe': 'new-flush', 'task_resume': {'offset': 384}}

        def train(model, directory, optimizer, **options):
            captured.update(options)
            self.assertIs(options['resume'], experiment.original_resume)
            for microstep in range(1, 25):
                updated = microstep == 24
                inner = {'optimizer_updated': updated, 'accumulation_microstep': microstep,
                    'accumulation_steps': 24, 'accumulation_group_size': 24,
                    'optimizer_group_loss': .25 if updated else None}
                options['on_chunk']({'chunk': inner, 'resume_state': final_resume if updated else None,
                    'metrics': {'loss': 10.+microstep, 'state': microstep/100},
                    'completed_transitions': microstep*16, 'total_transitions': 9000,
                    'horizon': 9000, 'eta_seconds': 10.})
                if not updated:
                    self.assertIs(experiment.extra['long_curriculum_state'], experiment.original_resume)
            return {'metrics': {'loss': .25}, 'resume_state': final_resume,
                    'completed_transitions': 384, 'total_transitions': 9000}

        return train, final_resume

    def test_23_pending_fragments_never_advance_optimizer_clocks_or_save_partial_state(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, boundary_stop = self.harness(folder, pause_at_flush=True)
            model, captured = self.model(), {}
            fake, final_resume = self.fake_curriculum(experiment, captured)
            with patch('turkeyworld.vjepa_training.EMATargetEncoder', self.ema()), \
                 patch('turkeyworld.vjepa_long_rollout.train_curriculum', side_effect=fake):
                with self.assertRaises(boundary_stop):
                    experiment.long_curriculum(model, 'teacher500vq')
            self.assertEqual(captured['gradient_accumulation_steps'], 24)
            self.assertEqual(captured['chunk_size'], 16)
            self.assertTrue(captured['aligned_supervision'])
            self.assertEqual(experiment.global_step, 1001)
            self.assertEqual(experiment.fragments_seen, 2024)
            self.assertEqual(len(experiment.ema.updates), 1)
            self.assertIs(experiment.ema.updates[0], model.core.encoder)
            self.assertEqual(experiment.previews, [1001])
            self.assertEqual(experiment.stops, [('teacher500vq_vjepa_temporal', 1, 1001)])
            self.assertEqual(len(experiment.saves), 1)
            self.assertEqual(experiment.saves[0]['resume'], final_resume)
            self.assertEqual(experiment.saves[0]['fragments_seen'], 2024)
            self.assertEqual(len(experiment.updates), 24)
            self.assertEqual([row['batch']['microstep'] for row in experiment.updates], list(range(1, 25)))
            for row in experiment.updates[:23]:
                self.assertEqual(row['global_training_step'], 1000)
                self.assertEqual(row['batch']['status'], 'accumulating')
                self.assertEqual(row['resume'], experiment.original_resume)
            self.assertEqual(experiment.updates[-1]['batch']['status'], 'updated')
            self.assertEqual(experiment.updates[-1]['loss'], .25)
            self.assertEqual(experiment.updates[-1]['current'], 384)
            self.assertEqual(len(experiment.metrics), 1)
            self.assertEqual(experiment.metrics[0]['step'], 1)
            self.assertEqual(experiment.metrics[0]['loss'], .25)
            self.assertEqual(experiment.metrics[0]['loss_aggregation'], 'mean_of_tbptt_chunks')
            self.assertFalse(experiment.state['adversarial']['active'])

    def test_normal_completion_adds_only_the_safe_final_checkpoint_after_one_update(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment, _ = self.harness(folder, pause_at_flush=False)
            fake, final_resume = self.fake_curriculum(experiment, {})
            with patch('turkeyworld.vjepa_training.EMATargetEncoder', self.ema()), \
                 patch('turkeyworld.vjepa_long_rollout.train_curriculum', side_effect=fake):
                experiment.long_curriculum(self.model(), 'teacher500vq')
            self.assertEqual(experiment.global_step, 1001)
            self.assertEqual(len(experiment.ema.updates), 1)
            self.assertEqual(experiment.fragments_seen, 2024)
            # One ordinary flush save, then the existing phase-final save.
            self.assertEqual(len(experiment.saves), 2)
            self.assertFalse(experiment.saves[0]['phase_complete'])
            self.assertTrue(experiment.saves[1]['phase_complete'])
            self.assertTrue(all(value['resume'] == final_resume for value in experiment.saves))
            self.assertTrue(all(value['step'] == 1 for value in experiment.saves))
            self.assertEqual(len(experiment.stops), 1)


if __name__ == '__main__':
    unittest.main()
