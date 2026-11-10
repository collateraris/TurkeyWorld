import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from turkeyworld.engine import STATE_DIM, Game
from turkeyworld.staged_curriculum import (STAGES, SnapshotData, StagedCurriculum,
    StagedSequenceData, StagedTrainingPools, renderer_snapshot_loss)


class StagedCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root/'plan.json'
        plan = {'version': 1, 'target_steps': 450, 'validation_every': 100,
            'replay_fraction': .25, 'required_passes': 2,
            'datasets': {stage['id']: {'directory': str(self.root/stage['id'])} for stage in STAGES}}
        self.path.write_text(json.dumps(plan), encoding='utf-8')

    def test_quality_not_elapsed_updates_controls_promotion(self):
        c = StagedCurriculum(self.path, start_step=0)
        self.assertEqual([s['id'] for s in STAGES], ['navigation', 'static_enemy', 'moving1', 'full_game'])
        self.assertTrue(c.validation_due(0))
        c.record_validation({'passed': False, 'failures': ['collision']}, 600)
        self.assertEqual(c.stage['id'], 'navigation')
        self.assertFalse(c.validation_due(699))
        self.assertTrue(c.validation_due(700))
        self.assertFalse(c.record_validation({'passed': True}, 700))
        self.assertTrue(c.record_validation({'passed': True}, 800))
        self.assertEqual(c.stage['id'], 'static_enemy')
        self.assertFalse(c.state['final_qualified'])
        self.assertEqual(c.state['history'][0]['stage'], 'navigation')

    def test_minimum_and_saved_streak_are_respected(self):
        c = StagedCurriculum(self.path, start_step=0)
        for step in (0, 100, 200):
            c.record_validation({'passed': True}, step)
        self.assertEqual(c.state['pass_streak'], 0)
        c.record_validation({'passed': True}, 600)
        resumed = StagedCurriculum(self.path, start_step=650, previous=json.loads(json.dumps(c.state)))
        self.assertEqual(resumed.state['pass_streak'], 1)
        self.assertTrue(resumed.record_validation({'passed': True}, 700))
        self.assertEqual(resumed.state['stage_start_step'], 700)
        resumed.record_validation({'passed': True}, 700)  # New stage baseline.
        with self.assertRaises(ValueError):
            resumed.record_validation({'passed': True}, 700)

    def test_plan_change_rejected_on_resume(self):
        c = StagedCurriculum(self.path, start_step=0)
        plan = json.loads(self.path.read_text())
        plan['replay_fraction'] = .5
        self.path.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):
            StagedCurriculum(self.path, start_step=0, previous=c.state)

    def test_horizons_and_final_gate(self):
        c = StagedCurriculum(self.path, start_step=0)
        self.assertEqual([c.horizon(i) for i in (0, 100, 200, 300, 500)], [1, 4, 8, 16, 29])
        c.state.update(stage_index=3, stage_start_step=1000)
        c.sync(1200)
        c.record_validation({'passed': True}, 1200)
        self.assertFalse(c.state['final_qualified'])
        c.record_validation({'passed': True}, 1300)
        self.assertTrue(c.state['final_qualified'])

    def test_inline_labels_and_replay_stay_aligned(self):
        from turkeyworld.engine import BEHAVIOR_VERSION
        directory = self.root/'navigation/rgb/train'
        directory.mkdir(parents=True)
        game = Game(seed=0, endless=True)
        frames = np.stack([game.render(size=32, overlays=False)]*7)
        states = np.stack([game.state_vector()]*7)
        mask = np.zeros((7, 32, 32), dtype=bool)
        mask[2, 0, 0] = True
        np.savez(directory/'episode.npz', frames=frames, states=states, actions=np.zeros((6, 4), np.float32),
            events=np.zeros((6, 4), np.float32), hit_ids=np.full(6, -1), attack_flags=np.zeros((6, 1)),
            map=game.map, split='train', behavior_version=BEHAVIOR_VERSION,
            object_masks=mask, background_frames=np.zeros_like(frames), dt=1/30)
        data = StagedSequenceData(directory, mode='stream', episode_paths=['episode.npz'])
        batch = data.batch(1, context=2, horizon=4, indices=[(0, 0)])
        self.assertEqual(batch['states'].shape, (1, 6, STATE_DIM))
        self.assertEqual(batch['attack_flags'].shape, (1, 4, 1))
        self.assertEqual(batch['target_object_masks'].shape, (1, 4, 1, 32, 32))
        self.assertEqual(batch['target_object_masks'][0, 0, 0, 0, 0].item(), 1)
        self.assertEqual(batch['target_object_masks'][0, 1].sum().item(), 0)

    def test_renderer_snapshot_loss_does_not_touch_dynamics_or_map(self):
        class Flow(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.3))
            def loss(self, target, base, state, **unused):
                return (self.weight-target.mean()).square(), {}
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.map_weight = torch.nn.Parameter(torch.tensor(.5))
                self.renderer = torch.nn.Parameter(torch.tensor(.2))
                self.dynamics = torch.nn.Parameter(torch.tensor(.7))
                self.flow_renderer = Flow()
            def encode_map(self, value):
                return value.float()*self.map_weight
            def render_base(self, state, features, fmap):
                return self.renderer.expand(len(state), 3, 32, 32)+fmap.mean()*0
            def transition(self, *args):
                raise AssertionError('Static renderer data cannot train a transition')
        model = Model()
        batch = {'states': torch.zeros(2, STATE_DIM), 'frames': torch.ones(2, 3, 32, 32),
            'map': torch.zeros(2, 16, 16, dtype=torch.long), 'object_masks': torch.ones(2, 1, 32, 32),
            'background_frames': torch.zeros(2, 3, 32, 32)}
        loss, _, endpoint = renderer_snapshot_loss(model, None, batch)
        loss.backward()
        self.assertIsNone(model.dynamics.grad)
        self.assertIsNone(model.map_weight.grad)
        self.assertIsNotNone(model.renderer.grad)
        self.assertIsNotNone(model.flow_renderer.weight.grad)
        self.assertEqual(endpoint['state'].shape[-1], STATE_DIM)


if __name__ == '__main__':
    unittest.main()
