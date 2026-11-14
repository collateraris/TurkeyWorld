from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld import collect_policy_motion as collector
from turkeyworld.dreamer_actor_critic import ActorCriticConfig, FactorizedActor
from turkeyworld.engine import Game
from turkeyworld.jepa_training_utils import SequenceData


@dataclass
class Config:
    navigation_only: bool = False
    image_size: int = 32


class TinyWorld(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.ones(1))
        self.config = Config(); self.vjepa_backend = nn.Module(); self.vjepa_backend.config = Config()


class CausalAdapter:
    instances = []
    def __init__(self, world, heads=None, sample_latents=True):
        assert heads is None and sample_latents is False
        self.world, self.feature_dim = world, 23
        self.initializations, self.observations = [], []
        self.instances.append(self)

    def initialize(self, state, grid, rgb):
        self.initializations.append((state.clone(), rgb.clone()))
        return SimpleNamespace(state=state.clone(), memory=torch.zeros(len(state), 3))

    def features(self, current): return torch.cat((current.state, current.memory), -1)

    def observe(self, current, actions, next_state, next_rgb):
        self.observations.append((current.memory.clone(), actions.clone(), next_state.clone(), next_rgb.clone()))
        # Memory depends on every actual current Source image, and continues
        # instead of restarting from an initial observation each tick.
        return SimpleNamespace(state=next_state.clone(), memory=current.memory+next_rgb.mean((1, 2, 3))[:, None])

    def transition_prior(self, *a, **kw): raise AssertionError('Collector must use current Source posteriors')


class PolicyMotionCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.policy, self.world = self.root/'policy.pt', self.root/'world.pt'
        config = ActorCriticConfig(23, hidden_dim=8, layers=1)
        actor = FactorizedActor(config)
        with torch.no_grad():
            actor.output.weight.zero_(); actor.output.bias.zero_()
            actor.output.bias[[1, 4, 7, 10]] = 10  # no movement; requests fire, which collection masks
        torch.save(dict(protocol=collector.POLICY_PROTOCOL, step=200, actor_updates=100,
            actor_critic_config=config.to_dict(), policy={'actor.'+name: value for name, value in actor.state_dict().items()}), self.policy)
        torch.save(dict(stage='world', training_steps=6873, global_training_step=6873,
            run_state={'global_training_step': 6873}), self.world)
        CausalAdapter.instances = []

    def mock_models(self):
        def load(path, device): return TinyWorld(), torch.load(path, map_location='cpu', weights_only=False)
        return patch.multiple(collector, RSSMImaginationAdapter=CausalAdapter, load_world_checkpoint=load)

    def run_collection(self, directory='corpus', **kw):
        return collector.collect_policy_motion(self.policy, self.world, self.root/directory,
            cycle_index=0, episodes_per_stream=1, transitions=4, device='cpu', **kw)

    def test_real_Source_pair_files_fit_SequenceData_and_use_every_current_RGB_posterior(self):
        policy_sha, world_sha = collector.file_sha(self.policy), collector.file_sha(self.world)
        with self.mock_models(): receipt = self.run_collection(exploration=0.)
        self.assertEqual(len(receipt['episodes']), 2)
        self.assertEqual(receipt['total_real_Source_transitions'], 8)
        self.assertEqual(receipt['policy_checkpoint_sha256'], policy_sha)
        self.assertEqual(receipt['world_checkpoint_sha256'], world_sha)
        self.assertTrue(receipt['validation_unchanged'])
        adapter = CausalAdapter.instances[0]
        self.assertEqual(len(adapter.initializations), 2)
        self.assertEqual(len(adapter.observations), 8)
        for role, key in enumerate(('nav_train_directory', 'enemy_train_directory')):
            directory = Path(receipt[key]); self.assertEqual(directory.name, 'train')
            data = SequenceData(directory, split='train', image_size=32, mode='stream')
            batch = data.batch(1, context=1, horizon=4, indices=np.array([[0, 0]]))
            self.assertEqual(batch['states'].shape, (1, 5, 20))
            self.assertEqual(batch['frames'].shape, (1, 5, 3, 32, 32))
            self.assertTrue(bool(batch['actions'][..., 3].eq(0).all()))
            with np.load(receipt['episodes'][role]['path'], allow_pickle=False) as saved:
                self.assertEqual(int(saved['fire_masked_flags'].sum()), 4)
                for tick in range(4):
                    _, _, observed_state, observed_rgb = adapter.observations[role*4+tick]
                    torch.testing.assert_close(observed_state[0], torch.from_numpy(saved['states'][tick+1]), rtol=0, atol=0)
                    expected = torch.from_numpy(saved['frames'][tick+1]).permute(2, 0, 1).float()/255
                    torch.testing.assert_close(observed_rgb[0], expected, rtol=0, atol=0)
            self.assertGreater(float(adapter.observations[role*4+1][0].abs().sum()), 0.)
        self.assertEqual(collector.file_sha(self.policy), policy_sha)
        self.assertEqual(collector.file_sha(self.world), world_sha)

    def test_saved_full_initial_snapshot_and_actions_reproduce_Source_frames_states_and_events(self):
        with self.mock_models(): receipt = self.run_collection(exploration=1.)
        for row in receipt['episodes']:
            with np.load(row['path'], allow_pickle=False) as saved:
                initial = json.loads(str(saved['initial_state']))
                game = Game(seed=int(saved['seed']), endless=True, enemies_enabled=bool(saved['enemies_enabled']))
                game.map = saved['map'].copy(); game.map_height, game.map_width = game.map.shape
                game.set_state(initial)
                self.assertTrue(np.array_equal(game.state_vector(), saved['states'][0]))
                self.assertTrue(np.array_equal(game.render(size=32, overlays=False), saved['frames'][0]))
                self.assertTrue(np.all(np.linalg.norm(saved['actions'][:, :3], axis=-1) > 0))
                for tick, action in enumerate(saved['actions']):
                    event = game.step(action, dt=1/30)
                    self.assertTrue(np.array_equal(game.state_vector(), saved['states'][tick+1]))
                    self.assertTrue(np.array_equal(Game.event_vector(event), saved['events'][tick]))
                    self.assertTrue(np.array_equal(game.render(size=32, overlays=False), saved['frames'][tick+1]))

    def test_held_epsilon_vectors_are_nonzero_and_frozen_Torch_RNG_is_preserved(self):
        before = torch.get_rng_state().clone(); numpy_before = np.random.get_state()
        with self.mock_models(): receipt = self.run_collection(exploration=1.)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        after = np.random.get_state(); self.assertEqual(numpy_before[0], after[0])
        np.testing.assert_array_equal(numpy_before[1], after[1]); self.assertEqual(numpy_before[2:], after[2:])
        for row in receipt['episodes']:
            with np.load(row['path'], allow_pickle=False) as saved:
                self.assertTrue(bool(saved['exploration_flags'].all()))
                self.assertTrue(np.all(saved['actions'][1:] == saved['actions'][0]))  # hold at least4ticks
        self.assertTrue(receipt['world_parameters_bitexact']); self.assertTrue(receipt['policy_parameters_bitexact'])

    def test_completed_collection_is_idempotent_and_rejects_data_or_checkpoint_drift(self):
        with self.mock_models():
            first = self.run_collection(exploration=0.)
            calls = len(CausalAdapter.instances)
            repeated = self.run_collection(exploration=0.)
            self.assertEqual(first, repeated); self.assertEqual(len(CausalAdapter.instances), calls)
            changed = self.root/'replacement.pt'; torch.save({'changed': True}, changed)
            changed.replace(self.policy)
            with self.assertRaises(ValueError): self.run_collection(exploration=0.)

    def test_quiet_HP_changes_are_rejected_with_bounded_attempts_and_no_completed_receipt(self):
        original = collector._initial_source
        def near(seed, stream):
            game = original(seed, stream)
            if stream == 'enemy_motion':
                # Legal adjacent positions in an open part of the unchanged map.
                state = game.get_state(); state['player'].update(x=2.5, y=2.5, angle=0.)
                state['enemies'][0].update(x=3.3, y=2.5, attack_cooldown=0.)
                game.set_state(state)
            return game
        with self.mock_models(), patch.object(collector, '_initial_source', side_effect=near):
            with self.assertRaises(RuntimeError): self.run_collection(exploration=0., max_attempts=2)
        root = self.root/'corpus'
        self.assertFalse((root/'collection_receipt.json').exists())
        rejected = json.loads((root/'rejected_attempts.json').read_text())
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all(row['reason'] == 'player_HP_changed' for row in rejected))
        self.assertEqual(list((root/'enemy_motion/rgb/train').glob('*.npz')), [])


if __name__ == '__main__': unittest.main()
