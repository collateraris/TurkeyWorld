from types import SimpleNamespace
import unittest

import torch

from turkeyworld.vjepa_independent_panel import (
    PROBE_OFFSETS, axis_errors, common_start_probes, independent_windows, summarize_probes)


class IndependentPanelTests(unittest.TestCase):
    def test_episode_disjoint_and_order_invariant(self):
        data = SimpleNamespace(split='val', episodes=[
            dict(path=f'val/ep{i}.npz', frames=range(182+i)) for i in range(13)])
        excluded = [dict(episode='ep12.npz', start=3)]
        first = independent_windows(data, excluded, 'navigation')
        data.episodes.reverse()
        self.assertEqual(first, independent_windows(data, excluded, 'navigation'))
        self.assertEqual(len({w['episode'] for w in first}), 8)
        self.assertNotIn('ep12.npz', {w['episode'] for w in first})
        for w in first:
            i = int(w['episode'][2:-4])
            self.assertIn(w['start'], range(3, 182+i-180))
        with self.assertRaises(ValueError):
            independent_windows(data, excluded, 'navigation', count=14)
        data.split = 'train'
        with self.assertRaises(ValueError):
            independent_windows(data, excluded, 'navigation')

    def test_signed_axes_and_ambiguous_contacts(self):
        stats = axis_errors(torch.tensor([.1,-.1,-.1,.2,0]),
                            torch.tensor([.1,0,-.1,.1,.0]),
                            torch.tensor([0,-.1,-.1,.0,0]))
        self.assertEqual(stats, dict(free_axes=2, blocked_axes=1, partial_axes=1,
                                     false_stops=1, false_passes=1))
        summary = summarize_probes([stats, stats])
        self.assertEqual(summary['false_stop_rate'], .5)
        self.assertEqual(summary['false_pass_rate'], 1.)

    def test_every_probe_ends_its_context_before_target(self):
        seen, commands = [], []
        class Adapter:
            feedback_renderer = True
            def initialize(self, context, grid, placeholder):
                seen.append(context[0,:,0].tolist())
                return context[:,-1]
            def transition_prior(self, state, action):
                commands.append(int(action[0,0]))
                return SimpleNamespace(state=state)
        timeline = torch.zeros(1,184,20)
        timeline[0,:,0] = torch.arange(184)
        timeline[:,:,3:5] = 1
        actions = torch.zeros(1,180,4); actions[0,:,0]=torch.arange(180)
        common_start_probes(Adapter(),timeline[:,:4],torch.zeros(1,16,16),actions,timeline[:,4:])
        self.assertEqual(seen, [list(range(t,t+4)) for t in PROBE_OFFSETS])
        self.assertEqual(commands,list(PROBE_OFFSETS))
