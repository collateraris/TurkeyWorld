from copy import deepcopy
from dataclasses import dataclass
import json
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.rssm_motion_only_training import (HORIZONS, COMBAT_READOUT_PREFIXES,
    configure_motion_only_trainable, motion_router_terms, motion_only_backward, _stream_backward)
from turkeyworld.rssm_player_geometry_objectives import PLAYER_GEOMETRY_PROTOCOL
from turkeyworld.rssm_npc_geometry_objectives import NPC_NAVIGATION_PROTOCOL
from turkeyworld.rssm_player_local_motion import PLAYER_LOCAL_MOTION_PROTOCOL
from turkeyworld.rssm_enemy_local_recovery import ENEMY_LOCAL_RECOVERY_PROTOCOL


@dataclass
class Config:
    navigation_only: bool = False
    simulation_hz: float = 30.


class TinyMotionWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.vjepa_backend = nn.Module(); self.vjepa_backend.config = Config()
        self.vjepa_backend.core = nn.Linear(1, 1)
        self.vjepa_backend.enemy_rssm = nn.Module(); self.vjepa_backend.enemy_rssm.core = nn.Linear(1, 1)
        self.vjepa_backend.enemy_rssm.attack_head = nn.Linear(1, 1)
        self.vjepa_backend.combat_rssm = nn.Module(); self.vjepa_backend.combat_rssm.core = nn.Linear(1, 1)
        self.vjepa_backend.combat_rssm.task_head = nn.Linear(1, 3)
        self.vjepa_backend.fusion_moe = nn.Module(); self.vjepa_backend.fusion_moe.combat_head = nn.Linear(1, 2)
        self.vjepa_backend.five_role_fusion = nn.Module(); self.vjepa_backend.five_role_fusion.combat_head = nn.Linear(1, 3)
        self.vjepa_backend.hit_target_head = nn.Linear(1, 2)
        self.vjepa_backend.attack_head = nn.Linear(1, 1)
        self.vjepa_backend.spatial_combat_head = nn.Linear(1, 2)
        self.flow_renderer = nn.Linear(1, 1)
        self.player_head = nn.Linear(3, 2)
        self.enemy_head = nn.Linear(3, 2)
        self.base_router = nn.Parameter(torch.randn(3, 3)*.1)
        self.five_router = nn.Parameter(torch.randn(5, 5)*.1)
        self.mode_router = nn.Parameter(torch.randn(3)*.1)
        self.calls, self.seen_frames, self.memory_outputs = [], [], []
        for core in (self.vjepa_backend.core, self.vjepa_backend.enemy_rssm.core, self.vjepa_backend.combat_rssm.core):
            nn.init.constant_(core.weight, .8); nn.init.constant_(core.bias, .01)

    def encode_map(self, grid): return torch.zeros(len(grid), 1)
    def initialize_from_image(self, frames, states, fmap):
        self.seen_frames.append(frames.detach().clone())
        return torch.full((len(states), 3), .1)

    def transition(self, state, action, fmap, memory):
        self.calls.append((state.detach().clone(), memory.detach().clone(), self.config.navigation_only))
        p = self.vjepa_backend.core(memory[:, :1]).tanh()
        e = self.vjepa_backend.enemy_rssm.core(memory[:, 1:2]).tanh()
        c = self.vjepa_backend.combat_rssm.core(memory[:, 2:3]).tanh()
        packed = torch.cat((p, e, c), -1)
        logits = self.player_head(packed)
        delta = .0002*logits.sigmoid()
        new = torch.cat((state[:, :2]+delta, state[:, 2:]), -1)
        if not self.config.navigation_only:
            npc = state[:, 8:10]+.0002*self.enemy_head(packed).tanh()
            new = torch.cat((new[:, :8], npc, new[:, 10:]), -1)
        self.memory_outputs.append(packed.detach().clone())
        base = dict(router_logits=self.base_router[None].expand(len(state), -1, -1),
            router_role_loss=self.base_router.sum()*1e5, router_balance_loss=self.base_router.sum()*1e5,
            five_role_fusion=dict(router_logits=self.five_router[None].expand(len(state), -1, -1),
                player_mode_logits=self.mode_router[None].expand(len(state), -1),
                player_mode_targets=torch.full((len(state),), int(not self.config.navigation_only), dtype=torch.long)))
        return dict(state=new, memory=packed, player_pass_logits=logits,
            respawn_flags=torch.zeros(len(state), 1, dtype=torch.bool), rssm_fusion_moe=base)


def batch(horizon, quiet=False, count=8):
    states = torch.zeros(count, horizon+1, 20)
    states[..., :2] = .25; states[..., 3:6] = 1
    if quiet:
        states[..., 8:10] = .5; states[..., 11] = 1; states[..., 12] = .6; states[..., 13] = 1
    target = states[:, 1:].clone()
    target[..., :2] += .0003*torch.arange(1, horizon+1)[None, :, None]
    if quiet: target[..., 8:10] -= .0003*torch.arange(1, horizon+1)[None, :, None]
    frames = torch.full((count, horizon+1, 3, 8, 8), 99.); frames[:, 0] = .25
    return dict(states=states, frames=frames, target_states=target,
        actions=torch.tensor([.8, .2, 0., 0.])[None, None].expand(count, horizon, 4).clone(),
        map=torch.zeros(count, 16, 16))


def kwargs(horizon=16):
    return dict(horizon=horizon, player_navigation_protocol={'tbptt': horizon, 'geometry': PLAYER_GEOMETRY_PROTOCOL},
        npc_navigation_protocol=NPC_NAVIGATION_PROTOCOL, player_local_protocol=PLAYER_LOCAL_MOTION_PROTOCOL,
        enemy_local_protocol=ENEMY_LOCAL_RECOVERY_PROTOCOL, player_local_horizon=4, enemy_local_horizon=4)


class MotionOnlyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def local_patches(self):
        return patch.multiple('turkeyworld.rssm_motion_only_training',
            _player_local_backward=lambda *a, **kw: dict(player_contact_loss=0., branches=0),
            enemy_local_recovery_backward=lambda *a, **kw: dict(enemy_local_loss=0., branches=0))

    def test_no_combat_router_queries_or_classes_receive_direct_supervision(self):
        base = torch.randn(2, 3, 3, requires_grad=True)
        five = torch.randn(2, 5, 5, requires_grad=True)
        mode = torch.randn(2, 3, requires_grad=True)
        role, balance = motion_router_terms(dict(rssm_fusion_moe=dict(router_logits=base,
            router_role_loss=base.sum()*1e10, router_balance_loss=base.sum()*1e10,
            five_role_fusion=dict(router_logits=five, player_mode_logits=mode, player_mode_targets=torch.tensor([0, 2])))))
        (role+balance).backward()
        self.assertEqual(float(base.grad[:, 2].abs().sum()+base.grad[:, :, 2].abs().sum()), 0.)
        self.assertEqual(float(five.grad[:, [2, 4]].abs().sum()+five.grad[:, :, [2, 4]].abs().sum()), 0.)
        self.assertEqual(float(mode.grad[:, 2].abs().sum()+mode.grad[1].abs().sum()), 0.)
        self.assertGreater(float(base.grad[:, :2, :2].abs().sum()), 0.)
        self.assertGreater(float(five.grad[:, [0, 1, 3]].abs().sum()), 0.)

    def test_actual_two_stream_backward_preserves_three_memories_and_uses_no_future_frames(self):
        model = TinyMotionWorld(); nav, enemy = batch(16), batch(16, True)
        for data in (nav, enemy):
            for value in data.values(): value.requires_grad_()
        configs = model.config, model.vjepa_backend.config
        with self.local_patches(), patch('turkeyworld.rssm_moe_training._direct_task_terms', side_effect=AssertionError('No combat AUX')):
            result = motion_only_backward(model, nav, enemy, **kwargs())
        self.assertEqual(len(model.calls), 32)
        for start in (0, 16):
            for offset in range(1, 16):
                self.assertTrue(torch.equal(model.calls[start+offset][1], model.memory_outputs[start+offset-1]))
        self.assertTrue(all(float(frames.max()) == .25 for frames in model.seen_frames))
        self.assertIs(model.config, configs[0]); self.assertIs(model.vjepa_backend.config, configs[1])
        self.assertEqual(result['main_independent_actor_count'], 16)
        self.assertEqual(result['main_combat_actor_count'], 0)
        self.assertEqual(result['combat_auxiliary_actor_count'], 0)
        self.assertEqual(result['combat_late_actor_count'], 0)
        self.assertEqual(result['combat_batch_draws'], 0)
        self.assertEqual(result['extra_contact_fragment_count'], 0)
        self.assertEqual(result['contact_scout_transitions'], 0)
        self.assertEqual(result['moe_combat_loss'], 0.)
        self.assertAlmostEqual(result['train_loss'], sum(result[key] for key in
            ('moe_player_loss', 'moe_enemy_loss', 'moe_router_loss')), places=9)
        for data in (nav, enemy): self.assertTrue(all(value.grad is None for value in data.values()))
        self.assertGreater(float(model.vjepa_backend.combat_rssm.core.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if name.startswith(COMBAT_READOUT_PREFIXES)))
        json.dumps(result, allow_nan=False)

    def test_player_and_enemy_groups_have_independent_fixed_denominators(self):
        torch.manual_seed(173); first = TinyMotionWorld(); second = deepcopy(first)
        nav, enemy = batch(16), batch(16, True)
        changed = deepcopy(enemy); changed['target_states'][..., 8:10] += .1
        with self.local_patches():
            a = motion_only_backward(first, nav, enemy, **kwargs())
            b = motion_only_backward(second, nav, changed, **kwargs())
        self.assertEqual(a['moe_player_loss'], b['moe_player_loss'])
        self.assertGreater(b['moe_enemy_loss'], a['moe_enemy_loss'])
        torch.testing.assert_close(first.player_head.weight.grad, second.player_head.weight.grad, rtol=0, atol=0)

    def test_final180_horizon_is_full_graph_and_not_subject_to_legacy_horizon_whitelist(self):
        self.assertEqual(HORIZONS, (16, 48, 80, 112, 144, 180))
        model = TinyMotionWorld()
        with self.local_patches(): result = motion_only_backward(model, batch(180), batch(180, True), **kwargs(180))
        self.assertEqual(len(model.calls), 360)
        self.assertTrue(all(row['tbptt'] == 180 and row['memory_detaches_after_initialization'] == 0
            for row in result['motion_only_streams']))
        self.assertGreater(float(model.vjepa_backend.core.weight.grad.abs().sum()), 0.)

    def test_only_final180_coordinate_error_reaches_first_transition_parameter(self):
        class FirstTickPerturbation(TinyMotionWorld):
            def __init__(self):
                super().__init__(); self.first_tick_offset = nn.Parameter(torch.tensor(.001))
            def transition(self, *args):
                result = super().transition(*args)
                if len(self.calls) == 1:
                    state = result['state'].clone()
                    state[:, 0] = state[:, 0]+self.first_tick_offset
                    result['state'] = state
                return result
        model = FirstTickPerturbation(); source = batch(180)
        with torch.no_grad():
            current = source['states'][:, 0]; memory = torch.full((8, 3), .1); targets = []
            for tick in range(180):
                result = model.transition(current, source['actions'][:, tick], torch.zeros(8, 1), memory)
                current, memory = result['state'], result['memory']; targets.append(current.clone())
        source['target_states'] = torch.stack(targets, 1)
        source['target_states'][:, -1, 0] += .02
        model.calls.clear(); model.memory_outputs.clear()
        _stream_backward(model, source, stream='navigation', horizon=180, context=1, observed_context=1,
            player_navigation_protocol=kwargs(180)['player_navigation_protocol'],
            npc_navigation_protocol=NPC_NAVIGATION_PROTOCOL, captures=[], enemy_captures=[],
            check_gradients=True, progress_callback=None)
        self.assertGreater(float(model.first_tick_offset.grad.abs()), 1e-6)

    def test_pure_combat_readout_weights_and_named_Adam_states_remain_bitexact(self):
        model = TinyMotionWorld(); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
        sum(parameter.square().mean() for parameter in model.parameters()).backward(); optimizer.step()
        names = dict(model.named_parameters())
        frozen = {name: parameter.detach().clone() for name, parameter in names.items() if name.startswith(COMBAT_READOUT_PREFIXES)}
        states = {name: deepcopy(optimizer.state[names[name]]) for name in frozen}
        groups = deepcopy(optimizer.state_dict()['param_groups'])
        optimizer.zero_grad(set_to_none=True)
        with self.local_patches(): motion_only_backward(model, batch(16), batch(16, True), **kwargs())
        optimizer.step()
        self.assertEqual(groups, optimizer.state_dict()['param_groups'])
        for name, before in frozen.items():
            self.assertFalse(names[name].requires_grad); self.assertIsNone(names[name].grad)
            torch.testing.assert_close(names[name], before, rtol=0, atol=0)
            for key, value in states[name].items(): torch.testing.assert_close(optimizer.state[names[name]][key], value, rtol=0, atol=0)

    def test_wrong_actor_mix_fire_damage_or_extra_contact_fragments_are_rejected(self):
        for mutation in ('count', 'fire', 'damage', 'enemy_dead', 'extra_contact', 'horizon'):
            with self.subTest(mutation=mutation):
                nav, enemy = batch(16), batch(16, True); options = kwargs()
                extra = None
                if mutation == 'count': nav = batch(16, count=7)
                elif mutation == 'fire': enemy['actions'][0, 0, 3] = 1
                elif mutation == 'damage': enemy['target_states'][0, 0, 4] = .9
                elif mutation == 'enemy_dead': enemy['target_states'][0, 0, 13] = 0
                elif mutation == 'extra_contact': extra = nav
                else: options['horizon'] = 176
                with self.local_patches(), self.assertRaises(ValueError):
                    motion_only_backward(TinyMotionWorld(), nav, enemy, extra, **options)

    def test_invalid_quiet_stream_cannot_leave_a_partially_backward_navigation_batch(self):
        model = TinyMotionWorld(); enemy = batch(16, True)
        enemy['actions'][0, 0, 3] = 1
        with self.assertRaises(ValueError): motion_only_backward(model, batch(16), enemy, **kwargs())
        self.assertEqual(model.calls, [])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_real_MAIN_prefix_local_branches_train_axes_and_own_NPC_velocity_without_extra_scouts(self):
        model = TinyMotionWorld()
        result = motion_only_backward(model, batch(16), batch(16, True), **kwargs())
        player, enemy = result['player_contact_replay'], result['enemy_local_recovery']
        self.assertEqual(player['scout_forward_transitions'], 0)
        self.assertEqual(player['independent_contact_fragments'], 0)
        self.assertGreater(player.get('direct_source_valid', 0), 0)
        self.assertGreater(player.get('axis_bce_loss', 0), 0.)
        self.assertGreater(player.get('axis_velocity_loss', 0), 0.)
        self.assertGreater(result['enemy_local_valid_source_labels'], 0)
        self.assertGreater(enemy['velocity_loss'], 0.)
        self.assertGreater(enemy['speed_loss'], 0.)
        self.assertTrue(enemy['full_own_memory_preserved'])


if __name__ == '__main__': unittest.main()
