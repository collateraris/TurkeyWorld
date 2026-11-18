"""The production V-JEPA transition and checkpoint boundary for quiet chase."""
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

import torch
from test_player_axis_fraction import axis_config
from turkeyworld.attach_quiet_pair import apply
from turkeyworld.engine import Game
from turkeyworld.learned_axis_collision import (
    LocalAxisCollisionHead,
    PREFIX,
    with_local_axis_head,
    verify_frozen_resume,
)
from turkeyworld.learned_quiet_pursuit import QuietPursuitHead
from turkeyworld.models import NeuralWorldModel, WorldModelConfig
from turkeyworld.quiet_pair_profile import PREFIXES, compatible_policy_refresh, quiet_mask
from turkeyworld.renderer_student import _hash_tensors
from turkeyworld.rssm_combat_protocol import environment_profile


class QuietPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.base = NeuralWorldModel(axis_config()).eval()
        self.player = LocalAxisCollisionHead()
        self.npc = QuietPursuitHead()
        source = dict(
            config=asdict(self.base.config),
            model=self.base.state_dict(),
            training_steps=9,
            global_training_step=9,
            stage="world",
            optimizer={"state": {1: {"step": 9}}},
            optimizer_parameter_names=["other"],
            rng_torch=torch.get_rng_state(),
            rng_cuda=[],
            ema_target_encoder={},
            state_schema="test",
            vjepa_sr_feedback={},
        )
        self.nav = with_local_axis_head(source, self.player, nav_only=True)
        self.nav_model = self.model(self.nav)
        self.old_frozen = {
            n: p
            for n, p in self.nav_model.named_parameters()
            if n.startswith(PREFIX) or n == "vjepa_backend.delta_head.0.bias"
        }
        self.nav["vjepa_sr_training"] = dict(frozen_parameters_sha256=_hash_tensors(self.old_frozen))
        self.npc_weights = dict(model=self.npc.state_dict(), train_receipt={"training_uses_heldout": False})
        self.pair = apply(self.nav, self.npc_weights)
        self.model_pair = self.model(self.pair)
        game = Game(seed=31, endless=True)
        self.state = torch.tensor(game.state_vector())[None]
        self.state[0, :4] = torch.tensor([0.3, 0.3, 0, 1])
        self.state[0, 4:8] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        self.state[0, 8:10] = 0.75
        self.state[0, 12:17] = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
        self.grid = torch.zeros(1, 16, 16, dtype=torch.long)

    @staticmethod
    def model(payload):
        m = NeuralWorldModel(WorldModelConfig(**payload["config"])).eval()
        m.load_state_dict(payload["model"], strict=True)
        return m

    def test_live_state_and_melee_guard(self):
        s = self.state.repeat(11, 1)
        a = torch.zeros(11, 4)
        a[1, 3] = 0.5
        for i, k in enumerate((6, 7, 14, 15, 16), 2):
            s[i, k] = 1.0e-4
        for i, k in enumerate((4, 12, 13), 7):
            s[i, k] = 0
        s[10, 8:10] = s[10, :2] + torch.tensor([1.06, 0]) / 16
        self.assertEqual(quiet_mask(s, a, 1 / 30).tolist(), [True] + [False] * 10)

    def test_mixed_batch_moves_both_actors_but_preserves_combat_and_navigation(self):
        b = 3
        state = self.state.repeat(b, 1)
        state[2, 8:10] = state[2, :2] + 0.01
        a = torch.tensor([[1.0, 0.2, 0.1, 0.0], [-0.4, 0.5, 0.3, 1.0], [0.0, 1.0, 0.0, 0.0]])
        grid = self.grid.repeat(b, 1, 1)
        memory = torch.full((b, self.nav_model.core.memory_tokens, self.nav_model.core.width), 0.01)

        def step(m, nav):
            with environment_profile(m, navigation_only=nav):
                fmap = m.encode_map(grid)
                return m.transition(state, a, fmap, memory)

        seen = []

        def npc(s, goal, grid):
            seen.append(goal.detach().clone())
            return torch.full_like(goal, 0.00025)

        with torch.inference_mode(), patch.object(
            Game, "step", side_effect=AssertionError("runtime Source")
        ), patch.object(self.model_pair.vjepa_backend.quiet_pursuit_head, "forward", side_effect=npc):
            old, new = step(self.nav_model, False), step(self.model_pair, False)
            self.assertEqual(new["quiet_pair_active"].tolist(), [True, False, False])
            self.assertEqual(len(seen), 1)
            torch.testing.assert_close(seen[0][0], new["state"][0, :2], rtol=0, atol=1e-7)
            torch.testing.assert_close(
                new["npc_displacement"][0, 0], torch.full((2,), 0.00025), rtol=0, atol=1e-7
            )
            for k in ("state", "memory", "events", "target_logits", "attack_logits"):
                self.assertTrue(torch.equal(old[k][1:], new[k][1:]), k)
            old, new = step(self.nav_model, True), step(self.model_pair, True)
            for k in ("state", "memory", "events", "player_pass_logits"):
                self.assertTrue(torch.equal(old[k], new[k]), k)
            self.assertEqual(len(seen), 1)

    def test_game_step_and_policy_prior_share_quiet_motion(self):
        from turkeyworld.vjepa_policy_world import VJEPAImaginationAdapter

        world = self.model_pair
        with torch.inference_mode(), environment_profile(world, navigation_only=False):
            adapter = VJEPAImaginationAdapter(world)
            current = adapter.initialize(self.state, self.grid, torch.zeros(1, 3, 32, 32))
            action = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            with patch.object(Game, "step", side_effect=AssertionError("runtime Source")):
                gui = world.step(
                    current.state,
                    action,
                    memory=current.memory,
                    map_features=current.map_features,
                    render=False,
                )
                following = adapter.transition_prior(current, action)
            self.assertTrue(gui["quiet_pair_active"].item())
            self.assertTrue(torch.equal(gui["state"], following.state))
            self.assertTrue(torch.equal(gui["memory"].reshape(1, -1), following.memory))
            self.assertIsNotNone(world.vjepa_backend.quiet_pursuit_head._cache)

    def test_optimizer_boundary_policy_refresh_and_freeze(self):
        m = self.model_pair
        for key in (
            "optimizer",
            "optimizer_parameter_names",
            "rng_torch",
            "rng_cuda",
            "ema_target_encoder",
            "vjepa_sr_training",
        ):
            self.assertIs(self.nav[key], self.pair[key], key)
        for key in self.nav["model"]:
            self.assertIs(self.nav["model"][key], self.pair["model"][key])
        updated = apply(self.nav, self.npc_weights, validated=self.pair, expect_step=9)
        self.assertEqual(updated["quiet_pair_migration"], self.pair["quiet_pair_migration"])
        with self.assertRaises(ValueError):
            apply(self.nav, {**self.npc_weights, "train_receipt": {"training_uses_heldout": True}})
        for previous in (self.nav_model.config, self.base.config):
            self.assertTrue(compatible_policy_refresh(previous, m, self.pair))
        self.assertFalse(
            compatible_policy_refresh(replace(self.base.config, vjepa_context_frames=8), m, self.pair)
        )
        frozen = {**self.old_frozen, **{n: p for n, p in m.named_parameters() if n.startswith(PREFIXES)}}
        current_hash = _hash_tensors(frozen)
        self.assertTrue(verify_frozen_resume(m, self.pair, frozen, current_hash))
        later = {
            **self.pair,
            "training_steps": 10,
            "vjepa_sr_training": {"frozen_parameters_sha256": current_hash},
        }
        self.assertTrue(verify_frozen_resume(m, later, frozen, current_hash))
        with self.assertRaises(ValueError):
            verify_frozen_resume(m, {**self.pair, "training_steps": 10}, frozen, current_hash)
        with torch.no_grad():
            m.vjepa_backend.quiet_pursuit_head.rank.add_(0.01)
        self.assertFalse(compatible_policy_refresh(self.nav_model.config, m, self.pair))
        with self.assertRaises(ValueError):
            verify_frozen_resume(m, self.pair, frozen, _hash_tensors(frozen))
