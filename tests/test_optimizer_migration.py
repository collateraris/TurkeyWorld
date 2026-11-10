"""Tiny real AdamW migration when newly trainable Flow parameters precede core."""
import copy
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class OptimizerMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        class World(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.flow_renderer = torch.nn.Linear(3, 3)
                self.core = torch.nn.Linear(3, 2)
                self.vjepa_backend = torch.nn.Module()
                self.vjepa_backend.renderer = torch.nn.Linear(2, 2)
                self.vjepa_backend.renderer.requires_grad_(False)

            def forward(self, x):
                return self.core(self.flow_renderer(x))

        torch.manual_seed(79)
        old = World()
        old.flow_renderer.requires_grad_(False)
        optimizer = torch.optim.AdamW([p for p in old.parameters() if p.requires_grad],
            lr=.017, betas=(.8, .95), eps=1e-7, weight_decay=.03, amsgrad=True, foreach=False)
        x = torch.tensor([[1., -.5, .25], [.2, .7, -1.]])
        old(x).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        checkpoint = {"optimizer": copy.deepcopy(optimizer.state_dict()),
            "model": copy.deepcopy(old.state_dict()), "pipeline_phase": "teacher500vq_world"}
        new = World()
        new.load_state_dict(old.state_dict())
        current = torch.optim.AdamW([p for p in new.parameters() if p.requires_grad], lr=.9)
        return old, optimizer, new, current, checkpoint, x

    def assert_state_equal(self, left, right):
        self.assertEqual(left["param_groups"], right["param_groups"])
        self.assertEqual(set(left["state"]), set(right["state"]))
        for identifier, entry in left["state"].items():
            self.assertEqual(set(entry), set(right["state"][identifier]))
            for key, value in entry.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, right["state"][identifier][key], rtol=0, atol=0)
                else:
                    self.assertEqual(value, right["state"][identifier][key])

    def test_legacy_core_moments_map_to_correct_new_ids_and_new_flow_starts_fresh(self):
        from turkeyworld.optimizer_migration import optimizer_parameter_names, optimizer_state_for_trainables
        old, old_optimizer, new, optimizer, checkpoint, x = self.fixture()
        rng = torch.get_rng_state().clone()
        migrated, report = optimizer_state_for_trainables(checkpoint, new, optimizer)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(report["strategy"], "legacy_world_pretrain_order")
        self.assertEqual(report["preserved_parameters"], 2)
        self.assertEqual(report["added_parameters"], ["flow_renderer.weight", "flow_renderer.bias"])
        self.assertEqual(set(migrated["state"]), {2, 3})
        for old_id, new_id in ((0, 2), (1, 3)):
            for key, value in checkpoint["optimizer"]["state"][old_id].items():
                torch.testing.assert_close(migrated["state"][new_id][key], value, rtol=0, atol=0)
        optimizer.load_state_dict(migrated)
        for key, value in old_optimizer.param_groups[0].items():
            if key != "params":
                self.assertEqual(optimizer.param_groups[0][key], value)
        self.assertEqual(optimizer.param_groups[0]["lr"], .017)
        self.assertEqual(optimizer.param_groups[0]["betas"], (.8, .95))
        self.assertNotIn(new.flow_renderer.weight, optimizer.state)
        self.assertNotIn(new.flow_renderer.bias, optimizer.state)
        # Exact old core update proves moments stayed attached by name despite
        # two newly inserted positional IDs at the beginning of the optimizer.
        old(x).square().mean().backward()
        new(x).square().mean().backward()
        old_optimizer.step()
        optimizer.step()
        torch.testing.assert_close(old.core.weight, new.core.weight, rtol=0, atol=0)
        torch.testing.assert_close(old.core.bias, new.core.bias, rtol=0, atol=0)
        self.assertEqual(float(optimizer.state[new.core.weight]["step"]), 2.)
        self.assertEqual(float(optimizer.state[new.flow_renderer.weight]["step"]), 1.)
        self.assertEqual(optimizer_parameter_names(new, optimizer),
            [["flow_renderer.weight", "flow_renderer.bias", "core.weight", "core.bias"]])

    def test_explicit_names_save_resume_and_identical_order_keep_actual_adamw_continuation(self):
        from turkeyworld.optimizer_migration import optimizer_parameter_names, optimizer_state_for_trainables
        _, _, model, optimizer, checkpoint, x = self.fixture()
        mapped, _ = optimizer_state_for_trainables(checkpoint, model, optimizer)
        optimizer.load_state_dict(mapped)
        model(x).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        resumed_checkpoint = {"model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()), "pipeline_phase": "teacher_quality_500m",
            "optimizer_parameter_names": optimizer_parameter_names(model, optimizer)}
        restored = copy.deepcopy(model)
        restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.5)
        mapped, report = optimizer_state_for_trainables(resumed_checkpoint, restored, restored_optimizer)
        self.assertEqual(report["strategy"], "parameter_names")
        self.assertTrue(report["parameter_order_unchanged"])
        self.assertEqual(report["added_parameters"], [])
        self.assert_state_equal(mapped, resumed_checkpoint["optimizer"])
        restored_optimizer.load_state_dict(mapped)
        model(x).square().mean().backward()
        restored(x).square().mean().backward()
        optimizer.step()
        restored_optimizer.step()
        for actual, expected in zip(restored.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_state_equal(restored_optimizer.state_dict(), optimizer.state_dict())

    def test_explicit_names_also_support_reordered_parameters_without_losing_options(self):
        from turkeyworld.optimizer_migration import optimizer_state_for_trainables
        _, _, model, _, checkpoint, _ = self.fixture()
        checkpoint["optimizer_parameter_names"] = [["core.weight", "core.bias"]]
        checkpoint["pipeline_phase"] = "student_distillation"
        checkpoint["optimizer"]["param_groups"][0]["param_names"] = ["core.weight", "core.bias"]
        optimizer = torch.optim.AdamW([model.core.bias, model.flow_renderer.bias,
                                      model.core.weight, model.flow_renderer.weight], lr=.3)
        mapped, _ = optimizer_state_for_trainables(checkpoint, model, optimizer)
        self.assertEqual(set(mapped["state"]), {0, 2})
        self.assertEqual(mapped["param_groups"][0]["param_names"],
                         ["core.bias", "flow_renderer.bias", "core.weight", "flow_renderer.weight"])
        torch.testing.assert_close(mapped["state"][0]["exp_avg"],
                                   checkpoint["optimizer"]["state"][1]["exp_avg"], rtol=0, atol=0)

    def test_full_adamw_keeps_frozen_weights_moments_and_exact_unfrozen_resume(self):
        from turkeyworld.optimizer_migration import optimizer_parameter_names, optimizer_state_for_trainables
        _, _, model, optimizer, checkpoint, x = self.fixture()
        mapped, _ = optimizer_state_for_trainables(checkpoint, model, optimizer)
        optimizer.load_state_dict(mapped)
        # Initialize both components' actual AdamW state before the phase.
        model(x).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        original_names = optimizer_parameter_names(model, optimizer)
        frozen_weights = copy.deepcopy(model.flow_renderer.state_dict())
        frozen_moments = {name: copy.deepcopy(optimizer.state[parameter])
                          for name, parameter in model.flow_renderer.named_parameters()}
        model.flow_renderer.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'frozen parameter'):
            optimizer_parameter_names(model, optimizer)
        self.assertEqual(optimizer_parameter_names(model, optimizer, allow_frozen=True), original_names)
        # The same full optimizer updates core while inactive Flow grad=None.
        model(x).square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.flow_renderer.parameters()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in model.flow_renderer.named_parameters():
            torch.testing.assert_close(parameter, frozen_weights[name], rtol=0, atol=0)
            for key, value in frozen_moments[name].items():
                torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)
        saved = {'model': copy.deepcopy(model.state_dict()),
                 'optimizer': copy.deepcopy(optimizer.state_dict()),
                 'optimizer_parameter_names': optimizer_parameter_names(model, optimizer, allow_frozen=True),
                 'pipeline_phase': 'teacher175vq_world'}
        restored = copy.deepcopy(model)
        # Restore the full optimizer before configuring the next active phase.
        restored.flow_renderer.requires_grad_(True)
        resumed_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.9)
        rng = torch.get_rng_state().clone()
        state, report = optimizer_state_for_trainables(saved, restored, resumed_optimizer)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertTrue(report['parameter_order_unchanged'])
        self.assertEqual(report['added_parameters'], [])
        self.assert_state_equal(state, saved['optimizer'])
        resumed_optimizer.load_state_dict(state)
        model.flow_renderer.requires_grad_(True)
        model(x).square().mean().backward()
        restored(x).square().mean().backward()
        optimizer.step()
        resumed_optimizer.step()
        for actual, expected in zip(restored.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_state_equal(resumed_optimizer.state_dict(), optimizer.state_dict())

    def test_allow_frozen_still_rejects_foreign_and_duplicate_parameters(self):
        from turkeyworld.optimizer_migration import optimizer_parameter_names
        _, _, model, optimizer, _, _ = self.fixture()
        model.core.requires_grad_(False)
        self.assertEqual(len(optimizer_parameter_names(model, optimizer, allow_frozen=True)[0]), 4)
        group = optimizer.param_groups[0]['params']
        group.append(group[0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            optimizer_parameter_names(model, optimizer, allow_frozen=True)
        group.pop()
        group.append(torch.nn.Parameter(torch.ones(1), requires_grad=False))
        with self.assertRaisesRegex(ValueError, 'outside the model'):
            optimizer_parameter_names(model, optimizer, allow_frozen=True)

    def test_legacy_fallback_is_restricted_to_known_pretraining_phase_and_exact_order_size(self):
        from turkeyworld.optimizer_migration import optimizer_state_for_trainables
        _, _, model, optimizer, checkpoint, _ = self.fixture()
        for phase in ("teacher500vq_world", "teacher500_world", "teacher250_world_003"):
            with self.subTest(phase=phase):
                checkpoint["pipeline_phase"] = phase
                _, report = optimizer_state_for_trainables(checkpoint, model, optimizer)
                self.assertEqual(report["strategy"], "legacy_world_pretrain_order")
        for phase in ("student_world", "teacher_quality_500m", "teacher500vq_world_extra", ""):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                optimizer_state_for_trainables(dict(checkpoint, pipeline_phase=phase), model, optimizer)

    def test_corrupted_names_ids_shapes_and_unsupported_groups_reject(self):
        from turkeyworld.optimizer_migration import optimizer_state_for_trainables
        _, _, model, optimizer, original, _ = self.fixture()
        for problem in ("duplicate_names", "missing_name", "name_count", "duplicate_ids", "unknown_state_id",
                        "model_shape", "moment_shape", "step_shape", "group_count", "conflicting_names"):
            with self.subTest(problem=problem):
                checkpoint = copy.deepcopy(original)
                checkpoint["optimizer_parameter_names"] = [["core.weight", "core.bias"]]
                if problem == "duplicate_names":checkpoint["optimizer_parameter_names"] = [["core.weight"] * 2]
                elif problem == "missing_name":checkpoint["optimizer_parameter_names"][0][0] = "removed.weight"
                elif problem == "name_count":checkpoint["optimizer_parameter_names"][0].pop()
                elif problem == "duplicate_ids":checkpoint["optimizer"]["param_groups"][0]["params"] = [0, 0]
                elif problem == "unknown_state_id":checkpoint["optimizer"]["state"][999] = {}
                elif problem == "model_shape":checkpoint["model"]["core.weight"] = torch.zeros(1)
                elif problem == "moment_shape":checkpoint["optimizer"]["state"][0]["exp_avg"] = torch.zeros(9)
                elif problem == "step_shape":checkpoint["optimizer"]["state"][0]["step"] = torch.zeros(2)
                elif problem == "group_count":checkpoint["optimizer"]["param_groups"].append(copy.deepcopy(checkpoint["optimizer"]["param_groups"][0]))
                else:checkpoint["optimizer"]["param_groups"][0]["param_names"] = ["core.bias", "core.weight"]
                with self.assertRaises(ValueError):
                    optimizer_state_for_trainables(checkpoint, model, optimizer)

    def test_removed_trainable_and_foreign_or_multiple_current_groups_reject(self):
        from turkeyworld.optimizer_migration import optimizer_state_for_trainables
        _, _, model, _, checkpoint, _ = self.fixture()
        choices = [torch.optim.AdamW([model.core.bias, model.flow_renderer.weight], lr=.1),
            torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], lr=.1),
            torch.optim.AdamW([{"params": list(model.core.parameters())},
                               {"params": list(model.flow_renderer.parameters())}], lr=.1)]
        for optimizer in choices:
            with self.subTest(groups=len(optimizer.param_groups)), self.assertRaises(ValueError):
                optimizer_state_for_trainables(checkpoint, model, optimizer)


if __name__ == "__main__":
    unittest.main()
