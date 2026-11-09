"""Thread-scoped sleep protection keeps the display/lock policy unchanged."""
import unittest
from unittest.mock import Mock, call, patch


class KeepAwakeTests(unittest.TestCase):
    def test_requests_system_only_and_restores_the_exact_previous_requirement(self):
        from turkeyworld.power_guard import KeepAwake
        continuous, system, display, away = 0x80000000, 0x1, 0x2, 0x40
        api = Mock(side_effect=[continuous, continuous | system])
        with patch("subprocess.run", side_effect=AssertionError("No system power setting commands")), \
             patch("os.system", side_effect=AssertionError("No shell commands")):
            with KeepAwake(api=api) as guard:
                self.assertTrue(guard.active)
                self.assertEqual(guard.previous_state, continuous)
                self.assertFalse(guard.error)
                requested = api.call_args.args[0]
                self.assertEqual(requested, continuous | system)
                self.assertFalse(requested & display)
                self.assertFalse(requested & away)
            self.assertFalse(guard.active)
        self.assertEqual(api.call_args_list, [call(continuous | system), call(continuous)])

    def test_restoration_preserves_a_prior_request_and_adds_continuous(self):
        from turkeyworld.power_guard import KeepAwake
        # This synthetic prior value deliberately lacks CONTINUOUS, verifying
        # explicit restoration rather than a hard-coded release request.
        api = Mock(side_effect=[0x1, 0x80000001])
        with KeepAwake(api=api) as guard:
            self.assertEqual(guard.previous_state, 0x1)
        self.assertEqual(api.call_args_list, [call(0x80000001), call(0x80000001)])

    def test_body_exception_is_propagated_after_releasing_sleep_protection(self):
        from turkeyworld.power_guard import KeepAwake
        api = Mock(side_effect=[0x80000000, 0x80000001])
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            with KeepAwake(api=api) as guard:
                raise RuntimeError("training failed")
        self.assertFalse(guard.active)
        self.assertEqual(api.call_args_list, [call(0x80000001), call(0x80000000)])

    def test_failed_initial_request_is_reported_without_releasing_an_unowned_request(self):
        from turkeyworld.power_guard import KeepAwake
        api = Mock(return_value=0)
        with KeepAwake(api=api) as guard:
            self.assertFalse(guard.active)
            self.assertTrue(guard.error)
        self.assertEqual(api.call_args_list, [call(0x80000001)])

    def test_request_and_release_api_errors_do_not_replace_training_exceptions(self):
        from turkeyworld.power_guard import KeepAwake
        failing_request = Mock(side_effect=OSError("request unavailable"))
        with KeepAwake(api=failing_request) as guard:
            self.assertFalse(guard.active)
            self.assertTrue(guard.error)
        failing_release = Mock(side_effect=[0x80000000, OSError("release unavailable")])
        with self.assertRaisesRegex(ValueError, "original training exception"):
            with KeepAwake(api=failing_release) as guard:
                raise ValueError("original training exception")
        self.assertTrue(guard.error)
        self.assertEqual(failing_release.call_count, 2)

    def test_zero_return_on_release_is_reported_without_raising(self):
        from turkeyworld.power_guard import KeepAwake
        api = Mock(side_effect=[0x80000000, 0])
        with KeepAwake(api=api) as guard:
            self.assertTrue(guard.active)
        self.assertTrue(guard.error)
        self.assertEqual(api.call_count, 2)

    def test_non_windows_default_is_a_noop_without_loading_windows_libraries(self):
        from turkeyworld.power_guard import KeepAwake
        with patch("sys.platform", "linux"), \
             patch("ctypes.WinDLL", create=True, side_effect=AssertionError("Windows-only library load")) as library:
            with KeepAwake() as guard:
                self.assertFalse(guard.active)
                self.assertFalse(guard.error)
            library.assert_not_called()


if __name__ == "__main__":
    unittest.main()
