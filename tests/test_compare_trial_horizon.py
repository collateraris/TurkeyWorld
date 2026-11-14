"""Trial defaults and explicit diagnostic overrides without loading models or GUI."""
from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

from turkeyworld.compare_play import arguments


ROOT=Path(__file__).resolve().parents[1]


class CompareTrialHorizonTests(unittest.TestCase):
    def test_cli_defaults_to180_and_preserves_explicit_positive_trial_lengths(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            world,sr=root/'world.pt',root/'sr.pt'
            world.touch();sr.touch()
            base=['--world',str(world),'--sr',str(sr),'--smooth','--display-size','1024']
            for extra,expected in (([],180),(['--steps','450'],450),(['--steps','48'],48),(['--steps','1'],1)):
                with self.subTest(extra=extra):
                    gui,neural=arguments(base+extra)
                    self.assertEqual((gui.steps,neural.steps),(expected,expected))
                    self.assertEqual(neural.warmup,0)
                    self.assertTrue(neural.initial_navigation_only)
            for value in ('0','-1'):
                with self.subTest(value=value),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    arguments(base+['--steps',value])

    def test_current_launcher_has_same_default_and_explicit_legacy_override(self):
        path=ROOT/'training_runs/diagnostic_tools/launch_current_neural_game.py'
        spec=importlib.util.spec_from_file_location('current_game_trial_options',path)
        launcher=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        self.assertEqual(launcher.arguments([]).steps,180)
        self.assertEqual(launcher.arguments(['--steps','450']).steps,450)
        self.assertEqual(launcher.arguments(['--steps','1']).steps,1)
        for value in ('0','-1'):
            with self.subTest(value=value),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                launcher.arguments(['--steps',value])


if __name__=='__main__':unittest.main()
