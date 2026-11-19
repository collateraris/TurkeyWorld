import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from turkeyworld.run_vjepa_sr_cycle import cycle_plan, run
from turkeyworld.vjepa_training_profiles import THREE_MODE, THREE_MODE_ACTORS


class CycleMigrationTest(unittest.TestCase):
    def test_explicit_upgrade_and_inherit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = cycle_plan(root)
            mixed = cycle_plan(root, THREE_MODE)
            self.assertEqual(json.loads((root / 'cycle_plan_before_three_mode.json').read_text()), legacy)
            self.assertEqual(mixed['world_actors_by_stream'], THREE_MODE_ACTORS)
            self.assertEqual(sum(mixed['world_actors_by_stream'].values()), mixed['main_actors'])
            self.assertEqual(mixed['world_updates_per_round'], legacy['world_updates_per_round'])
            self.assertEqual(cycle_plan(root), mixed)
            self.assertEqual(mixed['order'][1], 'world_validation')

    def test_preserve_unknown_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'cycle_plan.json'
            path.write_text('{"external": true}')
            with self.assertRaises(ValueError):
                cycle_plan(root, THREE_MODE)
            self.assertEqual(json.loads(path.read_text()), {'external': True})

    def test_next_round_collects_combat_and_validates_before_policy(self):
        class NextRound(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dynamics').mkdir()
            modules = []
            def launch(command, **kwargs):
                module = command[2].split('.')[-1]
                if len(modules) == 6:
                    self.assertEqual(module, 'collect_targeted_motion')
                    raise NextRound
                modules.append(module)
                if module == 'vjepa_combat_replay':
                    self.assertEqual(command[command.index('--cycle-index')+1], '0')
                elif module == 'train_vjepa_sr_dynamics':
                    self.assertEqual(command[command.index('--training-profile')+1], THREE_MODE)
                    self.assertEqual(command[command.index('--until-updates')+1], '100')
                    (root/'dynamics/status.json').write_text(json.dumps(
                        {'world_updates': 100, 'training_plan': {'training_profile': THREE_MODE}}))
                elif module == 'validate_vjepa_combat':
                    Path(command[command.index('--output')+1]).write_text('{}')
                    self.assertEqual(command[command.index('--stop-file')+1], str(root/'stop.request'))
                elif module == 'train_dreamer_policy':
                    self.assertEqual(command[command.index('--until-actor-updates')+1], '50')
                return SimpleNamespace(pid=1, returncode=0, wait=lambda: 0)

            prefix = 'turkeyworld.run_vjepa_sr_cycle.'
            with patch(prefix+'subprocess.Popen', side_effect=launch), patch(prefix+'process_identity', return_value={}), \
                 patch(prefix+'process_identity_alive', return_value=False), self.assertRaises(NextRound):
                run(SimpleNamespace(run_dir=root, training_profile=THREE_MODE))
            self.assertEqual(modules, ['collect_targeted_motion','vjepa_combat_replay','train_vjepa_sr_dynamics',
                                      'validate_vjepa_combat','train_dreamer_policy','collect_policy_motion'])
            status=json.loads((root/'status.json').read_text())
            self.assertEqual(status['round_index'],1)
            self.assertEqual(status['world_actors_by_stream'],THREE_MODE_ACTORS)
