"""Moved files retain the authored VAL/preview pins; I/O uses the native root."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from turkeyworld import workspace_paths
from turkeyworld.rssm_motion_only_validation import bind_motion_validation_plan,verify_motion_validation_plan
from turkeyworld.rssm_preview_assets import read_preview_assets_plan,file_sha256,resolve_preview_assets_plan
import test_rssm_motion_only_validation as validation_helpers


CONFIG={'protocol':'workspace_paths_v1','version':1,'origin_root':r'E:\nau\TurkeyWorld'}


class PortableMotionDataTests(unittest.TestCase):
    def test_fixed_val_bytes_panel_and_mtimes_survive_workspace_move(self):
        fixture=validation_helpers.MotionOnlyValidationTests('runTest');fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        (fixture.root/'workspace_portability.json').write_text(json.dumps(CONFIG),encoding='utf-8')
        with patch.object(workspace_paths,'ROOT',fixture.root):
            plan=fixture.bind();raw=fixture.path.read_bytes()
            self.assertTrue(plan['plan_path'].startswith(CONFIG['origin_root']))
        with TemporaryDirectory()as temporary:
            moved=Path(temporary)/'Ubuntu workspace';shutil.copytree(fixture.root,moved,copy_function=shutil.copy2)
            with patch.object(workspace_paths,'ROOT',moved):
                self.assertTrue(verify_motion_validation_plan(plan))
                rebuilt=bind_motion_validation_plan(moved/'navigation/val',moved/'quiet/val',moved/fixture.path.name)
                self.assertEqual(rebuilt,plan)
                self.assertEqual((moved/fixture.path.name).read_bytes(),raw)
                # A changed file still fails rather than silently rewriting the pin.
                altered=moved/'navigation/val/episode_000.npz'
                altered.write_bytes(altered.read_bytes()+b'corruption')
                with self.assertRaises(ValueError):verify_motion_validation_plan(plan)
                self.assertEqual((moved/fixture.path.name).read_bytes(),raw)

    def test_recursive_preview_pins_keep_windows_identity_on_moved_root(self):
        with TemporaryDirectory()as first,TemporaryDirectory()as second:
            origin=Path(first);moved=Path(second)
            (origin/'workspace_portability.json').write_text(json.dumps(CONFIG),encoding='utf-8')
            with patch.object(workspace_paths,'ROOT',origin):
                assets={}
                for name in('renderer','sr'):
                    path=origin/(name+'.pt');path.write_bytes(name.encode())
                    assets[name]={'checkpoint':workspace_paths.identity_path(path),'file_sha256':file_sha256(path)}
                initial=origin/'initial.json';initial.write_text(json.dumps(dict(version=1,kind='frozen_preview_assets_v1',**assets)))
                selected=origin/'selected.json';selected.write_bytes(initial.read_bytes())
                revision=origin/'revision.json';revision.write_text(json.dumps(dict(version=1,kind='preview_assets_revision_v1',
                    initial_plan_sha256=file_sha256(initial),selected_plan=workspace_paths.identity_path(selected),
                    selected_plan_sha256=file_sha256(selected))))
                before=resolve_preview_assets_plan(initial,revision)
                raw=deepcopy(before)
            shutil.copytree(origin,moved,dirs_exist_ok=True)
            with patch.object(workspace_paths,'ROOT',moved):
                after=resolve_preview_assets_plan(moved/'initial.json',moved/'revision.json',before[2])
                self.assertEqual(after,before);self.assertEqual(before,raw)
                self.assertEqual(read_preview_assets_plan(before[1]['plan_path']),before[1])


if __name__=='__main__':unittest.main()
