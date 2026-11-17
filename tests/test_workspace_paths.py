from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from turkeyworld.workspace_paths import WorkspacePaths,resolve_path,identity_path,load_workspace_paths


class WorkspacePathsTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'moved'
        self.root.mkdir();self.origin=r'E:\nau\TurkeyWorld';self.paths=WorkspacePaths(self.root,self.origin)
        (self.root/'training_runs').mkdir();self.file=self.root/'training_runs'/'weights.pt';self.file.write_bytes(b'unchanged tensors')
        self.old=self.origin+r'\training_runs\weights.pt'

    def test_explicit_storage_root_only_allows_declared_external_subtree(self):
        storage=Path(self.temp.name)/'storage';storage.mkdir()
        (self.root/'mounted').symlink_to(storage,target_is_directory=True)
        with self.assertRaises(ValueError):self.paths.resolve('mounted/file')
        explicit=WorkspacePaths(self.root,self.origin,[storage])
        self.assertEqual(explicit.resolve('mounted/file'),storage/'file')
        other=Path(self.temp.name)/'other';other.mkdir()
        (self.root/'undeclared').symlink_to(other,target_is_directory=True)
        with self.assertRaises(ValueError):explicit.resolve('undeclared/file')

    def configure(self):
        (self.root/'workspace_portability.json').write_text(json.dumps(dict(protocol='workspace_paths_v1',origin_root=self.origin)),encoding='utf-8')

    def test_old_absolute_relative_and_native_paths_resolve_to_same_native_file(self):
        for path in(self.old,'training_runs/weights.pt',r'training_runs\weights.pt',self.file):
            self.assertEqual(self.paths.resolve(path,must_exist=True),self.file)
            self.assertEqual(self.paths.relative(path),'training_runs/weights.pt')

    def test_old_identity_is_kept_exact_while_native_identity_maps_to_original_windows_root(self):
        original='e:/NAU/TurkeyWorld/training_runs/weights.pt'
        self.assertEqual(self.paths.identity_path(original),original)
        self.assertEqual(self.paths.identity_path(self.file),self.old)
        self.assertEqual(self.paths.identity_path('training_runs/weights.pt'),self.old)

    def test_strict_resolver_refuses_foreign_root_drive_unc_and_parent_traversal(self):
        for bad in(r'C:\other\a.pt',r'E:relative.pt',r'\\server\share\a.pt',r'\rooted.pt',
                '../outside',r'training_runs\..\outside',self.origin+r'X\wrong.pt',self.root.parent/'outside'):
            with self.subTest(path=bad),self.assertRaises(ValueError):self.paths.resolve(bad)

    def test_receipt_and_json_verification_do_not_rewrite_old_paths_or_bytes(self):
        body=dict(path=self.old,raw_history={'0':{'same':True}});before=deepcopy(body)
        path=self.root/'plan.json';path.write_text(json.dumps(body),encoding='utf-8');raw=path.read_bytes()
        sha=hashlib.sha256(raw).hexdigest()
        self.assertEqual(self.paths.read_json('plan.json',sha256=sha),body)
        self.assertEqual(body,before);self.assertEqual(path.read_bytes(),raw)
        self.assertEqual(self.paths.verify_file('plan.json',sha,size=len(raw)),
            dict(relative_path='plan.json',sha256=sha,bytes=len(raw)))
        with self.assertRaises(ValueError):self.paths.read_json('plan.json',sha256='0'*64)
        with self.assertRaises(ValueError):self.paths.verify_file('plan.json',sha,size=len(raw)+1)

    def test_optional_configuration_maps_io_and_identity_but_allows_native_external_files(self):
        self.configure();outside=self.root.parent/'external.pt';outside.write_bytes(b'caller file')
        self.assertEqual(resolve_path(self.old,root=self.root),self.file)
        self.assertEqual(identity_path(self.file,root=self.root),self.old)
        self.assertEqual(identity_path(self.old,root=self.root),self.old)
        self.assertEqual(resolve_path(outside,root=self.root),outside)
        self.assertEqual(identity_path(outside,root=self.root),str(outside))
        with self.assertRaises(ValueError):load_workspace_paths(self.root).resolve(outside)

    def test_without_config_native_normal_run_is_unchanged(self):
        self.assertIsNone(load_workspace_paths(self.root))
        self.assertEqual(resolve_path(self.file,root=self.root),self.file)
        self.assertEqual(identity_path(self.file,root=self.root),str(self.file))

    def test_physical_root_is_not_part_of_stable_identity_or_content_receipt(self):
        other=self.root.parent/'another_mount';(other/'training_runs').mkdir(parents=True)
        counterpart=other/'training_runs/weights.pt';counterpart.write_bytes(self.file.read_bytes())
        relocated=WorkspacePaths(other,self.origin);sha=hashlib.sha256(self.file.read_bytes()).hexdigest()
        self.assertEqual(self.paths.verify_file(self.old,sha),relocated.verify_file(self.old,sha))
        self.assertEqual(self.paths.identity_path(self.file),relocated.identity_path(counterpart))

    def test_symlink_escape_is_refused(self):
        outside=self.root.parent/'outside';outside.mkdir();link=self.root/'escape'
        try:link.symlink_to(outside,target_is_directory=True)
        except OSError as error:self.skipTest(f'Symlink creation unavailable: {error}')
        for call in(self.paths.resolve,self.paths.relative,self.paths.identity_path):
            with self.assertRaises(ValueError):call('escape/secret.pt')

    def test_bad_configuration_and_missing_files_fail_loudly(self):
        config=self.root/'workspace_portability.json';config.write_text('{"protocol":"other","origin_root":"E:/nau/TurkeyWorld"}')
        with self.assertRaises(ValueError):resolve_path(self.old,root=self.root)
        with self.assertRaises(FileNotFoundError):self.paths.resolve('missing.pt',must_exist=True)


if __name__=='__main__':unittest.main()
