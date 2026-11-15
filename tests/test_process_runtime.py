"""Host-bound PID handoffs never adopt a foreign numeric PID after relocation."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from turkeyworld import process_runtime as runtime

HOST=dict(platform='Linux',hostname='ubuntu-workstation',boot_id='boot1')


class HostProcessTests(unittest.TestCase):
    def test_foreign_and_unreceipted_pids_require_explicit_migration_without_os_probe(self):
        foreign=dict(protocol=runtime.PROTOCOL,pid=123,host=dict(platform='Windows',hostname='old-pc',boot_id=None),start_token=None)
        with patch.object(runtime,'host_identity',return_value=HOST),\
                patch.object(runtime,'process_alive',side_effect=AssertionError('Foreign PID must not be inspected')):
            for record in(None,foreign):
                with self.subTest(record=record),self.assertRaisesRegex(RuntimeError,'discard-foreign-pids'):
                    runtime.process_identity_alive(record)
                self.assertFalse(runtime.process_identity_alive(record,discard_foreign=True))

    def test_matching_host_checks_boot_and_process_birth_before_adopting_live_pid(self):
        record=dict(protocol=runtime.PROTOCOL,pid=123,host=HOST,start_token='start1')
        with patch.object(runtime,'host_identity',return_value=HOST),\
                patch.object(runtime,'_start_token',return_value='start1'),\
                patch.object(runtime,'process_alive',return_value=True)as alive:
            self.assertTrue(runtime.process_identity_alive(record));alive.assert_called_once_with(123)
        for change in('boot','start'):
            old=deepcopy(record)
            if change=='boot':old['host']['boot_id']='previous-boot'
            else:old['start_token']='previous-worker'
            with self.subTest(change=change),patch.object(runtime,'host_identity',return_value=HOST),\
                    patch.object(runtime,'_start_token',return_value='start1'),\
                    patch.object(runtime,'process_alive',side_effect=AssertionError('Stale PID must not be adopted')):
                self.assertFalse(runtime.process_identity_alive(old))

    def test_pid_text_sidecar_write_read_and_legacy_migration_do_not_mutate_old_receipts(self):
        with TemporaryDirectory()as directory,patch.object(runtime,'host_identity',return_value=HOST),\
                patch.object(runtime,'_start_token',return_value='start1'),\
                patch.object(runtime,'process_alive',return_value=True):
            path=Path(directory)/'training.pid';path.write_text('123')
            with self.assertRaisesRegex(RuntimeError,'host receipt'):runtime.pid_file_alive(path)
            self.assertFalse(runtime.pid_file_alive(path,discard_foreign=True))
            self.assertEqual(path.read_text(),'123');self.assertFalse(runtime.pid_receipt_path(path).exists())
            record=runtime.write_pid_file(path,123)
            archived=list((path.parent/'process_history').glob('*/training.pid'))
            self.assertEqual(len(archived),1);self.assertEqual(archived[0].read_bytes(),b'123')
            self.assertEqual(int(path.read_text()),123)
            self.assertEqual(json.loads(runtime.pid_receipt_path(path).read_text()),record)
            self.assertTrue(runtime.pid_file_alive(path))
            path.write_text('124')
            with self.assertRaisesRegex(ValueError,'differ'):runtime.pid_file_alive(path)

    def test_windows_creation_timestamp_is_captured_and_handle_released(self):
        def times(handle,creation,*others):creation._obj.value=987654321;return 1
        kernel=SimpleNamespace(OpenProcess=Mock(return_value=99),GetProcessTimes=Mock(side_effect=times),CloseHandle=Mock())
        with patch.object(runtime.platform,'system',return_value='Windows'),patch.object(runtime.os,'name','nt'),\
                patch.object(runtime.ctypes,'WinDLL',return_value=kernel,create=True):
            self.assertEqual(runtime._start_token(123),'987654321')
        kernel.CloseHandle.assert_called_once_with(99)

    def test_unreadable_birth_of_live_pid_fails_conservatively(self):
        record=dict(protocol=runtime.PROTOCOL,pid=123,host=HOST,start_token='start1')
        with patch.object(runtime,'host_identity',return_value=HOST),patch.object(runtime,'_start_token',return_value=None),\
                patch.object(runtime,'process_alive',return_value=True),self.assertRaisesRegex(RuntimeError,'refusing'):
            runtime.process_identity_alive(record)


if __name__=='__main__':unittest.main()
