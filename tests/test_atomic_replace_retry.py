import ctypes
from ctypes import wintypes
import errno
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from turkeyworld import training_pipeline as pipeline


def windows_error(code):
    error = PermissionError(errno.EACCES, 'Windows reader/share lock')
    error.winerror = code
    return error


class Clock:
    def __init__(self): self.now = 0.; self.sleeps = []
    def monotonic(self): return self.now
    def sleep(self, delay):
        self.sleeps.append(delay); self.now += delay


class AtomicReplaceRetryTests(unittest.TestCase):
    def test_reader_lock_longer_than_old_two_second_budget_eventually_replaces_atomically(self):
        clock = Clock(); actual_replace = os.replace
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'new.tmp', Path(folder)/'status.json'
            source.write_bytes(b'new complete payload'); target.write_bytes(b'old complete payload')
            def replace(old, new):
                if clock.now < 3.:
                    self.assertEqual(target.read_bytes(), b'old complete payload')
                    self.assertEqual(source.read_bytes(), b'new complete payload')
                    raise windows_error(5)
                actual_replace(old, new)
            with patch.object(pipeline.sys, 'platform', 'win32'), \
                    patch.object(pipeline.os, 'replace', side_effect=replace), \
                    patch.object(pipeline.time, 'monotonic', side_effect=clock.monotonic), \
                    patch.object(pipeline.time, 'sleep', side_effect=clock.sleep):
                pipeline.replace_file(source, target)
            self.assertGreaterEqual(clock.now, 3.)
            self.assertEqual(clock.sleeps[:4], [.05, .1, .2, .4])
            self.assertLessEqual(max(clock.sleeps), .5)
            self.assertEqual(target.read_bytes(), b'new complete payload')
            self.assertFalse(source.exists())

    def test_permanent_windows_lock_has_bounded_deadline_preserves_both_files_and_raises_same_error(self):
        clock = Clock(); blocked = windows_error(32)
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'checkpoint.tmp', Path(folder)/'checkpoint.pt'
            source.write_bytes(b'new checkpoint'); target.write_bytes(b'last durable checkpoint')
            with patch.object(pipeline.sys, 'platform', 'win32'), \
                    patch.object(pipeline.os, 'replace', side_effect=blocked) as replace, \
                    patch.object(pipeline.time, 'monotonic', side_effect=clock.monotonic), \
                    patch.object(pipeline.time, 'sleep', side_effect=clock.sleep):
                with self.assertRaises(PermissionError) as caught: pipeline.replace_file(source, target)
            self.assertIs(caught.exception, blocked)
            self.assertLessEqual(replace.call_count, 64)
            self.assertAlmostEqual(clock.now, 25.)
            self.assertLessEqual(max(clock.sleeps), .5)
            self.assertEqual(source.read_bytes(), b'new checkpoint')
            self.assertEqual(target.read_bytes(), b'last durable checkpoint')

    def test_other_windows_transient_codes_retry_but_unknown_permissions_and_non_windows_fail_immediately(self):
        for code in (33, 1224):
            with self.subTest(code=code), patch.object(pipeline.sys, 'platform', 'win32'), \
                    patch.object(pipeline.os, 'replace', side_effect=[windows_error(code), None]) as replace, \
                    patch.object(pipeline.time, 'sleep') as sleep:
                pipeline.replace_file('new.tmp', 'old.json')
                self.assertEqual(replace.call_count, 2); sleep.assert_called_once_with(.05)
        for platform, error in (('linux', windows_error(5)), ('win32', windows_error(1314)),
                                ('win32', PermissionError(errno.EACCES, 'permissions')),
                                ('win32', FileNotFoundError(errno.ENOENT, 'missing source'))):
            with self.subTest(platform=platform, error=error), patch.object(pipeline.sys, 'platform', platform), \
                    patch.object(pipeline.os, 'replace', side_effect=error) as replace, \
                    patch.object(pipeline.time, 'sleep') as sleep:
                with self.assertRaises(type(error)) as caught: pipeline.replace_file('new.tmp', 'old.json')
                self.assertIs(caught.exception, error)
                self.assertEqual(replace.call_count, 1); sleep.assert_not_called()

    def test_atomic_json_keeps_previous_valid_report_until_transient_reader_releases(self):
        clock = Clock(); actual_replace = os.replace
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'status.json'; target.write_text('{"step":4982}', encoding='utf-8')
            def replace(source, destination):
                if clock.now < 2.5:
                    self.assertEqual(json.loads(target.read_text())['step'], 4982)
                    self.assertEqual(json.loads(Path(source).read_text())['step'], 4983)
                    raise windows_error(5)
                actual_replace(source, destination)
            with patch.object(pipeline.sys, 'platform', 'win32'), \
                    patch.object(pipeline.os, 'replace', side_effect=replace), \
                    patch.object(pipeline.time, 'monotonic', side_effect=clock.monotonic), \
                    patch.object(pipeline.time, 'sleep', side_effect=clock.sleep):
                pipeline.atomic_json(target, {'step': 4983, 'loss': 2.5})
            self.assertEqual(json.loads(target.read_text()), {'step': 4983, 'loss': 2.5})
            self.assertEqual(list(target.parent.glob('*.tmp')), [])

    @unittest.skipUnless(os.name == 'nt', 'Real Windows delete-sharing handle test')
    def test_real_windows_reader_handle_lasting_over_two_seconds(self):
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]; kernel.CloseHandle.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'new.tmp', Path(folder)/'status.json'
            source.write_bytes(b'new'); target.write_bytes(b'old')
            # Share reads/writes but deliberately omit FILE_SHARE_DELETE.
            handle = kernel.CreateFileW(str(target), 0x80000000, 3, None, 3, 128, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            release = threading.Event()
            def close_reader():
                release.wait(2.25)
                kernel.CloseHandle(handle)
            reader = threading.Thread(target=close_reader)
            reader.start(); started = time.monotonic()
            try: pipeline.replace_file(source, target)
            finally: release.set(); reader.join(timeout=5.)
            self.assertGreaterEqual(time.monotonic()-started, 2.)
            self.assertFalse(reader.is_alive())
            self.assertEqual(target.read_bytes(), b'new')
            self.assertFalse(source.exists())


if __name__ == '__main__': unittest.main()
