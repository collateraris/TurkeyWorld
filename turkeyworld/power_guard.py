"""Keep background training awake while Windows may remain locked/dark.

Requests only system availability, never display availability or user input.
The request is scoped to its thread/process lifetime; power plan settings and
Windows lock/security policies are left unchanged. Explicit user sleep and
shutdown remain effective.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class KeepAwake:
    def __init__(self, api=None):
        self.api = api
        self.active = False
        self.previous_state = None
        self.error = None

    def __enter__(self):
        try:
            if self.api is None:
                if sys.platform != 'win32':
                    return self
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                self.api = kernel.SetThreadExecutionState
                self.api.argtypes = [wintypes.DWORD]
                self.api.restype = wintypes.DWORD
            previous = self.api(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            if not previous:
                raise OSError('Windows did not accept the system-awake request')
            self.previous_state = int(previous)
            self.active = True
        except OSError as error:
            self.error = str(error)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active:
            try:
                restored = self.api(self.previous_state | ES_CONTINUOUS)
                if not restored:
                    self.error = 'Windows did not acknowledge releasing the system-awake request'
            except OSError as error:
                self.error = str(error)
            finally:
                self.active = False
        return False


def _status(path, value):
    value = {'updated_at': datetime.now(timezone.utc).isoformat(), **value}
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.025)


def watch_training(pid_file, status_file, interval=5.):
    """Protect an existing worker without restarting it or importing torch.

    A real process handle detects termination without being confused by PID
    reuse. If training.pid changes while this guard runs, follow the new owned
    worker. When the worker finishes, release the request and exit.
    """
    if sys.platform != 'win32':
        raise RuntimeError('The standalone sleep guard is for Windows')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    pid_file, status_file = Path(pid_file), Path(status_file)
    process_handle, tracked_pid = None, None
    with KeepAwake() as guard:
        if not guard.active:
            _status(status_file, {'active': False, 'error': guard.error, 'guard_pid': os.getpid()})
            raise RuntimeError(guard.error)
        try:
            while True:
                pid = int(pid_file.read_text(encoding='utf-8').strip())
                if pid != tracked_pid:
                    if process_handle:
                        kernel.CloseHandle(process_handle)
                    process_handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
                    if not process_handle:
                        raise OSError(f'Cannot monitor training process {pid}')
                    tracked_pid = pid
                if kernel.WaitForSingleObject(process_handle, 0) == 0:
                    break
                _status(status_file, {'active': True, 'trainer_pid': tracked_pid,
                    'guard_pid': os.getpid(), 'system_required': True, 'display_required': False,
                    'screen_lock_allowed': True, 'settings_modified': False,
                    'backend': 'SetThreadExecutionState'})
                time.sleep(interval)
        finally:
            if process_handle:
                kernel.CloseHandle(process_handle)
    _status(status_file, {'active': False, 'trainer_pid': tracked_pid, 'guard_pid': os.getpid(),
                          'reason': 'Training process exited; awake request released'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid-file', type=Path, required=True)
    parser.add_argument('--status-file', type=Path)
    args = parser.parse_args(argv)
    watch_training(args.pid_file, args.status_file or args.pid_file.with_name('power_guard.json'))


if __name__ == '__main__':
    main()
