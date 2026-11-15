"""Keep background training awake while Windows/Linux may remain locked/dark.

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
import queue
import shutil
import subprocess
import sys
import threading
import time
import warnings

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class KeepAwake:
    def __init__(self, api=None):
        self.api = api
        self.active = False
        self.previous_state = None
        self.error = None
        self.backend = None
        self.process = None

    def _linux_enter(self):
        self.backend = 'systemd-inhibit'
        inhibitor = shutil.which('systemd-inhibit')
        if inhibitor is None:
            raise OSError('systemd-inhibit is unavailable; install systemd or use an explicit host sleep-inhibition wrapper')
        # systemd starts the helper only after acquiring its sleep inhibitor.
        # EOF releases it even after a parent crash/kill; no display or lock
        # inhibition is requested, and no daemon survives its input pipe.
        helper = 'import sys; print("TURKEYWORLD_INHIBITOR_READY", flush=True); sys.stdin.buffer.read()'
        self.process = subprocess.Popen([inhibitor,'--what=sleep','--mode=block',
            '--who=TurkeyWorld','--why=Neural game training','--',sys.executable,'-u','-c',helper],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,bufsize=1,close_fds=True)
        messages = queue.Queue()
        def consume():
            try:
                for line in self.process.stdout:messages.put(line.strip())
            finally:messages.put(None)
        threading.Thread(target=consume,daemon=True,name='sleep-inhibitor-readiness').start()
        deadline,errors = time.monotonic()+5.,[]
        while True:
            try:line=messages.get(timeout=max(.001,deadline-time.monotonic()))
            except queue.Empty:raise OSError('Timed out acquiring systemd sleep inhibitor')from None
            if line=='TURKEYWORLD_INHIBITOR_READY':
                if self.process.poll()is not None:raise OSError('Sleep inhibitor exited during startup')
                self.active=True
                return
            if line is None:raise OSError('systemd-inhibit failed: '+'; '.join(errors[-4:]))
            errors.append(line)
            if time.monotonic()>=deadline:raise OSError('Timed out acquiring systemd sleep inhibitor')

    def _linux_release(self):
        process=self.process
        if process is None:return
        try:
            if process.stdin is not None:process.stdin.close()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
        finally:
            self.active=False
            if process.stdout is not None:process.stdout.close()
            self.process=None

    def __enter__(self):
        try:
            if self.api is None:
                if sys.platform.startswith('linux'):
                    self._linux_enter()
                    return self
                if sys.platform != 'win32':
                    raise OSError('Sleep inhibition is unavailable on '+sys.platform)
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                self.api = kernel.SetThreadExecutionState
                self.api.argtypes = [wintypes.DWORD]
                self.api.restype = wintypes.DWORD
            self.backend = 'SetThreadExecutionState'
            previous = self.api(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            if not previous:
                raise OSError('Windows did not accept the system-awake request')
            self.previous_state = int(previous)
            self.active = True
        except OSError as error:
            self.error = str(error)
            if self.backend=='systemd-inhibit':
                try:self._linux_release()
                except (OSError,subprocess.TimeoutExpired):pass
                warnings.warn('Sleep protection inactive: '+self.error,RuntimeWarning,stacklevel=2)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.backend=='systemd-inhibit':
            try:self._linux_release()
            except (OSError,subprocess.TimeoutExpired)as error:
                self.error=str(error)
                warnings.warn('Sleep inhibitor release failed: '+self.error,RuntimeWarning,stacklevel=2)
            return False
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

    def report(self):
        if self.backend=='systemd-inhibit'and self.active and self.process.poll()is not None:
            self.active=False
            self.error='Sleep inhibitor exited unexpectedly'
            warnings.warn(self.error,RuntimeWarning,stacklevel=2)
        return dict(active=self.active,error=self.error,backend=self.backend,
            system_required=self.active,display_required=False,screen_lock_allowed=True,
            settings_modified=False)


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
