"""Host-bound PID receipts; no model imports, process launches or signaling."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import platform
import socket
import time

from .workspace_paths import resolve_path

PROTOCOL='host_process_identity_v1'


def host_identity():
    boot=None
    if platform.system()=='Linux':
        try:boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except OSError:pass
    return dict(platform=platform.system(),hostname=socket.gethostname(),boot_id=boot)


def _proc_fields(pid):
    raw=Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
    return raw.rsplit(')',1)[1].split()


def _start_token(pid):
    if platform.system()=='Linux':
        try:return _proc_fields(pid)[19]  # starttime is field22; state is field3.
        except (OSError,IndexError):return None
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_uint,ctypes.c_int,ctypes.c_uint]
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.POINTER(ctypes.c_ulonglong)]*4
        kernel.GetProcessTimes.restype=ctypes.c_int
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return None
        try:
            creation,exit_time,kernel_time,user_time=(ctypes.c_ulonglong()for _ in range(4))
            if kernel.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit_time),
                    ctypes.byref(kernel_time),ctypes.byref(user_time)):
                return str(creation.value)
        finally:kernel.CloseHandle(handle)
    return None


def process_alive(pid):
    if type(pid)is not int or pid<1:return False
    if os.name!='nt':
        try:os.kill(pid,0)
        except ProcessLookupError:return False
        except PermissionError:pass
        if platform.system()=='Linux':
            try:
                if _proc_fields(pid)[0]in('Z','X','x'):return False
            except (OSError,IndexError):pass
        return True
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint,ctypes.c_int,ctypes.c_uint]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint]
    kernel.WaitForSingleObject.restype=ctypes.c_uint
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if not handle:return False
    try:return kernel.WaitForSingleObject(handle,0)==258
    finally:kernel.CloseHandle(handle)


def process_identity(pid=None):
    pid=os.getpid()if pid is None else pid
    if type(pid)is not int or pid<1:raise ValueError('A process receipt needs a positive integer PID')
    return dict(protocol=PROTOCOL,pid=pid,host=host_identity(),start_token=_start_token(pid))


def process_identity_alive(record,*,discard_foreign=False):
    """Never inspect/adopt a PID from another host or an unreceipted old run.

    discard_foreign is an explicit relocation decision, not a resume default.
    It ignores legacy/foreign records without querying their numeric PID.
    """
    if record is None:
        if discard_foreign:return False
        raise RuntimeError('PID has no host receipt. Verify all old workers are stopped, then use --discard-foreign-pids for explicit migration.')
    if(not isinstance(record,dict)or record.get('protocol')!=PROTOCOL
            or type(record.get('pid'))is not int or record['pid']<1
            or not isinstance(record.get('host'),dict)):
        raise ValueError('Invalid host-bound process receipt')
    current=host_identity();saved=record['host']
    if saved.get('platform')!=current['platform']or saved.get('hostname')!=current['hostname']:
        if discard_foreign:return False
        raise RuntimeError('PID belongs to another host/platform. Use --discard-foreign-pids only after verifying the old worker is stopped.')
    if saved.get('boot_id')is not None and saved['boot_id']!=current['boot_id']:return False
    token=record.get('start_token')
    if token is not None:
        actual=_start_token(record['pid'])
        if actual is None and process_alive(record['pid']):
            raise RuntimeError('A live PID creation time could not be verified; refusing to adopt or duplicate it')
        if token!=actual:return False
    return process_alive(record['pid'])


def pid_receipt_path(path):
    path=resolve_path(path)
    return path.with_name(path.name+'.host.json')


def pid_file_alive(path,*,discard_foreign=False):
    path=resolve_path(path)
    if not path.exists():return False
    try:pid=int(path.read_text(encoding='utf-8').strip())
    except (ValueError,UnicodeError)as error:raise ValueError('Invalid PID file')from error
    receipt=pid_receipt_path(path)
    if not receipt.exists():return process_identity_alive(None,discard_foreign=discard_foreign)
    record=json.loads(receipt.read_text(encoding='utf-8'))
    if record.get('pid')!=pid:raise ValueError('PID text and host receipt differ')
    return process_identity_alive(record,discard_foreign=discard_foreign)


def _atomic_text(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(text,encoding='utf-8')
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def write_pid_file(path,pid=None):
    path=resolve_path(path);record=process_identity(pid)
    if path.exists():
        # Keep the pre-transfer ownership evidence before replacing active PID
        # pointers. Never reinterpret a Windows PID as a Linux process.
        directory=path.parent/'process_history'/f'{time.time_ns()}_{path.name}'
        directory.mkdir(parents=True,exist_ok=False)
        for source in(path,pid_receipt_path(path)):
            if source.exists():(directory/source.name).write_bytes(source.read_bytes())
    _atomic_text(pid_receipt_path(path),json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
    _atomic_text(path,str(record['pid'])+'\n')
    return record
