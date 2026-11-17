"""Explicit workspace relocation without rewriting saved identities or files.

``workspace_portability.json`` declares the old Windows workspace prefix. File
I/O uses the current native root; identity fields keep their original Windows
spelling. No global pathlib hooks, checkpoint loads or implicit file writes.
"""
import hashlib
import json
import os
from pathlib import Path,PurePosixPath,PureWindowsPath
import re

PROTOCOL='workspace_paths_v1'
ROOT=Path(__file__).resolve().parents[1]


def _text(value):
    value=os.fspath(value)
    if not isinstance(value,str)or not value or '\x00'in value:
        raise ValueError('A workspace path must be a nonempty text path')
    return value


def _sha(value):return isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None


class WorkspacePaths:
    def __init__(self,root,origin_root,allowed_native_roots=()):
        self.root=Path(root).resolve()
        self.allowed_native_roots=tuple(Path(p).resolve() for p in allowed_native_roots)
        raw=_text(origin_root);origin=PureWindowsPath(raw)
        if not self.root.is_dir()or not origin.is_absolute()or not re.fullmatch('[A-Za-z]:',origin.drive):
            raise ValueError('Relocation needs an existing native root and an absolute Windows workspace origin')
        self.origin_root=str(origin);self.origin=origin

    def _origin_parts(self,raw):
        value=PureWindowsPath(raw)
        if not value.is_absolute():return None
        try:return value.relative_to(self.origin).parts
        except ValueError:return None

    def _parts(self,value):
        raw=_text(value);parts=self._origin_parts(raw)
        if parts is None:
            native=Path(raw)
            if native.is_absolute():
                try:parts=native.relative_to(self.root).parts
                except ValueError:raise ValueError('Path is outside this workspace')from None
            else:
                windows=PureWindowsPath(raw)
                if windows.drive or windows.root:raise ValueError('Foreign or drive-relative Windows path is outside this workspace')
                parts=windows.parts
        if any(part=='..'or ':'in part for part in parts):
            raise ValueError('Parent traversal and ambiguous drive fields are not workspace paths')
        return tuple(part for part in parts if part not in('','.'))

    def _native(self,parts,*,must_exist=False):
        native=self.root.joinpath(*parts).resolve(strict=must_exist)
        if not any(native.is_relative_to(base) for base in (self.root,*self.allowed_native_roots)):raise ValueError('Workspace symlink escapes the declared native root')
        return native

    def relative(self,value):
        parts=self._parts(value);self._native(parts)
        return PurePosixPath(*parts).as_posix()

    def resolve(self,value,*,must_exist=False):
        return self._native(self._parts(value),must_exist=must_exist)

    def identity_path(self,value):
        raw=_text(value);parts=self._parts(raw);self._native(parts)
        # Existing serialized spelling is data, including separators/case. It
        # is not normalized simply because the physical workspace has moved.
        if self._origin_parts(raw)is not None:return raw
        return str(self.origin.joinpath(*parts))

    def verify_file(self,value,sha256,*,size=None):
        if not _sha(sha256)or size is not None and(type(size)is not int or size<0):
            raise ValueError('A file receipt needs an exact SHA256 and optional nonnegative byte count')
        path=self.resolve(value,must_exist=True)
        with path.open('rb')as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        actual_size=path.stat().st_size
        if actual!=sha256 or size is not None and actual_size!=size:raise ValueError('Relocated file bytes changed')
        return dict(relative_path=self.relative(value),sha256=actual,bytes=actual_size)

    def read_json(self,value,*,sha256=None):
        path=self.resolve(value,must_exist=True);raw=path.read_bytes()
        if sha256 is not None and(not _sha(sha256)or hashlib.sha256(raw).hexdigest()!=sha256):
            raise ValueError('Relocated immutable JSON bytes changed')
        return json.loads(raw.decode('utf-8-sig'))


def load_workspace_paths(root=None):
    root=ROOT if root is None else Path(root).resolve()
    config=root/'workspace_portability.json'
    if not config.is_file():return None
    value=json.loads(config.read_text(encoding='utf-8-sig'))
    if(not isinstance(value,dict)or value.get('protocol',PROTOCOL)!=PROTOCOL
            or set(value)-{'protocol','version','origin_root','allowed_native_roots'}or value.get('version',1)!=1):
        raise ValueError('Unknown explicit workspace portability configuration')
    allowed=value.get('allowed_native_roots',[])
    if not isinstance(allowed,list) or any(not isinstance(p,str) or not Path(p).is_absolute() for p in allowed):raise ValueError('Explicit native storage roots must be absolute paths')
    return WorkspacePaths(root,value['origin_root'],allowed)


def resolve_path(value,*,root=None,must_exist=False):
    """Native I/O path; valid external native absolute paths remain available."""
    raw=_text(value);mapping=load_workspace_paths(root)
    if mapping is not None:
        if mapping._origin_parts(raw)is not None:return mapping.resolve(raw,must_exist=must_exist)
        native=Path(raw)
        if native.is_absolute():
            # External native paths are ordinary caller-supplied files. Strict
            # WorkspacePaths.resolve remains available for manifest members.
            return native.resolve(strict=must_exist)
        return mapping.resolve(raw,must_exist=must_exist)
    if os.name!='nt'and(PureWindowsPath(raw).drive or PureWindowsPath(raw).root and not Path(raw).is_absolute()):
        raise ValueError('Windows paths require an explicit workspace_portability.json on this host')
    return Path(raw).resolve(strict=must_exist)


def identity_path(value,*,root=None):
    """Stable saved identity spelling, separate from the native I/O location."""
    raw=_text(value);mapping=load_workspace_paths(root)
    if mapping is not None:
        if mapping._origin_parts(raw)is not None:return mapping.identity_path(raw)
        native=Path(raw)
        if not native.is_absolute()or native.is_relative_to(mapping.root):return mapping.identity_path(raw)
    return str(resolve_path(raw,root=root))
