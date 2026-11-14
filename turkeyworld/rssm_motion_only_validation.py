"""Immutable held-out files and exact state-trajectory panels for motion rounds.

The complete ordered VAL pools are hashed, including files not chosen among
the eight windows. This is a separate identity from the already-pinned motion
objective plan. Binding or verification never queries a model or the engine.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from .rssm_main_ladder_revision import digest,file_sha
from .streaming_sequences import _small_fields
from .structured_state import STATE_DIM,STATE_SCHEMA_VERSION,ENTITY_SLOTS

PROTOCOL='rssm_motion_only_validation_pin_v1'


def _headers(path):
    result={}
    with zipfile.ZipFile(path)as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.npy')or '/'in name or name[:-4]in result:
                raise ValueError('Held-out NPZ must contain unique plain NPY fields')
            with archive.open(name)as stream:
                version=np.lib.format.read_magic(stream)
                reader=(np.lib.format.read_array_header_1_0 if version==(1,0)
                    else np.lib.format.read_array_header_2_0)
                shape,fortran,dtype=reader(stream)
            if dtype.hasobject:raise ValueError('Held-out validation refuses object/pickle fields')
            result[name[:-4]]=dict(shape=list(shape),dtype=dtype.str,fortran_order=bool(fortran))
    return result


def _episode(path):
    before=path.stat();sha=file_sha(path);headers=_headers(path)
    with np.load(path,allow_pickle=False)as saved:
        if 'split'not in saved.files or str(saved['split'])!='val':
            raise ValueError('Motion validation requires explicitly declared VAL, never TRAIN')
        fields,_,transitions=_small_fields(saved,path,'val')
        schema={key:saved[key].item()for key in('state_dim','state_schema_version','entity_slots',
            'behavior_version','dataset_kind','world_transition_compatible')if key in saved.files}
    if(headers['frames']['dtype']!=np.dtype(np.uint8).str or len(headers['frames']['shape'])!=4
            or headers['frames']['shape'][0]!=transitions+1 or headers['frames']['shape'][-1]!=3
            or headers['frames']['shape'][1]<1 or headers['frames']['shape'][2]<1
            or schema.get('state_dim',STATE_DIM)!=STATE_DIM
            or schema.get('state_schema_version',STATE_SCHEMA_VERSION)!=STATE_SCHEMA_VERSION
            or schema.get('entity_slots',ENTITY_SLOTS)!=ENTITY_SLOTS
            or not np.isclose(fields['dt'],1/30,atol=1e-12,rtol=0)):
        raise ValueError('Motion validation has incompatible frame/state schema or timestep')
    after=path.stat()
    if(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('A held-out episode changed while its identity was being read')
    return dict(episode=path.name,path=str(path),sha256=sha,bytes=after.st_size,
        mtime_ns=after.st_mtime_ns,split='val',transitions=transitions,frames=transitions+1,
        dt=fields['dt'],array_schema=headers,declared_schema=schema)


def _cohort(directory,settings):
    directory=Path(directory).resolve()
    if directory.name!='val'or not directory.is_dir():
        raise ValueError('Motion validation pins one explicit VAL split directory')
    paths=sorted(directory.glob('*.npz'))
    if not paths or any(path.resolve().parent!=directory for path in paths):
        raise ValueError('Motion validation needs direct, nonsymlink held-out NPZ files')
    files=[_episode(path)for path in paths]
    context,horizon,count,seed=(settings[key]for key in('context','horizon','count','seed'))
    rng=np.random.default_rng(seed)
    eligible=sorted((row['episode'],i,row['frames'])for i,row in enumerate(files)
        if row['frames']>=context+horizon)
    if len(eligible)<count:raise ValueError('Motion validation needs count distinct complete VAL episodes')
    chosen=[eligible[int(rng.choice(group))]for group in np.array_split(np.arange(len(eligible)),count)]
    windows=[dict(episode=name,episode_index=index,start=int(rng.integers(frames-context-horizon+1)))
        for name,index,frames in chosen]
    content=hashlib.sha256();identities=[]
    for row in windows:
        file=files[row['episode_index']];start=row['start']
        identities.append({key:file[key]for key in('episode','bytes','mtime_ns')})
        with np.load(file['path'],allow_pickle=False)as saved:
            for name,value in(('context_rgb',saved['frames'][start:start+context]),
                    ('states',np.asarray(saved['states'],dtype=np.float32)[start:start+context+horizon]),
                    ('actions',np.asarray(saved['actions'],dtype=np.float32)[start+context-1:start+context-1+horizon]),
                    ('map',np.asarray(saved['map'],dtype=np.int64))):
                array=np.ascontiguousarray(value)
                content.update(name.encode()+array.dtype.str.encode()+str(array.shape).encode()+array.tobytes())
        after=Path(file['path']).stat()
        if(after.st_size,after.st_mtime_ns)!=(file['bytes'],file['mtime_ns']):
            raise ValueError('A held-out episode changed while its fixed panel was being read')
    panel=dict(version='heldout_free_state_v1',split='val',windows=windows,context=context,horizon=horizon,
        seed=seed,source_content_sha256=content.hexdigest(),
        episode_identity_sha256=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest())
    return dict(directory=str(directory),split='val',episode_paths=[str(path)for path in paths],
        ordered_filenames=[path.name for path in paths],files=files,panel=panel,
        expected_fingerprint=hashlib.sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest())


def _settings(*,context=1,horizon=180,count=8,batch_size=4,seed=173):
    if(any(type(v)is not int or v<1 for v in(context,horizon,count,batch_size))
            or context!=1 or horizon!=180 or type(seed)is not int or seed<0):
        raise ValueError('Motion validation requires context1/H180 and positive fixed panel settings')
    return dict(context=context,horizon=horizon,count=count,batch_size=batch_size,seed=seed,
        image_size=32,include_respawned_slots=False,split='val',selection='uniformly_stratified_distinct_episodes',
        all_ordered_VAL_files_hashed=True,source_future_feedback=False,renderer_evaluated=False)


def _build(navdir,enemydir,path,settings):
    nav=Path(navdir).resolve();enemy=Path(enemydir).resolve()
    if nav==enemy:raise ValueError('Navigation and quiet-enemy VAL pools must remain distinct')
    data=dict(navigation=_cohort(nav,settings),enemy_motion=_cohort(enemy,settings))
    return dict(kind=PROTOCOL,version=1,plan_path=str(Path(path).resolve()),settings=settings,
        state_schema=dict(state_dim=STATE_DIM,version=STATE_SCHEMA_VERSION,entity_slots=ENTITY_SLOTS),
        **data,data_identity_sha256=digest(data))


def _exclusive_atomic_json(path,value):
    """Link a complete same-volume temporary file, atomically refusing overwrite."""
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False)as stream:
            temp=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:os.link(temp,path)
        except FileExistsError:pass
    finally:
        if temp is not None:temp.unlink(missing_ok=True)
    if json.loads(path.read_text(encoding='utf-8'))!=value:
        raise ValueError('Motion validation immutable file/panel plan changed')


def bind_motion_validation_plan(navdir,enemydir,path,*,context=1,horizon=180,count=8,batch_size=4,seed=173):
    path=Path(path).resolve();settings=_settings(context=context,horizon=horizon,count=count,batch_size=batch_size,seed=seed)
    core=_build(navdir,enemydir,path,settings)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=core:
            raise ValueError('Motion validation files, panel, settings or directories changed')
    else:_exclusive_atomic_json(path,core)
    return dict(core,validation_plan_sha256=file_sha(path))


def verify_motion_validation_plan(plan):
    expected=deepcopy(plan);sha=expected.pop('validation_plan_sha256',None)
    path=Path(expected.get('plan_path',''))
    if(expected.get('kind')!=PROTOCOL or not isinstance(sha,str)or not path.is_file()
            or file_sha(path)!=sha or json.loads(path.read_text(encoding='utf-8'))!=expected):
        raise ValueError('Motion validation immutable plan bytes changed')
    settings=expected['settings']
    canonical=_settings(**{key:settings[key]for key in('context','horizon','count','batch_size','seed')})
    rebuilt=_build(expected['navigation']['directory'],expected['enemy_motion']['directory'],path,canonical)
    if rebuilt!=expected:raise ValueError('Motion validation data or fixed selected windows changed')
    return True
