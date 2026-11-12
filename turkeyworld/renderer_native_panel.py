"""Fixed, disjoint Source VAL area/distance supplement; no sampler mutation."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def build_native_panel(validation, excluded_indices=(), seed=173, per_cohort=64):
    """Return tiny/small/far/empty indices and a plain-JSON selection receipt.

    Far overlaps the area groups, so reserve its examples first. Each visible
    cohort prefers one-quarter screen-edge partial examples; shortages remain
    explicit. Source map-wall occlusion is not inferred from this partial flag.
    """
    if getattr(validation,'split',None)!='val':
        raise ValueError('Native-area guard requires explicit Source VAL')
    if type(per_cohort)is not int or per_cohort<1:
        raise ValueError('Positive integer per_cohort required')
    excluded=set()
    for value in excluded_indices:
        if isinstance(value,(bool,np.bool_)) or int(value)!=value or not 0<=int(value)<len(validation.rows):
            raise ValueError('Excluded panel index is outside Source VAL')
        excluded.add(int(value))
    rng=np.random.default_rng(seed)
    pools={key:[]for key in ('tiny','small','far','empty')}
    facts={};by_file=defaultdict(list)
    for index,row in enumerate(validation.rows):
        if index in excluded:continue
        area=int(round(float(row['mask_fraction'])*1024))
        if not 0<=area<=1024:raise ValueError('Invalid native32 Source area')
        if area==0 and row['category']in('empty_navigation','empty_hidden'):
            pools['empty'].append(index);facts[index]={'area':0,'partial':False,'distance':None}
        elif area>0 and row['category']!='corpse':
            by_file[int(row['file'])].append((index,row,area))
    for file_index,items in by_file.items():
        path=Path(validation.files[file_index]['path'])
        info=validation.files[file_index];stat=path.stat()
        if 'stat'in info and tuple(info['stat'])!=(stat.st_size,stat.st_mtime_ns):
            raise ValueError('Immutable Source VAL file changed')
        with np.load(path,allow_pickle=False)as source:
            states=source['states']  # No RGB, masks or model/engine forward.
        for index,row,area in items:
            state=states[int(row['frame'])]
            if state.shape!=(20,)or not np.isfinite(state).all():
                raise ValueError('Native guard requires finite Source state20')
            if not(state[13]>=.5 and state[12]>0):continue
            distance=float(np.linalg.norm((state[8:10]-state[:2])*16))
            facts[index]={'area':area,'partial':row['category']=='partial','distance':distance}
            if area<=16:pools['tiny'].append(index)
            elif area<=64:pools['small'].append(index)
            if distance>=5:pools['far'].append(index)
    used=set(excluded);groups={};partial_counts={};available_counts={};partial_available={}
    desired_partial=per_cohort//4
    for name in ('far','tiny','small','empty'):
        available=[index for index in pools[name]if index not in used]
        if len(available)<per_cohort:
            raise ValueError(f'Need{per_cohort} distinct {name} Source VAL rows after exclusions; found{len(available)}')
        partial=[i for i in available if facts[i]['partial']]
        other=[i for i in available if not facts[i]['partial']]
        rng.shuffle(partial);rng.shuffle(other)
        wanted=0 if name=='empty'else desired_partial
        first=partial[:min(wanted,len(partial))]
        selected=first+other[:per_cohort-len(first)]
        if len(selected)<per_cohort:
            selected+=partial[len(first):len(first)+per_cohort-len(selected)]
        groups[name]=list(map(int,selected));used.update(selected)
        partial_counts[name]=int(sum(facts[i]['partial']for i in selected))
        available_counts[name]=len(available);partial_available[name]=len(partial)
    indices=[index for name in ('tiny','small','far','empty')for index in groups[name]]
    if len(set(indices))!=4*per_cohort or excluded.intersection(indices):
        raise AssertionError('Native guard cohorts/main panel overlap')
    panel_fingerprint=validation.panel_fingerprint(indices)
    receipt={'protocol':'source_native_area_guard_v1','split':'val','seed':int(seed),
        'per_cohort':per_cohort,'frames':len(indices),'cohort_indices':groups,
        'cohort_counts':{name:len(ids)for name,ids in groups.items()},
        'requested_partial_per_visible_cohort':desired_partial,
        'actual_partial_counts':partial_counts,'available_after_prior_cohorts':available_counts,
        'partial_available_after_prior_cohorts':partial_available,
        'excluded_indices_count':len(excluded),'all_indices_distinct':True,
        'excluded_main_panel_disjoint':True,'selection_priority':['far','tiny','small','empty'],
        'partial_definition':'Source projected screen-edge partial category; not a wall-occlusion label',
        'tiny_definition':'live visible1..16 native pixels',
        'small_definition':'live visible17..64 native pixels',
        'far_definition':'live visible NPC centre distance>=5 map tiles',
        'empty_definition':'zero exact Source NPC pixels, including hidden/empty navigation',
        'source_label_reads':'NPZ states only; native area from validated row.mask_fraction',
        'parent_sampler_and_rng_unchanged':True,'gpu_used':False,
        'fingerprint':panel_fingerprint,'panel_fingerprint':panel_fingerprint}
    receipt['selection_sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
    return indices,receipt
