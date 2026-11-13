"""Bounded, independent Source geometry pools across all wall depth ranges.

Only TRAIN labels form the auxiliary sampler; only existing held-out VAL labels
form its fixed panel. Original renderer samplers, clocks and mixtures are never
drawn or changed. No model, rendered error, GPU or future RGB selects examples.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json

import numpy as np
import torch

from .renderer_wall_depth import source_wall_depth_labels

PROTOCOL = 'source_all_depth_geometry_aux_pool_v1'
PANEL_PROTOCOL = 'source_all_depth_geometry_validation_v1'
STRATA = ('near_corners', 'mid_corners', 'far_thin', 'other_far')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class GeometryPoolError(ValueError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def _explicit(parent, indices):
    before = deepcopy(parent.sampler_state())
    try:
        result = parent.batch(len(indices), indices=indices)
    except Exception:
        if parent.sampler_state() != before: parent.restore_sampler(before)
        raise
    if parent.sampler_state() != before:
        parent.restore_sampler(before)
        raise ValueError('Explicit geometry Source read changed the main parent sampler; restored')
    return result


def _candidates(parent, count, rng):
    count = min(count, len(parent.rows))
    nav = [i for i, row in enumerate(parent.rows) if row['category'] == 'empty_navigation']
    other = [i for i, row in enumerate(parent.rows) if row['category'] != 'empty_navigation']
    nav_target, other_target = count // 2, count-count // 2
    selected = list(map(int, rng.choice(nav, min(nav_target, len(nav)), replace=False)))
    selected += list(map(int, rng.choice(other, min(other_target, len(other)), replace=False)))
    chosen = set(selected)
    if len(selected) < count:
        remaining = [i for i in range(len(parent.rows)) if i not in chosen]
        selected += list(map(int, rng.choice(remaining, count-len(selected), replace=False)))
    selected.sort()
    return selected, dict(requested_empty_navigation=nav_target,
        actual_empty_navigation=sum(parent.rows[i]['category']=='empty_navigation' for i in selected),
        available_empty_navigation=len(nav), requested_uniform_other=other_target,
        shortage_fill_frames=count-min(nav_target,len(nav))-min(other_target,len(other)),
        policy='half_empty_navigation_half_uniform_other_without_replacement_explicit_shortage_fill')


def _inventory(parent, *, candidate_count, seed, label_batch_size, minimum):
    if (type(candidate_count) is not int or not 4 <= candidate_count <= 16384
            or type(label_batch_size) is not int or not 1 <= label_batch_size <= 64
            or type(seed) is not int or seed < 0 or not len(parent.rows)):
        raise ValueError('Geometry candidates must be bounded4..16384, CPU batches1..64 and an integer seed')
    fingerprint = getattr(parent, 'identity', {}).get('fingerprint')
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError('Geometry pool requires an immutable parent Source fingerprint')
    rng = np.random.default_rng(seed)
    candidates, selection = _candidates(parent, candidate_count, rng)
    facts, support_counts, pixel_counts = {}, Counter(), Counter()
    quadrants, empty_count = Counter(), 0
    for first in range(0, len(candidates), label_batch_size):
        ids = candidates[first:first+label_batch_size]
        batch = _explicit(parent, ids)
        inputs = [batch[key] for key in ('map', 'state', 'object_mask')]
        if any(not torch.is_tensor(value) or value.device.type != 'cpu' for value in inputs):
            raise ValueError('Geometry inventory uses CPU Source maps/poses/masks only')
        labels = source_wall_depth_labels(*inputs)
        for offset, index in enumerate(ids):
            bands, far = labels['bands'], labels['aggregates']['depth_ge_5']
            pixels = dict(near_corner=int(bands['near_0_2']['corner'][offset].sum()),
                near_boundary=int(bands['near_0_2']['boundary'][offset].sum()),
                mid_corner=int(bands['mid_2_5']['corner'][offset].sum()),
                mid_boundary=int(bands['mid_2_5']['boundary'][offset].sum()),
                far_corner=int(far['corner'][offset].sum()), far_thin=int(far['thin_face'][offset].sum()),
                far_boundary=int(far['boundary'][offset].sum()))
            thin = pixels['far_thin'] > 0 or pixels['far_corner'] > 0
            support = dict(near_corners=pixels['near_corner']>0 or pixels['near_boundary']>0,
                mid_corners=pixels['mid_corner']>0 or pixels['mid_boundary']>0,
                far_thin=thin, other_far=pixels['far_boundary']>0 and not thin)
            support_counts.update(key for key, present in support.items() if present)
            pixel_counts.update(pixels)
            area = int((batch['object_mask'][offset]>.5).sum())
            angle = float(torch.atan2(batch['state'][offset,2], batch['state'][offset,3]))
            quadrant = min(3, int((angle % (2*np.pi)) // (np.pi/2)))
            category = str(parent.rows[index]['category'])
            quadrants[str(quadrant)] += 1; empty_count += int(area == 0)
            facts[index] = dict(support=support, pixels=pixels, quadrant=quadrant,
                source_category=category, empty_navigation=category=='empty_navigation', object_area=area)
    receipt = dict(protocol=PROTOCOL, split=parent.split, parent_source_fingerprint=fingerprint,
        candidate_limit=candidate_count, candidate_count=len(candidates), seed=seed,
        label_batch_size=label_batch_size, candidate_indices=candidates,
        candidate_indices_sha256=_digest(candidates), candidate_selection=selection,
        support_frame_counts={name:int(support_counts[name]) for name in STRATA},
        support_pixel_counts={name:int(pixel_counts[name]) for name in
            ('near_corner','near_boundary','mid_corner','mid_boundary','far_corner','far_thin','far_boundary')},
        candidate_camera_quadrants={str(q):int(quadrants[str(q)]) for q in range(4)},
        candidate_no_enemy_frames=int(empty_count), labels_only=True,
        model_error_or_generated_image_mining=False, gpu_used=False, parent_sampler_unchanged=True)
    supported = {name:[i for i in candidates if facts[i]['support'][name]] for name in STRATA}
    # Reserve enough distinct examples for every stratum before assigning
    # overlapping near/mid/far views. This avoids a priority order starving a
    # rarer family that co-occurs with close foreground walls.
    for values in supported.values(): rng.shuffle(values)
    owners = {}
    def augment(name, seen):
        for index in supported[name]:
            if index in seen: continue
            seen.add(index)
            previous = owners.get(index)
            if previous is None or augment(previous, seen):
                owners[index] = name
                return True
        return False
    order = sorted(STRATA, key=lambda name:(len(supported[name]), STRATA.index(name)))
    for _ in range(minimum):
        for name in order:
            if not augment(name, set()):
                receipt.update(required_distinct_per_stratum=minimum,
                    feasible_disjoint_minimum=False, failed_stratum=name)
                raise GeometryPoolError('Bounded Source geometry candidates lack distinct '+name+
                    ' support; explicitly enlarge the candidate pool', receipt)
    buckets = {name:[] for name in STRATA}
    for index, name in owners.items(): buckets[name].append(index)
    remaining = [i for i in candidates if i not in owners and any(facts[i]['support'].values())]
    rng.shuffle(remaining)
    remaining.sort(key=lambda i:sum(facts[i]['support'].values()))
    for index in remaining:
        eligible = [name for name in STRATA if facts[index]['support'][name]]
        name = min(eligible, key=lambda name:(len(buckets[name]), STRATA.index(name)))
        buckets[name].append(index)
    for values in buckets.values(): values.sort()
    retained = {i:{**facts[i], 'stratum':name} for name, values in buckets.items() for i in values}
    def row_identity(index):
        row = parent.rows[index]; info = parent.files[int(row['file'])]
        return dict(index=int(index),file=str(info['path']),frame=int(row['frame']),
                    source_file_sha256=info.get('sha256'),facts=retained[index])
    receipt.update(pool_counts={name:len(values) for name,values in buckets.items()},
        pool_indices=deepcopy(buckets), pool_hashes={name:_digest([row_identity(i) for i in values])
            for name,values in buckets.items()},
        pool_navigation_counts={name:sum(retained[i]['empty_navigation'] for i in values)for name,values in buckets.items()},
        assignment='disjoint_supported_balanced_after_exact_minimum_matching',
        required_distinct_per_stratum=minimum, feasible_disjoint_minimum=True,
        stratum_definitions=dict(near_corners='depth<2 corner or boundary',
            mid_corners='depth2..5 corner or boundary',far_thin='depth>=5 thin face or corner',
            other_far='depth>=5 boundary with no far thin face or corner'),
        stratum_weights={name:.25 for name in STRATA})
    receipt['fingerprint'] = _digest(receipt)
    return retained, buckets, receipt


class GeometryWallFrames:
    protocol = PROTOCOL
    split = 'train'
    def __init__(self, parent, *, candidate_count=8192, seed=31415, label_batch_size=32):
        if getattr(parent, 'split', None) != 'train':
            raise ValueError('Geometry auxiliary pool requires TRAIN, never validation')
        self.parent = parent
        self.facts, self.pool_buckets, self.identity = _inventory(parent,
            candidate_count=candidate_count, seed=seed, label_batch_size=label_batch_size, minimum=1)
        self.identity.update(sampler_seed=seed+1, validation_used=False)
        self.identity['fingerprint'] = _digest({k:v for k,v in self.identity.items() if k!='fingerprint'})
        self.rng = np.random.default_rng(seed+1)
        self.draws = 0; self.stratum_counts = dict.fromkeys(STRATA,0)
        self.navigation_counts = dict.fromkeys(('empty_navigation','other'),0)
        self.quadrant_counts = dict.fromkeys(('0','1','2','3'),0)

    def batch(self, size=8, *, indices=None):
        if type(size) is not int or size < 1: raise ValueError('Positive geometry auxiliary batch required')
        if indices is not None:
            if len(indices)!=size or any(type(i) is not int or i not in self.facts for i in indices):
                raise ValueError('Explicit geometry reads require retained TRAIN pool indices')
            return _explicit(self.parent,indices)
        previous = self.sampler_state(); selected=[]
        for _ in range(size):
            name = min(STRATA,key=lambda key:self.stratum_counts[key])
            index = int(self.rng.choice(self.pool_buckets[name])); selected.append(index)
            self.stratum_counts[name] += 1; self.draws += 1
            fact=self.facts[index]
            self.navigation_counts['empty_navigation' if fact['empty_navigation'] else 'other'] += 1
            self.quadrant_counts[str(fact['quadrant'])] += 1
        self.rng.shuffle(selected)
        try:return _explicit(self.parent,selected)
        except Exception:
            self.restore_sampler(previous)
            raise

    def sampler_state(self):
        return dict(protocol=self.protocol,fingerprint=self.identity['fingerprint'],draws=self.draws,
            rng=deepcopy(self.rng.bit_generator.state),stratum_counts=dict(self.stratum_counts),
            navigation_counts=dict(self.navigation_counts),quadrant_counts=dict(self.quadrant_counts))

    def restore_sampler(self,state):
        counters=('stratum_counts','navigation_counts','quadrant_counts')
        if (state.get('protocol')!=self.protocol or state.get('fingerprint')!=self.identity['fingerprint']
                or type(state.get('draws'))is not int or state['draws']<0
                or any(set(state.get(name,{}))!=set(getattr(self,name))
                    or any(type(v)is not int or v<0 for v in state[name].values())
                    or sum(state[name].values())!=state['draws']for name in counters)
                or any(state['stratum_counts'][name] != state['draws']//4+int(i<state['draws']%4)
                       for i,name in enumerate(STRATA))):
            raise ValueError('Geometry private sampler identity, quotas or successful-draw ledger changed')
        private=np.random.default_rng()
        try:private.bit_generator.state=deepcopy(state['rng'])
        except (KeyError,ValueError,TypeError)as error:raise ValueError('Geometry private RNG state changed')from error
        self.rng.bit_generator.state=deepcopy(private.bit_generator.state); self.draws=state['draws']
        for name in counters:setattr(self,name,dict(state[name]))


def build_geometry_panel(validation, *, candidate_count=4096, frames=128, seed=31416):
    """Fixed balanced Source VAL; old renderer panels may overlap this panel."""
    if getattr(validation,'split',None)!='val':
        raise ValueError('Geometry validation panel requires held-out VAL, never TRAIN')
    if type(frames)is not int or frames<4 or frames%4 or frames>1024:
        raise ValueError('Geometry panel frames must be a bounded positive multiple of four')
    quota=frames//4
    facts,buckets,inventory=_inventory(validation,candidate_count=candidate_count,seed=seed,
                                      label_batch_size=32,minimum=quota)
    rng=np.random.default_rng(seed+1);groups={};representatives=[]
    for name in STRATA:
        available=list(buckets[name]);rng.shuffle(available)
        chosen=[];navigation=Counter();directions=Counter()
        for _ in range(quota):
            prefer_nav=navigation[True]<=navigation[False]
            preferred=[i for i in available if facts[i]['empty_navigation']==prefer_nav]
            candidates=preferred or available
            index=min(candidates,key=lambda i:directions[facts[i]['quadrant']])
            available.remove(index);chosen.append(index)
            navigation[facts[index]['empty_navigation']]+=1;directions[facts[index]['quadrant']]+=1
        groups[name]=chosen
        representatives.append(max(chosen,key=lambda i:(facts[i]['empty_navigation'],
            sum(facts[i]['pixels'].values()),-i)))
    indices=[i for name in STRATA for i in groups[name]]
    if len(set(indices))!=frames:raise AssertionError('Geometry panel must contain distinct Source VAL frames')
    receipt=dict(protocol=PANEL_PROTOCOL,split='val',seed=seed,frames=frames,
        parent_source_fingerprint=validation.identity['fingerprint'],inventory_fingerprint=inventory['fingerprint'],
        candidate_selection=inventory['candidate_selection'],candidate_count=inventory['candidate_count'],
        support_frame_counts=inventory['support_frame_counts'],pool_counts=inventory['pool_counts'],
        stratum_indices=groups,stratum_counts={name:len(v)for name,v in groups.items()},
        preview_indices=representatives,
        no_enemy_counts={name:sum(facts[i]['object_area']==0 for i in values)for name,values in groups.items()},
        navigation_counts={name:sum(facts[i]['empty_navigation']for i in values)for name,values in groups.items()},
        camera_quadrants={name:{str(q):sum(facts[i]['quadrant']==q for i in values)for q in range(4)}
            for name,values in groups.items()},
        selection='equal_strata_prefer_half_navigation_then_balance_supported_camera_quadrants',
        TRAIN_used=False,model_error_mining=False,GPU_used=False,parent_sampler_unchanged=True,
        existing_VAL_panel_overlap_allowed=True,source_labels_only=True)
    fingerprint=validation.panel_fingerprint(indices)
    receipt.update(panel_fingerprint=fingerprint,fingerprint=fingerprint)
    receipt['selection_sha256']=_digest(receipt)
    return indices,receipt
