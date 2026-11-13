"""RGB-only SR labels plus aligned, strictly past-derived RSSM hypotheses.

The base NPR/Source corpus is immutable. Bounding boxes and visibility are
supervision only; inferred state validity is explicit and never synthesized
from a target pose for independent snapshots.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from .small_enemy_data import digest
from .sr_neural_pairs_data import SemanticSRData, fingerprint
from .enemy_animation import ANIMATION_CLASSES,enemy_animation_index

HYPOTHESIS_PROTOCOL = 'rssm_one_step_causal_enemy_hypotheses_v1'
SAMPLER_PROTOCOL = 'enemy_decoder_actual_hr_visible_empty_animation_balanced_v2'
DISTANCES = ('near','mid','far')


def hr_box_and_visibility(mask):
    """Tight normalized xyxy boxes; right/bottom are exclusive pixel edges."""
    if mask.ndim != 4 or mask.shape[1:] != (1,128,128):
        raise ValueError('Enemy labels require a native128 foreground mask')
    visible = mask.bool().flatten(1).any(1)
    boxes = torch.zeros((len(mask),4),dtype=torch.float32,device=mask.device)
    for i in torch.nonzero(visible,as_tuple=False).flatten().tolist():
        ys,xs = torch.nonzero(mask[i,0].bool(),as_tuple=True)
        boxes[i] = torch.stack((xs.min(),ys.min(),xs.max()+1,ys.max()+1)).float()/128
    return boxes,visible[:,None]


class EnemyDecoderData:
    def __init__(self,data_root,split,*,hypotheses_root,seed=173,device='cpu'):
        self.base = SemanticSRData(data_root,split,seed=seed,device=device)
        self.root = Path(hypotheses_root).resolve(); self.split = split; self.device = self.base.device
        manifest_path = self.root/'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (manifest.get('status') != 'complete' or manifest.get('protocol') != HYPOTHESIS_PROTOCOL
                or manifest.get('base_manifest_sha256') != digest(self.base.root/'manifest.json')
                or manifest.get('base_data_sha256') != self.base.manifest['data_sha256']
                or manifest.get('future_state_or_rgb_conditioning') is not False
                or manifest.get('bootstrap_horizon') != 1
                or manifest.get('static_hypothesis_policy') != 'zero_state_invalid'):
            raise ValueError('Enemy hypotheses must pin this exact immutable paired corpus and causal one-step policy')
        self.rows = self.base.rows; self.scene_keys = self.base.scene_keys
        self._cache = OrderedDict(); self.shards = {}
        for entry in manifest['shards']:
            path = self.root/entry['path']; stat = path.stat()
            self.shards[entry['base_shard']] = {**entry,'stat':(stat.st_size,stat.st_mtime_ns)}
        self.visible = []; self.empty_kind = []; self.distance = []; self.animation_labels = []
        self.animation_pools = {distance:{name:[] for name in ANIMATION_CLASSES}for distance in DISTANCES}
        self.pools = {'visible':dict.fromkeys(DISTANCES,None),'empty':{}}
        self.pools['visible'] = {key:[] for key in DISTANCES}
        arrays = {}
        for index,row in enumerate(self.rows):
            name = row['shard']
            if name not in arrays:
                with np.load(self.base.root/name,allow_pickle=False) as z:
                    states = z['state']
                    arrays[name] = (z['mask_hr'].reshape(len(states),-1).any(1),states,z['dynamic'],
                        enemy_animation_index(torch.from_numpy(states)).numpy())
            flags,states,dynamic,animation = arrays[name]; offset = row['offset']
            if name not in self.shards:
                raise ValueError('Hypothesis sidecar misses an original paired shard')
            predicted = self._read(name)
            if (len(predicted['valid']) != len(flags) or bool(predicted['valid'][offset]) != bool(dynamic[offset])):
                raise ValueError('Sidecar validity must exactly identify genuine previous observations')
            state = states[offset]; visible = bool(flags[offset])
            distance = 'near' if row['distance_tiles'] < 2 else 'mid' if row['distance_tiles'] < 5 else 'far'
            empty = ('hp0' if state[4] <= 0 else 'occluded' if state[13] >= .5 and state[12] > 0
                     else 'navigation' if row['stratum']=='navigation' else 'dead')
            self.visible.append(visible); self.empty_kind.append(empty); self.distance.append(distance)
            self.animation_labels.append(int(animation[offset]))
            if visible:
                self.pools['visible'][distance].append(index)
                self.animation_pools[distance][ANIMATION_CLASSES[int(animation[offset])]].append(index)
            else:
                self.pools['empty'].setdefault(empty,[]).append(index)
        if (any(not ids for ids in self.pools['visible'].values()) or not any(self.pools['empty'].values())):
            raise ValueError('Enemy decoder needs actual visible near/mid/far and actual HR-empty frames')
        self.rng = np.random.default_rng(seed); self.draws = 0
        self.case_counts = {'visible':0,'empty':0}
        self.subcounts = {pool:dict.fromkeys(groups,0) for pool,groups in self.pools.items()}
        self.animation_counts = {distance:{name:len(ids)for name,ids in phases.items()}
            for distance,phases in self.animation_pools.items()}
        self.animation_subcounts = {distance:{name:0 for name,ids in phases.items()if ids}
            for distance,phases in self.animation_pools.items()}
        self.identity = {'protocol':'rssm_guided_learned_enemy_decoder_data_v1','base':self.base.identity,
            'hypotheses_manifest_sha256':digest(manifest_path),'hypotheses_fingerprint':manifest['fingerprint'],
            'world_sha256':manifest['world_sha256'],'bootstrap_horizon':1,
            'hypothesis_policy':'actual_previous_Source_RGB_state_action_map_only',
            'future_state_or_rgb_conditioning':False,'static_hypothesis_policy':'zero_state_invalid',
            'sampler_protocol':SAMPLER_PROTOCOL,'visible_empty_weights':{'visible':.5,'empty':.5},
            'animation_classes':list(ANIMATION_CLASSES),'animation_counts':self.animation_counts,
            'animation_sampling':'equal_present_Source_phases_within_each_visible_distance',
            'animation_labels_are_loss_only':True,
            'actual_hr_mask_pool_counts':{pool:{key:len(ids) for key,ids in groups.items()}
                for pool,groups in self.pools.items()},'visibility_labels_shape':['B',1],
            'bbox_format':'native128_tight_xyxy_normalized_exclusive_right_bottom'}
        self.identity['fingerprint'] = fingerprint(self.identity)

    def __len__(self):
        return len(self.rows)

    def _read(self,name):
        entry = self.shards[name]; path = self.root/entry['path']; stat = path.stat()
        if (stat.st_size,stat.st_mtime_ns) != entry['stat']:
            raise ValueError('Immutable RSSM hypothesis sidecar changed')
        if name not in self._cache:
            if digest(path) != entry['sha256']:
                raise ValueError('RSSM hypothesis sidecar hash changed')
            with np.load(path,allow_pickle=False) as z:
                if set(z.files) != {'pred_state','valid'}:
                    raise ValueError('RSSM hypothesis sidecar may contain only predicted state and explicit validity')
                values = {key:z[key] for key in z.files}
            if (values['pred_state'].dtype != np.float32 or values['pred_state'].shape != (entry['rows'],20)
                    or values['valid'].dtype != np.bool_ or values['valid'].shape != (entry['rows'],)
                    or not np.isfinite(values['pred_state']).all()
                    or np.any(values['pred_state'][~values['valid']] != 0)):
                raise ValueError('Invalid predicted state20 or static zero-validity fallback')
            self._cache[name] = values
            while len(self._cache) > 4:
                self._cache.popitem(last=False)
        self._cache.move_to_end(name)
        return self._cache[name]

    def fixed_panel(self,count=None,*,seed=173):
        return self.base.fixed_panel(count,seed=seed)

    def panel_fingerprint(self,panel):
        return fingerprint({'identity':self.identity,'base_panel':self.base.panel_fingerprint(panel)})

    def batch(self,size,indices=None,rng=None):
        if indices is None:
            rng = self.rng if rng is None else rng; indices = []
            for _ in range(size):
                pool = min(self.case_counts,key=lambda k:(self.case_counts[k],k))
                subgroup = min(self.subcounts[pool],key=lambda k:(self.subcounts[pool][k],k))
                if pool=='visible':
                    phase = min(self.animation_subcounts[subgroup],
                        key=lambda k:(self.animation_subcounts[subgroup][k],k))
                    choices = self.animation_pools[subgroup][phase]
                    self.animation_subcounts[subgroup][phase] += 1
                else:
                    choices = self.pools[pool][subgroup]
                indices.append(int(rng.choice(choices)))
                self.case_counts[pool] += 1; self.subcounts[pool][subgroup] += 1; self.draws += 1
            rng.shuffle(indices)
        if len(indices) != size:
            raise ValueError('Explicit enemy-decoder batch indices differ from size')
        batch = self.base.batch(size,indices=indices)
        predicted = []; valid = []
        for index in indices:
            row = self.rows[int(index)]; values = self._read(row['shard'])
            predicted.append(values['pred_state'][row['offset']]); valid.append(values['valid'][row['offset']])
        state = torch.as_tensor(np.stack(predicted),dtype=torch.float32,device=self.device)
        flags = torch.as_tensor(np.asarray(valid),dtype=torch.float32,device=self.device)[:,None]
        batch['hypothesis'] = torch.cat((state,flags),dim=1).detach()
        batch['bbox_hr'],batch['visible_labels'] = hr_box_and_visibility(batch['mask_hr'])
        batch['animation_label'] = torch.tensor([self.animation_labels[int(i)]for i in indices],
            dtype=torch.long,device=self.device)
        return batch

    def sampler_state(self):
        return {'protocol':SAMPLER_PROTOCOL,'identity_sha256':self.identity['fingerprint'],
            'rng':deepcopy(self.rng.bit_generator.state),'draws':int(self.draws),
            'case_counts':dict(self.case_counts),'subcounts':deepcopy(self.subcounts),
            'animation_subcounts':deepcopy(self.animation_subcounts)}

    def restore_sampler(self,state):
        if (state.get('protocol') != SAMPLER_PROTOCOL or state.get('identity_sha256') != self.identity['fingerprint']
                or type(state.get('draws')) is not int or state['draws'] < 0
                or set(state.get('case_counts',{})) != set(self.case_counts)
                or set(state.get('subcounts',{})) != set(self.subcounts)
                or set(state.get('animation_subcounts',{})) != set(self.animation_subcounts)
                or any(type(v) is not int or v < 0 for v in state['case_counts'].values())
                or sum(state['case_counts'].values()) != state['draws']):
            raise ValueError('Enemy decoder sampler identity or successful-draw clock changed')
        for pool in self.subcounts:
            counts = state['subcounts'][pool]
            if (set(counts) != set(self.subcounts[pool]) or any(type(v) is not int or v < 0 for v in counts.values())
                    or sum(counts.values()) != state['case_counts'][pool]):
                raise ValueError('Enemy decoder actual-visible/empty strata counters changed')
        for distance,expected in self.animation_subcounts.items():
            counts = state['animation_subcounts'][distance]
            if (set(counts) != set(expected) or any(type(v)is not int or v<0 for v in counts.values())
                    or sum(counts.values()) != state['subcounts']['visible'][distance]):
                raise ValueError('Enemy decoder visible distance/animation counters changed')
        self.rng.bit_generator.state = deepcopy(state['rng']); self.draws = state['draws']
        self.case_counts = dict(state['case_counts']); self.subcounts = deepcopy(state['subcounts'])
        self.animation_subcounts = deepcopy(state['animation_subcounts'])
