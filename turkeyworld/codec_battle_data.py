"""Explicit real HP0, recoil, NPC and empty replay for compact-codec adaptation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .renderer_combat_data import CombatRendererFrames


def render_condition_key(state, grid, *, enemies_enabled=True):
    """Source-visible health distinguishes terminal tint from the same alive pose.

    This is a training split guard, never a renderer or model input. It ignores
    unobservable positive player HP values, while retaining the actual sprite
    and weapon frames. It makes no claim that all different keys yield novel RGB.
    """
    state = np.asarray(state, np.float32)
    if state.shape != (20,) or not np.isfinite(state).all(): raise ValueError('Expected finite Source state20')
    sprite = -1
    if enemies_enabled:
        if state[13] < .5: sprite = 5 if state[16]*.6 < .35 else 6
        elif state[15] > 0: sprite = 4
        elif state[14]*1.2 > 1: sprite = 3
        elif abs(state[10])+abs(state[11]) > .025: sprite = 1 if state[17] < .5 else 2
        else: sprite = 0
    gun = 1 if state[7] > .45 else 2 if state[7] > 0 else 0
    return (state[[0,1,2,3,8,9]].tobytes()+bytes([int(state[4] <= 0),gun,sprite+1])
            +hashlib.sha256(np.asarray(grid, np.uint8).tobytes()).digest())


class CodecBattleFrames(CombatRendererFrames):
    BATTLE_WEIGHTS = {'terminal': .25, 'recoil_hurt': .25, 'npc': .25, 'empty': .25}

    def __init__(self, data_root, appearance_root, split, *, seed=173,
                 exclude_scene_keys=None, exclude_render_keys=None):
        # Coarse six-coordinate exclusion would conflate alive and HP0 tint.
        # Callers keep the historical main panel separately and use the more
        # specific render-condition guard for this supplemental HP0 panel.
        super().__init__(data_root, appearance_root, split, seed=seed, exclude_scene_keys=exclude_scene_keys)
        self.parent_identity = dict(self.identity)
        self.render_keys = set(); excluded = set(exclude_render_keys or ())
        self.battle_buckets = {key: [] for key in self.BATTLE_WEIGHTS}
        self.battle_weights = dict(self.BATTLE_WEIGHTS)
        self.battle_counts = dict.fromkeys(self.battle_weights, 0)
        arrays = {}
        for index, row in enumerate(self.rows):
            info = self.files[row['file']]
            if row['file'] not in arrays:
                with np.load(info['path'], allow_pickle=False) as source:
                    arrays[row['file']] = source['states'].copy(), source['map'].copy()
            states, grid = arrays[row['file']]; state = states[row['frame']]
            key = render_condition_key(state, grid, enemies_enabled=info['kind'] != 'empty_navigation')
            self.render_keys.add(key)
            row['player_hp0'] = bool(state[4] <= 0)
            row['recoil_hurt'] = bool(state[4] > 0 and (state[7] > 0 or 'player_hurt' in row['combat_cases']))
            if key in excluded: continue
            if row['player_hp0']: bucket = 'terminal'
            elif row['recoil_hurt']: bucket = 'recoil_hurt'
            elif row['mask_fraction'] > 0: bucket = 'npc'
            else: bucket = 'empty'
            self.battle_buckets[bucket].append(index)
        if any(not indices for indices in self.battle_buckets.values()):
            raise ValueError('Need genuine HP0/recoil/NPC/empty cohorts: '+str({k:len(v)for k,v in self.battle_buckets.items()}))
        self.terminal_indices = list(self.battle_buckets['terminal'])
        self.identity.update(protocol='codec_battle_terminal_source_v1',
            battle_weights=self.battle_weights,
            battle_cohort_counts={k:len(v) for k,v in self.battle_buckets.items()},
            excluded_render_conditions=len(excluded),
            render_key_definition='camera/NPCxy/map/playerHP0tint/gunframe/NPCsprite; Source labels only',
            quality_scope='episode-held-out, exact-render-condition-filtered supplement; not novel-image guarantee')
        self.identity['fingerprint'] = hashlib.sha256(json.dumps(
            {k:v for k,v in self.identity.items() if k!='fingerprint'},sort_keys=True).encode()).hexdigest()

    def _battle_select(self, size, rng, counts, draws, *, unique=False, excluded=None):
        chosen=[]; used=set(excluded or ())
        for _ in range(size):
            choices={name:([i for i in ids if i not in used] if unique else ids)
                     for name,ids in self.battle_buckets.items()}
            active=[name for name,ids in choices.items() if ids]
            if not active: raise ValueError('No distinct real Source frames left for codec panel')
            name=max(active,key=lambda key:(draws+1)*self.battle_weights[key]-counts[key])
            index=int(rng.choice(choices[name]));chosen.append(index);used.add(index)
            counts[name]+=1;draws+=1
        rng.shuffle(chosen);return chosen,draws

    def fixed_panel(self, count, *, seed=173):
        if count < len(self.terminal_indices): raise ValueError('Panel must include every held-out HP0 once')
        ids=list(self.terminal_indices);counts=dict.fromkeys(self.battle_weights,0);counts['terminal']=len(ids)
        extra,_=self._battle_select(count-len(ids),np.random.default_rng(seed),counts,len(ids),unique=True,excluded=ids)
        return ids+extra

    def batch(self, size, *, indices=None):
        if indices is None:
            indices,self.draws=self._battle_select(size,self.rng,self.battle_counts,self.draws)
            for index in indices:self.case_counts[self.rows[index]['category']]+=1
        # Skip CombatRendererFrames' mixture counters: this is a distinct plan.
        from .renderer_enemy_data import EnemyFrames
        return EnemyFrames.batch(self,size,indices=indices)

    def sampler_state(self):
        from .renderer_enemy_data import EnemyFrames
        return {**EnemyFrames.sampler_state(self),'battle_counts':dict(self.battle_counts)}

    def restore_sampler(self,state):
        from .renderer_enemy_data import EnemyFrames
        if set(state['battle_counts'])!=set(self.battle_weights) or sum(state['battle_counts'].values())!=state['draws']:
            raise ValueError('Codec battle sampling clocks changed')
        EnemyFrames.restore_sampler(self,state);self.battle_counts=dict(state['battle_counts'])

    def cohort_rows(self, indices):
        return {name:[pos for pos,index in enumerate(indices) if test(self.rows[index])]
            for name,test in {'player_hp0':lambda r:r['player_hp0'],
                'player_alive':lambda r:not r['player_hp0'], 'recoil_hurt':lambda r:r['recoil_hurt'],
                'npc_visible':lambda r:r['mask_fraction']>0,
                'navigation_empty':lambda r:r['category'] in ('empty_navigation','empty_hidden')}.items()}
