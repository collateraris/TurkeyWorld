"""Real held-out low-HP/death calibration of sidecar reward/continue heads.

Matched pre-death and terminal windows intentionally overlap. Counts disclose
measured rows and unique Source transitions; this is not population prevalence.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import random

import numpy as np
import torch

from .jepa_training_utils import SequenceData
from .dreamer_reward_continue import source_reward_continue,reward_continue_loss,reward_settings

KIND='dreamer_real_Source_reward_continue_validation_v1'


def _sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


@contextmanager
def _preserve(adapter):
    python=random.getstate();numpy=np.random.get_state();cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all()if torch.cuda.is_initialized()else None
    modules=list(adapter.heads.modules())
    if hasattr(adapter,'world'):modules+=list(adapter.world.modules())
    modes={module:module.training for module in modules}
    try:
        adapter.heads.eval()
        yield
    finally:
        random.setstate(python);np.random.set_state(numpy);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
        for module,training in modes.items():module.training=training


def _panel(directory,horizon):
    directory=Path(directory).resolve()
    if(directory.name!='val'or type(horizon)is not int or horizon<1):
        raise ValueError('Reward calibration requires explicit VAL and a positive horizon')
    records=[];windows=[];skipped=[]
    for path in sorted(directory.glob('*.npz')):
        with np.load(path,allow_pickle=False)as data:
            if str(data['split'])!='val':raise ValueError('Reward calibration refuses mixed TRAIN/VAL')
            states=np.asarray(data['states']);events=np.asarray(data['events'])
            if(states.ndim!=2 or states.shape[1]!=20 or len(states)!=len(events)+1 or events.shape[1:]!=(4,)
                    or not np.isfinite(states).all()or not np.isfinite(events).all()):
                raise ValueError('Malformed held-out Source reward records')
            deaths=np.flatnonzero((states[:-1,4]>0)&(states[1:,4]<=0))
            if not len(deaths):continue
            if len(deaths)!=1:raise ValueError('A held-out Source episode contains multiple deaths/resets')
            tick=int(deaths[0]);terminal=tick-horizon+1
            alive=[start for start in range(max(0,len(events)-horizon+1))
                if bool((states[start:start+horizon+1,4]>0).all())]
            if terminal<0 or terminal+horizon>len(events)or not alive:
                skipped.append(dict(path=str(path),reason='no_full_terminal_and_matched_alive_window',death_tick=tick))
                continue
            index=len(records);records.append(dict(path=str(path),sha256=_sha(path),death_tick=tick,
                seed=int(data['seed'])if 'seed'in data else None,transitions=len(events),
                initial_HP=float(states[0,4])))
            windows.extend([dict(episode_index=index,start=terminal,role='real_player_death'),
                dict(episode_index=index,start=alive[0],role='matched_low_HP_alive')])
    if not records:raise ValueError('No real held-out deaths with complete matched alive windows; no perfect recall is inferred')
    return records,windows,skipped


@torch.no_grad()
def evaluate_reward_heads(adapter,directory,*,horizon=16,batch_size=4,config=None):
    if adapter.heads is None:raise ValueError('Reward calibration needs learned sidecar heads')
    if type(batch_size)is not int or batch_size<1:raise ValueError('Reward calibration batch must be positive')
    records,windows,skipped=_panel(directory,horizon)
    paths=[row['path']for row in records];device=next(adapter.heads.parameters()).device
    data=SequenceData(directory,device=device,image_size=32,split='val',mode='stream',cache_bytes=32*1024**2,
        episode_paths=paths,refresh_interval=0)
    settings=reward_settings(config);predictions=[];label_rows=[];roles=[];case_results=[]
    with _preserve(adapter):
        for first in range(0,len(windows),batch_size):
            chosen=windows[first:first+batch_size]
            panel=np.asarray([(row['episode_index'],row['start'])for row in chosen],dtype=np.int64)
            batch=data.batch(len(chosen),context=1,horizon=horizon,indices=panel)
            features=adapter.observe_batch(batch)
            if features.shape[:2]!=(len(chosen),horizon+1):raise ValueError('Reward VAL posterior feature alignment changed')
            output=adapter.heads(features[:,1:])
            labels=source_reward_continue(batch['previous_states'],batch['target_states'],batch['events'],config=settings)
            predictions.append(output);label_rows.append(labels);roles.extend([row['role']for row in chosen])
            for local,row in enumerate(chosen):
                single={key:(value[local:local+1]if torch.is_tensor(value)and value.ndim>1 else value)
                    for key,value in output.items()}
                single_labels={key:(value[local:local+1]if torch.is_tensor(value)else
                    {k:v[local:local+1]for k,v in value.items()}if key=='components'else value)
                    for key,value in labels.items()}
                _,metrics=reward_continue_loss(single,single_labels)
                case_results.append(dict(**row,path=records[row['episode_index']]['path'],metrics=metrics))
        joined={key:torch.cat([p[key]for p in predictions],0)for key in('reward_logits','continue_logit')}
        joined.update(reward_bins=predictions[0]['reward_bins'],encoding=predictions[0]['encoding'])
        labels={key:torch.cat([row[key]for row in label_rows],0)
            for key in('reward','continue_target','valid','terminated','truncated')}
        labels['components']={key:torch.cat([row['components'][key]for row in label_rows],0)for key in label_rows[0]['components']}
        _,metrics=reward_continue_loss(joined,labels)
        by_role={}
        for role in('real_player_death','matched_low_HP_alive'):
            select=torch.tensor([value==role for value in roles],device=device)
            selected={key:(value[select]if key in('reward_logits','continue_logit')else value)for key,value in joined.items()}
            selected_labels={key:({k:v[select]for k,v in value.items()}if key=='components'else value[select])for key,value in labels.items()}
            _,by_role[role]=reward_continue_loss(selected,selected_labels)
    # Re-read file identities after measurement so replacing any VAL input fails.
    if any(_sha(row['path'])!=row['sha256']for row in records):raise ValueError('Held-out reward files changed during calibration')
    unique={(row['episode_index'],tick)for row in windows for tick in range(row['start'],row['start']+horizon)}
    raw=len(windows)*horizon
    return dict(kind=KIND,source_held_out=True,split='val',directory=str(Path(directory).resolve()),horizon=horizon,
        feature_scope='complete_three_RSSM_posterior_on_recorded_Source_RGB_state_actions',
        reward_settings=settings,windows=len(windows),unique_Source_episodes=len(records),
        window_transition_measurements=raw,unique_source_transitions=len(unique),
        duplicate_window_measurements=raw-len(unique),balanced_diagnostic_panel=True,
        population_prevalence_estimate=False,importance_weights_used=False,
        alive_control='same_Source_file_low_HP_predeath_window_no_terminal_transition',
        labels='recorded_real_HP_death_not_time_limit_or_fabricated_done',
        no_world_gradients=True,no_optimizer_updates=True,RNG_preserved=True,
        metrics=metrics,by_role=by_role,Source_files=records,cases=case_results,skipped=skipped,
        gameplay_qualified=False)
