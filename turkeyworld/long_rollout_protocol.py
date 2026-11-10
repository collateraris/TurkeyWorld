"""Immutable opt-in450-step protocol, separate from the original scene plan."""
from pathlib import Path
import copy
import hashlib
import json

from .navigation_experiment import experiment_fingerprint


def bind_long_rollout(components, path, step):
    path=Path(path)
    previous=components.state.get('dynamics_rollout')
    if not path.exists():
        if previous:
            raise ValueError('Saved long rollout requires its immutable protocol file')
        return None
    plan=json.loads(path.read_text(encoding='utf-8'))
    if step!=components.state['last_optimizer_step'] or step<plan.get('start_optimizer_step',step+1):
        raise ValueError('Long dynamics must bind to its completed optimizer boundary')
    if (plan.get('version')!=1 or plan.get('horizon')!=450 or plan.get('simulation_hz')!=30
            or plan.get('tbptt_chunk')!=16 or plan.get('microbatch_size')!=16
            or plan.get('component_plan_fingerprint')!=components.state['fingerprint']
            or plan.get('environment_profile')!='navigation_no_enemies_v1'
            or plan.get('joint_start_horizon')!=450):
        raise ValueError('Unsupported or changed long dynamics protocol')
    fingerprint=experiment_fingerprint(plan)
    root=Path(plan['dataset_directory']).resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('status')!='complete' or manifest['config']['fingerprint']!=plan['dataset_config_fingerprint']
            or manifest['config'].get('environment_profile')!='navigation_no_enemies_v1'
            or manifest['config'].get('enemies_enabled') is not False):
        raise ValueError('Long navigation source is incomplete or has a different profile')
    identity=[]
    for item in manifest['records']:
        source=(root/item['relative_path']).resolve()
        if root not in source.parents or not source.is_file():
            raise ValueError('Long navigation source is missing or escapes its dataset')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest!=item['sha256']:
            raise ValueError('Long navigation source content changed')
        identity.append((item['relative_path'],digest))
    content=experiment_fingerprint(sorted(identity))
    if content!=plan['dataset_content_fingerprint']:
        raise ValueError('Long navigation content identity changed')
    if previous:
        if previous.get('fingerprint')!=fingerprint:
            raise ValueError('Long dynamics protocol changed on resume')
    else:
        if components.name!='dynamics' or step!=plan['start_optimizer_step']:
            raise ValueError('New long dynamics protocol requires its completed dynamics boundary')
        old=components.phase
        old.setdefault('validation_protocol_history',[]).append(copy.deepcopy(old['gate']))
        old.update(gate={'status':'pending','horizon':450},pass_streak=0,best_score=None,
            checks_without_improvement=0,validation_by_case={})
        components.state['dynamics_rollout']={**plan,'fingerprint':fingerprint,'enabled':True,
            'simulated_seconds':15,'optimizer_updates_per_rollout':1,'source_future_feedback':False,
            'numerical_state_resets':0,'long_update_count':0,'committed_long_update_count':0,
            'short_update_count':old['updates'],'joint_long_update_count':0,
            'long_navigation_update_count':0,
            'dataset':{'directory':str(root),'train_clips':manifest['config']['train_clips'],
                'validation_clips':manifest['config']['val_clips'],'profile':plan['environment_profile'],
                'fingerprint':content}}
        components.sync()
    return components.state['dynamics_rollout']
