"""Dreamer-style actor/critic in the learned three-RSSM environment.

The live dynamics trainer remains the sole owner of world-model optimization.
This independent learner pins complete world checkpoints, fits reward and
continuation on real TRAIN replay, and optimizes a discrete policy exclusively
in prior imagination. Image models and the user's manual controls are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .dreamer_actor_critic import ActorCriticConfig, DreamerActorCritic
from .dreamer_actor_critic_math import imagination_weights
from .dreamer_policy_data import PolicyReplay, digest, file_sha
from .dreamer_policy_world import RSSMImaginationAdapter
from .dreamer_reward_continue import (DreamerRewardContinue, RewardContinueConfig,
    source_reward_continue, reward_continue_loss, reward_settings)
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_student import _hash_tensors
from .training_pipeline import atomic_json, now


PROTOCOL = 'dreamer_actor_critic_imagination_v1'
ROOT = Path(__file__).resolve().parents[1]
MAIN_RUN = ROOT/'training_runs/rssm_moe_nav15_enemy25_cap20_20261113'
TRAIN_DIRS = [
    ROOT/'training_runs/staged_single_20261110/data_navigation_recovery_v1/long/rgb/train',
    ROOT/'training_runs/rssm_enemy_motion_20261113/data/full_game/rgb/train',
    ROOT/'training_runs/combat_source_20261111/data/full_game/rgb/train',
]
HEAD_EVENT_DIR=ROOT/'training_runs/combat_source_20261111/data/moving1/rgb/train'


def atomic_torch(path, payload):
    temporary = Path(str(path)+f'.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def pinned_world(source, destination):
    """Hard-link one atomic boundary before reading a continuously replaced file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        os.link(source, destination)
    sha = file_sha(destination)
    # mmap avoids copying the retained main Adam states into private RAM.
    saved = torch.load(destination, map_location='cpu', weights_only=False, mmap=True)
    if saved.get('stage') != 'world': raise ValueError('Policy needs a complete world checkpoint')
    step = saved.get('step', saved.get('global_training_step'))
    if type(step) is not int: raise ValueError('Pinned world lacks an integer completed step')
    if any(saved.get(key) != step for key in ('training_steps','global_training_step')):
        raise ValueError('Pinned dynamics checkpoint clocks disagree')
    if saved.get('run_state',{}).get('global_training_step') != step:
        raise ValueError('Pinned dynamics run state is not a completed optimizer boundary')
    return dict(path=str(destination.resolve()), sha256=sha, step=step,
        config_sha256=digest(saved['config']), tensor_count=len(saved['model']))


def floats(metrics):
    result = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().float().item()
        if isinstance(value, (int, float, str, bool)) or value is None:
            if isinstance(value, float) and not np.isfinite(value):
                raise FloatingPointError(f'Nonfinite policy metric {key}')
            result[key] = value
    return result


def optimizer_step(loss, optimizer, parameters):
    parameters = list(parameters)
    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite policy objective')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 100., error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def restore_optimizer(optimizer, saved, module, names, expected_steps):
    canonical = [name for name,_ in module.named_parameters()]
    if names != canonical or len(saved['param_groups']) != 1:
        raise ValueError('Policy Adam parameter identities changed')
    ids = saved['param_groups'][0]['params']
    if ids != list(range(len(canonical))):
        raise ValueError('Policy Adam parameter order changed')
    if expected_steps == 0 and saved['state']:
        raise ValueError('Untrained policy branch has unexpected Adam history')
    if expected_steps and len(saved['state']) != len(canonical):
        raise ValueError('Policy Adam history is incomplete')
    expected_group=optimizer.state_dict()['param_groups'][0]
    if {k:v for k,v in saved['param_groups'][0].items()if k!='params'} != {
            k:v for k,v in expected_group.items()if k!='params'}:
        raise ValueError('Policy Adam hyperparameters changed')
    parameters=list(module.parameters())
    for index,state in saved['state'].items():
        if (set(state)!={'step','exp_avg','exp_avg_sq'}
                or float(state['step']) != expected_steps or any(not bool(torch.isfinite(value).all())
                for value in state.values() if torch.is_tensor(value))):
            raise ValueError('Policy Adam clocks or moments are invalid')
        if (any(state[key].shape != parameters[index].shape for key in ('exp_avg','exp_avg_sq'))
                or bool((state['exp_avg_sq']<0).any())):
            raise ValueError('Policy Adam moment shape or variance is invalid')
    optimizer.load_state_dict(saved)


def archive_unsaved_history(root, completed_step):
    """A crash can leave published, unsaved steps; never append duplicates."""
    path = root/'history.jsonl'
    if not path.exists(): return
    keep, tail, in_tail, previous = [], [], False, -1
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
            valid = type(row.get('step')) is int and previous<row['step']<=completed_step
        except (ValueError,AttributeError): valid=False
        if in_tail or not valid:
            in_tail=True;tail.append(line)
        else:
            keep.append(line);previous=row['step']
    if tail:
        archive = root/f'history_unsaved_tail_before_resume_{time.time_ns()}.jsonl'
        archive.write_text('\n'.join(tail)+'\n',encoding='utf-8')
        temporary = path.with_suffix('.jsonl.tmp')
        temporary.write_text('\n'.join(keep)+('\n'if keep else''),encoding='utf-8')
        os.replace(temporary,path)


class PolicyTraining:
    def __init__(self, args):
        self.args, self.root = args, args.run_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(args.device)
        self.step = 0
        self.state = dict(protocol=PROTOCOL, run_id=self.root.name, status='initializing',
            step=0, phase='prepare', imagination_horizon=args.horizon,
            batch_size=3*args.per_stream, main_dynamics_run=str(MAIN_RUN),
            main_dynamics_unchanged=True, renderer_frozen=True,
            actor_updates=0, quality_qualified=False,
            metrics={}, validation=None, keep_awake={'system_required': True, 'display_required': False})
        self.publish()

    def publish(self):
        self.state['updated_at'] = now()
        atomic_json(self.root/'status.json', self.state)

    def load_world(self, identity):
        model, _ = load_world_checkpoint(identity['path'], self.device)
        model.requires_grad_(False).eval()
        # The adapter controls stochastic prior sampling without training NN assets.
        if hasattr(self, 'world'):
            if digest(model.config.to_dict() if hasattr(model.config, 'to_dict')
                    else vars(model.config)) != self.world_config_sha:
                raise ValueError('Refreshed world architecture changed; explicit policy migration required')
        self.world, self.world_identity = model, identity
        self.world_config_sha = digest(vars(model.config))
        self.adapter = RSSMImaginationAdapter(model, getattr(self, 'heads', None), sample_latents=True)
        self.state.update(world_step=identity['step'], world_sha256=identity['sha256'],
            world_checkpoint=identity['path'])
        self.world_fingerprint = _hash_tensors(model.state_dict())

    def prepare(self):
        resume = self.root/'checkpoints/policy.pt'
        saved = torch.load(resume, map_location=self.device, weights_only=False) if resume.exists() else None
        plan_path = self.root/'training_plan.json'
        if saved is None and plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding='utf-8'))
            identity = plan['world_initial']
        elif saved is not None:
            plan = json.loads(plan_path.read_text(encoding='utf-8'))
            identity = saved['world_identity']
        else:
            identity = pinned_world(self.args.world_source, self.root/'world_snapshots/initial.pt')
            plan = None
        if file_sha(identity['path']) != identity['sha256']:
            raise ValueError('Policy world snapshot changed')
        self.load_world(identity)
        torch.manual_seed(self.args.seed); np.random.seed(self.args.seed); random.seed(self.args.seed)
        config = ActorCriticConfig(feature_dim=self.adapter.feature_dim,
            value_low=-math.log1p(700), value_high=math.log1p(700))
        self.policy = DreamerActorCritic(config).to(self.device)
        # Actual per-transition rewards are bounded by[-2,1.1]. A known bounded
        # support avoids enormous raw expectations from untrained extreme bins.
        self.head_config = RewardContinueConfig(feature_dim=self.adapter.feature_dim,
            low=-math.log1p(2), high=math.log1p(2))
        self.heads = DreamerRewardContinue(config=self.head_config).to(self.device)
        self.adapter.heads = self.heads
        self.reward_config = reward_settings()
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=3e-5, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=3e-5, eps=1e-5)
        self.head_optimizer = torch.optim.Adam(self.heads.parameters(), lr=1e-4, eps=1e-5)
        self.replay = PolicyReplay(TRAIN_DIRS, device=self.device, horizon=self.args.replay_horizon,
            seed=self.args.seed, identities=plan['data'] if plan else None)
        self.head_replay = PolicyReplay([HEAD_EVENT_DIR], device=self.device,
            horizon=self.args.replay_horizon,seed=self.args.seed+2,
            identities=plan['head_event_data'] if plan else None)
        from .dreamer_policy_validation import build_policy_validation_panel
        panel_path = self.root/'validation_panel.json'
        panel = build_policy_validation_panel(ROOT/'training_runs/combat_source_20261111/data',
            count=4, seed=self.args.seed+1, include_terminal=True)
        if panel_path.exists() and json.loads(panel_path.read_text(encoding='utf-8')) != panel:
            raise ValueError('Policy held-out validation panel changed')
        if not panel_path.exists(): atomic_json(panel_path,panel)
        self.validation_panel = panel
        declared_reward_settings = self.reward_config
        proposed = dict(protocol=PROTOCOL, world_initial=plan['world_initial'] if plan else identity,
            world_source=str(self.args.world_source.resolve()), actor_critic=config.to_dict(),
            reward_settings=declared_reward_settings, reward_continue_config=vars(self.head_config),
            data=self.replay.identities,
            head_event_data=self.head_replay.identities,
            head_event_data_identity_sha256=self.head_replay.identity_sha256,
            data_identity_sha256=self.replay.identity_sha256,
            validation_data_identity_sha256=panel['data_identity_sha256'],
            validation_panel_fingerprint=panel['fingerprint'],
            settings=dict(imagination_horizon=self.args.horizon, replay_horizon=self.args.replay_horizon,
                per_stream=self.args.per_stream, head_warmup=self.args.warmup,
                head_loss_mixture=dict(main=.8, short_events=.2),
                short_event_actors=self.args.per_stream, imagination_seeds_from_short_event_replay=False,
                head_refresh_warmup=self.args.refresh_warmup, refresh_every=self.args.refresh_every,
                validation_every=self.args.validate_every, validation_horizon=self.args.validation_horizon,
                validation_starts=4, seed=self.args.seed,
                real_labels='recorded Source TRAIN only; alive cutoff is not terminal',
                actor_training='prior-only learned environment; no Source feedback or future RGB',
                reward_head_importance='uniform-window target with exact rare-stratum importance',
                online_collection=False, policy_controls_game=False))
        if plan is not None and plan != proposed:
            raise ValueError('Policy arguments or immutable training plan changed')
        if plan is None: atomic_json(plan_path, proposed)
        self.plan, self.plan_sha = proposed, digest(proposed)
        self.state.update(plan_sha256=self.plan_sha, data_identity_sha256=self.replay.identity_sha256,
            head_event_data_identity_sha256=self.head_replay.identity_sha256,
            validation_data_identity_sha256=panel['data_identity_sha256'],
            validation_panel_fingerprint=panel['fingerprint'],
            reward_settings=declared_reward_settings, data_coverage=[{k:v for k,v in row.items() if k!='episodes'}
                for row in self.replay.identities], actor_critic_config=config.to_dict())
        self.state['head_event_data_coverage']=[{k:v for k,v in row.items() if k!='episodes'}
            for row in self.head_replay.identities]
        self.refresh_at = 0
        self.actor_updates = 0
        self.cycle_data=None
        if saved is not None:
            if (saved['protocol'] != PROTOCOL or saved['plan_sha256'] != self.plan_sha
                    or saved['actor_critic_config'] != config.to_dict()
                    or saved['reward_continue_config'] != vars(self.head_config)
                    or saved['reward_settings'] != self.reward_config):
                raise ValueError('Policy checkpoint protocol, plan or branch configuration mismatch')
            self.policy.load_state_dict(saved['policy'])
            self.heads.load_state_dict(saved['reward_continue'])
            for optimizer, name, module, steps in (
                    (self.actor_optimizer,'actor_optimizer',self.policy.actor,saved['actor_updates']),
                    (self.critic_optimizer,'critic_optimizer',self.policy.critic,saved['actor_updates']),
                    (self.head_optimizer,'head_optimizer',self.heads,saved['step'])):
                restore_optimizer(optimizer,saved[name],module,saved[name+'_names'],steps)
            self.replay.load_state_dict(saved['replay'])
            self.head_replay.load_state_dict(saved['head_event_replay'])
            self.step, self.actor_updates, self.refresh_at = saved['step'], saved['actor_updates'], saved['refresh_at']
            torch.set_rng_state(saved['torch_rng'].cpu())
            if self.device.type == 'cuda': torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
            np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
            archive_unsaved_history(self.root,self.step)
            checks=dict(policy_tensors_bitexact=_hash_tensors(self.policy.state_dict())==_hash_tensors(saved['policy']),
                reward_continue_tensors_bitexact=_hash_tensors(self.heads.state_dict())==_hash_tensors(saved['reward_continue']),
                torch_rng_bitexact=torch.equal(torch.get_rng_state(),saved['torch_rng'].cpu()),
                cuda_rng_bitexact=(self.device.type!='cuda' or all(torch.equal(a,b.cpu())
                    for a,b in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng']))),
                main_replay_rng_exact=self.replay.state_dict()==saved['replay'],
                event_replay_rng_exact=self.head_replay.state_dict()==saved['head_event_replay'],
                python_rng_exact=random.getstate()==saved['python_rng'],
                named_Adam_groups_clocks_and_moments_valid=True,
                world_identity_verified=file_sha(self.world_identity['path'])==self.world_identity['sha256'])
            if not all(checks.values()): raise AssertionError('Policy resume changed a saved training boundary')
            atomic_json(self.root/'resume_verification.json',dict(step=self.step,
                actor_updates=self.actor_updates,world_step=self.world_identity['step'],
                passed=True,checks=checks,checked_before_SGD=True))
        cycle_path=getattr(self.args,'cycle_corpus',None)
        previous_cycle=saved.get('cycle_policy_data')if saved is not None else None
        if cycle_path is None and previous_cycle is not None:
            cycle_path=Path(previous_cycle['receipt_path'])
        if cycle_path is not None:
            from .dreamer_policy_cycle_data import PolicyCycleData
            self.cycle_data=PolicyCycleData(cycle_path,device=self.device,horizon=self.args.replay_horizon,
                seed=self.args.seed,previous=previous_cycle)
            self.state['cycle_policy_data']=self.cycle_data.state_dict()
        self.state.update(step=self.step, actor_updates=self.actor_updates, status='running')
        receipt_path=self.root/'validation_receipt.json'
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
            if receipt['policy_step']<=self.step and receipt['plan_sha256']==self.plan_sha:
                self.state['validation_receipt']=receipt
                self.state['validation']=json.loads(Path(receipt['report_file']).read_text(encoding='utf-8'))
                reward_path=self.root/f"validation/reward_step_{receipt['policy_step']:06d}.json"
                if reward_path.exists():
                    self.state['reward_validation']=json.loads(reward_path.read_text(encoding='utf-8'))
        self.state['parameters']=dict(actor=sum(p.numel()for p in self.policy.actor.parameters()),
            critic=sum(p.numel()for p in self.policy.critic.parameters()),
            slow_critic=sum(p.numel()for p in self.policy.slow_critic.parameters()),
            reward_continue=sum(p.numel()for p in self.heads.parameters()))
        self.save()
        self.publish()

    def save(self):
        directory = self.root/'checkpoints'
        directory.mkdir(exist_ok=True)
        payload = dict(protocol=PROTOCOL, plan_sha256=self.plan_sha, step=self.step,
            actor_updates=self.actor_updates, refresh_at=self.refresh_at,
            world_identity=self.world_identity, actor_critic_config=self.policy.config.to_dict(),
            reward_continue_config=vars(self.head_config),
            reward_settings=self.reward_config, policy=self.policy.state_dict(),
            reward_continue=self.heads.state_dict(), actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(), head_optimizer=self.head_optimizer.state_dict(),
            actor_optimizer_names=[name for name,_ in self.policy.actor.named_parameters()],
            critic_optimizer_names=[name for name,_ in self.policy.critic.named_parameters()],
            head_optimizer_names=[name for name,_ in self.heads.named_parameters()],
            replay=self.replay.state_dict(), torch_rng=torch.get_rng_state(),
            head_event_replay=self.head_replay.state_dict(),
            cycle_policy_data=self.cycle_data.state_dict()if self.cycle_data is not None else None,
            cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],
            numpy_rng=np.random.get_state(), python_rng=random.getstate())
        atomic_torch(directory/'policy.pt', payload)
        self.state.update(checkpoint_sha256=file_sha(directory/'policy.pt'), checkpoint_step=self.step)

    def refresh(self,*,force=False):
        if not force and(not self.args.refresh_every or not self.step or self.step % self.args.refresh_every):
            return
        path = self.root/f'world_snapshots/refresh_{self.step:06d}_{time.time_ns()}.pt'
        identity = pinned_world(self.args.world_source, path)
        if identity['sha256'] == self.world_identity['sha256']:
            # This redundant link is ours and cannot affect the running world file.
            path.unlink()
            return
        previous = self.world_identity
        devices=[self.device.index if self.device.index is not None else torch.cuda.current_device()]if self.device.type=='cuda'else[]
        with torch.random.fork_rng(devices=devices):
            self.load_world(identity)
        self.refresh_at = self.step
        with (self.root/'world_refreshes.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(step=self.step, previous=previous, current=identity,
                head_warmup=self.args.refresh_warmup))+'\n')

    @torch.no_grad()
    def imagine(self, initial):
        current = initial
        features = [self.adapter.features(current).detach()]
        actions, rewards, continues = [], [], []
        self.heads.eval()
        for _ in range(self.args.horizon):
            action = self.policy.actor.act(features[-1])
            result = self.adapter.step(current, action)
            current = result['world_state']
            actions.append(action.detach()); rewards.append(result['reward'].detach())
            continues.append(result['continue_prob'].detach())
            features.append(result['next_features'].detach())
        return dict(features=torch.stack(features,1), actions=torch.stack(actions,1),
            rewards=torch.stack(rewards,1), continues=torch.stack(continues,1),
            initial_alive=(initial.state[:,4]>0).float())

    def train_update(self):
        batch = self.replay.batch(self.args.per_stream)
        if self.cycle_data is not None:
            batch=self.cycle_data.merge(batch,self.args.per_stream)
        event_batch=self.head_replay.batch(self.args.per_stream)
        with torch.no_grad():
            features, observed = self.adapter.observe_batch(batch, return_states=True)
            labels = source_reward_continue(batch['previous_states'], batch['target_states'],
                batch['events'], config=self.reward_config)
            event_features=self.adapter.observe_batch(event_batch)
            event_labels=source_reward_continue(event_batch['previous_states'],event_batch['target_states'],
                event_batch['events'],config=self.reward_config)
        self.heads.train()
        head_losses, head_metrics = reward_continue_loss(self.heads(features[:,1:].detach()),
            labels, weights=batch['sampling_weights'])
        event_loss,event_metrics=reward_continue_loss(self.heads(event_features[:,1:].detach()),
            event_labels,weights=event_batch['sampling_weights'])
        head_total=.8*head_losses+.2*event_loss
        metrics = floats(head_metrics)
        metrics.update({'event_'+key:value for key,value in floats(event_metrics).items()})
        for key in ('loss','reward_loss','continue_loss'):
            metrics[key]=.8*head_metrics[key]+.2*event_metrics[key]
        metrics['head_grad_norm'] = optimizer_step(head_total, self.head_optimizer, self.heads.parameters())
        warmup = self.step < self.args.warmup or (self.refresh_at > 0
            and self.step-self.refresh_at < self.args.refresh_warmup)
        self.state['phase'] = 'reward_continue_warmup' if warmup else 'actor_critic_imagination'
        if not warmup:
            with torch.no_grad():
                # Choose only alive posterior states from the real replay sequence.
                valid = batch['states'][:,:,4] > 0
                seeds = torch.multinomial(valid.float(), 1).squeeze(-1)
                initial = self.adapter.select_observed(observed, seeds)
                imagined = self.imagine(initial)
            losses, policy_metrics = self.policy.losses(imagined)
            metrics.update(floats(policy_metrics))
            metrics['actor_grad_norm'] = optimizer_step(losses['actor'], self.actor_optimizer,
                self.policy.actor.parameters())
            metrics['critic_grad_norm'] = optimizer_step(losses['critic'], self.critic_optimizer,
                self.policy.critic.parameters())
            self.policy.update_slow_critic()
            self.actor_updates += 1
            metrics.update(actor_loss=float(losses['actor'].detach()),
                critic_loss=float(losses['critic'].detach()),
                imagined_return=float((imagination_weights(imagined['continues'],
                    gamma=self.policy.config.gamma,initial_alive=imagined['initial_alive'])*imagined['rewards']).sum(1).mean()),
                imagined_unweighted_reward_sum=float(imagined['rewards'].sum(1).mean()),
                imagined_continue=float(imagined['continues'].mean()))
        if any(p.grad is not None for p in self.world.parameters()):
            raise AssertionError('Actor or critic wrote gradients into the pinned world')
        metrics['world_gradients_absent'] = True
        return metrics

    def validate(self):
        """Called only at a saved boundary; validation must not alter TRAIN RNG."""
        from .dreamer_policy_validation import evaluate_policy
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],
            np.random.get_state(), random.getstate())
        try:
            panel = self.validation_panel
            directory = self.root/'validation'; directory.mkdir(exist_ok=True)
            validated_policy = directory/f'policy_step_{self.step:06d}.pt'
            if not validated_policy.exists(): os.link(self.root/'checkpoints/policy.pt',validated_policy)
            self.policy.eval(); self.heads.eval()
            report = evaluate_policy(lambda f: self.policy.actor.act(f, deterministic=True),
                self.adapter, panel, step=self.step,
                policy_identity=dict(checkpoint_sha256=self.state['checkpoint_sha256'],
                    step=self.step, plan_sha256=self.plan_sha),
                world_identity=dict(checkpoint_sha256=self.world_identity['sha256'],
                    step=self.world_identity['step']), reward_settings=self.reward_config,
                horizon=self.args.validation_horizon, seed=self.args.seed+1)
            atomic_json(directory/f'step_{self.step:06d}.json', report)
            atomic_json(self.root/'validation_latest.json', report)
            self.state['validation'] = report
            from .dreamer_reward_validation import evaluate_reward_heads
            reward_report=evaluate_reward_heads(self.adapter,
                ROOT/'training_runs/combat_source_20261111/data/moving1/rgb/val',horizon=16)
            reward_report.update(policy_step=self.step,
                policy_checkpoint_sha256=self.state['checkpoint_sha256'],
                world_sha256=self.world_identity['sha256'],world_step=self.world_identity['step'],
                plan_sha256=self.plan_sha)
            atomic_json(directory/f'reward_step_{self.step:06d}.json',reward_report)
            self.state['reward_validation']=reward_report
            self.state['validation_receipt']=dict(report_file=str((directory/f'step_{self.step:06d}.json').resolve()),
                report_sha256=file_sha(directory/f'step_{self.step:06d}.json'),
                policy_checkpoint_file=str(validated_policy.resolve()),
                policy_checkpoint_sha256=file_sha(validated_policy),
                world_checkpoint_file=self.world_identity['path'],world_sha256=self.world_identity['sha256'],
                world_step=self.world_identity['step'],policy_step=self.step,
                panel_fingerprint=panel['fingerprint'],validation_data_identity_sha256=panel['data_identity_sha256'],
                plan_sha256=self.plan_sha)
            atomic_json(self.root/'validation_receipt.json',self.state['validation_receipt'])
        finally:
            torch.set_rng_state(rng[0])
            if self.device.type=='cuda': torch.cuda.set_rng_state_all(rng[1])
            np.random.set_state(rng[2]); random.setstate(rng[3])
            self.policy.train()

    def run(self):
        self.prepare()
        if(self.root/'stop.request').exists():
            self.state['status']='stopped';self.save();self.publish();return
        if getattr(self.args,'refresh_world_now',False):
            self.refresh(force=True);self.save();self.publish()
        started_step = self.step
        while ((self.args.updates == 0 or self.step-started_step < self.args.updates)
                and(not getattr(self.args,'until_step',0)or self.step<self.args.until_step)
                and(not getattr(self.args,'until_actor_updates',0)or self.actor_updates<self.args.until_actor_updates)):
            if(self.root/'stop.request').exists():break
            self.refresh()
            started = time.perf_counter()
            metrics = self.train_update()
            self.step += 1
            metrics['update_seconds'] = time.perf_counter()-started
            if self.device.type=='cuda':
                metrics['cuda_peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
            self.state.update(status='running', step=self.step, actor_updates=self.actor_updates, metrics=metrics)
            row = dict(protocol=PROTOCOL, run_id=self.root.name, step=self.step, timestamp=now(),
                phase=self.state['phase'], world_step=self.world_identity['step'],
                world_sha256=self.world_identity['sha256'], plan_sha256=self.plan_sha,
                data_identity_sha256=self.replay.identity_sha256, metrics=metrics)
            with (self.root/'history.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            if self.step<=3 or self.step%self.args.save_every==0 or self.step==self.args.warmup:
                self.save()
            if self.step%self.args.validate_every==0:
                self.save()
                previous_phase=self.state['phase'];self.state['phase']='validation';self.publish()
                self.validate();self.state['phase']=previous_phase
            if self.step<=3 or self.step%250==0:
                if _hash_tensors(self.world.state_dict()) != self.world_fingerprint:
                    raise AssertionError('Pinned environment weights changed inside the policy learner')
                self.state['frozen_world_verified_step'] = self.step
            self.publish()
            if (self.root/'stop.request').exists(): break
        self.save()
        self.state['status'] = 'stopped' if (self.root/'stop.request').exists() else 'completed'
        self.publish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'training_runs/dreamer_actor_critic_20261114')
    parser.add_argument('--world-source', type=Path, default=MAIN_RUN/'checkpoints/rssm_world.pt')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--replay-horizon', type=int, default=16)
    parser.add_argument('--per-stream', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--refresh-warmup', type=int, default=20)
    parser.add_argument('--refresh-every', type=int, default=250)
    parser.add_argument('--validate-every', type=int, default=100)
    parser.add_argument('--validation-horizon', type=int, default=32)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--updates', type=int, default=0, help='0: continue until stop.request')
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--refresh-world-now',action='store_true',help='Explicit serial-cycle world handoff before SGD')
    parser.add_argument('--cycle-corpus',type=Path,help='Immutable appended TRAIN Source collection')
    parser.add_argument('--until-step',type=int,default=0,help='Idempotent absolute update target for a serial cycle')
    parser.add_argument('--until-actor-updates',type=int,default=0,help='Absolute actor/critic update target excluding head warmup')
    args = parser.parse_args(argv)
    if min(args.horizon,args.replay_horizon,args.per_stream,args.validate_every,args.save_every,args.validation_horizon)<1:
        parser.error('Batch, horizons and intervals must be positive')
    if args.validation_horizon>450: parser.error('Validation is limited to450 transitions')
    if min(args.warmup,args.refresh_warmup,args.refresh_every,args.updates,args.until_step,args.until_actor_updates)<0:
        parser.error('Budgets and intervals must be nonnegative')
    if args.device=='cuda': torch.cuda.init()
    torch.set_num_threads(2)
    args.run_dir = args.run_dir.resolve(); args.world_source=args.world_source.resolve()
    args.run_dir.mkdir(parents=True,exist_ok=True)
    from .vqgan_pipeline import process_alive
    pid_path=args.run_dir/'training.pid'
    if pid_path.exists() and process_alive(int(pid_path.read_text())):
        parser.error('Policy worker is already running')
    pid_path.write_text(str(os.getpid()))
    trainer = PolicyTraining(args)
    try:
        with KeepAwake(): trainer.run()
    except BaseException as error:
        trainer.state.update(status='failed', error=f'{type(error).__name__}: {error}')
        trainer.publish()
        raise


if __name__=='__main__': main()
