"""Serial real-Source collection -> motion RSSM round -> imagined actor/critic.

The scheduler owns no model optimizer. Complete child checkpoints and immutable
Source receipts are the handoffs; stage recovery never restarts a finished
world round or adds an accidental extra policy budget.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .power_guard import KeepAwake
from .process_runtime import pid_file_alive,write_pid_file,process_identity,process_identity_alive
from .workspace_paths import resolve_path,identity_path

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='dreamer_motion_Source_world_policy_cycle_v1'
DEFAULT_WORLD=ROOT/'training_runs/rssm_moe_nav15_enemy25_cap20_20261113'
DEFAULT_POLICY=ROOT/'training_runs/dreamer_actor_critic_20261114'
IO_ATTEMPTS=24
IO_INITIAL_DELAY=.025
IO_MAX_DELAY=.5


def now():return datetime.now(timezone.utc).isoformat()
def retry_shared_file(operation):
    """Windows may briefly deny access while a status file is atomically replaced."""
    delay=IO_INITIAL_DELAY
    for attempt in range(IO_ATTEMPTS):
        try:return operation()
        except PermissionError:
            if attempt+1==IO_ATTEMPTS:raise
            time.sleep(delay);delay=min(delay*2,IO_MAX_DELAY)

def read(path):return json.loads(retry_shared_file(lambda:resolve_path(path).read_text(encoding='utf-8')))
def sha(path):
    def calculate():
        with resolve_path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    return retry_shared_file(calculate)
def atomic(path,value):
    path=resolve_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    body=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    try:
        retry_shared_file(lambda:temporary.write_text(body,encoding='utf-8'))
        retry_shared_file(lambda:os.replace(temporary,path))
    except BaseException:
        try:temporary.unlink(missing_ok=True)
        except OSError:pass
        raise
def _posix_alive(pid):
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM proves the PID exists; it must not trigger a second worker.
        pass
    if sys.platform.startswith('linux'):
        try:
            # kill(pid,0) also succeeds for an unreaped child. Its process name
            # may contain spaces/parentheses, so parse after the final ')'.
            record=Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
            state=record.rsplit(')',1)[1].split()[0]
            if state in ('Z','X','x'):return False
        except (OSError,IndexError):
            # /proc may be unavailable; the signal permission probe still
            # established existence. A process-exit race is rechecked next poll.
            pass
    return True


def alive(pid):
    if not isinstance(pid,int)or isinstance(pid,bool)or pid<1:return False
    if os.name!='nt':return _posix_alive(pid)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.OpenProcess.argtypes=[ctypes.c_uint,ctypes.c_int,ctypes.c_uint]
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint]
    kernel.WaitForSingleObject.restype=ctypes.c_uint
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x00100000,False,int(pid))
    if not handle:return False
    try:return kernel.WaitForSingleObject(handle,0)==258
    finally:kernel.CloseHandle(handle)


def child_process_options():
    """Suppress Windows console helpers; POSIX requires zero creation flags."""
    return dict(creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)if os.name=='nt'else 0)


class MotionCycle:
    def __init__(self,args):
        self.args=args;self.root=resolve_path(args.run_dir);self.root.mkdir(parents=True,exist_ok=True)
        self.status_path=self.root/'status.json'
        settings=dict(protocol=PROTOCOL,version=1,world_run=identity_path(args.world_run),policy_run=identity_path(args.policy_run),
            motion_plan_sha256=sha(args.motion_plan),horizons=[16,48,80,112,144,180],updates_per_horizon=30,
            world_updates_per_round=180,actor_critic_updates_per_round=args.policy_updates,
            episodes_per_stream=args.episodes_per_stream,episode_transitions=180,
            target_game_seconds=6.,cohorts=['navigation','enemy_motion'],combat_loss=False,
            collect_fire_mask=True,policy_rebinds_new_world_before_SGD=True,
            stage_order=['collect_source','world_training','policy_training'],
            immutable_TRAIN_append=True,heldout_VAL_unchanged=True)
        plan=self.root/'cycle_plan.json'
        if plan.exists()and read(plan)!=settings:raise ValueError('Serial motion cycle plan changed')
        if not plan.exists():atomic(plan,settings)
        self.state=read(self.status_path)if self.status_path.exists()else dict(protocol=PROTOCOL,
            status='preparing',round_index=0,phase='collect_source',source_episodes=0,source_transitions=0,
            world_updates=0,policy_updates=0,world_target_updates=180,policy_target_updates=args.policy_updates,
            completed_rounds=0,started_at=now(),active_pid=None)
        if self.state['protocol']!=PROTOCOL:raise ValueError('Unknown cycle state')
        self.state['plan_sha256']=sha(plan)
        from .rssm_motion_round_schedule import read_revision
        self.round_schedule_revision=read_revision(args.world_run/'motion_round_schedule.json',self.state.get('round_schedule_revision'))
        if self.round_schedule_revision is not None:self.state['round_schedule_revision']=self.round_schedule_revision

    def world_budget(self):
        from .rssm_motion_round_schedule import schedule_for_round
        schedule=schedule_for_round(getattr(self,'round_schedule_revision',None),self.state['round_index'])
        return schedule['world_updates_per_round'] if schedule else 180

    def publish(self,*,mirror=False):
        self.state['updated_at']=now();atomic(self.status_path,self.state)
        if mirror:
            path=self.args.world_run/'status.json'
            if path.exists():
                status=read(path);status['dreamer_cycle']=self.state;atomic(path,status)

    def bootstrap(self,module,*arguments):
        return [sys.executable,'-X','utf8','-u','-m','turkeyworld.bootstrap',
            '--module',module,*map(str,arguments)]

    def retire_initial_flags(self):
        if getattr(self.args,'resume_after_user_stop',False):
            for name,flag in(('cycle',self.root/'stop.request'),
                    ('world',self.args.world_run/'stop_training'),('policy',self.args.policy_run/'stop.request')):
                if flag.exists():
                    destination=self.root/'authorized_resume_flags'/f'{time.time_ns()}_{name}'
                    destination.parent.mkdir(exist_ok=True)
                    destination.write_bytes(flag.read_bytes());flag.unlink()
            self.state.pop('error',None)
            self.state['initial_flags_retired']=True;self.publish();return
        if self.state.get('initial_flags_retired'):return
        for name,flag in(('world_stop',self.args.world_run/'stop_training'),
                ('policy_stop',self.args.policy_run/'stop.request')):
            if flag.exists():
                destination=self.root/'retired_initial_flags'/name
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(flag.read_bytes());flag.unlink()
        self.state['initial_flags_retired']=True;self.publish()

    def pin_assets(self,directory):
        assets=directory/'assets';assets.mkdir(parents=True,exist_ok=True)
        for source,name in((self.args.world_run/'checkpoints/rssm_world.pt','world.pt'),
                (self.args.policy_run/'checkpoints/policy.pt','policy.pt')):
            target=assets/name
            if not target.exists():os.link(source,target)
        return assets

    def execute(self,directory,name,command,*,finished,progress):
        """Adopt an already launched collection or resume a finite child stage."""
        child_stop=(self.args.world_run/'stop_training')if name=='world'else(
            self.args.policy_run/'stop.request')if name=='policy'else None
        if(self.root/'stop.request').exists()or(child_stop is not None and child_stop.exists()):
            raise InterruptedError('Serial cycle stopped before launching its next child')
        if self.state.get('error'):
            self.state['previous_error']=dict(error=self.state.pop('error'),recovered_at=now())
        self.state['status']='running'
        if finished():
            progress();self.state['active_pid']=None;self.publish(mirror=True);return
        launch_path=directory/(name+'_launch.json')
        launch=read(launch_path)if launch_path.exists()else{}
        child=None;pid=launch.get('pid')
        if launch.get('process_identity')is not None and launch['process_identity'].get('pid')!=pid:
            raise ValueError('Child PID and host-bound launch receipt differ')
        owned=(process_identity_alive(launch.get('process_identity'),
            discard_foreign=getattr(self.args,'discard_foreign_pids',False))if pid is not None else False)
        if not owned and not finished():
            if launch_path.exists():
                archive=directory/'launch_history'/f'{time.time_ns()}_{launch_path.name}'
                archive.parent.mkdir(exist_ok=True)
                archive.write_bytes(launch_path.read_bytes())
            with(directory/(name+'.stdout.log')).open('ab')as out,(directory/(name+'.stderr.log')).open('ab')as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    **child_process_options())
            pid=child.pid;atomic(launch_path,dict(pid=pid,argv=command,started_at=now(),process_identity=process_identity(pid)))
        self.state.update(status='running',active_pid=pid);self.publish(mirror=name!='world')
        while alive(pid):
            progress();self.publish(mirror=name!='world')
            if(self.root/'stop.request').exists():
                flag=(self.args.world_run/'stop_training')if name=='world'else(
                    self.args.policy_run/'stop.request')if name=='policy'else None
                if flag is not None:flag.write_text('User requested stopping the serial cycle',encoding='utf-8')
            time.sleep(2)
        if child is not None:child.wait()
        progress();self.state['active_pid']=None;self.publish(mirror=True)
        if(self.root/'stop.request').exists():raise InterruptedError('Serial cycle stopped by user')
        if not finished():
            raise RuntimeError(f'{name} stage did not publish a complete handoff; inspect {directory/(name+".stderr.log")}')

    def collect(self,directory):
        assets=self.pin_assets(directory);output=directory/'source';receipt=output/'corpus_receipt.json'
        # The collector's canonical receipt filename is part of the stage API.
        def receipt_path():
            return next((path for path in(output/'collection_receipt.json',output/'corpus_receipt.json',output/'receipt.json')if path.exists()),None)
        def finished():return receipt_path()is not None
        def progress():
            path=output/'collection_progress.json'
            if path.exists():
                row=read(path);self.state.update(source_episodes=row.get('accepted',0),
                    source_transitions=row.get('accepted',0)*180,collection_progress=row)
        command=self.bootstrap('turkeyworld.collect_policy_motion',
            '--policy-checkpoint',assets/'policy.pt','--world-checkpoint',assets/'world.pt',
            '--output-dir',output,'--cycle-index',self.state['round_index'],
            '--episodes-per-stream',self.args.episodes_per_stream,'--transitions',180)
        self.execute(directory,'collection',command,finished=finished,progress=progress)
        path=receipt_path();value=read(path)
        if(value.get('split')!='train'or value.get('validation_unchanged')is not True
                or len(value.get('episodes',[]))!=2*self.args.episodes_per_stream
                or value.get('policy_checkpoint_sha256')!=sha(assets/'policy.pt')
                or value.get('world_checkpoint_sha256')!=sha(assets/'world.pt')
                or value.get('cycle_index')!=self.state['round_index']
                or value.get('transitions_per_episode')!=180
                or value.get('total_real_Source_transitions')!=2*self.args.episodes_per_stream*180):
            raise ValueError('Collected Source corpus is not the complete declared TRAIN handoff')
        for row in value['episodes']:
            if(row['split']!='train'or sha(row['path'])!=row['sha256']or row.get('transitions')!=180
                    or row.get('policy_checkpoint_sha256')!=value['policy_checkpoint_sha256']
                    or row.get('world_checkpoint_sha256')!=value['world_checkpoint_sha256']):
                raise ValueError('Collected Source data changed')
        for stream in('navigation','enemy_motion'):
            if sum(row.get('stream')==stream for row in value['episodes'])!=self.args.episodes_per_stream:
                raise ValueError('Collected Source data has incorrect motion cohort counts')
        self.state.update(source_receipt=identity_path(path),source_receipt_sha256=sha(path),
            source_episodes=len(value['episodes']),source_transitions=value['total_real_Source_transitions'])
        return path

    def world(self,directory):
        budget=self.world_budget();self.state['world_target_updates']=budget
        path=self.args.world_run/'status.json'
        def current():return read(path)
        def finished():
            row=current();r=row.get('rssm_training',{}).get('motion_only_revision',{})
            return(row.get('status')=='world_round_complete'and r.get('gate',{}).get('round_index')==self.state['round_index']
                and r.get('gate',{}).get('updates')==budget)
        def progress():
            row=current();r=row.get('rssm_training',{}).get('motion_only_revision',{})
            if r.get('gate',{}).get('round_index')==self.state['round_index']:
                self.state.update(world_updates=r['gate']['updates'],world_step=row.get('global_training_step'))
        command=self.bootstrap('turkeyworld.train_rssm_motion_only','--run-dir',self.args.world_run,
            '--source-checkpoint',self.args.initial_world,'--motion-plan',self.args.motion_plan,
            '--round-corpus',resolve_path(self.state['source_receipt']),'--cycle-status',self.status_path)
        if getattr(self.args,'discard_foreign_pids',False):command.append('--discard-foreign-pids')
        self.execute(directory,'world',command,finished=finished,progress=progress)
        final=current();self.state.update(world_step=final['global_training_step'],
            trained_world_sha256=sha(self.args.world_run/'checkpoints/rssm_world.pt'))
        atomic(directory/'world_complete.json',dict(step=self.state['world_step'],
            sha256=self.state['trained_world_sha256'],world_updates=budget,budget_is_quality_pass=False))

    def policy(self,directory):
        path=self.args.policy_run/'status.json'
        if 'policy_actor_target'not in self.state:
            current=read(path);start=current['actor_updates']
            self.state.update(policy_actor_start=start,policy_actor_target=start+self.args.policy_updates)
            self.publish(mirror=True)
        def finished():
            row=read(path)
            return(row.get('status')=='completed'and row.get('actor_updates',-1)>=self.state['policy_actor_target']
                and row.get('world_sha256')==self.state['trained_world_sha256'])
        def progress():
            row=read(path);self.state.update(policy_updates=max(0,row.get('actor_updates',0)-self.state['policy_actor_start']),
                policy_step=row.get('step'),policy_phase=row.get('phase'))
        command=self.bootstrap('turkeyworld.train_dreamer_policy','--run-dir',self.args.policy_run,
            '--world-source',self.args.world_run/'checkpoints/rssm_world.pt','--refresh-world-now',
            '--cycle-corpus',resolve_path(self.state['source_receipt']),
            '--until-actor-updates',self.state['policy_actor_target'])
        if getattr(self.args,'discard_foreign_pids',False):command.append('--discard-foreign-pids')
        self.execute(directory,'policy',command,finished=finished,progress=progress)
        atomic(directory/'policy_complete.json',dict(actor_updates=read(path)['actor_updates'],
            step=read(path)['step'],world_sha256=self.state['trained_world_sha256'],
            policy_checkpoint_sha256=sha(self.args.policy_run/'checkpoints/policy.pt')))

    def run(self):
        self.retire_initial_flags()
        start_round=self.state['round_index']
        while self.args.cycles==0 or self.state['round_index']-start_round<self.args.cycles:
            if(self.root/'stop.request').exists():raise InterruptedError('Serial cycle stopped by user')
            directory=self.root/f'cycles/cycle_{self.state["round_index"]:06d}';directory.mkdir(parents=True,exist_ok=True)
            phase=self.state['phase']
            if phase=='collect_source':
                self.collect(directory);self.state['phase']='world_training';self.publish(mirror=True)
            if self.state['phase']=='world_training':
                self.world(directory);self.state['phase']='policy_training';self.publish(mirror=True)
            if self.state['phase']=='policy_training':
                self.policy(directory)
                self.state.update(round_index=self.state['round_index']+1,
                    completed_rounds=self.state['completed_rounds']+1,phase='collect_source',
                    source_episodes=0,source_transitions=0,world_updates=0,policy_updates=0)
                for key in('policy_actor_start','policy_actor_target'):self.state.pop(key,None)
                self.publish(mirror=True)
        self.state['status']='completed';self.publish(mirror=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'training_runs/dreamer_motion_cycle_20261114')
    parser.add_argument('--world-run',type=Path,default=DEFAULT_WORLD)
    parser.add_argument('--policy-run',type=Path,default=DEFAULT_POLICY)
    parser.add_argument('--motion-plan',type=Path,default=DEFAULT_WORLD/'motion_only_plan.json')
    parser.add_argument('--initial-world',type=Path)
    parser.add_argument('--episodes-per-stream',type=int,default=16)
    parser.add_argument('--policy-updates',type=int,default=50)
    parser.add_argument('--cycles',type=int,default=0)
    parser.add_argument('--resume-after-user-stop',action='store_true',help='Explicitly authorize clearing saved stop flags and resuming the same stage')
    parser.add_argument('--discard-foreign-pids',action='store_true',help='Explicit migration only: ignore foreign/legacy PID receipts after confirming old workers are stopped')
    args=parser.parse_args(argv)
    if min(args.episodes_per_stream,args.policy_updates)<1 or args.cycles<0:parser.error('Invalid cycle budgets')
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,resolve_path(value))
    if args.initial_world is None:args.initial_world=resolve_path(read(args.world_run/'motion_only_source.json')['source_checkpoint'])
    args.run_dir.mkdir(parents=True,exist_ok=True)
    pid=args.run_dir/'cycle.pid'
    if pid_file_alive(pid,discard_foreign=args.discard_foreign_pids):parser.error('Serial cycle controller already active')
    write_pid_file(pid)
    cycle=MotionCycle(args)
    try:
        with KeepAwake():cycle.run()
    except BaseException as error:
        cycle.state.update(status='stopped'if isinstance(error,InterruptedError)else'failed',
            error=f'{type(error).__name__}: {error}',active_pid=None)
        cycle.publish(mirror=True)
        if not isinstance(error,InterruptedError):raise


if __name__=='__main__':main()
