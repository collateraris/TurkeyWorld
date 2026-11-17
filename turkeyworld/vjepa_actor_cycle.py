"""Frozen restored V-JEPA: actor/critic imagination -> genuine Source -> repeat."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from .training_pipeline import atomic_json,now
from .power_guard import KeepAwake
from .process_runtime import pid_file_alive,write_pid_file


def run(a):
    root=a.run_dir.resolve();root.mkdir(parents=True,exist_ok=True);policy=root/'policy';world=a.world.resolve()
    plan=dict(protocol='vjepa_actor_Source_cycle_v1',world=str(world),world_updates=False,
        actor_updates_per_round=50,episodes_per_stream=64,transitions=360,head_warmup=20,
        actor_features='state20_plus_VJEPA_latent_features128',imagination_horizon=15,
        previous_RSSM_policy_reused=False,Source_future_in_imagination=False)
    path=root/'cycle_plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('V-JEPA cycle plan changed')
    atomic_json(path,plan);status_path=root/'status.json'
    state=json.loads(status_path.read_text()) if status_path.exists() else dict(round_index=0,phase='policy_training')
    def publish(**kw):state.update(protocol=plan['protocol'],status='running',updated_at=now(),**kw);atomic_json(status_path,state)
    def child(module,args,logname):
        with (root/(logname+'.stdout.log')).open('ab') as out,(root/(logname+'.stderr.log')).open('ab') as err:
            proc=subprocess.Popen([sys.executable,'-m',module,*map(str,args)],stdout=out,stderr=err)
            publish(active_pid=proc.pid)
            while proc.poll() is None:
                if (root/'stop.request').exists():
                    flag=policy/'stop.request' if module.endswith('train_dreamer_policy') else root/'cycles'/f"cycle_{state['round_index']:06d}"/'source/stop.request'
                    flag.parent.mkdir(parents=True,exist_ok=True);flag.write_text('User requested stopping V-JEPA actor cycle\n')
                time.sleep(1)
            if (root/'stop.request').exists():raise InterruptedError('V-JEPA actor cycle stopped')
            if proc.returncode:raise RuntimeError(f'{module} exited {proc.returncode}; inspect {logname}.stderr.log')
    while not (root/'stop.request').exists():
        i=state['round_index'];directory=root/'cycles'/f'cycle_{i:06d}';directory.mkdir(parents=True,exist_ok=True)
        receipt=directory/'source/collection_receipt.json'
        if state['phase']=='policy_training':
            args=['--run-dir',policy,'--world-source',world,'--horizon',15,'--replay-horizon',8,'--per-stream',2,
                '--warmup',20,'--validate-every',50,'--validation-horizon',16,'--save-every',10,'--until-actor-updates',(i+1)*50]
            if i:args+=['--cycle-corpus',root/'cycles'/f'cycle_{i-1:06d}'/'source/collection_receipt.json']
            publish(phase='policy_training',actor_target=(i+1)*50);child('turkeyworld.train_dreamer_policy',args,f'policy_{i:06d}')
            state['phase']='collect_source';publish()
        if not receipt.exists():
            publish(phase='collect_source')
            child('turkeyworld.collect_policy_motion',['--policy-checkpoint',policy/'checkpoints/policy.pt','--world-checkpoint',world,
                '--output-dir',directory/'source','--cycle-index',i,'--episodes-per-stream',64,'--transitions',360,
                '--target-hard-cases','--max-attempts',64,'--inference-batch-size',8],f'source_{i:06d}')
        publish(round_index=i+1,phase='policy_training',last_source_receipt=str(receipt))
    raise InterruptedError('V-JEPA actor cycle stopped')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--world',type=Path,required=True);a=p.parse_args()
    a.run_dir.mkdir(parents=True,exist_ok=True)
    if pid_file_alive(a.run_dir/'cycle.pid'):raise ValueError('V-JEPA actor cycle is already running')
    write_pid_file(a.run_dir/'cycle.pid')
    try:
        with KeepAwake():run(a)
    except BaseException as e:
        status=a.run_dir/'status.json';v=json.loads(status.read_text()) if status.exists() else {}
        atomic_json(status,{**v,'status':'stopped' if isinstance(e,InterruptedError) else 'failed','error':str(e),'active_pid':None,'updated_at':now()})
        if not isinstance(e,InterruptedError):raise

if __name__=='__main__':main()
