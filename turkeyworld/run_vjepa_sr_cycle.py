"""Source -> pooled V-JEPA/SR-feedback dynamics -> imagined actor/critic cycle."""
import argparse,json,subprocess,sys,time,os
from pathlib import Path
from .training_pipeline import atomic_json,now
from .process_runtime import process_identity,process_identity_alive,pid_file_alive,write_pid_file
from .power_guard import KeepAwake


def run(a):
    root=a.run_dir.absolute();world=root/'dynamics/checkpoints/vjepa_world.pt';policy=root/'policy'
    plan=dict(protocol='vjepa_SR_world_actor_Source_cycle_v1',world_updates_per_round=100,horizon=180,
        main_actors=16,actor_updates_per_round=50,episodes_per_stream=64,transitions=360,
        order=['world_training','policy_training','collect_source'],feedback_every=8,feedback_blend=.1)
    path=root/'cycle_plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('V-JEPA SR cycle plan changed')
    atomic_json(path,plan);sp=root/'status.json';state=json.loads(sp.read_text()) if sp.exists() else dict(round_index=0,phase='world_training')
    def publish(**kw):
        if state.get('error'):state['previous_error']=state.pop('error')
        state.update(status='running',protocol=plan['protocol'],updated_at=now(),**kw);atomic_json(sp,state)
    def execute(module,args,name,stop):
        launch=root/(name+'_launch.json');v=json.loads(launch.read_text()) if launch.exists() else {}
        adopted=process_identity_alive(v.get('process_identity')) if v.get('process_identity') else False
        proc=None
        if not adopted:
            with (root/(name+'.stdout.log')).open('ab') as out,(root/(name+'.stderr.log')).open('ab') as err:
                proc=subprocess.Popen([sys.executable,'-m',module,*map(str,args)],stdout=out,stderr=err)
            v=dict(pid=proc.pid,process_identity=process_identity(proc.pid),argv=[module,*map(str,args)]);atomic_json(launch,v)
        publish(active_pid=v['pid'])
        while process_identity_alive(v['process_identity']):
            if (root/'stop.request').exists():stop.parent.mkdir(parents=True,exist_ok=True);stop.write_text('User requested stopping V-JEPA SR cycle\n')
            time.sleep(1)
        if proc is not None:proc.wait()
        if (root/'stop.request').exists():raise InterruptedError('V-JEPA SR cycle stopped')
        if proc is not None and proc.returncode:raise RuntimeError(name+' failed; inspect stderr log')
    while not (root/'stop.request').exists():
        i=state['round_index'];directory=root/'cycles'/f'cycle_{i:06d}';directory.mkdir(parents=True,exist_ok=True)
        if state['phase']=='world_training':
            targeted=directory/'targeted_v2'
            if not (targeted/'collection_receipt.json').exists():
                publish(phase='world_training',world_target=(i+1)*100)
                execute('turkeyworld.collect_targeted_motion',['--output-dir',targeted,'--cycle-index',i],f'targeted_{i:06d}',targeted/'stop.request')
            publish(phase='world_training',world_target=(i+1)*100)
            selection=root/'dynamics/wall_selection.json'
            wall=json.loads(selection.read_text()).get('wall_coefficient',.25) if selection.exists() else .25
            execute('turkeyworld.train_vjepa_sr_dynamics',['--run-dir',root/'dynamics','--until-updates',(i+1)*100,'--wall-weight',wall],f'world_{i:06d}',root/'dynamics/stop_training')
            v=json.loads((root/'dynamics/status.json').read_text())
            if v.get('world_updates',0)<(i+1)*100:raise RuntimeError('World did not complete its optimizer budget')
            publish(phase='policy_training')
        if state['phase']=='policy_training':
            args=['--run-dir',policy,'--world-source',world,'--horizon',15,'--replay-horizon',8,'--per-stream',2,
                '--warmup',20,'--validate-every',50,'--validation-horizon',16,'--save-every',10,'--until-actor-updates',(i+1)*50,'--refresh-world-now']
            if i:args+=['--cycle-corpus',root/'cycles'/f'cycle_{i-1:06d}'/'source/collection_receipt.json']
            publish(phase='policy_training',actor_target=(i+1)*50)
            execute('turkeyworld.train_dreamer_policy',args,f'policy_{i:06d}',policy/'stop.request')
            publish(phase='collect_source')
        receipt=directory/'source/collection_receipt.json'
        if not receipt.exists():
            execute('turkeyworld.collect_policy_motion',['--policy-checkpoint',policy/'checkpoints/policy.pt','--world-checkpoint',world,
                '--output-dir',directory/'source','--cycle-index',i,'--episodes-per-stream',64,'--transitions',360,
                '--target-hard-cases','--max-attempts',64,'--inference-batch-size',8],f'source_{i:06d}',directory/'source/stop.request')
        publish(round_index=i+1,phase='world_training',last_source_receipt=str(receipt))
    raise InterruptedError('V-JEPA SR cycle stopped')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();a.run_dir.mkdir(parents=True,exist_ok=True)
    if pid_file_alive(a.run_dir/'cycle.pid'):raise ValueError('V-JEPA SR cycle already active')
    write_pid_file(a.run_dir/'cycle.pid')
    try:
        with KeepAwake():run(a)
    except BaseException as e:
        path=a.run_dir/'status.json';s=json.loads(path.read_text()) if path.exists() else {}
        atomic_json(path,{**s,'status':'stopped' if isinstance(e,InterruptedError) else 'failed','error':str(e),'active_pid':None,'updated_at':now()})
        if not isinstance(e,InterruptedError):raise

if __name__=='__main__':main()
