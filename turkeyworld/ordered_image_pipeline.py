"""Base -> RF3 -> VQGAN decoder -> SR, exactly3000 production updates each."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .power_guard import KeepAwake
from .training_pipeline import atomic_json,now
from .vqgan_pipeline import process_alive

PROTOCOL='ordered_image_training_v1'
ORDER=('base','rf3','decoder','sr')
LABELS=dict(base='Base',rf3='RF3',decoder='Декодер VQGAN',sr='Апскейлер')
BUDGET=3000
OUTPUTS={
    'base':('base_last.pt','base_best.pt','base_best_for_preview.pt'),
    'rf3':('renderer_last.pt','renderer_best.pt','renderer_best_for_preview.pt'),
    'decoder':('decoder_last.pt','decoder_best.pt','decoder_best_for_preview.pt'),
    'sr':('sr_last.pt','sr_best.pt','sr_for_game.pt'),
}


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    try:
        value=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return value if isinstance(value,dict)else {}
    except(OSError,ValueError):return {}


def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Immutable image-pipeline plan changed: '+str(path))
    else:atomic_json(path,value)


def pin_file(source,destination):
    source,destination=Path(source).resolve(strict=True),Path(destination).resolve()
    value=sha(source);destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        try:os.link(source,destination)
        except OSError:shutil.copy2(source,destination)
    if sha(destination)!=value:raise ValueError('Pinned image-pipeline asset changed: '+str(destination))
    return dict(path=str(destination),sha256=value,source=str(source))


def stage_complete(status,budget=BUDGET):
    return(status.get('status')=='completed'and type(status.get('current'))is int
           and status['current']==budget and status.get('total')==budget)


def validated_prefix(stages):
    completed=[key for key in ORDER if stages[key].get('status')=='completed']
    if completed!=list(ORDER[:len(completed)]):raise ValueError('Image stages cannot be skipped or reordered')
    return len(completed)


def verified_pid(launch,argv):
    pid=launch.get('pid')
    if type(pid)is not int or not process_alive(pid):return None
    if launch.get('argv')!=argv:raise ValueError('Live image-stage command changed')
    import psutil
    actual=psutil.Process(pid).cmdline()
    if actual[1:]!=argv[1:]:raise ValueError('Image-stage PID belongs to a different process')
    return pid


class MonitorMirror:
    """Mirror one stage for the already-running read-only HTTP server."""
    def __init__(self,root):self.root=Path(root);self.source=None;self.offset=0;self.tail=b'';self.preview_stamp=None

    def update(self,source):
        source=Path(source).resolve()
        if self.source!=source:
            self.source=source;self.offset=0;self.tail=b'';self.preview_stamp=None
            (self.root/'metrics.jsonl').write_bytes(b'')
            for name in('preview.png','preview.json'):(self.root/name).unlink(missing_ok=True)
        metrics=source/'metrics.jsonl'
        if metrics.exists():
            if metrics.stat().st_size<self.offset:
                self.offset=0;self.tail=b'';(self.root/'metrics.jsonl').write_bytes(b'')
            with metrics.open('rb')as stream:
                stream.seek(self.offset);chunk=stream.read(8*1024*1024);self.offset=stream.tell()
            chunk=self.tail+chunk;end=chunk.rfind(b'\n')
            if end>=0:
                with(self.root/'metrics.jsonl').open('ab')as stream:stream.write(chunk[:end+1])
                self.tail=chunk[end+1:]
            else:self.tail=chunk
        image,meta=source/'preview.png',source/'preview.json'
        if image.exists()and meta.exists():
            stamp=(image.stat().st_mtime_ns,meta.stat().st_mtime_ns)
            if stamp[0]<=stamp[1]and stamp!=self.preview_stamp:
                header=read(meta);pixels=image.read_bytes()
                if header and read(meta)==header and stamp==(image.stat().st_mtime_ns,meta.stat().st_mtime_ns):
                    temporary=self.root/'preview.tmp.png';temporary.write_bytes(pixels);temporary.replace(self.root/'preview.png')
                    atomic_json(self.root/'preview.json',header);self.preview_stamp=stamp
        header=read(self.root/'preview.json')
        if header and(self.root/'preview.png').exists():
            return dict(step=header.get('step'),sha256=sha(self.root/'preview.png'),metadata=header)
        return None


class OrderedImagePipeline:
    def __init__(self,args):
        self.args=args;self.root=args.run_dir.resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.assets={}
        for key,name in(('world','world.pt'),('source_recipe_parent','source_recipe_parent.pt'),
                         ('decoder_parent','decoder_parent.pt'),('sr_parent','sr_parent.pt')):
            self.assets[key]=pin_file(getattr(args,key),self.root/'assets'/name)
        base_plan=args.base_run/'budget_3000_plan.json'
        if not base_plan.is_file():raise ValueError('Base must already have its explicit TOTAL3000 revision')
        self.runs={key:(args.base_run if key=='base'else self.root/key).resolve()for key in ORDER}
        for run in self.runs.values():run.mkdir(parents=True,exist_ok=True)
        self.plan=dict(protocol=PROTOCOL,version=1,order=list(ORDER),updates_per_stage=BUDGET,total_updates=12000,
            base_updates_include_already_completed=True,other_stage_updates_are_new=True,
            assets=self.assets,base_run=str(args.base_run),base_budget_plan_sha256=sha(base_plan),
            run_directories={k:str(v)for k,v in self.runs.items()},sr_source_root=str(args.sr_source_root),
            monitor_control_dir=str(args.monitor_control_dir),device=args.device,
            solver='euler',solver_steps=3,restorer_active=False,automatic_game_promotion=False,
            best_selection='whole_best_validation_checkpoint_with_matching_export',
            quality_thresholds_do_not_extend_fixed_budgets=True,preflight_updates_discarded=3,
            preview_every=dict(base=10,rf3=10,decoder=10,sr=15),validation_every=100)
        immutable_json(self.root/'ordered_plan.json',self.plan);self.plan_sha=sha(self.root/'ordered_plan.json')
        self.state=read(self.root/'supervisor_state.json')or dict(protocol=PROTOCOL,plan_sha256=self.plan_sha,
            started_at=now(),status='running',stages={k:dict(key=k,label=LABELS[k],status='queued',current=0,budget=BUDGET,
            run_dir=str(self.runs[k]))for k in ORDER},events=[])
        if self.state.get('plan_sha256')!=self.plan_sha:raise ValueError('Image-pipeline resume changed its plan')
        validated_prefix(self.state['stages']);self.mirror=MonitorMirror(self.root)

    def store(self):self.state['updated_at']=now();atomic_json(self.root/'supervisor_state.json',self.state)

    def event(self,message):self.state['events']=(self.state['events']+[dict(timestamp=now(),message=message)])[-50:];self.store()

    def handoff(self,key):
        value=self.state['stages'][key].get('handoff')
        if not isinstance(value,dict):raise ValueError('Missing previous image-stage handoff: '+key)
        for entry in value['files'].values():
            if sha(entry['path'])!=entry['sha256']:raise ValueError('Pinned best-stage artifact changed')
        return value['files']

    def command(self,key,*,probe=False):
        run=self.root/(key+'_preflight')if probe else self.runs[key]
        common=[sys.executable,'-X','utf8','-u','-m']
        asset=lambda key:self.assets[key]['path']
        if key=='base':return read(self.runs[key]/'launch_command.json')['argv']
        parent=self.handoff(ORDER[ORDER.index(key)-1])
        if key=='rf3':
            argv=common+['turkeyworld.train_renderer_rf_chain','--run-dir',str(run),'--world',asset('world'),
                '--parent-checkpoint',parent['training']['path'],'--renderer-checkpoint',parent['renderer']['path'],
                '--source-recipe-parent',asset('source_recipe_parent'),'--steps','3000','--threads','2','--seed','173','--device',self.args.device]
        elif key=='decoder':
            argv=common+['turkeyworld.train_decoder_rf_chain','--run-dir',str(run),'--world',asset('world'),
                '--generator-checkpoint',parent['renderer']['path'],'--generator-full-checkpoint',parent['training']['path'],
                '--decoder-parent',asset('decoder_parent'),'--source-recipe-parent',asset('source_recipe_parent'),
                '--steps','3000','--batch','8','--lr','0.0001','--device',self.args.device]
        else:
            argv=common+['turkeyworld.train_final_renderer_sr','--run-dir',str(run),
                '--parent-checkpoint',asset('sr_parent'),'--world',asset('world'),'--renderer',parent['renderer']['path'],
                '--source-root',str(self.args.sr_source_root),'--steps','3000','--cache-batch','16','--device',self.args.device]
        return argv+(['--probe']if probe else[])

    def publish(self,key,*,phase='training',error=None):
        child=read(self.runs[key]/'status.json')
        preview=self.mirror.update(self.runs[key])
        record=self.state['stages'][key]
        current=child.get('current',0)
        if type(current)is not int or not 0<=current<=BUDGET:current=0
        if phase=='preflight':current=0
        if record.get('status')!='completed':
            state='preflight'if phase=='preflight'else child.get('status','initializing')
            # Completion is sealed only after the matching best weights/report
            # are pinned. A restart during handoff must retry that handoff.
            if state=='completed':state='finalizing'
            record.update(current=current,status=state)
        containers={'base':'base_geometry_training','rf3':'rf_chain_training',
                    'decoder':'decoder_rf_chain_training','sr':'sr_moe_training'}
        training=child.get(containers[key],{})
        preview_kinds={'base':'base_geometry_GT_query','rf3':'renderer_rf_chain_GT_query',
                       'decoder':'decoder_rf_chain_GT_query','sr':'sr_moe_five_preview_v1'}
        validation_kinds={'base':'base_geometry_validation_v1','rf3':'renderer_rf_codec_validation_v1',
                          'decoder':'decoder_rf_adaptation_validation_v1','sr':'sr_moe_five_validation_v1'}
        panel=child.get('validation_panel_fingerprint')or training.get('validation_panel_fingerprint')
        identity=dict(stage=child.get('stage'),loss_protocol=child.get('loss_protocol'),
            plan_sha256=training.get('plan_sha256'),
            world_sha256=training.get('feature_world_sha256')or training.get('world_sha256'),
            panel_fingerprint=panel,preview_kind=preview_kinds[key],validation_kind=validation_kinds[key],
            objective_protocol='renderer_rf_codec_geometry_v1'if key=='rf3'else child.get('loss_protocol'))
        rows=[deepcopy(self.state['stages'][name])for name in ORDER]
        q=dict(protocol=PROTOCOL,plan_sha256=self.plan_sha,status=self.state['status'],active_stage=key,
            active_phase=phase,stages=rows,updates_per_stage=BUDGET,total_updates=12000,
            completed_updates=sum(min(BUDGET,row.get('current',0))for row in rows),
            active_run_dir=str(self.runs[key]),preview_identity=preview,restorer_active=False,
            automatic_game_promotion=False,validation_every=100,
            preview_every=15 if key=='sr'else 10,
            training_component=LABELS[key],quality_thresholds_do_not_extend_fixed_budgets=True)
        q['active_child_identity']=identity
        result=deepcopy(child)
        result.update(run_id=self.root.name,active_run_id=child.get('run_id'),image_pipeline=q,
            active_stage_run_dir=str(self.runs[key]),current=current,total=BUDGET,
            stage_label=f'{ORDER.index(key)+1}/4 · {LABELS[key]} · '+('GPU preflight'if phase=='preflight'else child.get('stage_label','Обучение')),
            log=self.state['events'],updated_at=child.get('updated_at',now()))
        if panel:result['validation_panel_fingerprint']=panel
        if preview:result.update(prediction_preview=preview['metadata'],preview_url='/preview.png')
        else:result.pop('prediction_preview',None);result.pop('preview_url',None)
        if error:result.update(status='failed',error=str(error),message=str(error))
        elif self.state['status']in('completed','stopped'):result['status']=self.state['status']
        atomic_json(self.root/'status.json',result);self.store()

    def launch(self,key,*,probe=False):
        if any(sha(entry['path'])!=entry['sha256']for entry in self.assets.values()):
            raise ValueError('A pinned ordered image-pipeline input changed')
        argv=self.command(key,probe=probe)
        run=self.root/(key+'_preflight')if probe else self.runs[key];run.mkdir(parents=True,exist_ok=True)
        launch_path=run/('launch_command.json'if key!='base'else'launch_command.json')
        launch=read(launch_path)
        if launch:
            if launch.get('argv')!=argv:raise ValueError('Completed/live image-stage command changed')
            pid=verified_pid(launch,argv)
            if pid:return pid
            status=read(run/'status.json')
            if status.get('status')=='completed':return None
            raise RuntimeError('Image-stage worker stopped unexpectedly; refusing an automatic reset: '+key)
        if key=='base':raise ValueError('Adopt only the already-running revised Base worker')
        if(run/'stop_training').exists():raise RuntimeError('Image stage was explicitly stopped: '+key)
        with(run/'training.stdout.log').open('ab')as out,(run/'training.stderr.log').open('ab')as err:
            process=subprocess.Popen(argv,cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,
                stdout=out,stderr=err,close_fds=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        atomic_json(launch_path,dict(pid=process.pid,argv=argv,started_at=now(),pipeline_plan_sha256=self.plan_sha,
            discarded_preflight=probe,parent_artifacts=self.handoff(ORDER[ORDER.index(key)-1])))
        return process.pid

    def wait(self,key,pid,*,probe=False):
        run=self.root/(key+'_preflight')if probe else self.runs[key]
        while True:
            status=read(run/'status.json')
            if(self.root/'stop_pipeline').exists():
                if pid and process_alive(pid):atomic_json(run/'stop_training',dict(reason='ordered_image_pipeline_stop',time=now()))
                self.state['status']='stopped';self.publish(key,phase='preflight'if probe else'training');return False
            if status.get('status')=='failed':raise RuntimeError(key+': '+str(status.get('error','worker failed')))
            self.publish(key,phase='preflight'if probe else'training')
            alive=bool(pid and process_alive(pid))
            if not alive:
                if probe:
                    report=read(run/'gpu_warmstart.json')
                    if(status.get('status')!='completed'or report.get('passed')is not True
                            or report.get('discarded_optimizer_updates')!=3):
                        raise RuntimeError('Actual GPU preflight did not pass three discarded updates: '+key)
                    return True
                if stage_complete(status):return True
                if status.get('status')=='stopped':
                    self.state['status']='stopped';self.publish(key);return False
                raise RuntimeError('Worker exited without exactly3000 completed updates: '+key)
            time.sleep(self.args.poll_seconds)

    def seal(self,key):
        import torch
        if self.args.device=='cuda':torch.cuda.init()
        torch.set_num_threads(2)
        run=self.runs[key];last,best,bundle=OUTPUTS[key]
        final=torch.load(run/'checkpoints'/last,map_location='cpu',weights_only=False)
        if final.get('step')!=BUDGET:raise ValueError('Completed stage checkpoint is not update3000: '+key)
        selected=torch.load(run/'checkpoints'/best,map_location='cpu',weights_only=False)
        selected_step=selected.get('step')
        if type(selected_step)is not int or not 0<=selected_step<=BUDGET:raise ValueError('Invalid best whole-checkpoint step')
        best_meta=read(run/'best.json')
        if best_meta.get('step')!=selected_step:raise ValueError('Best checkpoint and best.json differ')
        export=torch.load(run/'checkpoints'/bundle,map_location='cpu',weights_only=False)
        if key!='sr':
            expected=dict(base=6,rf3=7,decoder=8)[key]
            if export.get('format_version')!=expected or export.get('local_updates')!=selected_step:
                raise ValueError('Best renderer export and full checkpoint differ: '+key)
            from .models import load_world_checkpoint
            from .renderer_student import load_renderer_student,_hash_tensors
            model,_=load_world_checkpoint(self.assets['world']['path'],'cpu')
            load_renderer_student(model,run/'checkpoints'/bundle,allow_unqualified_codec=True)
            if _hash_tensors(model.base_student.state_dict())!=_hash_tensors(selected['base']):
                raise ValueError('Selected renderer Base weights do not match full Adam checkpoint')
            if _hash_tensors(model.flow_renderer.state_dict())!=_hash_tensors(selected['flow']):
                raise ValueError('Selected renderer RF/codec weights do not match full Adam checkpoint')
            del model
        else:
            from .renderer_student import _hash_tensors
            from .models import load_sr_checkpoint
            if (not selected.get('model')or not export.get('model')
                    or type(selected.get('training_steps'))is not int
                    or selected['training_steps']!=export.get('training_steps')):
                raise ValueError('Selected SR export has incomplete weights or training provenance')
            loaded,_=load_sr_checkpoint(run/'checkpoints'/bundle,'cpu')
            if(_hash_tensors(loaded.state_dict())!=_hash_tensors(selected['model'])
                    or _hash_tensors(export['model'])!=_hash_tensors(selected['model'])):
                raise ValueError('Selected SR weights do not match their whole Adam checkpoint')
        report=run/'validation'/f'step_{selected_step:06d}.json'
        if not report.is_file():raise ValueError('Best selection is missing its exact validation report: '+key)
        if best_meta.get('report_sha256')not in(None,sha(report)):raise ValueError('Best validation SHA changed')
        root=self.root/'handoffs'/key
        files=dict(training=pin_file(run/'checkpoints'/best,root/'training.pt'),
            renderer=pin_file(run/'checkpoints'/bundle,root/('sr.pt'if key=='sr'else'renderer.pt')),
            validation=pin_file(report,root/'validation.json'))
        record=dict(protocol=PROTOCOL,stage=key,completed_updates=BUDGET,selected_step=selected_step,
            best=best_meta,files=files,whole_checkpoint_selected=True,automatic_game_promotion=False)
        immutable_json(root/'handoff.json',record)
        self.state['stages'][key].update(status='completed',current=BUDGET,handoff=record)
        self.store();return record

    def run(self):
        self.state['status']='running';self.store()
        atomic_json(self.args.monitor_control_dir/'active_run.json',dict(run_dir=str(self.root),
            reason='Ordered Base -> RF3 -> VQGAN decoder -> SR;3000 updates per stage',activated_at=now()))
        active=ORDER[min(validated_prefix(self.state['stages']),len(ORDER)-1)]
        try:
            for key in ORDER:
                active=key
                if self.state['stages'][key].get('status')=='completed':self.handoff(key);continue
                if key!='base'and not self.state['stages'][key].get('preflight_passed'):
                    self.event('GPU-проверка перед этапом '+LABELS[key]+'; тестовые обновления отбрасываются.')
                    pid=self.launch(key,probe=True)
                    if not self.wait(key,pid,probe=True):return
                    self.state['stages'][key]['preflight_passed']=True;self.store()
                self.event('Этап '+LABELS[key]+':3000 обновлений, передача лучших полных весов по валидации.')
                pid=self.launch(key)
                self.state['stages'][key]['pid']=pid;self.store()
                if not self.wait(key,pid):return
                self.seal(key);self.event('Завершён бюджет '+LABELS[key]+':3000; лучшие веса закреплены для следующего этапа.')
            self.state['status']='completed';self.event('Все четыре этапа завершили по3000 обновлений.');self.publish('sr')
        except BaseException as error:
            self.state['status']='failed';self.state['error']=f'{type(error).__name__}: {error}'
            self.publish(active,error=self.state['error']);raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('run-dir','base-run','world','source-recipe-parent','decoder-parent','sr-parent','sr-source-root','monitor-control-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--poll-seconds',type=float,default=2.)
    args=parser.parse_args(argv)
    if not 0<args.poll_seconds<=10:parser.error('Polling interval must be in(0,10]')
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    args.run_dir.mkdir(parents=True,exist_ok=True);pid_path=args.run_dir/'pipeline.pid'
    if pid_path.exists()and process_alive(int(pid_path.read_text())):parser.error('Ordered image supervisor is already running')
    pid_path.write_text(str(os.getpid()),encoding='ascii')
    with KeepAwake()as guard:
        power=dict(active=guard.active,system_required=guard.active,display_required=False,
            guard_pid=os.getpid(),scope=PROTOCOL,error=guard.error,updated_at=now())
        atomic_json(args.run_dir/'power_guard.json',power)
        try:OrderedImagePipeline(args).run()
        finally:atomic_json(args.run_dir/'power_guard.json',{**power,'active':False,'system_required':False,'updated_at':now()})


if __name__=='__main__':main()
