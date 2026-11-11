"""Ordered tiny-enemy adaptation: fixed-Q codec, then SR, then Base/RF1."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .distill_renderer import _digest
from .power_guard import KeepAwake
from .small_enemy_validation import relative_small_enemy_gate
from .training_pipeline import atomic_json,now
from .vqgan_pipeline import process_alive


ORDER=('codec','sr','renderer')


def stage_can_advance(stage,report,baseline):
    if stage not in ORDER:return False
    names={'codec':('source_reconstruction',),'sr':('clean','codec'),'renderer':('base','rf1')}[stage]
    checked=relative_small_enemy_gate(report,baseline,predictions=names,
        compatibility=('full_rf1',)if stage=='codec'else())
    return bool(report.get('small_enemy_stage_gate',{}).get('passed')is True and checked['passed'])


def next_stage(completed):
    if list(completed)!=list(ORDER[:len(completed)]):raise ValueError('Ordered stages cannot be skipped')
    return ORDER[len(completed)]if len(completed)<len(ORDER)else None


def immutable_json(path,value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError('Immutable pipeline plan changed: '+str(path))
    else:atomic_json(path,value)


def verified_live_child(path,argv):
    if not path.exists():return None
    launch=json.loads(path.read_text(encoding='utf-8'));pid=launch.get('pid')
    if not isinstance(pid,int)or not process_alive(pid):return None
    if launch.get('argv')!=argv:raise ValueError('A live child belongs to a different immutable stage command')
    try:
        import psutil
        actual=psutil.Process(pid).cmdline()
    except Exception as error:
        raise RuntimeError('Cannot verify live ordered-stage process; refusing a duplicate worker')from error
    if actual[1:]!=argv[1:]:raise ValueError('Live PID command differs; refusing duplicate or unrelated process adoption')
    return pid


def milestone_complete(stage,run,budget):
    status=run/'status.json';launch=run/'launch.json'
    if not status.exists():return False
    value=json.loads(status.read_text(encoding='utf-8'))
    current=value.get('current')
    required=budget+(8000 if stage=='renderer'else 0)
    if (value.get('status')!='completed'or not isinstance(current,int)or isinstance(current,bool)or current<required):
        return False
    if launch.exists():
        pid=json.loads(launch.read_text(encoding='utf-8')).get('pid')
        if isinstance(pid,int)and process_alive(pid):return False
    return True


def validate_stage_artifact(stage,run,report):
    import torch
    names={'codec':'codec_student.pt','sr':'sr_for_game.pt','renderer':'renderer_student.pt'}
    path=run/'checkpoints'/names[stage]
    if not path.is_file():raise ValueError('Completed stage is missing its selected artifact: '+str(path))
    saved=torch.load(path,map_location='cpu',weights_only=False)
    expected={'codec':'distilled_vqgan_codec','sr':'sr','renderer':'renderer_student'}[stage]
    if saved.get('stage')!=expected or not isinstance(saved.get('training_steps'),int)or saved['training_steps']<1:
        raise ValueError('Selected stage artifact has invalid training provenance')
    embedded=(saved.get('quality_gate',{})if stage=='codec'else saved.get('validation',{})if stage=='sr'
        else saved.get('metrics',{}).get('source_validation',{}))
    if embedded.get('step')!=report.get('step')or embedded.get('panel_fingerprint')!=report.get('panel_fingerprint'):
        raise ValueError('Selected artifact and held-out report do not describe the same checkpoint/panel')
    if (stage=='codec'and(saved.get('frozen')is not True or saved.get('fixed_codebook')is not True)):
        raise ValueError('Selected codec must retain a frozen initialized codebook')
    if stage=='renderer'and(saved.get('solver')!='euler'or saved.get('solver_steps')!=1):
        raise ValueError('Keep the one-Euler compact neural renderer')
    return {'path':str(path),'sha256':_digest(path),'training_steps':saved['training_steps']}


class SmallEnemyPipeline:
    def __init__(self,args):
        self.args=args;self.root=args.run_dir.resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.state={'run_id':self.root.name,'stage':'ordered_small_enemy_pipeline','status':'preparing',
            'order':list(ORDER),'completed_stages':[],'started_at':now(),'automatic_deployment':False,
            'gameplay_qualified':False,'source_quality_passed':False}
        self.publish()

    def publish(self,**values):
        self.state.update(values,updated_at=now());atomic_json(self.root/'status.json',self.state)

    def assets(self):
        args=self.args;root=self.root/'assets';root.mkdir(exist_ok=True)
        entries={'world_combat.pt':args.world,'renderer_parent_combat.pt':args.renderer_parent,
            'renderer_parent_navigation.pt':args.renderer_navigation_parent,'sr_parent.pt':args.sr_parent,
            'codec_parent.pt':args.codec_parent}
        identities={}
        for name,source in entries.items():
            destination=root/name;expected=_digest(source)
            if not destination.exists():shutil.copy2(source,destination)
            if _digest(destination)!=expected:raise ValueError('Pipeline parent asset changed: '+name)
            identities[name]={'path':str(destination),'sha256':expected,'original':str(source.resolve())}
        nav=root/'world_navigation.pt'
        if not nav.exists():
            import torch
            saved=torch.load(root/'world_combat.pt',map_location='cpu',weights_only=False)
            if saved['config']['navigation_only']is not False:raise ValueError('Expected explicit combat-profile source')
            saved['config']={**saved['config'],'navigation_only':True}
            saved['small_enemy_training_view']={'profile_only':True,'all_world_tensors_unchanged':True}
            torch.save(saved,nav);del saved
        identities['world_navigation.pt']={'path':str(nav),'sha256':_digest(nav),'config_change_only':'navigation_only=True'}
        immutable_json(self.root/'parent_assets.json',identities)
        return root

    def command(self,stage,budget,assets):
        args=self.args;run=self.root/stage
        common=[sys.executable,'-X','utf8','-m']
        if stage=='codec':
            return common+['turkeyworld.train_codec_enemy','--run-dir',str(run),'--world',str(assets/'world_combat.pt'),
                '--renderer-checkpoint',str(assets/'renderer_parent_combat.pt'),'--data-root',str(args.data_root),
                '--empty-data-root',str(args.data_root),'--small-data-root',str(args.data_root),
                '--object-crop-weight','.1','--steps',str(budget),'--batch','8','--validation-frames','512',
                '--validation-batch','8','--validate-every','100','--preview-every','100','--threads','2','--device',args.device]
        if stage=='sr':
            return common+['turkeyworld.train_small_enemy_sr','--run-dir',str(run),'--data-root',str(args.data_root),
                '--parent',str(assets/'sr_parent.pt'),'--codec-checkpoint',str(self.root/'codec/checkpoints/codec_student.pt'),
                '--steps',str(budget),'--batch','8','--validation-frames','512','--device',args.device]
        from .small_enemy_renderer_init import prepare_small_renderer_initial
        initial=prepare_small_renderer_initial(assets,self.root/'codec/checkpoints/codec_student.pt',
            self.root/'renderer_initial',args.renderer_adam_parent)
        return common+['turkeyworld.train_renderer_enemy','--run-dir',str(run),'--world',str(assets/'world_navigation.pt'),
            '--renderer-checkpoint',str(initial),'--data-root',str(args.data_root),'--empty-data-root',str(args.data_root),
            '--small-data-root',str(args.data_root),'--parent-adam-checkpoint',str(self.root/'renderer_initial/renderer_adam_parent.pt'),
            '--steps',str(8000+budget),'--batch','16','--validation-frames','512','--validation-batch','8',
            '--validate-every','100','--preview-every','100','--save-every','10','--threads','2',
            '--base-regression-weight','1','--source-latent-weight','.1','--allow-unqualified-codec','--device',args.device,
            '--preserve-distance-panel',str(Path(__file__).resolve().parents[1]/'training_runs/renderer_distance_refine_20261111/source_distance_panel.json'),
            '--preserve-distance-baseline',str(Path(__file__).resolve().parents[1]/'training_runs/renderer_distance_refine_20261111/training_rebalance_only/validation/step_008000.json'),
            '--frozen-sr-checkpoint',str(self.root/'sr/checkpoints/sr_for_game.pt')]

    def run(self):
        args=self.args
        if(self.root/'stop_pipeline').exists():self.publish(status='stopped');return
        assets=self.assets()
        plan={'version':1,'order':list(ORDER),'milestone_updates':2000,'extension_updates':1000,
            'advance_only_on_relative_heldout_gate':True,'data_root':str(args.data_root.resolve()),
            'assets':json.loads((self.root/'parent_assets.json').read_text(encoding='utf-8')),
            'renderer_Adam_parent_sha256':_digest(args.renderer_adam_parent),
            'automatic_deployment':False,'native_size':32,'paired_hr_size':128,
            'codec_Q_and_latent_norm_fixed':True,'codec_SR_frozen_during_renderer':True}
        immutable_json(self.root/'ordered_plan.json',plan)
        while True:
            if(self.root/'stop_pipeline').exists():self.publish(status='stopped');return
            manifest=args.data_root/'manifest.json'
            if manifest.exists():
                data=json.loads(manifest.read_text(encoding='utf-8'))
                if data.get('status')=='complete':break
                if data.get('status')=='failed':raise RuntimeError('Source tiny-enemy collection failed')
            self.publish(status='waiting_for_source_data',current_stage='data')
            time.sleep(2)
        immutable_json(self.root/'data_binding.json',{'manifest_sha256':_digest(manifest),'fingerprint':data['fingerprint']})
        completed=[]
        for stage in ORDER:
            if next_stage(completed)!=stage:raise AssertionError('Stage order changed')
            budget=2000;run=self.root/stage;run.mkdir(exist_ok=True)
            while True:
                if _digest(manifest)!=json.loads((self.root/'data_binding.json').read_text())['manifest_sha256']:
                    raise ValueError('Source data manifest changed')
                selected=run/'validation_selected.json';baseline=run/'validation_initial.json'
                ledger=run/'milestone.json'
                if ledger.exists():budget=json.loads(ledger.read_text())['budget']
                if selected.exists()and baseline.exists():
                    report=json.loads(selected.read_text(encoding='utf-8'));old=json.loads(baseline.read_text(encoding='utf-8'))
                    if milestone_complete(stage,run,budget)and stage_can_advance(stage,report,old):
                        artifact=validate_stage_artifact(stage,run,report);break
                argv=self.command(stage,budget,assets)
                if(self.root/'stop_pipeline').exists():self.publish(status='stopped');return
                logs=self.root/'logs';logs.mkdir(exist_ok=True)
                adopted=verified_live_child(run/'launch.json',argv)
                process=None
                if adopted is None:
                    with(logs/f'{stage}_{budget}.stdout.log').open('a',encoding='utf-8')as stdout,(logs/f'{stage}_{budget}.stderr.log').open('a',encoding='utf-8')as stderr:
                        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt'else 0
                        process=subprocess.Popen(argv,cwd=Path(__file__).resolve().parents[1],stdout=stdout,stderr=stderr,creationflags=flags)
                    adopted=process.pid
                    atomic_json(run/'launch.json',{'pid':adopted,'argv':argv,'budget':budget})
                self.publish(status='training',current_stage=stage,child_pid=adopted,milestone=budget,completed_stages=completed,
                    adopted_existing_child=process is None)
                atomic_json(self.root/'active_run.json',{'run_dir':str(run),'scope':'monitoring_only; no game asset changes'})
                while (process.poll()is None if process is not None else process_alive(adopted)):
                    if(self.root/'stop_pipeline').exists():
                        (run/('stop_requested'if stage=='codec'else'stop_training')).touch()
                    time.sleep(2)
                if(self.root/'stop_pipeline').exists():self.publish(status='stopped');return
                if process is not None and process.returncode:raise RuntimeError(f'{stage} worker exited{process.returncode}; see logs')
                if process is None:
                    status=json.loads((run/'status.json').read_text(encoding='utf-8'))
                    if status.get('status')!='completed':raise RuntimeError(f'Adopted{stage} worker did not finish its milestone')
                if not milestone_complete(stage,run,budget):
                    raise RuntimeError(f'{stage} exited without durable completion of its {budget}-update milestone')
                if not selected.exists():raise RuntimeError(stage+' produced no held-out selected report')
                report=json.loads(selected.read_text(encoding='utf-8'));old=json.loads(baseline.read_text(encoding='utf-8'))
                if stage_can_advance(stage,report,old):
                    artifact=validate_stage_artifact(stage,run,report);break
                immutable_json(run/f'extension_{budget:06d}_{budget+1000:06d}.json',{
                    'from_budget':budget,'to_budget':budget+1000,'selected_report_sha256':_digest(selected),
                    'reason':'Relative held-out small-enemy criterion not met; do not advance the ordered pipeline'})
                budget+=1000;atomic_json(ledger,{'budget':budget})
            completed.append(stage)
            atomic_json(run/'ordered_stage_pass.json',{'stage':stage,'source_quality_passed':report.get('source_quality_passed',False),
                'relative_small_enemy_gate_passed':True,'selected_report_sha256':_digest(selected),
                'artifact':artifact,'completed_milestone':budget,'gameplay_qualified':False})
            self.publish(completed_stages=list(completed))
        self.publish(status='completed',current_stage='completed',completed_stages=completed,
            message='All three relative adaptation gates passed; no automatic deployment or gameplay qualification')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','data-root','world','renderer-parent','renderer-navigation-parent','codec-parent','sr-parent','renderer-adam-parent'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda');args=parser.parse_args(argv)
    args.run_dir.mkdir(parents=True,exist_ok=True);pid=args.run_dir/'pipeline.pid'
    if pid.exists()and process_alive(int(pid.read_text())):parser.error('Ordered pipeline is already running')
    atomic_json(pid,os.getpid());pipeline=SmallEnemyPipeline(args)
    try:
        with KeepAwake():pipeline.run()
    except BaseException as error:
        pipeline.publish(status='failed',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
