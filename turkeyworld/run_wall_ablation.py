"""Matched full-H180 wall-weight pilot; promote a measured branch and resume cycle."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import torch
from .training_pipeline import atomic_json,now
from .process_runtime import pid_file_alive,write_pid_file


def decide(reports):
    baseline,candidate=reports['025'],reports['050']
    streams=('navigation','enemy_motion')
    loss=lambda report:sum(report[s]['objective_mean']['weighted_motion_loss'] for s in streams)
    walls=lambda report:sum(report[s]['player_wall_overlap_ticks_mean'] for s in streams)
    drift_ok=all(candidate[s][key]<=baseline[s][key]*1.05+1e-6 for s,key in [('navigation','player_XY_H180_mean_tiles'),('enemy_motion','player_XY_H180_mean_tiles'),('enemy_motion','enemy_XY_H180_mean_tiles')])
    improved_walls=walls(candidate)<walls(baseline)*.95
    loss_ok=loss(candidate)<=loss(baseline)*1.02
    return dict(selected='050' if drift_ok and improved_walls and loss_ok else '025',candidate_drift_within5pct=drift_ok,candidate_wall_ticks_improved5pct=improved_walls,candidate_common_loss_within2pct=loss_ok,common_loss_025=loss(baseline),common_loss_050=loss(candidate),wall_ticks_025=walls(baseline),wall_ticks_050=walls(candidate),pilot_only=True)


def run(root,experiment,updates):
    root=Path(root).absolute();experiment=Path(experiment).absolute();experiment.mkdir(parents=True,exist_ok=True)
    if pid_file_alive(experiment/'worker.pid'):raise ValueError('Wall pilot already running')
    write_pid_file(experiment/'worker.pid')
    dynamics=root/'dynamics';source=experiment/'initial.pt'
    if not source.exists():os.link(dynamics/'checkpoints/vjepa_world.pt',source)
    saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True);start_updates=saved['vjepa_sr_training']['updates'];start_step=saved['training_steps'];del saved
    plan=dict(protocol='matched_wall_H180_blocks16_v1',initial_step=start_step,initial_updates=start_updates,updates_per_branch=updates,wall_coefficients=[.25,.5],horizon=180,MAIN=16,lr=1e-5,common_VAL_wall_coefficient=.25,common_VAL_auxiliary_mode='blocks16',selection='wall_ticks_minus5pct_each_drift_plus5pct_common_loss_plus2pct',renderer_frozen=True)
    atomic_json(experiment/'plan.json',plan)
    def publish(**kwargs):atomic_json(dynamics/'wall_ablation.json',dict(plan,status='running',experiment=str(experiment),updated_at=now(),**kwargs))
    def execute(module,args,branch,stop):
        publish(branch=branch,phase=module)
        with (experiment/(branch+'_'+module.rsplit('.',1)[-1]+'.stdout.log')).open('ab') as out,(experiment/(branch+'_'+module.rsplit('.',1)[-1]+'.stderr.log')).open('ab') as err:
            proc=subprocess.Popen([sys.executable,'-m',module,*map(str,args)],stdout=out,stderr=err)
        while proc.poll() is None:
            if (root/'stop.request').exists():
                if stop is not None:stop.write_text('Stop wall pilot at optimizer boundary\n')
                else:proc.terminate()
            time.sleep(1)
        if (root/'stop.request').exists():raise InterruptedError('Wall pilot stopped')
        if proc.returncode:raise RuntimeError('Wall pilot child failed: '+branch+'/'+module)
    reports={};metrics={}
    for name,weight in [('025',.25),('050',.5)]:
        branch=experiment/name;(branch/'checkpoints').mkdir(parents=True,exist_ok=True);checkpoint=branch/'checkpoints/vjepa_world.pt'
        if not checkpoint.exists():os.link(source,checkpoint)
        execute('turkeyworld.train_vjepa_sr_dynamics',['--run-dir',branch,'--until-updates',start_updates+updates,'--wall-weight',weight],name,branch/'stop_training')
        rows=[json.loads(s) for s in (branch/'metrics.jsonl').read_text().splitlines()];metrics[name]=rows
        execute('turkeyworld.validate_vjepa_sr',['--world',checkpoint,'--output',branch/'validation.json','--auxiliary-mode','blocks16','--wall-weight',.25],name,None)
        reports[name]=json.loads((branch/'validation.json').read_text())
    def sampling(rows):return [[(s['stream'],s['file'],s['start'],s['corpus'],s['motion_kind']) for s in row['samples']] for row in rows]
    if sampling(metrics['025'])!=sampling(metrics['050']):raise AssertionError('A/B TRAIN windows differ')
    result=decide(reports);result.update(matched_TRAIN_windows=True,completed_at=now(),plan=plan)
    atomic_json(experiment/'result.json',result)
    lines=['# Wall loss0.25/0.5 · matched H180 pilot','',f'По{updates} обновлений MAIN16 от checkpoint{start_step},LR1e-5. Идентичные TRAIN окна подтверждены. VAL:8NAV+8quiet,H180,common wall0.25,blocks16.','',
        '| Метрика |0.25|0.5|','|---|---:|---:|']
    for stream,key,title in [('navigation','player_XY_H180_mean_tiles','NAV player drift'),('enemy_motion','player_XY_H180_mean_tiles','Quiet player drift'),('enemy_motion','enemy_XY_H180_mean_tiles','NPC drift'),('navigation','player_wall_overlap_ticks_mean','NAV overlap ticks'),('enemy_motion','player_wall_overlap_ticks_mean','Quiet overlap ticks')]:
        lines.append(f'|{title}|{reports["025"][stream][key]:.4f}|{reports["050"][stream][key]:.4f}|')
    lines+=['',f'Выбрана ветка {result["selected"]}. Это короткий пилот, не окончательная квалификация. Подробные условия выбора:result.json. Общий цикл продолжает выбранные веса/Adam.']
    (experiment/'REPORT_RU.md').write_text('\n'.join(lines)+'\n')
    selected=experiment/result['selected'];checkpoint=selected/'checkpoints/vjepa_world.pt'
    # Retain the original live checkpoint as initial.pt; atomic same-filesystem promotion.
    live=torch.load(dynamics/'checkpoints/vjepa_world.pt',map_location='cpu',weights_only=False,mmap=True)
    if live['training_steps']!=start_step:raise RuntimeError('Live training advanced during isolated pilot; refusing to replace newer weights')
    del live
    temporary=dynamics/'checkpoints/wall_selected.tmp.pt'
    temporary.unlink(missing_ok=True);os.link(checkpoint,temporary);temporary.replace(dynamics/'checkpoints/vjepa_world.pt')
    with (dynamics/'metrics.jsonl').open('a') as stream:
        for row in metrics[result['selected']]:stream.write(json.dumps(dict(row,wall_ablation_promoted=True))+'\n')
    weight=.5 if result['selected']=='050' else .25
    atomic_json(dynamics/'wall_selection.json',dict(wall_coefficient=weight,experiment=str(experiment),result=result))
    status=json.loads((selected/'status.json').read_text());status.update(run_id='dynamics',status='running',wall_ablation=result);atomic_json(dynamics/'status.json',status)
    atomic_json(dynamics/'wall_ablation.json',dict(plan,status='complete',experiment=str(experiment),result=result,updated_at=now()))
    (dynamics/'stop_training').unlink(missing_ok=True)
    with (root/'resume_motion_v3.stdout.log').open('ab') as out,(root/'resume_motion_v3.stderr.log').open('ab') as err:
        subprocess.Popen([sys.executable,'-m','turkeyworld.run_vjepa_sr_cycle','--run-dir',str(root)],stdout=out,stderr=err,start_new_session=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--updates',type=int,default=4);a=p.parse_args()
    try:run(a.run_dir,a.experiment,a.updates)
    except BaseException as error:
        atomic_json(a.run_dir/'dynamics/wall_ablation.json',dict(status='stopped' if isinstance(error,InterruptedError) else 'failed',error=repr(error),experiment=str(a.experiment),updated_at=now()));raise
if __name__=='__main__':main()
