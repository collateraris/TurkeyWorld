"""Pair frozen V-JEPA checkpoints across navigation, pursuit and contact validation."""
import argparse
import json
from pathlib import Path
from .training_pipeline import atomic_json


def qualify(root, baseline='base', candidate='candidate', *, hard_gate=None):
    read=lambda name:json.loads((root/name).read_text())
    metrics={}; checks={}; pursuit={}
    for panel in ('fixed','independent','hard'):
        if panel=='hard' and hard_gate is not None:
            # Reuse only the previously qualified player-only hard panel. Caller
            # must independently check the attached head and architecture lineage.
            checks['hard_prequalified']=hard_gate['nav_pass'] and hard_gate['checks']['hard_strafe_wall'] and hard_gate['checks']['hard_corner_exit']
            continue
        a,b=(read(n+'_'+panel+'.json') for n in (baseline,candidate))
        if a['step']!=b['step']:raise ValueError('Different parent training steps')
        if panel=='hard':
            assert a['panel_sha256']==b['panel_sha256'] and a['closed_loop']==b['closed_loop']==180
            assert not a['source_future_feedback'] and not b['source_future_feedback']
            for group in ('strafe_wall','corner_exit'):
                vals=[v['per_class'][group] for v in (a,b)]
                if any(vals[0]['probes'][k]!=vals[1]['probes'][k] for k in ('free_axes','blocked_axes')):raise ValueError('Different hard-panel axes')
                mx={k:[r[k] for r in vals] for k in ('H180','wall_ticks','mean_path_error')}
                mx.update({k:[r['probes'][k] for r in vals] for k in ('false_stops','false_passes','free_axes','blocked_axes')})
                checks['hard_'+group]=(mx['H180'][1]<mx['H180'][0] and all(mx[k][1]<=mx[k][0] for k in ('wall_ticks','false_stops','false_passes')))
                metrics['hard_'+group]=mx
            continue
        assert a['panel_fingerprint']==b['panel_fingerprint'] and a['panel']==b['panel']==panel
        for d in (a,b):
            assert not d['Source_future_feedback'] and d['feedback_mode']=='on' and d['horizon']==180
        for stream in ('navigation','enemy_motion'):
            k=panel+'_'+stream
            if a[stream]['windows']!=b[stream]['windows'] or a[stream]['count']!=b[stream]['count'] or a[stream]['count']!=8:raise ValueError('Unpaired episodes')
            v=metrics[k]={name:[r[stream][orig] for r in (a,b)] for name,orig in [('H180','player_XY_H180_mean_tiles'),('wall_ticks','player_wall_overlap_ticks_mean')]}
            if panel=='independent':
                for name in ('false_stops','false_passes','free_axes','blocked_axes'):
                    v[name]=[r[stream]['common_start_axis'][name] for r in (a,b)]
                if any(v[key][0]!=v[key][1] for key in ('free_axes','blocked_axes')):raise ValueError('Unpaired common-start probes')
            if stream=='navigation':
                checks[k]=v['H180'][1]<v['H180'][0] and v['wall_ticks'][1]<=v['wall_ticks'][0]
                if panel=='independent':checks['nav_probes']=v['false_stops'][1]<v['false_stops'][0] and v['false_passes'][1]<=v['false_passes'][0]
            else:
                v['npc_H180']=[r[stream]['enemy_XY_H180_mean_tiles'] for r in (a,b)]
                pursuit[k]=v['npc_H180'][1]<=1.05*v['npc_H180'][0] and v['H180'][1]<=v['H180'][0]+1e-6 and v['wall_ticks'][1]<=v['wall_ticks'][0]
                if panel=='independent':pursuit['pursuit_probes']=v['false_passes'][1]<=v['false_passes'][0]
    return dict(checks=checks,nav_pass=all(checks.values()),pursuit_checks=pursuit,
                pursuit_pass=all(pursuit.values()),measurements=metrics)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);a=p.parse_args()
    result=qualify(a.experiment);atomic_json(a.experiment/'qualification.json',result)
    print(json.dumps(result,indent=2))
