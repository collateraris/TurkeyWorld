"""Held-out RF/Base preservation, independent of the moving training hinge."""
import math

CRITICAL_COHORTS = ('overall','alive','npc_hurt','player_hurt','hp0','corpse','empty','navigation','large')


def validate_base_anchor_identity(anchor,current):
    """A codec revision may reuse only the same source world and exact panels.

    Codec/renderer bundle hashes may intentionally differ; neither changes
    the original compact Base outputs used as the frozen reference.
    """
    for key in ('world_sha256','panel_fingerprint'):
        if not anchor.get(key) or anchor.get(key)!=current.get(key):
            raise ValueError('Frozen Base anchor source/panel differs: '+key)
    for key in ('train','val'):
        if (not anchor.get(key,{}).get('fingerprint')
                or anchor[key]['fingerprint']!=current.get(key,{}).get('fingerprint')):
            raise ValueError('Frozen Base anchor dataset differs: '+key)
    if anchor.get('hp0_panel_fingerprint')!=current.get('hp0_panel_fingerprint'):
        raise ValueError('Frozen Base anchor HP0 panel differs')


def base_preservation_assessment(report,baseline=None,*,cohorts=CRITICAL_COHORTS,
                                 mae_tolerance=.002,ssim_tolerance=.01,region_tolerance=.002):
    def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
    def compare(now,old,name):
        failures=[]
        if not now.get('frames') or not old.get('frames'):return [name+':missing_frames']
        for key,tolerance,lower in [('mae_mean',mae_tolerance,True),('ssim_mean',ssim_tolerance,False),
                ('foreground_mae',region_tolerance,True),('background_mae',region_tolerance,True)]:
            if key.startswith('foreground') and now.get('foreground_frames')==old.get('foreground_frames')==0:continue
            if key.startswith('background') and now.get('background_frames')==old.get('background_frames')==0:continue
            a,b=now.get(key),old.get(key)
            if not finite(a) or not finite(b):failures.append(name+':missing_'+key)
            elif (a>b+tolerance if lower else a<b-tolerance):failures.append(name+':'+key)
        return failures
    rf_failures,base_failures=[],[]
    current,parent=report.get('preservation_cohorts',{}),(baseline or {}).get('preservation_cohorts',{})
    for cohort in cohorts:
        row,old=current.get(cohort,{}),parent.get(cohort,{})
        rf_failures.extend(compare(row.get('rf1',{}),row.get('base',{}),cohort))
        base_failures.extend(compare(row.get('base',{}),old.get('base',{}),cohort))
    return {'passed':not rf_failures and not base_failures,
        'rf_not_worse_than_current_base':{'passed':not rf_failures,'failures':rf_failures},
        'base_not_worse_than_frozen_parent':{'passed':not base_failures,'failures':base_failures},
        'parent_reference_available':baseline is not None,'cohorts':list(cohorts),
        'tolerances':{'mae':mae_tolerance,'ssim':ssim_tolerance,'region_mae':region_tolerance},
        'scope':'Fixed held-out Source render queries; not an inference guarantee or gameplay qualification'}
