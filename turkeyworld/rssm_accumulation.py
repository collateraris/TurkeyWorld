"""Transparent metric accounting for sequential RSSM microbatch accumulation."""
from __future__ import annotations

from copy import deepcopy
import math


def accumulated_metrics(rows):
    """Average objectives; sum named event/work counts; retain each full row."""
    if not rows:raise ValueError('At least one completed backward microbatch is required')
    if len(rows)==1:return dict(rows[0])
    output=deepcopy(rows[-1]);keys=set.union(*(set(row)for row in rows))
    count_suffixes=('_count','_sum','_calls','_probes','_draws','_chunks','_branches','_forward_transitions','_queries','_attempts')
    summed=[];averaged=[]
    for key in keys:
        if key.endswith('_direct_per_step_counts'):
            present=[row[key] for row in rows if key in row]
            if not present or not all(isinstance(value,list) for value in present):
                raise ValueError('Expected per-step direct-event count lists')
            length=len(present[0])
            if any(len(value)!=length for value in present):
                raise ValueError('Direct-event horizons differ across accumulated microbatches')
            combined=[]
            for index in range(length):
                names=set().union(*(set(value[index]) for value in present))
                if not all(type(value[index].get(name,0))is int and value[index].get(name,0)>=0
                           for value in present for name in names):
                    raise ValueError('Direct-event counts must be nonnegative integers')
                combined.append({name:sum(value[index].get(name,0) for value in present) for name in names})
            output[key]=combined;summed.append(key)
            continue
        is_count=key.endswith(count_suffixes)
        is_count_dict=key.endswith(('_counts','_cohorts'))
        if not is_count and not is_count_dict and any(key not in row for row in rows):
            continue
        values=[row.get(key,{}if is_count_dict else 0)for row in rows]
        if all(isinstance(value,bool)for value in values):
            output[key]=all(values)
        elif all(isinstance(value,(int,float))and not isinstance(value,bool)for value in values):
            if not all(math.isfinite(value)for value in values):raise FloatingPointError('Nonfinite accumulated metric '+key)
            if is_count:output[key]=sum(values);summed.append(key)
            else:output[key]=sum(values)/len(values);averaged.append(key)
        elif is_count_dict and all(isinstance(value,dict)for value in values):
            names=set().union(*(set(value)for value in values))
            if all(isinstance(value.get(name,0),(int,float))for value in values for name in names):
                if not all(math.isfinite(value.get(name,0))for value in values for name in names):
                    raise FloatingPointError('Nonfinite accumulated count dictionary '+key)
                output[key]={name:sum(value.get(name,0)for value in values)
                    /(len(values)if name.endswith(('_ce','_loss','_component'))else 1)for name in names}
                summed.append(key)
        elif key.endswith('_selected_anchors')and all(isinstance(value,list)for value in values):
            output[key]=[{'microbatch_index':index+1,**anchor}for index,value in enumerate(values)
                for anchor in value]
    # Variable branch selections and cohorts remain unabridged in these rows;
    # their per-microbatch actor indices never masquerade as global actor IDs.
    output['microbatch_metrics']=deepcopy(rows)
    output['metric_aggregation']={'objectives':'mean_over_microbatches','named_counts':'sum',
        'averaged_keys':sorted(averaged),'summed_keys':sorted(summed),
        'variable_keys_not_averaged':sorted(key for key in keys if key not in summed
            and key not in averaged and any(key not in row for row in rows)),
        'other_diagnostics':'last_microbatch; complete values retained in microbatch_metrics'}
    return output


def average_accumulated_gradients(parameters, accumulation):
    """One optimizer update uses the mean of all completed microbatch gradients."""
    if type(accumulation)is not int or accumulation<1:raise ValueError('Positive accumulation required')
    if accumulation==1:return
    for value in parameters:
        if value.grad is not None:value.grad.mul_(1/accumulation)
