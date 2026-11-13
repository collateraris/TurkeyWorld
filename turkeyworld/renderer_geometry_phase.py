"""Strict v6-to-v7 wall-geometry continuation with the complete parent Adam.

No checkpoint is read here. The caller supplies the pinned completed parent
and persists the returned receipt. All RNG streams and the existing Source
sampler's private PCG clock continue from that completed boundary.
"""
from __future__ import annotations

import copy
import re

import torch


CONFIGURATION_CHANGES = frozenset({'run_dir', 'steps', 'geometry_phase_parent', 'geometry_wall_supervision'})
PARENT_PROTOCOL = 'one_enemy_native_area_balanced_renderer_v5'
QUOTAS = frozenset({'tiny_1_16', 'small_17_64', 'far', 'replay'})
IDENTITY_ADDITIONS = frozenset({'geometry_wall_protocol', 'geometry_wall_aux_pool', 'geometry_wall_panel', 'geometry_wall_receipt', 'geometry_wall_fingerprint'})


from .renderer_native_phase import (_same, _sampler_view, _validate_rng, restore_renderer_native_rng)


def restore_renderer_geometry_parent(saved, model, optimizer, data, far_data, identity, configuration, frozen, *,
                                     parent_checkpoint_sha256, geometry_protocol):
    """Load the full completed v6 parent after whitelisted v7 preflight checks.

    This restores weights, all Adam states, private sampler PCG, and the global
    Torch/NumPy/Python/CUDA RNG streams. Validation/probing after adoption must
    preserve those streams before the first actual update.
    """
    from .renderer_geometry_objectives import geometry_wall_protocol as canonical_protocol
    if not isinstance(parent_checkpoint_sha256, str) or not re.fullmatch('[0-9a-f]{64}', parent_checkpoint_sha256):
        raise ValueError('Native continuation requires a pinned parent checkpoint SHA256')
    if not _same(geometry_protocol, canonical_protocol()):
        raise ValueError('Geometry continuation protocol differs from the canonical immutable settings')
    required = {'base', 'flow', 'optimizer', 'step', 'sampler', 'far_wall_aux_sampler', 'configuration', 'identity', 'frozen_fingerprint'}
    if not isinstance(saved, dict) or not required <= saved.keys():
        raise ValueError('Native continuation requires the full parent, including Adam/sampler/settings')
    _validate_rng(saved)
    device = next(model.parameters()).device
    if device.type == 'cuda' and not saved['rng_cuda']:
        raise ValueError('CUDA continuation requires the saved parent CUDA RNG stream')
    old = saved['configuration']
    step = saved['step']
    if (type(step) is not int or step < 1 or type(old.get('steps')) is not int or step != old['steps']
            or type(configuration.get('steps')) is not int or configuration['steps'] <= step):
        raise ValueError('The new native phase must extend a completed positive parent optimizer boundary')
    if (old.get('small_enemy_rebalance') is not True or configuration.get('small_enemy_rebalance') is not True
            or old.get('far_wall_supervision') is not True or configuration.get('far_wall_supervision') is not True
            or old.get('geometry_wall_supervision', False) is not False
            or configuration.get('geometry_wall_supervision') is not True
            or not isinstance(configuration.get('geometry_phase_parent'), str)
            or not configuration['geometry_phase_parent']
            or old.get('run_dir') == configuration.get('run_dir')):
        raise ValueError('Only an explicit completed v6-to-geometry v7 phase is permitted')
    before = {key: value for key, value in old.items() if key not in CONFIGURATION_CHANGES}
    after = {key: value for key, value in configuration.items() if key not in CONFIGURATION_CHANGES}
    if not _same(before, after):
        changed = sorted(k for k in before.keys() | after.keys() if not _same(before.get(k), after.get(k)))
        raise ValueError('Geometry continuation changed inherited settings: '+','.join(changed))
    if (IDENTITY_ADDITIONS & saved['identity'].keys()
            or not IDENTITY_ADDITIONS <= identity.keys()
            or not _same(identity['geometry_wall_protocol'], geometry_protocol)
            or not _same(saved['identity'], {key: value for key, value in identity.items() if key not in IDENTITY_ADDITIONS})):
        raise ValueError('Geometry parent Source/frozen-feature/codec/old-panel identity changed')
    pool = identity['geometry_wall_aux_pool']
    source_fingerprints = {saved['identity'].get('train', {}).get('fingerprint'),
                           saved['sampler'].get('fingerprint')}
    if (not isinstance(pool, dict) or pool.get('split') != 'train'
            or pool.get('parent_source_fingerprint') not in source_fingerprints
            or not isinstance(pool.get('fingerprint'), str)
            or not re.fullmatch('[0-9a-f]{64}', pool['fingerprint'])):
        raise ValueError('Geometry auxiliary identity must reference the unchanged TRAIN Source pool')
    if (not isinstance(identity['geometry_wall_panel'], list) or not identity['geometry_wall_panel']
            or len(identity['geometry_wall_panel']) != len(set(identity['geometry_wall_panel']))
            or not isinstance(identity['geometry_wall_fingerprint'], str)
            or not re.fullmatch('[0-9a-f]{64}', identity['geometry_wall_fingerprint'])
            or identity['geometry_wall_receipt'].get('split') != 'val'):
        raise ValueError('Geometry validation requires a pinned distinct held-out VAL panel')
    if far_data is None or saved['identity'].get('far_wall_aux_pool') != far_data.identity:
        raise ValueError('The original independent far-wall pool identity changed')
    if saved['frozen_fingerprint'] != frozen:
        raise ValueError('Native parent frozen world/codec identity changed')

    # Check layout, finite parent tensors, and the immutable codec/norm/noise
    # BEFORE allowing any parent Base/RF weight to enter the target.
    states = [model.base_student.state_dict(), model.flow_renderer.state_dict()]
    expected_groups = [list(model.base_student.parameters()),
                       [p for n, p in model.flow_renderer.named_parameters()
                        if not n.startswith('codec.') and p.requires_grad]]
    noncodec_names = {n for n, p in model.flow_renderer.named_parameters()
                      if not n.startswith('codec.') and p.requires_grad}
    for label, target in zip(('base', 'flow'), states):
        parent = saved[label]
        if (not isinstance(parent, dict) or parent.keys() != target.keys()
                or any(not torch.is_tensor(v) or v.shape != target[k].shape
                       or v.dtype != target[k].dtype or not bool(torch.isfinite(v).all())
                       for k, v in parent.items())):
            raise ValueError('Native parent '+label+' tensor architecture/dtype/finiteness changed')
    frozen_flow = {key: value for key, value in saved['flow'].items() if key not in noncodec_names}
    if not _same(frozen_flow, {key: states[1][key] for key in frozen_flow}):
        raise ValueError('Native parent codec/all buffers/latent normalization/fixed noise changed')
    groups = saved['optimizer'].get('param_groups', [])
    if (len(groups) != 2 or len(optimizer.param_groups) != 2
            or any([id(p) for p in current['params']] != [id(p) for p in wanted]
                   for current, wanted in zip(optimizer.param_groups, expected_groups))
            or any(len(group['params']) != len(wanted) for group, wanted in zip(groups, expected_groups))
            or any(not _same({k: v for k, v in old_group.items() if k != 'params'},
                             {k: v for k, v in current.items() if k != 'params'})
                   for old_group, current in zip(groups, optimizer.param_groups))):
        raise ValueError('Native parent ordered Base/non-codec RF Adam groups or options changed')
    ids = [i for group in groups for i in group['params']]
    optimizer_states = saved['optimizer'].get('state', {})
    if len(ids) != len(set(ids)) or set(ids) != set(optimizer_states):
        raise ValueError('Native continuation needs every parent Adam state exactly once')
    for group, parameters in zip(groups, expected_groups):
        for index, parameter in zip(group['params'], parameters):
            state = optimizer_states[index]
            if (not isinstance(state, dict) or not all(k in state for k in ('step', 'exp_avg', 'exp_avg_sq'))
                    or float(state['step']) != step or any(not torch.is_tensor(state[k])
                        or state[k].shape != parameter.shape or state[k].dtype != parameter.dtype
                        or not bool(torch.isfinite(state[k]).all()) for k in ('exp_avg', 'exp_avg_sq'))):
                raise ValueError('Native parent Adam moments, shape or completed-step clock changed')
    sampler = saved['sampler']
    if (sampler.get('protocol') != PARENT_PROTOCOL or sampler.get('parent_draws') != 0
            or sampler.get('parent_sampler_draws_at_start') != 0
            or sampler.get('phase_draws') != step*old['batch']
            or set(sampler.get('distance_counts', {})) != QUOTAS):
        raise ValueError('Native parent sampler protocol or historical draw clock changed')
    private = _sampler_view(data)
    private.restore_sampler(copy.deepcopy(sampler))
    if not _same(private.sampler_state(), sampler):
        raise ValueError('Native sampler cannot restore the exact saved PCG/counters')

    private_far = _sampler_view(far_data)
    private_far.restore_sampler(copy.deepcopy(saved['far_wall_aux_sampler']))
    if not _same(private_far.sampler_state(), saved['far_wall_aux_sampler']):
        raise ValueError('Far-wall sampler cannot restore its exact saved PCG/counters')

    rng = torch.get_rng_state().clone()
    model.base_student.load_state_dict(saved['base'], strict=True)
    model.flow_renderer.load_state_dict(saved['flow'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    data.restore_sampler(copy.deepcopy(sampler))
    far_data.restore_sampler(copy.deepcopy(saved['far_wall_aux_sampler']))
    if (not _same(saved['base'], model.base_student.state_dict())
            or not _same(saved['flow'], model.flow_renderer.state_dict())
            or not _same(saved['optimizer'], optimizer.state_dict())
            or not _same(sampler, data.sampler_state())
            or not _same(saved['far_wall_aux_sampler'], far_data.sampler_state())):
        raise AssertionError('Native continuation changed parent weights, Adam, or sampler during adoption')
    if not torch.equal(rng, torch.get_rng_state()):
        raise AssertionError('Native parent adoption consumed CPU Torch RNG')
    restored_rng = restore_renderer_native_rng(saved, device=device)
    return {
        'version': 1, 'kind': 'far_geometry_v6_to_all_depth_geometry_v7_full_Adam_continuation',
        'start_step': step, 'parent_optimizer_step': step, 'parent_checkpoint_sha256': parent_checkpoint_sha256,
        'identity': copy.deepcopy(identity), 'geometry_wall_protocol': copy.deepcopy(geometry_protocol),
        'allowed_configuration_changes': sorted(CONFIGURATION_CHANGES),
        'all_base_flow_tensors_bitexact': True, 'codec_normalization_noise_bitexact': True,
        'adam_all_moments_options_bitexact': True, 'adam_state_count': len(optimizer_states),
        'parent_sampler_and_private_rng_bitexact': True, 'sampler_draws_preserved': sampler['phase_draws'],
        'parent_draws': sampler['parent_draws'], 'parent_sampler_draws_at_start': 0,
        'parent_mix_counts': copy.deepcopy(sampler['parent'].get('mix_counts', {})),
        'parent_sampler_counts_at_boundary': copy.deepcopy(sampler),
        'parent_best': copy.deepcopy(saved.get('best')), 'parent_streak': saved.get('streak', 0),
        'parent_enemy_streak': saved.get('enemy_streak', 0),
        'source_identity_preserved': True,
        'identity_additions': sorted(IDENTITY_ADDITIONS & identity.keys()),
        'new_geometry_aux_sampler_is_independent': True,
        'main_sampler_and_far_auxiliary_bitexact': True,
        'far_wall_auxiliary_draws_at_adoption': int(saved['far_wall_aux_sampler']['draws']),
        'global_rng_restored': not restored_rng['cuda_rng_restore_required'],
        'global_rng_restoration_required': restored_rng['cuda_rng_restore_required'],
        **restored_rng,
        'new_phase_progress_does_not_reset_optimizer_or_sampler': True,
        'automatic_active_promotion': False, 'gameplay_qualified': False,
    }


def geometry_candidate_assessment(report, baseline):
    """Wall-only experimental selection; old release/quality gates stay separate."""
    import math
    from .renderer_geometry_objectives import geometry_preservation_assessment, geometry_report_score
    failures=[]
    current=report['geometry_wall_guard_validation']; original=baseline['geometry_wall_guard_validation']
    guard=geometry_preservation_assessment(current,original,tolerance=.002)
    if not guard['passed']:failures.extend(guard['failure_reasons'])
    score=geometry_report_score(current);initial=geometry_report_score(original)
    delta=initial-score
    if not all(math.isfinite(v)for v in (score,initial))or not(delta>.001 or delta>.01*initial):
        failures.append('geometry_score_not_improved')
    panels=[('main',report,baseline)]
    for key in ('native_area_guard_validation','distance_validation','terminal_supplement'):
        if key in baseline:
            if key not in report:failures.append(key+'_missing')
            else:panels.append((key,report[key],baseline[key]))
    for panel,now,old in panels:
        for head in ('base','rf1'):
            a,b=now[head],old[head]
            for cohort,field,count in (('visible_enemy','foreground_mae','foreground_frames'),
                                      ('overall','mae_mean','frames')):
                x,y=a[cohort],b[cohort]
                if not y.get(count):continue
                value,reference=x.get(field),y.get(field)
                if (x.get(count)!=y[count] or type(value)not in(int,float)
                        or type(reference)not in(int,float) or not math.isfinite(value)
                        or not math.isfinite(reference) or value>reference+.002):
                    failures.append(panel+'_'+head+'_'+cohort+'_regressed_or_support_changed')
            for name,y in b.get('categories',{}).items():
                if not(name.startswith('empty')or name=='navigation')or not y.get('frames'):continue
                x=a.get('categories',{}).get(name,{})
                if (x.get('frames')!=y['frames'] or any(type(x.get(k))not in(int,float)
                        or not math.isfinite(x[k])for k in ('mae_mean','ssim_mean'))
                        or x['mae_mean']>y['mae_mean']+.002 or x['ssim_mean']<y['ssim_mean']-.01):
                    failures.append(panel+'_'+head+'_'+name+'_regressed_or_support_changed')
    return dict(passed=not failures,failure_reasons=failures,score=score,baseline_score=initial,
        score_improvement=delta,relative_improvement=delta/max(initial,1e-12),
        geometry_preservation=guard,npc_empty_mae_tolerance=.002,empty_ssim_tolerance=.01,
        scope='Wall-only experimental candidate, same held-out support and NPC/empty preservation',
        release_qualified=False,gameplay_qualified=False,automatic_active_promotion=False)


GEOMETRY_PREVIEW_COLUMNS = ('near_corners','mid_corners','far_thin','other_far')


def geometry_preview_arrays(model,data,panel,*,conditions_fn=None):
    """Four held-out render queries; no NPC-presence or quality assumptions."""
    from .renderer_enemy_objectives import enemy_conditions
    from .renderer_refinement import refinement_forward
    from .staged_validation import _isolated_validation
    if getattr(data,'split',None)not in('val','test')or len(panel)!=4 or len(set(panel))!=4:
        raise ValueError('Geometry preview requires four distinct held-out queries')
    with _isolated_validation(model,data,173):
        batch=data.batch(len(panel),indices=panel)
        conditions=(conditions_fn or enemy_conditions)(model,batch)
        base,rgb,_=refinement_forward(model,conditions,detach_base=True)
        codec=model.flow_renderer.codec
        reference=codec.decode(codec.encode(conditions['target'])['quantized'])
        values={'source':conditions['target'],'base':base,'rgb':rgb,'codec_reference':reference}
        if any(not bool(torch.isfinite(v).all())for v in values.values()):
            raise FloatingPointError('Nonfinite held-out geometry preview')
        return {key:value.detach().cpu().clone()for key,value in values.items()}


def save_geometry_preview(arrays,path,step,*,panel,receipt,mirror=None,every=10,
                          source_quality_gate_evaluated=False):
    """Keep pinned geometry columns in order, including all-empty navigation."""
    from .renderer_enemy_validation import save_enemy_preview
    if (type(step)is not int or step<0 or type(every)is not int or every<1
            or receipt.get('split')!='val' or len(panel)!=len(set(panel))):
        raise ValueError('Geometry preview needs a valid step and pinned VAL receipt')
    representatives=receipt.get('preview_indices')
    groups=receipt.get('stratum_indices',{})
    if (not isinstance(representatives,list)or len(representatives)!=4
            or len(set(representatives))!=4
            or any(index not in groups.get(name,[])or index not in panel
                   for name,index in zip(GEOMETRY_PREVIEW_COLUMNS,representatives))):
        raise ValueError('Geometry preview columns differ from the pinned stratum representatives')
    positions=[panel.index(index)for index in representatives]
    view={}
    for key in ('source','base','rgb','codec_reference'):
        value=arrays[key]
        if value.shape!=(len(panel),3,32,32)or not bool(torch.isfinite(value).all()):
            raise ValueError('Geometry preview requires aligned finite native32 RGB')
        view[key]=value[positions].detach().cpu()
    view['categories']=list(GEOMETRY_PREVIEW_COLUMNS)
    save_enemy_preview(view,path,step)
    if mirror is not None:
        from shutil import copyfile
        copyfile(path,mirror)
    return dict(step=step,every=every,kind='Source_enemy_GT_query',
        source_held_out=True,renderer_only=True,world_rollout=False,codec_frozen=True,
        columns=list(GEOMETRY_PREVIEW_COLUMNS),source_query_indices=list(representatives),
        geometry_wall_fingerprint=receipt['panel_fingerprint'],
        loss_protocol='one_enemy_all_depth_wall_geometry_renderer_v7',
        feature_policy='static_zero128_or_dynamic_frozen_past_prior',
        source_quality_gate_evaluated=source_quality_gate_evaluated,
        generated_render_queries=4,full_validation=False,source_pixels_resized_for_display_only=True)
