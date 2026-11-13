"""Held-out Source NPC/empty/feature-domain reports for RF1 refinement."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw

from .object_losses import (balanced_region_losses,balanced_image_l1,balanced_mse,
    balanced_edge_l1,decoded_ssim_loss,foreground_contrastive_loss)
from .renderer_enemy_objectives import enemy_conditions
from .renderer_refinement import refinement_forward
from .staged_validation import _isolated_validation
from .visual_quality import image_quality_metrics


def _summary(prediction,target,mask,subset):
    selected=torch.as_tensor(subset,dtype=torch.bool)
    if not selected.any():return {'frames':0}
    pred,truth,roi=prediction[selected],target[selected],mask[selected]
    quality=image_quality_metrics(pred,truth)
    regions=balanced_region_losses(pred,truth,roi)
    result={'frames':int(selected.sum()),**{k+'_mean':float(v.mean()) for k,v in quality.items() if k in {'psnr','ssim','mae'}}}
    for name,present in (('foreground','foreground_present'),('background','background_present')):
        valid=regions[present]
        result[name+'_frames']=int(valid.sum())
        result[name+'_mae']=float(regions[name][valid].mean()) if valid.any() else None
    return result


def renderer_metrics(prediction,source,mask,background,categories,dynamic):
    visible=mask.flatten(1).any(-1)
    summaries={'overall':_summary(prediction,source,mask,np.ones(len(source),bool)),
        'visible_enemy':_summary(prediction,source,mask,visible),
        'static_zero_features':_summary(prediction,source,mask,~dynamic),
        'dynamic_past_prior_features':_summary(prediction,source,mask,dynamic),
        'categories':{name:_summary(prediction,source,mask,np.asarray(categories)==name) for name in sorted(set(categories))}}
    if visible.any():
        fg=balanced_region_losses(prediction,source,mask)['foreground'][visible]
        empty=balanced_region_losses(prediction,background,mask)['foreground'][visible]
        summaries['visible_enemy'].update(closer_to_source_than_without_npc_rate=float((fg<empty).float().mean()),
            presence_distance_gap=float((empty-fg).mean()))
    return summaries


def candidate_assessment(report,baseline,*,empty_mae_tolerance=.005,empty_ssim_tolerance=.02):
    reasons=[]
    for model in ('base','rf1'):
        now,old=report[model],baseline[model]
        current=now['visible_enemy']['foreground_mae'];initial=old['visible_enemy']['foreground_mae']
        if current is None or initial is None or current>=initial-.0005:reasons.append(model+'_npc_not_improved')
        if now['overall']['mae_mean']>old['overall']['mae_mean']+.002:reasons.append(model+'_full_image_regression')
        for category in ('empty_navigation','empty_hidden'):
            a,b=now['categories'].get(category,{}),old['categories'].get(category,{})
            if not a.get('frames') or not b.get('frames'):reasons.append(model+'_'+category+'_missing');continue
            if a['mae_mean']>b['mae_mean']+empty_mae_tolerance or a['ssim_mean']<b['ssim_mean']-empty_ssim_tolerance:
                reasons.append(model+'_'+category+'_regression')
    return {'validated_improvement':not reasons,'failure_reasons':reasons,
        'criteria':{'npc_foreground_mae_improvement':.0005,'full_mae_tolerance':.002,
                    'empty_mae_tolerance':empty_mae_tolerance,'empty_ssim_tolerance':empty_ssim_tolerance}}


def evaluate_enemy_renderer(model,data,panel,*,step=0,batch_size=8,baseline=None,
                           require_base_preservation=False,preservation_cohorts=None,base_anchor=None,
                           small_enemy_rebalance=False,far_wall_supervision=False,conditions_fn=None):
    if far_wall_supervision and not small_enemy_rebalance:
        raise ValueError('Far-depth validation requires the native-area renderer protocol')
    parts={key:[] for key in ('base','rgb','source','mask','background','codec_reference','dynamic',
                              'states','endpoint','source_endpoint')}
    if small_enemy_rebalance: parts['map']=[]
    with _isolated_validation(model,data,173):
        for first in range(0,len(panel),batch_size):
            ids=panel[first:first+batch_size]
            batch=data.batch(len(ids),indices=ids)
            conditions=(conditions_fn or enemy_conditions)(model,batch)
            base,rgb,endpoint=refinement_forward(model,conditions,detach_base=True)
            # Diagnostic reference only: Source RGB never enters RF conditions.
            codec=model.flow_renderer.codec
            source_codes=codec.encode(conditions['target'])['quantized']
            reference=codec.decode(source_codes)
            values={'base':base,'rgb':rgb,'source':conditions['target'],'mask':conditions['object_mask'],
                'background':conditions['background'],'codec_reference':reference,'dynamic':conditions['dynamic'],
                'states':conditions['state'],'endpoint':endpoint,
                'source_endpoint':model.flow_renderer.normalize_latents(source_codes)}
            if small_enemy_rebalance: values['map']=conditions['map']
            if any(not bool(torch.isfinite(value).all()) for value in values.values()):
                raise FloatingPointError('Nonfinite Source enemy rendering validation')
            for key,value in values.items():parts[key].append(value.detach().cpu())
    arrays={key:torch.cat(value) for key,value in parts.items()}
    categories=[data.rows[int(i)]['category'] for i in panel]
    arrays['categories']=categories
    report={'step':step,'frames':len(panel),'source_held_out':True,'renderer_only':True,
        'scope':'Native32 GT-pose one-NPC Source; static zero appearance and causal past-prior feature domains',
        'panel_fingerprint':data.panel_fingerprint(panel),'solver_steps':1,
        'static_feature_policy':'snapshot_zero128_no_target_encoding',
        'dynamic_feature_policy':'frozen_prior_previous_Source_RGB_state_action_only',
        'future_source_rgb_in_prior':False,'future_source_pose_in_prior':False,
        'gameplay_qualified':False,'sr_quality_evaluated':False,
        'codec_reference_is_a_proven_ceiling':False,
        'codec_reference_scope':'Frozen compact Source encoder+quantizer+decoder reconstruction; empirical reference, not a mathematical upper bound'}
    for name,key in (('base','base'),('rf1','rgb'),('codec_reference','codec_reference')):
        report[name]=renderer_metrics(arrays[key],arrays['source'],arrays['mask'],arrays['background'],
            categories,arrays['dynamic'])
    pixel_terms={}
    for name,key in (('base','base'),('rf1','rgb')):
        prediction=arrays[key]
        pixel_terms[name+'_l1']=float(balanced_image_l1(prediction,arrays['source'],arrays['mask']))
        pixel_terms[name+'_ssim_loss']=float(decoded_ssim_loss(prediction,arrays['source'],arrays['mask']))
        pixel_terms[name+'_edge']=float(balanced_edge_l1(prediction,arrays['source'],arrays['mask']))
        pixel_terms[name+'_contrast']=float(foreground_contrastive_loss(prediction,arrays['source'],arrays['background'],arrays['mask']))
    common_rgb=sum(pixel_terms[name+'_l1']+.2*pixel_terms[name+'_ssim_loss']
        +.1*pixel_terms[name+'_edge']+.1*pixel_terms[name+'_contrast'] for name in ('base','rf1'))
    latent=float(balanced_mse(arrays['endpoint'],arrays['source_endpoint'],arrays['mask']))
    report['common_objective']={**pixel_terms,'source_latent_mse':latent,
        'rgb_objective':common_rgb,'original_latent01_objective':common_rgb+.1*latent,
        'weights':{'l1':1.,'ssim':.2,'edge':.1,'contrast':.1,'latent':.1},
        'preservation_penalty_included':False,'extra_rf_forward_calls':0}
    states,visible=arrays['states'],arrays['mask'].flatten(1).any(-1)
    alive=(states[:,13]>=.5)&(states[:,12]>0)
    conditions={'overall':np.ones(len(panel),bool),'alive':visible&alive,
        'npc_hurt':visible&alive&(states[:,15]>0),'hp0':states[:,4]<=0,'corpse':visible&~alive,
        'empty':~visible,'navigation':np.asarray(categories)=='empty_navigation',
        'large':np.asarray(categories)=='large'}
    player_hurt=np.zeros(len(panel),bool)
    groups=data.cohort_rows(panel) if hasattr(data,'cohort_rows') else {}
    if groups and not all(isinstance(value,dict) for value in groups.values()):groups={'source':groups}
    if groups:
        player_hurt[groups.get('combat',{}).get('player_hurt',[])]=True
    conditions['player_hurt']=player_hurt
    report['preservation_cohorts']={name:{'base':_summary(arrays['base'],arrays['source'],arrays['mask'],selected),
        'rf1':_summary(arrays['rgb'],arrays['source'],arrays['mask'],selected)} for name,selected in conditions.items()}
    if groups:
        report['source_cohorts']={}
        for pool,cohorts in groups.items():
            report['source_cohorts'][pool]={}
            for case,indices in cohorts.items():
                selected=np.zeros(len(panel),bool);selected[indices]=True
                report['source_cohorts'][pool][case]={'frames':len(indices),
                    **{name:_summary(arrays[key],arrays['source'],arrays['mask'],selected)
                       for name,key in (('base','base'),('rf1','rgb'))}}
        report['source_cohort_scope']='Recorded Source event/navigation labels select held-out GT-pose images; not combat-policy accuracy'
    report['summary']=report['rf1']['overall']
    report['passed']=all(report[name]['overall']['ssim_mean']>=.9 and report[name]['overall']['mae_mean']<=.03
        and report[name]['visible_enemy']['foreground_mae']<=.05 for name in ('base','rf1'))
    report['thresholds']={'source_ssim_mean':.9,'source_mae_mean':.03,'npc_foreground_mae':.05,
                          'applies_to':'both Base and RF1; experimental relative improvement reported separately'}
    if baseline is not None:
        report['experimental_selection']=candidate_assessment(report,baseline)
    from .renderer_preservation_validation import base_preservation_assessment
    report['base_preservation']=base_preservation_assessment(report,base_anchor if base_anchor is not None else baseline,
        **({'cohorts':preservation_cohorts} if preservation_cohorts is not None else {}))
    report['base_parent_anchor']='external_original_parent' if base_anchor is not None else 'initial_current_run_parent'
    report['base_preservation_enforced']=require_base_preservation
    if require_base_preservation:
        report['passed']=report['passed'] and report['base_preservation']['passed']
        if baseline is not None:
            selection=report['experimental_selection']
            selection['validated_improvement']=selection['validated_improvement'] and report['base_preservation']['passed']
            if not report['base_preservation']['passed']:selection['failure_reasons'].append('base_preservation_release_gate')
    report['selection_score']=(report['rf1']['visible_enemy']['foreground_mae']
        +.5*report['base']['visible_enemy']['foreground_mae']
        +.25*(report['rf1']['overall']['mae_mean']+report['base']['overall']['mae_mean']))
    if small_enemy_rebalance:
        from .renderer_small_enemy_protocol import native_area_report
        from .renderer_wall_supervision import wall_validation_report
        report['native_area_validation']=native_area_report(arrays)
        report['source_wall_validation']=wall_validation_report(arrays)
    if far_wall_supervision:
        from .renderer_wall_depth import depth_banded_wall_report
        report['depth_banded_wall_validation']=depth_banded_wall_report(arrays)
    return report,arrays


def save_enemy_preview(arrays,path,step):
    categories=arrays['categories'];chosen=[]
    for name in ('large','partial','corpse','empty_navigation'):
        matches=[i for i,value in enumerate(categories) if value==name and i not in chosen]
        if matches:chosen.append(matches[0])
    chosen += [i for i in range(len(categories)) if i not in chosen][:4-len(chosen)]
    size=160;labels=[('source','SOURCE32'),('base','TRAINED BASE'),('rgb','RF1'),('codec_reference','FROZEN CODEC REFERENCE')]
    image=Image.new('RGB',(len(chosen)*size,len(labels)*(size+25)),(12,17,25));draw=ImageDraw.Draw(image)
    for row,(key,title) in enumerate(labels):
        for column,index in enumerate(chosen):
            value=arrays[key][index].float().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy()
            image.paste(Image.fromarray(value).resize((size,size),Image.Resampling.NEAREST),(column*size,row*(size+25)+25))
            draw.text((column*size+3,row*(size+25)+3),f'{title} g{step} | {categories[index]}',fill='white')
    path=Path(path);tmp=path.with_name(path.name+'.tmp.png');image.save(tmp);tmp.replace(path)

