"""Fixed held-out NPR validation for frozen Atlas versus the new SR expert append."""
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .atlas_enemy_validation import evaluate_atlas_enemy_sr, preview_atlas_enemy_sr
from .sr_moe_objectives import sr_moe_loss, LOSS_WEIGHTS, source_sr_moe_labels, _per_image_region
from .sr_moe_phase import PROTOCOL
from .training_pipeline import atomic_json, now, replace_file


KIND = 'sr_moe_five_validation_v1'
PREVIEW_KIND = 'sr_moe_five_preview_v1'
ROLES = ('near', 'mid', 'far', 'empty')


@torch.no_grad()
def wall_image_metrics(model, output, batch):
    labels = source_sr_moe_labels(batch, device=output['image'].device)
    image, target = output['image'].float(), batch['hr_source'].detach().to(output['image'])
    rows = [{} for _ in image]
    for region in ('near_wall', 'far_wall'):
        mask = labels[region]
        values, present = _per_image_region((image-target).abs(), mask)
        for i, row in enumerate(rows): row[region+'_mae'] = float(values[i]) if present[i] else None
        numer = image.new_zeros(len(image)); denom = image.new_zeros(len(image))
        for axis in (-1,-2):
            a,b=[slice(None)]*4,[slice(None)]*4;a[axis],b[axis]=slice(1,None),slice(None,-1)
            support=torch.maximum(mask[tuple(a)],mask[tuple(b)])
            error=(image.diff(dim=axis)-target.diff(dim=axis)).abs()
            numer+=(error*support).flatten(1).sum(-1);denom+=3*support.flatten(1).sum(-1)
        for i,row in enumerate(rows):row[region+'_edge_mae']=float(numer[i]/denom[i]) if denom[i]>0 else None
    return rows


def _metadata(preview, step):
    preview['metadata'].update(kind=PREVIEW_KIND, objective_protocol=PROTOCOL, step=step,
        row_labels=['true Source128','actual current NPR32','frozen AtlasSR106105','new five-expert SR128'],
        inference_input='actual_NPR_RGB32_with_optional_causal_RSSM_hint_and_embedded_fixed_atlas')
    return preview


def evaluate_sr_moe(model, reference, data, panel, *, step, batch_size=8, baseline_report=None):
    old_identity=None if baseline_report is None else baseline_report['atlas_validation_identity']
    report,preview=evaluate_atlas_enemy_sr(model,reference,data,panel,step=step,batch_size=batch_size,
        baseline_report=old_identity,objective_fn=sr_moe_loss,objective_weights=LOSS_WEIGHTS,
        objective_protocol=PROTOCOL,extra_metrics_fn=wall_image_metrics)
    identity={k:deepcopy(report[k]) for k in ('kind','step','data_identity_sha256','panel_fingerprint',
        'learned_reference_model_sha256','learned_reference_config_sha256','preview_cases')}
    report.update(kind=KIND, atlas_validation_identity=identity,
        models={'current':report['models']['atlas'],'frozen_atlas_parent':report['models']['learned_reference']},
        reference_scope='frozen_Atlas106105_same_current_NPR_inputs_not_a_matched_training_control',
        wall_region_scope='Source32_DDA_regions_nearest128_excluding_true_SourceHR_enemy_and_Source_weapon',
        old_SR_parameters_frozen=True, Source_targets_are_labels_only=True)
    report['temporal']['arms']={'current':report['temporal']['arms']['atlas'],
                              'frozen_atlas_parent':report['temporal']['arms']['learned_reference']}
    summary=report['models']['current']['summary']
    ssim=summary.get('ssim_mean');mae=summary.get('mae_mean')
    passed=type(ssim) in (int,float) and type(mae) in (int,float) and ssim>.91 and mae*255<2.5
    report['quality_goal']=dict(passed=passed,strict=True,input_kind='actual_NPR_RGB32',
        ssim_strictly_greater_than=.91,mae_255_strictly_less_than=2.5,
        observed_ssim=ssim,observed_mae_255=None if mae is None else mae*255,
        scope='held_out_current_NPR_only; clean/codec training guards never inflate this score',
        gameplay_qualified=False,automatic_deployment=False)
    report['training_goal']={**report['quality_goal'],'budget_completion_is_quality':False}
    return report,_metadata(preview,step)


def preview_sr_moe(model,reference,data,panel,*,step,baseline_report,batch_size=8):
    preview=preview_atlas_enemy_sr(model,reference,data,panel,step=step,batch_size=batch_size,
                                  baseline_report=baseline_report['atlas_validation_identity'])
    return _metadata(preview,step)


def save_sr_moe_preview(preview,path,*,every=15):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);meta=preview['metadata']
    if meta.get('kind')!=PREVIEW_KIND or meta.get('row_count')!=4:
        raise ValueError('Expected a four-row held-out current-NPR SR expert preview')
    canvas=Image.new('RGB',(1024,1120),(14,22,31));draw=ImageDraw.Draw(canvas)
    for row,label in enumerate(meta['row_labels']):
        draw.text((4,row*280+4),label,fill=(235,239,245))
        for column,role in enumerate(ROLES):
            case=preview['cases'].get(role)
            if case is None:continue
            image=(case['source'],case['npr'],case['predictions']['learned_reference'],case['predictions']['atlas'])[row]
            canvas.paste(Image.fromarray(np.asarray(image)).resize((256,256),Image.Resampling.NEAREST),(column*256,row*280+24))
            draw.text((column*256+4,row*280+13),role+' | row '+str(case['row_index']),fill=(220,225,235))
    temporary=path.with_name(path.stem+'.tmp.png');canvas.save(temporary,format='PNG');replace_file(temporary,path)
    result={**meta,'path':str(path),'every':every,'generated_at':now(),'preview_image_size':[1024,1120],
        'caption':'Same Source128/current NPR32; frozen Atlas parent versus new five Transformer experts'}
    atomic_json(path.with_suffix('.json'),result)
    return result
