"""Matched full-state SR twins and a small, explicit adversarial schedule."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from .distill_renderer import _save
from .models import build_upscaler
from .sr_semantic_migration import restore_semantic_optimizer, _sha
from .training_pipeline import atomic_json

PROTOCOL='enemy_roi_patchgan_ab_v1'
REPORT_KIND='enemy_roi_patchgan_ab_validation_v1'
PREVIEW_KIND='enemy_roi_patchgan_ab_preview_v1'
SETTINGS=dict(version=1,warmup_updates=100,ramp_updates=400,max_weight=.01,
    gradient_ratio_limit=.1,gradient_probe='shared_RGBA_RGB_alpha_readout_weight_L2',
    feature_matching_weight=.1,discriminator_lr=.0002,discriminator_betas=[0.,.99],
    validation_every=100,prediction_every=15,batch_size=8,
    control='same_initial_model_Adam_same_completed_steps_same_actual_inputs',
    discriminator_canvas='same_detached_NPR_crop_for_real_and_fake',
    GAN_affects_inference_cost=False,automatic_deployment=False)


def scheduled_gan_weight(update):
    if type(update)is not int or update<0:raise ValueError('GAN update must be a completed nonnegative integer')
    return SETTINGS['max_weight']*max(0.,min(1.,(update-SETTINGS['warmup_updates'])/SETTINGS['ramp_updates']))


def ledger_update(previous,step,indices,domains,inputs,hints):
    """One receipt for the actual shared query, not one separately drawn per arm."""
    data=json.dumps(dict(previous=previous,step=step,indices=list(map(int,indices)),domains=list(domains)),sort_keys=True).encode()
    digest=hashlib.sha256(data)
    for value in (inputs,hints):digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare_patchgan_continuation(source,destination,*,seed=3173):
    from .sr_enemy_patchgan import EnemyPatchDiscriminator
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists()or source==destination:raise ValueError('PatchGAN requires a new immutable continuation')
    parent=torch.load(source,map_location='cpu',weights_only=False)
    if (parent.get('config',{}).get('enemy_decoder_restoration')is not True
            or parent['config'].get('restoration_blend_scale')!=1.
            or parent.get('restoration_schedule',{}).get('phase')!='restoration'
            or any(k not in parent for k in ('optimizer','optimizer_parameter_names','sampler',
                                             'rng_numpy','rng_python','rng_torch','rng_cuda'))):
        raise ValueError('Matched GAN continuation requires a complete full-blend restoration source')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model=build_upscaler(parent['config']);model.load_state_dict(parent['model'],strict=True)
        discriminator=EnemyPatchDiscriminator()
    # Validate named mixed-age Adam restoration; retain the serialized states exactly.
    restore_semantic_optimizer(model,parent)
    optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=SETTINGS['discriminator_lr'],
                                 betas=tuple(SETTINGS['discriminator_betas']))
    sha=_sha(source)
    receipt=dict(kind=PROTOCOL,source_checkpoint=str(source),source_checkpoint_sha256=sha,
        source_training_steps=parent['training_steps'],source_semantic_updates=parent['semantic_updates'],
        model_tensors_preserved=len(parent['model']),Adam_states_preserved=len(parent['optimizer']['state']),
        source_sampler_draws=parent['sampler']['draws'],generator_new_parameters=0,
        model_parameters=sum(p.numel()for p in model.parameters()),
        discriminator_parameters=sum(p.numel()for p in discriminator.parameters()),
        inherited_model_Adam_RNG_sampler_preserved=True,matched_control_same_initial_state=True,
        seed=seed,settings=deepcopy(SETTINGS),automatic_deployment=False)
    payload=deepcopy(parent)
    payload.update(patchgan_migration=receipt,parent_protocol=deepcopy(parent.get('protocol')),
        parent_best=deepcopy(parent.get('best')),patchgan_updates=0,
        patchgan_state=dict(version=1,updates=0,discriminator_steps=0,
            control_model=deepcopy(parent['model']),control_optimizer=deepcopy(parent['optimizer']),
            discriminator=discriminator.state_dict(),discriminator_optimizer=optimizer_d.state_dict(),
            batch_ledger_sha256=hashlib.sha256(('shared-input-ledger:'+sha).encode()).hexdigest(),
            settings=deepcopy(SETTINGS)),best=None,gameplay_qualified=False,automatic_deployment=False)
    payload.pop('protocol',None)
    _save(destination,payload)
    atomic_json(destination.with_suffix(destination.suffix+'.patchgan.json'),receipt)
    return receipt
