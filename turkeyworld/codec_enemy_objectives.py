"""Source enemy adaptation of compact E/D in the unchanged RF latent space.

The full quantizer state stays frozen. Original/previous codec targets are
prepared by the caller, detached here, and never become trainable teachers.
There is no discriminator or world-model update in this objective.
"""
from __future__ import annotations

import torch
import math
from torch.nn import functional as F

from .object_losses import (balanced_edge_l1, balanced_image_l1, balanced_mse,
                            decoded_ssim_loss, foreground_contrastive_loss)


def _label(value, device, shape, name):
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
        raise ValueError(name+' has an incompatible Source/reference shape')
    value = value.detach().to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name+' must be finite')
    return value


def _pixel_terms(prediction, target, background, mask, prefix):
    return {prefix+'_l1': (balanced_image_l1(prediction, target, mask), 1.),
            prefix+'_ssim': (decoded_ssim_loss(prediction, target, mask), .2),
            prefix+'_edge': (balanced_edge_l1(prediction, target, mask), .1),
            prefix+'_contrast': (foreground_contrastive_loss(prediction, target, background, mask), .1)}


def native_object_crop_terms(prediction,target,mask,*,padding=2):
    """Native pixel crops around visible Source masks; no resize or data grads."""
    target=target.detach();mask=mask.detach();l1=[];edges=[]
    if prediction.shape!=target.shape or mask.shape!=(len(prediction),1,*prediction.shape[-2:]):
        raise ValueError('Native object crops require aligned RGB and single-channel masks')
    if type(padding)is not int or padding<0:raise ValueError('Crop padding must be nonnegative')
    height,width=prediction.shape[-2:]
    for i in range(len(prediction)):
        y,x=torch.where(mask[i,0]>.5)
        if not len(x):continue
        x0=max(0,int(x.min())-padding);x1=min(width,int(x.max())+1+padding)
        y0=max(0,int(y.min())-padding);y1=min(height,int(y.max())+1+padding)
        p=prediction[i:i+1,:,y0:y1,x0:x1].float();t=target[i:i+1,:,y0:y1,x0:x1].float()
        l1.append(F.l1_loss(p,t));parts=[]
        if x1-x0>1:parts.append(F.l1_loss(p[:,:,:,1:]-p[:,:,:,:-1],t[:,:,:,1:]-t[:,:,:,:-1]))
        if y1-y0>1:parts.append(F.l1_loss(p[:,:,1:]-p[:,:,:-1],t[:,:,1:]-t[:,:,:-1]))
        edges.append(sum(parts)/len(parts)if parts else p.sum()*0)
    zero=prediction.sum()*0
    return (torch.stack(l1).mean()if l1 else zero,torch.stack(edges).mean()if edges else zero,len(l1))


def codec_enemy_loss(student, batch, references, *, decoder_only=False,object_crop_weight=0.,object_crop_edge_weight=.1):
    """Warm up D on original Source codes, then adapt E/D with RF replay.

    ``decoder_only=True`` performs no encoder/quantizer forward at all. It
    supervises D(original_quantized) against real Source pixels, avoiding a
    request to paint an enemy from the current compact encoder's wrong codes.
    Joint training adds Source reconstruction, original latent alignment and
    old-Base encoder alignment. Reference tensors and Source arrays never get
    gradients. The caller owns E/D optimizer groups and phase requires_grad.
    """
    if not isinstance(decoder_only, bool):
        raise ValueError('decoder_only must be an explicit boolean')
    if any(not isinstance(v,(int,float))or isinstance(v,bool)or not math.isfinite(v)or v<0
           for v in (object_crop_weight,object_crop_edge_weight)):
        raise ValueError('Native object crop weights must be finite and nonnegative')
    quantizer = student.quantize
    if (quantizer.training or any(value.requires_grad for value in quantizer.parameters())
            or not bool(quantizer.initialized)):
        raise ValueError('Compact enemy adaptation requires an initialized frozen quantizer in eval')
    device = next(student.parameters()).device
    target_value = batch['target']
    if not torch.is_tensor(target_value) or target_value.ndim != 4 or target_value.shape[1] != 3:
        raise ValueError('Codec enemy Source targets require RGB[B,3,H,W]')
    size, count = student.config.image_size, len(target_value)
    if count < 1:
        raise ValueError('Codec enemy batch must be nonempty')
    rgb_shape = (count, 3, size, size)
    target = _label(target_value, device, rgb_shape, 'target')
    background = _label(batch['background'], device, rgb_shape, 'background')
    mask = _label(batch['object_mask'], device, (count, 1, size, size), 'object_mask')
    if any(bool(((value < 0) | (value > 1)).any()) for value in (target, background, mask)):
        raise ValueError('Source RGB/background/mask labels must lie in[0,1]')
    latent_shape = (count, student.config.embedding_dim, student.config.latent_size, student.config.latent_size)
    original_z = _label(references['original_quantized'], device, latent_shape, 'original_quantized')
    rf_z = _label(references['rf_quantized'], device, latent_shape, 'rf_quantized')
    old_rf = _label(references['old_rf_rgb'], device, rgb_shape, 'old_rf_rgb')
    original_rf = _label(references['original_rf_rgb'], device, rgb_shape, 'original_rf_rgb')
    decoded_source = student.decode(original_z)
    decoded_rf = student.decode(rf_z)
    if not bool(torch.isfinite(decoded_source).all() and torch.isfinite(decoded_rf).all()):
        raise FloatingPointError('Nonfinite compact decoder enemy/RF reconstruction')
    terms = _pixel_terms(decoded_source, target, background, mask, 'decoder_source')
    if object_crop_weight:
        crop,edge,crop_count=native_object_crop_terms(decoded_source,target,mask)
        terms.update(decoder_source_crop_l1=(crop,object_crop_weight),
                     decoder_source_crop_edge=(edge,object_crop_weight*object_crop_edge_weight))
    terms.update(rf_old_replay_l1=(F.l1_loss(decoded_rf.float(), old_rf), .25),
                 rf_original_replay_l1=(F.l1_loss(decoded_rf.float(), original_rf), .05))
    if not decoder_only:
        original_prequant = _label(references['original_prequant'], device, latent_shape, 'original_prequant')
        base_rgb = _label(references['base_rgb'], device, rgb_shape, 'base_rgb')
        old_base_prequant = _label(references['old_base_prequant'], device, latent_shape, 'old_base_prequant')
        encoded = student.encode(target)
        reconstructed = student.decode(encoded['quantized'])
        base = student.encode(base_rgb)
        if not bool(torch.isfinite(reconstructed).all() and torch.isfinite(encoded['prequant']).all()
                    and torch.isfinite(base['prequant']).all()):
            raise FloatingPointError('Nonfinite compact encoder enemy/Base reconstruction')
        terms.update(_pixel_terms(reconstructed, target, background, mask, 'encoder_source'))
        if object_crop_weight:
            crop,edge,_=native_object_crop_terms(reconstructed,target,mask)
            terms.update(encoder_source_crop_l1=(crop,object_crop_weight),
                         encoder_source_crop_edge=(edge,object_crop_weight*object_crop_edge_weight))
        terms.update(encoder_prequant_mse=(balanced_mse(encoded['prequant'], original_prequant, mask), .25),
                     encoder_vq_loss=(encoded['vq_loss'], .1),
                     base_encoder_compat_mse=(F.mse_loss(base['prequant'].float(), old_base_prequant), .5))
    total = sum(value*weight for value, weight in terms.values())
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite compact codec enemy objective')
    metrics = {}
    for name, (value, weight) in terms.items():
        metrics[name] = float(value.detach())
        metrics['weight_'+name] = weight
        metrics['weighted_'+name] = float(value.detach())*weight
    metrics.update(train_loss=float(total.detach()), decoder_only=decoder_only,
        encoder_forward_active=not decoder_only, codebook_trainable=False, quantizer_eval=True,
        gan_active=False, source_examples=count, visible_enemy_examples=int(mask.flatten(1).any(-1).sum()),
        foreground_pixel_fraction=float(mask.mean()), reference_gradients=False,
        region_reduction='equal_present_foreground_background_per_image',
        rf_replay_scope='same fixed actual RF quantized endpoints; full-pipeline new-encoder regression is separate')
    if object_crop_weight:
        metrics.update(object_crop_active=True,object_crop_examples=crop_count,
            object_crop_padding=2,object_crop_resize=False,object_crop_scope='Source decoded/reconstructed native RGB only; no RF crop')
    return total, metrics
