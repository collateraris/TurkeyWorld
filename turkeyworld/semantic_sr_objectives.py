"""Source-only image/shape labels for RGB-only semantic super-resolution."""
import torch
from torch.nn import functional as F

from .codec_enemy_objectives import native_object_crop_terms
from .object_losses import (balanced_image_l1, balanced_edge_l1, decoded_ssim_loss,
                            foreground_contrastive_loss)

LOSS_WEIGHTS = {'pixel': 1., 'ssim': .2, 'edge': .1, 'crop': .1,
                'crop_edge': .01, 'foreground_contrast': .1,
                'mask_bce': .1, 'mask_dice': .05, 'presence_bce': .05}


def semantic_sr_loss(output, batch):
    """Only predictions receive gradients; Source masks never enter the model."""
    image = output['image']
    target = batch['hr_source'].detach().float()
    high_mask, low_mask = (batch[key].detach().float() for key in ('mask_hr', 'mask_lr'))
    logits, presence = output['mask_logits_lr'].float(), output['presence_logits'].float()
    if (image.shape != target.shape or logits.shape != low_mask.shape
            or high_mask.shape != (len(image), 1, 128, 128)
            or presence.shape != (len(image), 1)):
        raise ValueError('Semantic SR image/mask/presence labels are not aligned')
    mask = low_mask.expand_as(logits)
    errors = F.binary_cross_entropy_with_logits(logits, mask, reduction='none').flatten(1)
    flat = mask.flatten(1)
    pos, neg = flat.sum(-1), (1-flat).sum(-1)
    mask_bce = (((errors*flat).sum(-1)/pos.clamp_min(1)
                +(errors*(1-flat)).sum(-1)/neg.clamp_min(1))
                /((pos > 0).float()+(neg > 0).float()).clamp_min(1)).mean()
    probabilities = logits.sigmoid().flatten(1)
    dice = (1-(2*(probabilities*flat).sum(-1)+1e-6)
            /(probabilities.sum(-1)+pos+1e-6)).mean()
    presence_label = (pos > 0).float()[:, None]
    crop, crop_edge, crop_count = native_object_crop_terms(image, target, high_mask, padding=8)
    terms = {'pixel': balanced_image_l1(image, target, high_mask),
        'ssim': decoded_ssim_loss(image, target, high_mask),
        'edge': balanced_edge_l1(image, target, high_mask),
        'crop': crop, 'crop_edge': crop_edge,
        'foreground_contrast': foreground_contrastive_loss(image, target,
            batch['hr_background'].detach(), high_mask),
        'mask_bce': mask_bce, 'mask_dice': dice,
        'presence_bce': F.binary_cross_entropy_with_logits(presence, presence_label)}
    loss = sum(LOSS_WEIGHTS[name]*value for name, value in terms.items())
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite semantic SR objective')
    metrics = {name: float(value.detach()) for name, value in terms.items()}
    metrics.update({name+'_weighted': LOSS_WEIGHTS[name]*value for name, value in metrics.copy().items()})
    metrics.update(train_loss=float(loss.detach()), visible_images=int((pos > 0).sum()),
                   empty_images=int((pos == 0).sum()), native_crop_images=int(crop_count),
                   source_labels_detached=True, inference_rgb_only=True)
    return loss, metrics
