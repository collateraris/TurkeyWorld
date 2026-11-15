"""Direct supervision for RGB sprite geometry, visibility and seven-way phase."""
import torch
from torch.nn import functional as F
from .atlas_enemy_objectives import source_full_sprite_geometry
from .enemy_decoder_objectives import balanced_binary_loss,_mask_terms
from .enemy_decoder_sr import box_iou

PREFIXES=('enemy_decoder_visual.','enemy_decoder_heatmap.','enemy_decoder_size.',
    'enemy_decoder_box.','enemy_decoder_animation.','enemy_decoder_visible.',
    'enemy_decoder_visual_confirmation.','enemy_decoder_global_presence.')
PROTOCOL='patrol_sprite_heads_refinement_v1'
WEIGHTS=dict(presence=.5,confirmation=.5,phase=1.,bbox_l1=.5,bbox_iou=.25,
    height=.1,visibility_bce=.5,visibility_dice=.25)
GROUND_PROTOCOL='patrol_sprite_ground_anchor_v2'
GROUND_WEIGHTS={**WEIGHTS,'height':.5,'visibility_bce':1.,'visibility_dice':.5,
    'bottom_anchor':2.,'vertical_center':1.,'square_shape':.5}


def configure(model):
    model.eval().requires_grad_(False)
    for name,p in model.sprite_renderer.named_parameters():p.requires_grad_(name.startswith(PREFIXES))
    return [name for name,p in model.named_parameters() if p.requires_grad]


def objective(output,batch,*,ground=False):
    mask=batch['mask_hr'].detach().float();visible=mask.flatten(1).any(-1)
    labels=source_full_sprite_geometry(batch['state'].detach())
    geometry=visible&labels['geometry_valid'];box=output['visual_full_bbox'].float();target=labels['full_bbox'].detach()
    def selected(values,selection):return values[selection].mean() if bool(selection.any()) else values.sum()*0
    bce,dice=_mask_terms(output['visual_visible_logits_hr'],mask,torch.ones_like(visible))
    overlap=box_iou(box,target)
    terms=dict(presence=balanced_binary_loss(output['presence_logits'],visible),
        confirmation=balanced_binary_loss(output['visual_confirmation_logits'],visible&(overlap.detach()>=.1)),
        phase=selected(F.cross_entropy(output['visual_animation_logits'],labels['phase'],reduction='none'),visible),
        bbox_l1=selected((box-target).abs().mean(-1),geometry),bbox_iou=selected(1-overlap,geometry),
        height=selected(F.smooth_l1_loss((box[:,3]-box[:,1]).clamp_min(.001).log(),
            labels['full_sprite_height'].clamp_min(.001).log(),reduction='none'),geometry),
        visibility_bce=bce,visibility_dice=dice)
    if ground:
        terms.update(bottom_anchor=selected((box[:,3]-target[:,3]).abs(),geometry),
            vertical_center=selected(((box[:,1]+box[:,3]-target[:,1]-target[:,3])/2).abs(),geometry),
            square_shape=selected(((box[:,2]-box[:,0])-(box[:,3]-box[:,1])).abs(),geometry))
    weights=GROUND_WEIGHTS if ground else WEIGHTS
    loss=sum(weights[k]*v for k,v in terms.items())
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite sprite-head objective')
    return loss,{**{k:float(v.detach()) for k,v in terms.items()},'train_loss':float(loss.detach()),
        'visible':int(visible.sum()),'Source_inputs':False,'GAN_active':False}


def sample_indices(data,rng,step):
    normal=[i for i in range(len(data)) if data.visible[i] and data.animation_labels[i]<3]
    combat=[i for i in range(len(data)) if data.visible[i] and data.animation_labels[i]>=3]
    empty=[i for i in range(len(data)) if not data.visible[i]]
    if not normal or not combat or not empty:raise ValueError('Need patrol/combat-negative/empty training cases')
    result=[]
    for n in range(4):
        distance=('near','mid','far')[(step+n)%3]
        ids=[i for i in normal if data.distance[i]==distance] or normal
        result.append(int(rng.choice(ids)))
    result.extend(int(i) for i in rng.choice(combat,2));result.extend(int(i) for i in rng.choice(empty,2))
    rng.shuffle(result);return result
