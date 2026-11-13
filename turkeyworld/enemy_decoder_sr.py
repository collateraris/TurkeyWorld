"""Learned RGBA enemy restoration from RGB and a *predicted* RSSM hypothesis.

Camera geometry proposes a region. RGB-only confirmation and a learned visible
mask decide where a decoded layer is admissible. No atlas, Source image, map,
wall query or ground-truth mask participates in inference.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .models import SRResidualBlock
from .semantic_sr import SemanticEDSRUpscaler
from .enemy_animation import enemy_animation_index


def box_iou(first, second):
    intersection = (torch.minimum(first[:, 2:], second[:, 2:])
                    -torch.maximum(first[:, :2], second[:, :2])).clamp_min(0).prod(-1)
    area1=(first[:,2:]-first[:,:2]).clamp_min(0).prod(-1)
    area2=(second[:,2:]-second[:,:2]).clamp_min(0).prod(-1)
    return intersection/(area1+area2-intersection).clamp_min(1e-8)


def sample_box(image, box, size):
    """Differentiable RGB/feature ROI query. Boxes use normalized xyxy."""
    axis=(torch.arange(size,device=image.device,dtype=torch.float32)+.5)/size
    y,x=torch.meshgrid(axis,axis,indexing='ij')
    points=torch.stack((x,y),-1)[None]
    grid=box[:,None,None,:2]+points*(box[:,None,None,2:]-box[:,None,None,:2]).clamp_min(1e-6)
    with torch.autocast(image.device.type,enabled=False):
        result=F.grid_sample(image.float(),grid.float()*2-1,align_corners=False,padding_mode='zeros')
    return result.to(image.dtype)


def warp_box(canonical, box, size=128):
    axis=(torch.arange(size,device=canonical.device,dtype=torch.float32)+.5)/size
    y,x=torch.meshgrid(axis,axis,indexing='ij')
    points=torch.stack((x,y),-1)[None]
    grid=(points-box[:,None,None,:2])/(box[:,None,None,2:]-box[:,None,None,:2]).clamp_min(1e-6)
    with torch.autocast(canonical.device.type,enabled=False):
        return F.grid_sample(canonical.float(),grid.float()*2-1,align_corners=False,padding_mode='zeros')


def project_enemy_hypothesis(hypothesis, *, batch, device):
    """68-degree camera proposal from detached predicted state20+validbit."""
    if hypothesis is None:
        values=torch.zeros(batch,21,device=device,dtype=torch.float32)
    else:
        if (not torch.is_tensor(hypothesis) or hypothesis.shape!=(batch,21)
                or not hypothesis.is_floating_point() or not bool(torch.isfinite(hypothesis).all())
                or bool(((hypothesis[:,-1]!=0)&(hypothesis[:,-1]!=1)).any())):
            raise ValueError('Hypothesis must be finite predicted state[B,20]+binary validbit[B,1]')
        values=hypothesis.detach().to(device=device,dtype=torch.float32)
        if not bool(torch.isfinite(values).all()):
            raise ValueError('Hypothesis must have a finite FP32 representation')
    state=values[:,:20]
    heading=state[:,2:4]
    norm=heading.norm(dim=-1,keepdim=True)
    sine,cosine=(heading/norm.clamp_min(1e-6)).unbind(-1)
    dx,dy=((state[:,8:10]-state[:,:2])*16).unbind(-1)
    depth=dx*cosine+dy*sine
    lateral=-dx*sine+dy*cosine
    projection=1/(2*math.tan(math.radians(68)/2))
    height=(projection*.93/depth.clamp_min(.05)).clamp_max(5.)
    center=.5+projection*lateral/depth.clamp_min(.05)
    bottom=.5+projection*.5/depth.clamp_min(.05)
    box=torch.stack((center-height/2,bottom-height,center+height/2,bottom),-1).clamp(0,1)
    xy=torch.cat((state[:,:2],state[:,8:10]),-1)
    valid=((values[:,-1]>0)&(norm[:,0]>1e-6)&(depth>.05)
        &(xy>=0).all(-1)&(xy<=1).all(-1)&((box[:,2:]-box[:,:2])>1e-5).all(-1))
    return values,torch.where(valid[:,None],box,torch.zeros_like(box)),valid[:,None]


class EnemyDecoderEDSRUpscaler(SemanticEDSRUpscaler):
    def __init__(self, channels=64,blocks=16,scale=4,res_scale=.1,
                 semantic_channels=32,semantic_blocks=4,enemy_decoder_channels=24,
                 enemy_decoder_stable=False,enemy_decoder_visual_first=False,
                 enemy_decoder_visual_threshold=.9,enemy_decoder_visual_size_max=1.5,
                 enemy_decoder_restoration=False,restoration_blend_scale=0.0):
        if type(enemy_decoder_channels)is not int or enemy_decoder_channels<4:
            raise ValueError('Enemy decoder needs at least4 feature channels')
        super().__init__(channels,blocks,scale,res_scale,semantic_channels,semantic_blocks)
        if type(enemy_decoder_stable)is not bool:
            raise ValueError('Enemy decoder stable profile must be a boolean')
        if (type(enemy_decoder_visual_first) is not bool
                or enemy_decoder_visual_first and not enemy_decoder_stable):
            raise ValueError('Visual-first decoding requires the explicit frozen-parent stable profile')
        if (type(enemy_decoder_restoration) is not bool
                or enemy_decoder_restoration and not (enemy_decoder_stable and enemy_decoder_visual_first)):
            raise ValueError('Restoration decoding requires the explicit stable visual-first profile')
        if (type(restoration_blend_scale) not in (float,int)
                or not math.isfinite(restoration_blend_scale) or not 0 <= restoration_blend_scale <= 1
                or not enemy_decoder_restoration and restoration_blend_scale != 0):
            raise ValueError('Restoration blend must be a finite explicit scale in[0,1] for its opt-in profile')
        if (type(enemy_decoder_visual_threshold) not in (float,int)
                or not .5 < enemy_decoder_visual_threshold < 1):
            raise ValueError('Visual-first confidence threshold must be finite and between.5 and1')
        if (type(enemy_decoder_visual_size_max) not in (float,int)
                or not .75 <= enemy_decoder_visual_size_max <= 2):
            raise ValueError('Visual-first size upper range must be finite between.75 and2 frames')
        if enemy_decoder_stable:
            # Old classifiers participate in the old RGB gate too. Freeze the
            # ENTIRE parent, not merely its RGB convolution readouts.
            for parameter in self.parameters():parameter.requires_grad_(False)
        self.config.update(architecture='enemy_decoder_edsr',enemy_decoder_channels=enemy_decoder_channels)
        if enemy_decoder_stable:self.config['enemy_decoder_stable']=True
        width=enemy_decoder_channels
        self.enemy_decoder_visual=nn.Sequential(nn.Conv2d(3,width,3,padding=1),nn.GELU(),
            SRResidualBlock(width),SRResidualBlock(width))
        self.enemy_decoder_heatmap=nn.Conv2d(width,1,1)
        self.enemy_decoder_size=nn.Sequential(nn.Linear(width*2,32),nn.GELU(),nn.Linear(32,2))
        self.enemy_decoder_box=nn.Sequential(nn.Linear(width*2,32),nn.GELU(),nn.Linear(32,4))
        self.enemy_decoder_confirmation=nn.Sequential(nn.Linear(width*2,32),nn.GELU(),nn.Linear(32,1))
        nn.init.constant_(self.enemy_decoder_confirmation[-1].bias,-2.)
        self.enemy_decoder_condition=nn.Sequential(nn.Linear(31 if enemy_decoder_stable else 24,64),
            nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.enemy_decoder_seed=nn.Linear(64+width*2,width*8*8)
        def upsample():
            return nn.Sequential(nn.Conv2d(width,width*4,3,padding=1),nn.PixelShuffle(2),nn.GELU(),
                nn.Conv2d(width,width*4,3,padding=1),nn.PixelShuffle(2),nn.GELU())
        self.enemy_decoder_rgba=nn.Sequential(upsample(),nn.Conv2d(width,4,3,padding=1))
        self.enemy_decoder_visible=nn.Sequential(upsample(),nn.Conv2d(width,1,1))
        # A module parameter, appended after all old modules: canonical optimizer
        # ordering remains parent-first instead of putting a root parameter first.
        self.enemy_decoder_gain=nn.Linear(1,1,bias=False)
        nn.init.zeros_(self.enemy_decoder_gain.weight)
        if enemy_decoder_stable:
            self.enemy_decoder_global_presence=nn.Linear(width*2,1)
            self.enemy_decoder_animation=nn.Linear(width*2,7)
        if enemy_decoder_visual_first:
            self.config.update(enemy_decoder_visual_first=True,
                enemy_decoder_visual_threshold=float(enemy_decoder_visual_threshold),
                enemy_decoder_visual_size_max=float(enemy_decoder_visual_size_max))
            # Append after ALL149 inherited parameters so their named optimizer
            # groups and mixed-age moments are reusable without reordering.
            self.enemy_decoder_visual_condition=nn.Sequential(
                nn.Linear(width*2+4+7,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
            nn.init.zeros_(self.enemy_decoder_visual_condition[2].weight)
            nn.init.zeros_(self.enemy_decoder_visual_condition[2].bias)
            self.enemy_decoder_visual_confirmation=nn.Sequential(
                nn.Linear(width*2,32),nn.GELU(),nn.Linear(32,1))
            self.enemy_decoder_visual_confirmation.load_state_dict(self.enemy_decoder_confirmation.state_dict())
            self.enemy_decoder_visual_gain=nn.Linear(1,1,bias=False)
            nn.init.zeros_(self.enemy_decoder_visual_gain.weight)
        if enemy_decoder_restoration:
            self.config['enemy_decoder_restoration']=True
            self.set_restoration_blend_scale(restoration_blend_scale)

    def set_restoration_blend_scale(self,value):
        """Persist the explicit curriculum blend; loading never implies full blend.

        This operational scalar adds no parameter, buffer or random draw. A
        saved standalone config carries the exact same inference scale.
        """
        if (not self.config.get('enemy_decoder_restoration',False)
                or type(value) not in (float,int) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Restoration blend requires its opt-in profile and a finite scale in[0,1]')
        self.restoration_blend_scale=float(value)
        self.config['restoration_blend_scale']=float(value)

    def _visual_first_output(self,result,base,visual,pooled,visual_proposal,visible_logits):
        """Select one RGB-confirmed layer; the visual path never reads state.

        Routing confidence is a learned score, not a calibrated probability.
        Train mixes uncertain routes softly so a zero visual gain learns from
        the first update. Confident vision has absolute priority in both modes;
        inference selects one hard route. The hint cannot veto vision.
        """
        roi=sample_box(visual,visual_proposal,8)
        summary=torch.cat((roi.mean((-2,-1)),roi.amax((-2,-1))),-1)
        confirmation_logits=self.enemy_decoder_visual_confirmation(summary).float()
        animation_logits=self.enemy_decoder_animation(summary).float()
        phase_probability=animation_logits.softmax(-1)
        condition=torch.cat((summary.float(),visual_proposal,phase_probability),-1)
        embedding=self.enemy_decoder_visual_condition(
            condition.to(self.enemy_decoder_visual_condition[0].weight.dtype))
        seed=self.enemy_decoder_seed(torch.cat((embedding,summary),-1))
        seed=seed.reshape(-1,self.config['enemy_decoder_channels'],8,8)
        rgba=self.enemy_decoder_rgba(seed).sigmoid()
        extent=(visual_proposal[:,2:]-visual_proposal[:,:2]).clamp_min(1e-5)
        correction=.15*self.enemy_decoder_box(summary).float().tanh()
        box=(visual_proposal+correction*extent.repeat(1,2)).clamp(0,1)
        box=torch.cat((torch.minimum(box[:,:2],box[:,2:]),torch.maximum(box[:,:2],box[:,2:])),-1)
        box_valid=((box[:,2:]-box[:,:2])>1e-5).all(-1,keepdim=True)
        warped=warp_box(rgba,box)
        visual_alpha=warped[:,3:4]*visible_logits.sigmoid()*box_valid[:,:,None,None].float()
        presence=result['presence_logits'].float().sigmoid()
        visual_confirmation=confirmation_logits.sigmoid()
        confidence=torch.minimum(presence,visual_confirmation)
        threshold=self.config['enemy_decoder_visual_threshold']
        visual_hard=(presence>=threshold)&(visual_confirmation>=threshold)&box_valid
        guided_hard=(~visual_hard)&(presence>=.5)&result['enemy_applicable']
        guided_hard=guided_hard&(result['enemy_confirmation_logits']>=0)&(result['projection_agreement']>=.05)
        visual_soft=presence*visual_confirmation*box_valid.float()
        guided_soft=((1-visual_soft)*presence*result['enemy_confirmation_logits'].sigmoid()
                     *result['projection_agreement']*result['enemy_applicable'].float())
        if self.training:
            visual_weight=torch.where(visual_hard,torch.ones_like(visual_soft),visual_soft)
            guided_weight=torch.where(visual_hard,torch.zeros_like(guided_soft),guided_soft)
        else:
            visual_weight=visual_hard.float()
            guided_weight=guided_hard.float()
        raw_gain=self.enemy_decoder_visual_gain.weight[0,0].tanh()
        hard_gain=raw_gain.clamp(0,1)
        gain=hard_gain.detach()+(raw_gain-raw_gain.detach()) if self.training else hard_gain
        if self.config.get('enemy_decoder_restoration',False):
            # Direct raw-layer losses train RGB and alpha even at scale zero.
            # The display curriculum cannot be defeated by a learned tiny gain.
            result['restoration_visual_legacy_gain']=hard_gain.detach()
            gain=raw_gain.new_tensor(self.restoration_blend_scale)
        visual_composite_alpha=visual_alpha*gain.float()*visual_weight[:,:,None,None]
        guided_image=result['image']
        proposed=(base.float()+visual_composite_alpha*(warped[:,:3]-base.float())
                  +guided_weight[:,:,None,None]*(guided_image.float()-base.float()))
        bounded=proposed.clamp(0,1)
        image=(bounded.detach()+(proposed-proposed.detach()) if self.training else bounded).to(base.dtype)
        guided={
            'guided_rgba':result['enemy_rgba'],'guided_bbox':result['enemy_bbox'],
            'guided_hypothesis_bbox':result['hypothesis_bbox'],
            'guided_confirmation_logits':result['enemy_confirmation_logits'],
            'guided_applicable':result['enemy_applicable'],
            'guided_alpha_hr':result['enemy_alpha_hr'],
            'guided_animation_logits':result['enemy_animation_logits'],
            'guided_animation_index':result['enemy_animation_index'],
        }
        result.update(**guided,image=image,visual_proposal_bbox=visual_proposal,
            visual_bbox=box,visual_rgba=rgba,visual_confirmation_logits=confirmation_logits,
            visual_visible_logits_hr=visible_logits,visual_animation_logits=animation_logits,
            visual_animation_index=animation_logits.argmax(-1),
            visual_confidence=confidence,visual_route=visual_hard,guided_route=guided_hard,
            visual_route_weight=visual_weight,guided_route_weight=guided_weight,
            visual_alpha_hr=visual_alpha,visual_composite_alpha_hr=visual_composite_alpha,
            visual_gain=gain.float(),visual_route_threshold=threshold,
            visual_animation_source='RGB_only',visual_first=True,
            routing='soft_uncertain_routes_confident_visual_priority' if self.training else 'one_hard_route')
        if self.config.get('enemy_decoder_restoration',False):
            result.update(restoration=True,restoration_blend_scale=self.restoration_blend_scale,
                restoration_base_rgb=base,visual_raw_warped_rgba=warped)
        return result

    def forward_with_aux(self,low_rgb,hypothesis=None):
        result=super().forward_with_aux(low_rgb)
        values,proposal,applicable=project_enemy_hypothesis(hypothesis,batch=len(low_rgb),device=low_rgb.device)
        visual=self.enemy_decoder_visual(low_rgb)
        pooled=torch.cat((visual.mean((-2,-1)),visual.amax((-2,-1))),-1)
        heat=self.enemy_decoder_heatmap(visual).float().flatten(1).softmax(-1).reshape(-1,32,32)
        axis=(torch.arange(32,device=low_rgb.device,dtype=torch.float32)+.5)/32
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        center=torch.stack(((heat*xx).sum((-2,-1)),(heat*yy).sum((-2,-1))),-1)
        size_range=(self.config['enemy_decoder_visual_size_max']
                    if self.config.get('enemy_decoder_visual_first',False)else .75)
        size=.02+size_range*self.enemy_decoder_size(pooled).float().sigmoid()
        visual_box=torch.cat((center-size/2,center+size/2),-1).clamp(0,1)
        roi=sample_box(visual,proposal,8)
        roi_summary=torch.cat((roi.mean((-2,-1)),roi.amax((-2,-1))),-1)
        # This head sees queried RGB features ONLY, not the state embedding.
        confirmation_logits=self.enemy_decoder_confirmation(roi_summary).float()
        correction=.15*self.enemy_decoder_box(roi_summary).float().tanh()
        extent=(proposal[:,2:]-proposal[:,:2]).clamp_min(1e-5)
        box=(proposal+correction*extent.repeat(1,2)).clamp(0,1)
        box=torch.cat((torch.minimum(box[:,:2],box[:,2:]),torch.maximum(box[:,:2],box[:,2:])),-1)
        safe_state=torch.where(applicable,values[:,:20],torch.zeros_like(values[:,:20])).clamp(-10,10)
        cond=torch.cat((safe_state,proposal),-1)
        animation=None
        if self.config.get('enemy_decoder_stable',False):
            predicted=enemy_animation_index(values[:,:20])
            animation=torch.where(applicable[:,0],predicted,torch.full_like(predicted,-1))
            onehot=F.one_hot(predicted,7).float()*applicable.float()
            cond=torch.cat((cond,onehot),-1)
        cond=cond.to(self.enemy_decoder_condition[0].weight.dtype)
        embedding=self.enemy_decoder_condition(cond)
        seed=self.enemy_decoder_seed(torch.cat((embedding,roi_summary),-1)).reshape(-1,self.config['enemy_decoder_channels'],8,8)
        rgba=self.enemy_decoder_rgba(seed).sigmoid()
        visible_logits=self.enemy_decoder_visible(visual).float()
        proposal_center=(proposal[:,:2]+proposal[:,2:])/2
        agreement=torch.exp(-.5*(((proposal_center-center)/(.1+extent/2)).square().sum(-1,keepdim=True)))
        confirmation=confirmation_logits.sigmoid()if self.training else(confirmation_logits>=0).float()
        warped=warp_box(rgba,box)
        alpha=(warped[:,3:4]*visible_logits.sigmoid()*confirmation[:,:,None,None]
               *agreement[:,:,None,None]*applicable[:,:,None,None].float())
        raw_gain=self.enemy_decoder_gain.weight[0,0].tanh()
        hard_gain=raw_gain.clamp(0,1)
        gain=hard_gain.detach()+(raw_gain-raw_gain.detach())if self.training else hard_gain
        if self.config.get('enemy_decoder_restoration',False):
            result['restoration_guided_legacy_gain']=hard_gain.detach()
            gain=raw_gain.new_tensor(self.restoration_blend_scale)
        composite_alpha=alpha*gain.float()
        base=result['image']
        proposed=base.float()+composite_alpha*(warped[:,:3]-base.float())
        bounded=proposed.clamp(0,1)
        image=(bounded.detach()+(proposed-proposed.detach())if self.training else bounded).to(base.dtype)
        result.update(image=image,enemy_rgba=rgba,enemy_bbox=box,enemy_visual_bbox=visual_box,
            enemy_alpha_hr=alpha,enemy_composite_alpha_hr=composite_alpha,
            enemy_visible_logits_hr=visible_logits,enemy_confirmation_logits=confirmation_logits,
            enemy_applicable=applicable,hypothesis_bbox=proposal,
            projection_agreement=agreement,enemy_gain=gain.float())
        if self.config.get('enemy_decoder_restoration',False):
            result['guided_raw_warped_rgba']=warped
        if self.config.get('enemy_decoder_stable',False):
            # Recognition losses now train independent new RGB features, while
            # super().forward_with_aux used the frozen old mask/presence for RGB.
            result.update(presence_logits=self.enemy_decoder_global_presence(pooled).float(),
                mask_logits_lr=F.avg_pool2d(visible_logits,kernel_size=4,stride=4),
                enemy_animation_index=animation,
                enemy_animation_logits=self.enemy_decoder_animation(roi_summary).float(),
                legacy_RGB_parent_frozen=True)
        if self.config.get('enemy_decoder_visual_first',False):
            return self._visual_first_output(result,base,visual,pooled,visual_box,visible_logits)
        return result

    def forward(self,low_rgb,hypothesis=None):
        return self.forward_with_aux(low_rgb,hypothesis)['image']
