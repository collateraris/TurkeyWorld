"""Conditional wall-only PatchGAN: Source masks select labels, never SR inputs."""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .renderer_wall_supervision import source_wall_masks

SETTINGS=dict(protocol='wall_patchgan_sr_v1',patch_size=32,patches_per_image=2,
    wall_coverage=1.,erosion_pixels=2,hinge=True,feature_matching_weight=.1,
    warmup_updates=25,ramp_updates=100,max_gan_weight=.002,gradient_ratio_limit=.05,
    wall_l1_weight=.5,wall_edge_weight=.25,wall_high_frequency_weight=.25,
    generator_lr=.00001,discriminator_lr=.000005,discriminator_betas=[0.,.99],batch=8)


class WallPatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([
            nn.Sequential(spectral_norm(nn.Conv2d(6,24,4,2,1)),nn.LeakyReLU(.2)),
            nn.Sequential(spectral_norm(nn.Conv2d(24,48,4,2,1)),nn.LeakyReLU(.2)),
            nn.Sequential(spectral_norm(nn.Conv2d(48,96,3,1,1)),nn.LeakyReLU(.2))])
        self.logits=spectral_norm(nn.Conv2d(96,1,3,1,1))

    def forward(self,condition,candidate,*,features=False):
        value=torch.cat((condition,candidate),1);outputs=[]
        for block in self.blocks:
            value=block(value);outputs.append(value)
        logits=self.logits(value)
        return (logits,outputs) if features else logits


def wall_labels(batch):
    return source_wall_masks(batch['map'],batch['state'],batch['mask_hr'],native_size=128)['wall'].detach()


def wall_reconstruction(image,target,mask):
    target=target.detach();mask=mask.detach()
    def mean(error,support):
        mass=support.flatten(1).sum(-1)*error.shape[1]
        values=(error*support).flatten(1).sum(-1)/mass.clamp_min(1)
        present=mass>0
        return values[present].mean() if bool(present.any()) else error.sum()*0
    l1=mean((image-target).abs(),mask)
    edges=[]
    for axis in (-1,-2):
        a,b=[slice(None)]*4,[slice(None)]*4;a[axis]=slice(1,None);b[axis]=slice(None,-1)
        support=torch.minimum(mask[tuple(a)],mask[tuple(b)])
        edges.append(mean((image.diff(dim=axis)-target.diff(dim=axis)).abs(),support))
    edge=sum(edges)/2
    highpass=lambda x:x-F.avg_pool2d(x,5,1,2,count_include_pad=False)
    interior=1-F.max_pool2d(1-mask,5,1,2)
    high=mean((highpass(image)-highpass(target)).abs(),interior)
    loss=SETTINGS['wall_l1_weight']*l1+SETTINGS['wall_edge_weight']*edge+SETTINGS['wall_high_frequency_weight']*high
    return loss,dict(wall_l1=float(l1.detach()),wall_edge=float(edge.detach()),wall_high_frequency=float(high.detach()),
        wall_reconstruction_loss=float(loss.detach()),wall_images=int(mask.flatten(1).any(-1).sum()))


def wall_patch_pairs(low,fake,target,mask,rng):
    """Identical coordinates/condition for real/fake; only actual exposed walls."""
    size=SETTINGS['patch_size'];radius=SETTINGS['erosion_pixels']
    interior=1-F.max_pool2d(1-mask.detach(),2*radius+1,1,radius)
    coverage=F.avg_pool2d(interior,size,stride=8)
    condition=F.interpolate(low.detach(),size=(128,128),mode='bilinear',align_corners=False)
    fields={name:[] for name in ('condition','fake','real')};coordinates=[]
    for i in range(len(fake)):
        positions=(coverage[i,0]>=SETTINGS['wall_coverage']).nonzero(as_tuple=False).cpu().numpy()
        if not len(positions):continue
        for j in rng.choice(len(positions),size=min(SETTINGS['patches_per_image'],len(positions)),replace=False):
            y,x=(positions[int(j)]*8).tolist();coordinates.append([i,y,x])
            fields['condition'].append(condition[i,:,y:y+size,x:x+size])
            fields['fake'].append(fake[i,:,y:y+size,x:x+size])
            fields['real'].append(target[i,:,y:y+size,x:x+size].detach())
    if not coordinates:return None
    return {**{k:torch.stack(v) for k,v in fields.items()},'coordinates':coordinates}


@contextmanager
def frozen_discriminator(model):
    mode=model.training;flags=[p.requires_grad for p in model.parameters()]
    model.eval().requires_grad_(False)
    try:yield
    finally:
        model.train(mode)
        for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)


def discriminator_loss(model,pairs):
    real=model(pairs['condition'].detach(),pairs['real'].detach())
    fake=model(pairs['condition'].detach(),pairs['fake'].detach())
    return F.relu(1-real).mean()+F.relu(1+fake).mean()


def generator_loss(model,pairs):
    with frozen_discriminator(model):
        logits,fake=model(pairs['condition'].detach(),pairs['fake'],features=True)
        with torch.no_grad():_,real=model(pairs['condition'].detach(),pairs['real'].detach(),features=True)
        adversarial=-logits.mean();matching=torch.stack([(a-b).abs().mean() for a,b in zip(fake,real)]).mean()
        loss=adversarial+SETTINGS['feature_matching_weight']*matching
    return loss,dict(adversarial_loss=float(adversarial.detach()),feature_matching_loss=float(matching.detach()))


def gan_weight(updates):
    return SETTINGS['max_gan_weight']*max(0.,min(1.,(updates-SETTINGS['warmup_updates'])/SETTINGS['ramp_updates']))
