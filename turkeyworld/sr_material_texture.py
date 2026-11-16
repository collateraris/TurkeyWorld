"""Training-only Source material/texture strata; runtime remains RGB32-only."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from .renderer_wall_depth import _source_ray_faces
from .sr_wall_patchgan import wall_labels


def material_labels(batch):
    exposed=wall_labels(batch)
    tangent=math.tan(math.radians(34))
    screen=(np.arange(128)+.5)/128*2-1
    fields={k:[] for k in ('brick','mortar','earth','grass','grass_edge')}
    for tilemap,state in zip(batch['map'].detach().cpu().numpy(),batch['state'].detach().cpu().double().numpy()):
        x,y=state[:2]*16;angle=math.atan2(state[2],state[3]);c,s=math.cos(angle),math.sin(angle)
        rx,ry=c-s*tangent*screen,s+c*tangent*screen
        depth,faces=_source_ray_faces(tilemap,x,y,rx,ry)
        height=128/(2*tangent)/depth
        v=(np.arange(128)[:,None]-(64-height/2)[None,:])/height[None,:]
        wall=(v>=0)&(v<1)
        u=(((np.where(faces[:,0],x+depth*rx,y+depth*ry))%1)*64).astype(int)
        ty=np.clip((v*64).astype(int),0,63)
        brick=wall&(faces[:,2][None,:]==1)
        earthwall=wall&(faces[:,2][None,:]==2)
        green=ty<10+((u//5)%3)[None,:]*3
        grass=earthwall&green
        edge=np.zeros_like(grass);edge[1:]|=grass[1:]!=grass[:-1];edge[:-1]|=grass[1:]!=grass[:-1]
        values=dict(brick=brick,mortar=brick&((ty%16<2)|((u[None,:]+(ty//16%2)*16)%32<2)),
            earth=earthwall&~green,grass=grass,grass_edge=edge&earthwall)
        for key,value in values.items():fields[key].append(value)
    return {key:torch.as_tensor(np.stack(value)[:,None],device=exposed.device,dtype=exposed.dtype)*exposed for key,value in fields.items()}


def texture_loss(image,target,labels):
    """Equal stratum weighting prevents large flat walls hiding narrow seams."""
    def mean(error,mask):
        count=mask.flatten(1).sum(1)*error.shape[1];present=count>0
        values=(error*mask).flatten(1).sum(1)/count.clamp_min(1)
        return values[present].mean() if bool(present.any()) else image.sum()*0
    total=image.sum()*0;metrics={}
    for name,mask in labels.items():
        color=mean((image-target.detach()).abs(),mask)
        gradients=[]
        for axis in (-1,-2):
            a=[slice(None)]*4;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
            support=torch.maximum(mask[tuple(a)],mask[tuple(b)])
            gradients.append(mean((image.diff(dim=axis)-target.detach().diff(dim=axis)).abs(),support))
        edge=sum(gradients)/2
        hp=lambda value:value-F.avg_pool2d(value,3,1,1,count_include_pad=False)
        high=mean((hp(image)-hp(target.detach())).abs(),mask)
        total=total+color+2*edge+high
        metrics.update({name+'_l1':float(color.detach()),name+'_edge':float(edge.detach()),name+'_high_frequency':float(high.detach())})
    return total,metrics
