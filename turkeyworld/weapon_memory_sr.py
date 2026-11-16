"""Learned screen-space weapon memories; phase inferred exclusively from RGB32.

Source weapon pixels initialize trainable memories during migration only.
No asset loading, Source states or Source renderer are used by forward().
"""
import torch
from torch import nn
from torch.nn import functional as F

class WeaponMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_logits=nn.Parameter(torch.zeros(3,3,128,128))
        self.alpha_logits=nn.Parameter(torch.full((3,1,128,128),-10.))
        self.phase_head=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.LeakyReLU(.2),nn.Conv2d(16,32,3,stride=2,padding=1),nn.LeakyReLU(.2),nn.AdaptiveAvgPool2d((6,8)),nn.Flatten(),nn.Linear(32*6*8,64),nn.LeakyReLU(.2),nn.Linear(64,3))
        nn.init.zeros_(self.phase_head[-1].weight);nn.init.zeros_(self.phase_head[-1].bias)
        with torch.no_grad():self.phase_head[-1].bias[0]=2.
    def forward(self,low):
        logits=self.phase_head(low[:,:,20:32,:]);prob=logits.softmax(-1)
        hard=F.one_hot(logits.argmax(-1),3).to(prob)
        weights=hard.detach()+prob-prob.detach() if self.training else hard
        rgb=torch.einsum('bk,kchw->bchw',weights,self.rgb_logits.sigmoid())
        alpha=self.alpha_logits.sigmoid()
        alpha=torch.where(alpha>1e-4,alpha,torch.zeros_like(alpha))
        alpha=torch.einsum('bk,kchw->bchw',weights,alpha)
        return dict(rgb=rgb,alpha=alpha,phase_logits=logits)

from functools import lru_cache

@lru_cache(maxsize=1)
def source_weapon_rgba():
    """Offline labels/migration only; same nearest rasterizer as Source."""
    import numpy as np
    from .engine import Game
    game=Game(seed=0);size=int(128*.67);poses=[]
    for sprite in game._guns:
        sx=(np.arange(size)*sprite.shape[1]/size).astype(int)
        sy=(np.arange(size)*sprite.shape[0]/size).astype(int)
        rgba=np.zeros((128,128,4),dtype=np.uint8)
        x=128//2-size//2;y=128-size
        rgba[y:y+size,x:x+size]=sprite[sy[:,None],sx[None,:]]
        poses.append(torch.from_numpy(rgba.copy()).permute(2,0,1).float()/255)
    return torch.stack(poses)

@torch.no_grad()
def initialize_weapon_memory(module):
    rgba=source_weapon_rgba().to(module.rgb_logits)
    module.rgb_logits.copy_(torch.logit(rgba[:,:3].clamp(1e-5,1-1e-5)))
    module.alpha_logits.copy_(torch.logit(rgba[:,3:].clamp(1e-5,1-1e-5)))
    return dict(poses=3,initialization='Source_weapon_atlas_pixels_to_trainable_memory',Source_runtime_inputs=False)


def weapon_labels(states):
    from .engine import Game
    from .renderer_wall_supervision import _gun_masks
    flash=states[:,7].detach().clamp(0,1)*Game.FLASH_DURATION
    phase=torch.where(flash>.045,1,torch.where(flash>0,2,0)).long()
    masks=torch.as_tensor(__import__('numpy').stack(_gun_masks(128)),device=states.device,dtype=states.dtype)[:,None]
    return phase,masks[phase]


def weapon_objective(output,batch):
    phase,mask=weapon_labels(batch['state']);target=batch['hr_source'].detach();memory=output['weapon_memory']
    ce=F.cross_entropy(memory['phase_logits'],phase)
    mass=mask.flatten(1).sum(1)*3
    l1=(((output['image']-target).abs()*mask).flatten(1).sum(1)/mass.clamp_min(1)).mean()
    edge=[]
    for axis in (-1,-2):
        a=[slice(None)]*4;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
        support=torch.minimum(mask[tuple(a)],mask[tuple(b)])
        error=(output['image'].diff(dim=axis)-target.diff(dim=axis)).abs()
        edge.append((error*support).sum()/(support.sum()*3).clamp_min(1))
    edge=sum(edge)/2
    alpha_target=source_weapon_rgba()[:,3:].to(mask)[phase]
    alpha=F.binary_cross_entropy(memory['alpha'].clamp(1e-5,1-1e-5),alpha_target)
    loss=ce+2*l1+edge+.1*alpha
    return loss,dict(weapon_loss=float(loss.detach()),weapon_l1=float(l1.detach()),weapon_edge=float(edge.detach()),weapon_phase_ce=float(ce.detach()),weapon_phase_accuracy=float((memory['phase_logits'].argmax(-1)==phase).float().mean()),weapon_alpha_bce=float(alpha.detach()))
