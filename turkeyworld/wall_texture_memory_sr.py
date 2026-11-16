"""RGB-predicted wall geometry and explicit learned game texture memories.

Source map/pose is used by labels() only, never by WallTextureMemory.forward.
Green earth tops are constrained to the top of predicted earth-wall geometry.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .renderer_wall_depth import _source_ray_faces

class WallTextureMemory(nn.Module):
    def __init__(self,channels,geometry_supervised=False,coherent_uv=False):
        super().__init__();self.geometry_supervised=geometry_supervised;self.coherent_uv=coherent_uv
        self.geometry=nn.Sequential(nn.Conv1d(channels,48,3,padding=1),nn.LeakyReLU(.2),nn.Conv1d(48,3,3,padding=1))
        nn.init.zeros_(self.geometry[-1].weight);nn.init.zeros_(self.geometry[-1].bias)
        with torch.no_grad():self.geometry[-1].bias.copy_(torch.tensor([math.log(3.),0.,1.]))
        self.rgb_logits=nn.Parameter(torch.zeros(2,3,64,64))
    def forward(self,features,material_logits,low_rgb=None):
        geometry=self.geometry(features.mean(2))
        render_geometry=geometry.detach() if self.geometry_supervised else geometry
        depth=render_geometry[:,:1].clamp(math.log(.1),math.log(40)).exp()
        angle=torch.atan2(render_geometry[:,1:2],render_geometry[:,2:3])
        u=torch.remainder(angle/(2*math.pi),1)
        if self.coherent_uv:
            depth=F.avg_pool1d(depth,7,1,3,count_include_pad=False)
            # Perspective footprint along a locally planar wall; one phase anchor
            # avoids independent per-column UV predictions producing oscillations.
            step=torch.sqrt(F.pad(depth.diff(dim=-1),(1,0)).square()+(depth/(128/(2*math.tan(math.radians(34))))).square())
            position=step.cumsum(-1)
            u=torch.remainder(position-position[:,:,64:65]+u[:,:,64:65],1)
        height=(128/(2*math.tan(math.radians(34))))/depth
        yy=torch.arange(128,device=features.device,dtype=features.dtype)[None,None,:,None]
        v=(yy-(64-height[:,:,None,:]/2))/height[:,:,None,:]
        # Nearest texture pixels with differentiable bilinear sampling during training.
        uq=(torch.floor(u*64)+.5)/64
        vq=(torch.floor(v*64)+.5)/64
        if self.training:
            uq=uq.detach()+u-u.detach();vq=vq.detach()+v-v.detach()
        grid=torch.stack((uq[:,0,None,:].expand(-1,128,-1)*2-1,vq[:,0].clamp(0,1)*2-1),-1)
        # Tile u wraps; grid_sample has border handling for native texture edges.
        textures=self.rgb_logits.sigmoid()
        samples=[F.grid_sample(textures[i:i+1].expand(len(features),-1,-1,-1),grid,mode='bilinear' if self.training else 'nearest',padding_mode='border',align_corners=False) for i in range(2)]
        prob=(material_logits.detach() if self.geometry_supervised else material_logits).softmax(1);earth=prob[:,2:4].sum(1,keepdim=True)
        wall=prob[:,1:2]+earth
        soft_material=torch.cat((prob[:,1:2],earth),1)/(wall+1e-6)
        if self.coherent_uv:
            # Original material chromaticity is invariant to scalar wall shading.
            # Use only RGB32 to prevent a red brick wall selecting the green atlas.
            low_hr=F.interpolate(low_rgb,size=(128,128),mode='bilinear',align_corners=False)
            prototypes=torch.stack((textures[0].mean((1,2)),textures[1,:,32:].mean((1,2))))
            chroma=F.normalize(low_hr,dim=1,eps=1e-5);prototypes=F.normalize(prototypes,dim=1)
            similarity=torch.einsum('bchw,kc->bkhw',chroma,prototypes)
            soft_material=(soft_material.clamp_min(1e-5).log()+80*similarity).softmax(1)
        hard_material=F.one_hot(soft_material.argmax(1),2).permute(0,3,1,2).to(features)
        material=hard_material.detach()+soft_material-soft_material.detach() if self.training else hard_material
        rgb=samples[0]*material[:,:1]+samples[1]*material[:,1:2]
        visible=((v>=0)&(v<1)).to(features)
        confidence=(wall>.65).to(features)
        # Green comes from earth texture UV only, never a free green overlay.
        alpha=visible*confidence*(torch.ones_like(wall) if self.coherent_uv else wall)
        return dict(rgb=rgb,alpha=alpha,geometry=geometry,depth=depth,u=u,v=v)

@torch.no_grad()
def initialize_wall_memory(module):
    from .engine import Game
    textures=torch.as_tensor(Game(seed=0)._textures[1:].copy()).permute(0,3,1,2).to(module.rgb_logits)/255
    module.rgb_logits.copy_(torch.logit(textures.clamp(1e-5,1-1e-5)))


def geometry_labels(batch):
    tangent=math.tan(math.radians(34));screen=(np.arange(128)+.5)/128*2-1
    depths=[];phases=[]
    for tilemap,state in zip(batch['map'].detach().cpu().numpy(),batch['state'].detach().cpu().double().numpy()):
        x,y=state[:2]*16;angle=math.atan2(state[2],state[3]);c,s=math.cos(angle),math.sin(angle)
        rx,ry=c-s*tangent*screen,s+c*tangent*screen
        depth,faces=_source_ray_faces(tilemap,x,y,rx,ry)
        u=np.where(faces[:,0],x+depth*rx,y+depth*ry)%1
        depths.append(np.log(depth));phases.append(np.stack((np.sin(u*2*math.pi),np.cos(u*2*math.pi))))
    to=lambda a:torch.as_tensor(np.stack(a),device=batch['state'].device,dtype=batch['state'].dtype)
    return to(depths)[:,None],to(phases)


def wall_memory_objective(output,batch):
    memory=output['wall_memory'];depth,phase=geometry_labels(batch)
    from .sr_wall_patchgan import wall_labels
    exposed=wall_labels(batch);columns=exposed.amax(2)
    def mean(value):return (value*columns).sum()/(columns.sum()*value.shape[1]).clamp_min(1)
    logdepth=mean((memory['geometry'][:,:1]-depth).abs())
    direction=F.normalize(memory['geometry'][:,1:3],dim=1,eps=1e-5)
    uv=mean((direction-phase).square())
    image=output['image'];target=batch['hr_source'].detach()
    chroma=lambda x:F.relu(x[:,1:2]-torch.maximum(x[:,:1],x[:,2:3]))
    excess=F.relu(chroma(image)-chroma(target)-.02)
    negative=(chroma(target)<.03).to(image)
    green=(excess*negative).sum()/negative.sum().clamp_min(1)
    loss=.5*logdepth+.2*uv+3*green
    return loss,dict(wall_depth_log_mae=float(logdepth.detach()),wall_uv_direction_error=float(uv.detach()),false_green_excess=float(green.detach()),wall_memory_loss=float(loss.detach()))

class WallTextureMemoryV2(WallTextureMemory):
    """Column-consistent material and Source-style shading, RGB-predicted only."""
    def __init__(self,channels,geometry_supervised=False,coherent_uv=False):
        super().__init__(channels,geometry_supervised,coherent_uv)
        self.column_head=nn.Sequential(nn.Conv1d(channels,32,3,padding=1),nn.LeakyReLU(.2),nn.Conv1d(32,3,3,padding=1))
        nn.init.zeros_(self.column_head[-1].weight);nn.init.zeros_(self.column_head[-1].bias)
    def forward(self,features,material_logits,low_rgb=None):
        # Interior band excludes the grass top. One material choice per column
        # avoids selecting green earth pixels independently on a brick face.
        wall_prob=material_logits.softmax(1)[:,1:].sum(1,keepdim=True)
        weight=wall_prob[:,:,52:76].detach()
        pooled=(features[:,:,52:76]*weight).sum(2)/weight.sum(2).clamp_min(1e-5)
        column=self.column_head(pooled)
        textures=self.rgb_logits.sigmoid()
        prototypes=torch.stack((textures[0].mean((1,2)),textures[1,:,32:].mean((1,2))))
        low_hr=F.interpolate(low_rgb,size=(128,128),mode='bilinear',align_corners=False)
        rgb_column=(low_hr[:,:,52:76]*weight).sum(2)/weight.sum(2).clamp_min(1e-5)
        similarity=torch.einsum('bcw,kc->bkw',F.normalize(rgb_column,dim=1),F.normalize(prototypes,dim=1))
        type_logits=column[:,:2]+80*similarity
        chosen=type_logits.argmax(1,keepdim=True)
        # Retain independently predicted wall/other support, replace only type.
        prob=material_logits.softmax(1)
        wall=prob[:,1:].sum(1,keepdim=True)
        brick=wall*(chosen[:, :,None,:]==0).to(wall)
        earth=wall*(chosen[:, :,None,:]==1).to(wall)
        corrected=torch.cat((prob[:,:1],brick,earth,earth*0),1).clamp_min(1e-8).log()
        # Disable the older per-pixel chromaticity prior by using a column RGB.
        column_rgb=rgb_column[:,:,None,:].expand(-1,-1,128,-1)
        result=super().forward(features,corrected,column_rgb)
        # Exact Source depth attenuation, with neural side-of-face prediction.
        side=column[:,2:3].sigmoid()
        shade=(1-.24*side)*(1-result['depth']*.024).clamp(.5,1)
        result['rgb']=result['rgb']*shade[:,:,None,:]
        result.update(column_logits=type_logits,side_logits=column[:,2:3],shade=shade)
        return result


def column_labels(batch):
    tangent=math.tan(math.radians(34));screen=(np.arange(128)+.5)/128*2-1
    materials=[];sides=[]
    for tilemap,state in zip(batch['map'].detach().cpu().numpy(),batch['state'].detach().cpu().double().numpy()):
        x,y=state[:2]*16;angle=math.atan2(state[2],state[3]);c,s=math.cos(angle),math.sin(angle)
        _,faces=_source_ray_faces(tilemap,x,y,c-s*tangent*screen,s+c*tangent*screen)
        materials.append(faces[:,2]-1);sides.append(faces[:,0])
    return (torch.as_tensor(np.stack(materials),device=batch['state'].device,dtype=torch.long),torch.as_tensor(np.stack(sides)[:,None],device=batch['state'].device,dtype=batch['state'].dtype))


def column_objective(memory,batch):
    from .sr_wall_patchgan import wall_labels
    material,side=column_labels(batch);support=wall_labels(batch).amax(2)
    ce=F.cross_entropy(memory['column_logits'],material,reduction='none')
    loss_material=(ce*support[:,0]).sum()/support.sum().clamp_min(1)
    side_error=F.binary_cross_entropy_with_logits(memory['side_logits'],side,reduction='none')
    loss_side=(side_error*support).sum()/support.sum().clamp_min(1)
    accuracy=(((memory['column_logits'].argmax(1)==material).to(support)*support[:,0]).sum()/support.sum().clamp_min(1))
    return loss_material+.2*loss_side,dict(wall_column_material_ce=float(loss_material.detach()),wall_column_material_accuracy=float(accuracy.detach()),wall_side_bce=float(loss_side.detach()))
