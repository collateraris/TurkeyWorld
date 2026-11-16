"""RGB32 -> RRDB -> x4 -> learned materials/texture -> separate patrol sprites."""
import torch
from torch import nn
from torch.nn import functional as F
from .patrol_sprite_sr import patrol_alpha

ARCHITECTURE='rrdb_material_sprite_sr'

class DenseBlock(nn.Module):
    def __init__(self,c,g):
        super().__init__()
        self.layers=nn.ModuleList(nn.Conv2d(c+i*g,g if i<4 else c,3,padding=1) for i in range(5))
    def forward(self,x):
        values=[x]
        for i,layer in enumerate(self.layers):
            out=layer(torch.cat(values,1))
            if i<4:values.append(F.leaky_relu(out,.2))
        return x+.2*out

class RRDB(nn.Module):
    def __init__(self,c,g):
        super().__init__();self.body=nn.Sequential(*[DenseBlock(c,g) for _ in range(3)])
    def forward(self,x):return x+.2*self.body(x)

class RRDBMaterialSR(nn.Module):
    def __init__(self,*,sprite_config,channels=48,growth=24,blocks=6,scale=4,weapon_memory=False,wall_texture_memory=False,wall_geometry_supervised=False,wall_coherent_uv=False,wall_texture_v2=False):
        super().__init__()
        if scale!=4 or channels<8 or growth<4 or not 1<=blocks<=24:raise ValueError('Invalid RRDB x4 configuration')
        from .models import build_upscaler
        self.config=dict(architecture=ARCHITECTURE,sprite_config=sprite_config,channels=channels,growth=growth,blocks=blocks,scale=scale)
        self.stem=nn.Conv2d(3,channels,3,padding=1)
        self.trunk=nn.Sequential(*[RRDB(channels,growth) for _ in range(blocks)],nn.Conv2d(channels,channels,3,padding=1))
        self.up=nn.Sequential(nn.Conv2d(channels,channels*4,3,padding=1),nn.PixelShuffle(2),nn.LeakyReLU(.2),nn.Conv2d(channels,channels*4,3,padding=1),nn.PixelShuffle(2),nn.LeakyReLU(.2))
        self.material_head=nn.Conv2d(channels,4,3,padding=1) # other, brick, earth, grass
        self.seam_head=nn.Conv2d(channels,2,3,padding=1) # mortar, grass boundary
        self.color_head=nn.Conv2d(channels,3,3,padding=1)
        self.texture_head=nn.Sequential(nn.Conv2d(channels+6,channels,3,padding=1),nn.LeakyReLU(.2),nn.Conv2d(channels,3,3,padding=1))
        for layer in (self.color_head,self.texture_head[-1]):nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.sprite_renderer=build_upscaler(sprite_config).eval().requires_grad_(False)
        self.weapon_memory=None
        if weapon_memory:
            from .weapon_memory_sr import WeaponMemory
            self.weapon_memory=WeaponMemory()
            self.config['weapon_memory']=True
        self.wall_memory=None
        if wall_texture_memory:
            from .wall_texture_memory_sr import WallTextureMemory,WallTextureMemoryV2
            self.wall_memory=(WallTextureMemoryV2 if wall_texture_v2 else WallTextureMemory)(channels,geometry_supervised=wall_geometry_supervised,coherent_uv=wall_coherent_uv)
            if wall_texture_v2:self.config["wall_texture_v2"]=True
            if wall_coherent_uv:self.config["wall_coherent_uv"]=True
            if wall_geometry_supervised:self.config["wall_geometry_supervised"]=True
            self.config['wall_texture_memory']=True
        self.enemies_enabled=True;self.predicted_sprite_state=None
    def configure_training(self):
        for name,p in self.named_parameters():p.requires_grad_(not name.startswith('sprite_renderer.'))
        return self.train()
    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'sprite_renderer'):self.sprite_renderer.eval()
        return self
    def forward_with_aux(self,low_rgb,hypothesis=None):
        if low_rgb.ndim!=4 or low_rgb.shape[1:]!=(3,32,32) or not bool(torch.isfinite(low_rgb).all()):raise ValueError('Expected finite RGB32')
        low=low_rgb.to(self.stem.weight);x=self.stem(low);features=self.up(x+self.trunk(x))
        materials=self.material_head(features);seams=self.seam_head(features)
        base=F.interpolate(low,scale_factor=4,mode='bilinear',align_corners=False)
        color=self.color_head(features);support=1-materials.softmax(1)[:,:1]
        texture=.25*self.texture_head(torch.cat((features,materials.softmax(1),seams.sigmoid()),1)).tanh()*support
        raw=base+color+texture;image=raw.clamp(0,1)
        if self.training:image=image.detach()+raw-raw.detach()
        wall_memory=None
        if self.wall_memory is not None:
            wall_memory=self.wall_memory(features,materials,low)
            image=image*(1-wall_memory['alpha'])+wall_memory['rgb']*wall_memory['alpha']
        with torch.no_grad():
            sprite=self.sprite_renderer.forward_with_aux(low)
            alpha=patrol_alpha(sprite['visual_composite_alpha_hr'],sprite['visual_animation_index'],enemies_enabled=self.enemies_enabled,predicted_state=self.predicted_sprite_state)
        final=(image*(1-alpha)+sprite['visual_raw_warped_rgba'][:,:3]*alpha).clamp(0,1)
        weapon=None
        if self.weapon_memory is not None:
            weapon=self.weapon_memory(low)
            final=(final*(1-weapon['alpha'])+weapon['rgb']*weapon['alpha']).clamp(0,1)
        return dict(wall_memory=wall_memory,weapon_memory=weapon,image=final,pre_sprite_image=image,background=base.detach(),residual=color+texture,material_logits=materials,seam_logits=seams,sprite_alpha=alpha,patrol_sprites_only=True,combat_animation_sprites=False,inference_rgb_only=True)
    def forward(self,low_rgb,hypothesis=None):return self.forward_with_aux(low_rgb,hypothesis)['image']
