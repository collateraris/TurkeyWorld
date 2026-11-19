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
    def __init__(self,*,sprite_config,channels=48,growth=24,blocks=6,scale=4,weapon_memory=False,wall_texture_memory=False,wall_geometry_supervised=False,wall_coherent_uv=False,wall_texture_v2=False,combat_appearance=False,source_sprite_compositor=False):
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
        self.source_sprite_compositor = source_sprite_compositor
        self.predicted_sprite_map = None
        self.combat_weapon_phase = None
        if source_sprite_compositor:
            if combat_appearance or not weapon_memory:
                raise ValueError("The seven-pose compositor requires a single enemy layer and weapon memory")
            from .source_sprite_compositor import SourceSpriteCompositor
            self.sprite_renderer = SourceSpriteCompositor(sprite_config)
            self.combat_weapon_phase = nn.Sequential(nn.Linear(1,16), nn.Tanh(), nn.Linear(16,3))
            self.config["source_sprite_compositor"] = True
        else:
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
        self.combat_appearance=None
        if combat_appearance:
            from .combat_appearance_sr import CombatAppearanceSR
            self.combat_appearance=CombatAppearanceSR()
            self.config['combat_appearance']=True
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
        combat_state = self.predicted_sprite_state if self.enemies_enabled else None
        sprite_aux = {}
        if self.source_sprite_compositor:
            sprite_aux = self.sprite_renderer(image, self.predicted_sprite_state, self.predicted_sprite_map,
                                              enabled=self.enemies_enabled)
            from .source_sprite_compositor import composite_source_sprite
            alpha = sprite_aux["sprite_alpha"]
            final = composite_source_sprite(image, sprite_aux["sprite_rgb"], alpha)
        else:
            with torch.no_grad():
                sprite=self.sprite_renderer.forward_with_aux(low)
                alpha=patrol_alpha(sprite['visual_composite_alpha_hr'],sprite['visual_animation_index'],enemies_enabled=self.enemies_enabled,predicted_state=self.predicted_sprite_state)
            sprite_rgb=sprite['visual_raw_warped_rgba'][:,:3]
            final=(image*(1-alpha)+sprite_rgb*alpha).clamp(0,1)
        if self.combat_appearance is not None:
            final = self.combat_appearance(low, final, combat_state)
        scene_before_weapon = final
        weapon=None
        if self.weapon_memory is not None:
            if combat_state is not None and (self.combat_appearance is not None or self.combat_weapon_phase is not None):
                base_logits=self.weapon_memory.phase_head(low[:,:,20:32,:])
                phase_logits=(self.combat_appearance.phase_logits(combat_state) if self.combat_weapon_phase is None
                              else self.combat_weapon_phase(combat_state[:,7:8].to(self.combat_weapon_phase[0].weight)))
                weapon=self.weapon_memory.compose(torch.where(combat_state[:,7:8]>0,phase_logits,base_logits))
            else:
                weapon=self.weapon_memory(low)
            if self.source_sprite_compositor and self.predicted_sprite_state is not None:
                gun = weapon['rgb'].float()
                tint=(gun.mul(255).round()+gun.new_tensor([70,8,5])[None,:,None,None]).mul(.5).floor()/255
                weapon['rgb']=torch.where((self.predicted_sprite_state[:,4]<=0)[:,None,None,None],tint,gun)
            final=(final*(1-weapon['alpha'])+weapon['rgb']*weapon['alpha']).clamp(0,1)
        result=dict(wall_memory=wall_memory,weapon_memory=weapon,image=final,scene_before_weapon=scene_before_weapon,
                    pre_sprite_image=image,background=base.detach(),residual=color+texture,material_logits=materials,
                    seam_logits=seams,sprite_alpha=alpha,patrol_sprites_only=not self.source_sprite_compositor,
                    combat_animation_sprites=self.source_sprite_compositor,
                    inference_rgb_only=self.combat_appearance is None and not self.source_sprite_compositor)
        return {**result,**sprite_aux}
    def forward(self,low_rgb,hypothesis=None):return self.forward_with_aux(low_rgb,hypothesis)['image']
