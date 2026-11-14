"""Bounded learned detail corrections over the preferred frozen neural SR.

Only RGB32 is an inference input. The exact copied EDSR/Semantic/RGB-MoE base
is frozen; a zero RGB readout starts at its displayed image bit for bit.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_sr import SemanticEDSRUpscaler
from .sr_transformer_moe import SRTransformerMoE

ARCHITECTURE='background_detail_sr'
TRAINABLE_PREFIX='detail_residual.'
MIGRATION_KIND='frozen_preferred_background_detail_v1'


class DetailBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.SiLU(),
            nn.Conv2d(channels,channels,3,padding=1))
    def forward(self,value):return value+.1*self.body(value)


class BackgroundDetailResidual(nn.Module):
    """LR features plus the actual HR base; no opacity or spatial replacement."""
    def __init__(self,channels=48,blocks=5):
        super().__init__()
        if(type(channels)is not int or channels<8 or channels%8
                or type(blocks)is not int or not 1<=blocks<=12):
            raise ValueError('Detail width must be divisible by8 and depth must be1..12')
        c=channels
        self.stem=nn.Sequential(nn.Conv2d(6,c,3,padding=1),nn.SiLU())
        self.body=nn.Sequential(*[DetailBlock(c)for _ in range(blocks)])
        self.up64=nn.Sequential(nn.Conv2d(c,4*c,3,padding=1),nn.PixelShuffle(2),nn.SiLU(),DetailBlock(c))
        self.up128=nn.Sequential(nn.Conv2d(c,4*c,3,padding=1),nn.PixelShuffle(2),nn.SiLU())
        self.merge=nn.Sequential(nn.Conv2d(c+6,c,3,padding=1),nn.SiLU(),DetailBlock(c),DetailBlock(c))
        self.output=nn.Conv2d(c,3,3,padding=1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,low_rgb,base_rgb):
        coarse_base=F.avg_pool2d(base_rgb,4)
        x=self.body(self.stem(torch.cat((low_rgb,coarse_base),1)))
        x=self.up128(self.up64(x))
        low_hr=F.interpolate(low_rgb,size=(128,128),mode='bilinear',align_corners=False)
        return self.output(self.merge(torch.cat((x,base_rgb,low_hr),1)))


class BackgroundDetailSR(SemanticEDSRUpscaler):
    def __init__(self,channels=64,blocks=16,scale=4,res_scale=.1,
                 semantic_channels=32,semantic_blocks=4,detail_channels=48,
                 detail_blocks=5,residual_bound=.25,frozen_sr_moe=True):
        if(frozen_sr_moe is not True or type(residual_bound)not in(int,float)
                or not math.isfinite(residual_bound)or not 0<residual_bound<=.5):
            raise ValueError('Detail SR requires frozen RGB MoE and a fixed residual bound in(0,.5]')
        super().__init__(channels,blocks,scale,res_scale,semantic_channels,semantic_blocks)
        self.sr_moe=SRTransformerMoE()
        self.detail_residual=BackgroundDetailResidual(detail_channels,detail_blocks)
        self.residual_bound=float(residual_bound)
        self.config.update(architecture=ARCHITECTURE,detail_channels=detail_channels,
            detail_blocks=detail_blocks,residual_bound=self.residual_bound,frozen_sr_moe=True)
        self.configure_training()

    def configure_training(self):
        for name,p in self.named_parameters():p.requires_grad_(name.startswith(TRAINABLE_PREFIX))
        self.train(True)
        return self

    def train(self,mode=True):
        super().train(mode)
        for name,module in self.named_children():
            if name!='detail_residual':module.eval()
        return self

    def forward_with_aux(self,low_rgb,hypothesis=None):
        if(not torch.is_tensor(low_rgb)or low_rgb.ndim!=4 or low_rgb.shape[1:]!=(3,32,32)
                or not len(low_rgb)or not low_rgb.is_floating_point()or not bool(torch.isfinite(low_rgb).all())
                or bool(((low_rgb<0)|(low_rgb>1)).any())):
            raise ValueError('Background detail SR accepts finite RGB[B,3,32,32] in[0,1] only')
        parameter=next(self.detail_residual.parameters());low=low_rgb.to(parameter)
        with torch.no_grad():
            background_raw=SemanticEDSRUpscaler.forward_with_aux(self,low)['image']
            background_raw=background_raw+self.sr_moe(low)['base_rgb_residual'].to(background_raw)
            background=background_raw.clamp(0,1)
        raw=self.detail_residual(low,background)
        residual=self.residual_bound*raw.tanh()
        proposed=background+residual;bounded=proposed.clamp(0,1)
        image=bounded.detach()+(proposed-proposed.detach())if self.training else bounded
        return dict(image=image,background=background.detach(),background_raw=background_raw.detach(),
            residual=residual,raw_residual=raw,residual_bound=self.residual_bound,
            inference_rgb_only=True,hypothesis_used=False)

    def forward(self,low_rgb,hypothesis=None):return self.forward_with_aux(low_rgb,hypothesis)['image']


def _hash(values):
    from .renderer_student import _hash_tensors
    return _hash_tensors(values)


def frozen_background_state(model):
    return {name:value for name,value in model.state_dict().items()if not name.startswith(TRAINABLE_PREFIX)}


def background_detail_metadata(model):
    if not isinstance(model,BackgroundDetailSR):raise ValueError('Unknown background detail SR architecture')
    if any('atlas'in name.lower()or name.startswith('enemy_restoration.')for name in model.state_dict()):
        raise ValueError('Background detail SR contains forbidden fixed pixels or enemy-replacement branch')
    names=[name for name,p in model.named_parameters()if p.requires_grad]
    if names!=[name for name,_ in model.named_parameters()if name.startswith(TRAINABLE_PREFIX)]:
        raise ValueError('Only the new detail residual may train')
    return dict(architecture=ARCHITECTURE,input_size=32,output_size=128,
        trainable_prefix=TRAINABLE_PREFIX,trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),
        frozen_parameters=sum(p.numel()for p in model.parameters()if not p.requires_grad),
        parameter_tensors=len(list(model.parameters())),trainable_parameter_tensors=len(names),
        frozen_background_sha256=_hash(frozen_background_state(model)),residual_bound=model.residual_bound,
        residual_parameterization='bound_times_tanh',initial_output='exact_display_clipped_frozen_background',
        inference_inputs='RGB32_only',hypothesis_used=False,atlas_class_used=False,
        atlas_buffers_present=False,atlas_asset_required=False,alpha_used=False,enemy_replacement_used=False,
        fullframe_native_subpixels=True,automatic_deployment=False)


def initialize_background_detail(parent,*,seed=173,detail_channels=48,detail_blocks=5,residual_bound=.25):
    """Copy only the preferred frozen base from a complete neural SR payload."""
    config=parent.get('config',{});weights=parent.get('model');step=parent.get('training_steps')
    if(config.get('architecture')!='neural_enemy_restoration_sr'or config.get('frozen_sr_moe')is not True
            or not isinstance(weights,dict)or type(step)is not int or step<0 or type(seed)is not int
            or any('atlas'in name.lower()for name in weights)):
        raise ValueError('Background detail SR requires a complete RGB-only neural restoration source')
    constructor={key:config[key]for key in('channels','blocks','scale','res_scale','semantic_channels','semantic_blocks')}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model=BackgroundDetailSR(**constructor,detail_channels=detail_channels,detail_blocks=detail_blocks,
            residual_bound=residual_bound)
    target=model.state_dict();copied=[]
    for name,value in target.items():
        if name.startswith(TRAINABLE_PREFIX):continue
        source=weights.get(name)
        if(not torch.is_tensor(source)or source.shape!=value.shape or source.dtype!=value.dtype
                or not bool(torch.isfinite(source).all())):
            raise ValueError('Missing/incompatible preferred background tensor: '+name)
        target[name]=source.detach().cpu().clone();copied.append(name)
    if any(not name.startswith('enemy_restoration.')for name in set(weights)-set(copied)):
        raise ValueError('Source contains tensors outside the preferred base and omitted RGB/alpha head')
    model.load_state_dict(target,strict=True);model.configure_training()
    if not all(torch.equal(model.state_dict()[name],weights[name].detach().cpu())for name in copied):
        raise AssertionError('Preferred frozen background copy is not bit exact')
    report=dict(kind=MIGRATION_KIND,version=1,seed=seed,parent_training_steps=step,
        source_model_sha256=_hash(weights),copied_tensor_names=copied,copied_tensors=len(copied),
        copied_background_sha256=_hash(frozen_background_state(model)),target_config=deepcopy(model.config),
        new_detail_parameters=sum(p.numel()for p in model.detail_residual.parameters()),
        new_detail_parameter_tensors=len(list(model.detail_residual.parameters())),
        parent_omitted_tensor_names=sorted(set(weights)-set(copied)),frozen_background_bitexact=True,
        zero_initialized_residual_output=True,old_Adam_reused=False,new_optimizer='fresh_Adam',
        atlas_class_used=False,atlas_buffers_copied=False,atlas_asset_read=False,
        hypothesis_required=False,alpha_used=False,enemy_replacement_used=False,automatic_deployment=False)
    return model,report
