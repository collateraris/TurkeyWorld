"""RGB-only full-frame enemy restoration on an explicitly frozen SR background.

The background copies semantic EDSR and the existing RGB Transformer adapter.
This architecture has no fixed pixels, atlas buffers, crop placement, map or
RSSM input. Visible RGB and alpha are independently supervised during training.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_sr import SemanticEDSRUpscaler
from .sr_transformer_moe import SRTransformerMoE,CONFIG as SR_MOE_CONFIG


ARCHITECTURE='neural_enemy_restoration_sr'
TRAINABLE_PREFIX='enemy_restoration.'
MIGRATION_KIND='frozen_background_fullframe_neural_enemy_v1'


class RestorationResidual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.SiLU(),
                                nn.Conv2d(width,width,3,padding=1))
    def forward(self,value):return value+.2*self.body(value)


class FullFrameEnemyDecoder(nn.Module):
    """Spatially aligned 32-to128 U-Net with learned native subpixels."""
    def __init__(self,channels=48,bottleneck_blocks=1):
        super().__init__()
        if(type(channels)is not int or channels<8 or channels%8
                or type(bottleneck_blocks)is not int or not 1<=bottleneck_blocks<=4):
            raise ValueError('Enemy restoration needs a positive width divisible by8 and1..4 bottleneck blocks')
        c=channels
        self.stem=nn.Sequential(nn.Conv2d(3,c,3,padding=1),nn.SiLU(),RestorationResidual(c),RestorationResidual(c))
        self.down16=nn.Sequential(nn.Conv2d(c,2*c,3,stride=2,padding=1),nn.SiLU(),
                                  RestorationResidual(2*c),RestorationResidual(2*c))
        self.down8=nn.Sequential(nn.Conv2d(2*c,4*c,3,stride=2,padding=1),nn.SiLU(),
                                 *[RestorationResidual(4*c)for _ in range(bottleneck_blocks)])
        self.up16=nn.Conv2d(4*c,2*c,3,padding=1)
        self.merge16=nn.Sequential(nn.Conv2d(4*c,2*c,3,padding=1),nn.SiLU(),RestorationResidual(2*c))
        self.up32=nn.Conv2d(2*c,c,3,padding=1)
        self.merge32=nn.Sequential(nn.Conv2d(2*c,c,3,padding=1),nn.SiLU(),RestorationResidual(c))
        self.native64=nn.Sequential(nn.Conv2d(c,4*c,3,padding=1),nn.PixelShuffle(2),nn.SiLU(),RestorationResidual(c))
        self.native128=nn.Sequential(nn.Conv2d(c,4*c,3,padding=1),nn.PixelShuffle(2),nn.SiLU(),RestorationResidual(c))
        self.rgb=nn.Conv2d(c,3,3,padding=1)
        self.opacity=nn.Conv2d(c,1,3,padding=1)
        nn.init.normal_(self.rgb.weight,std=.001);nn.init.zeros_(self.rgb.bias)
        nn.init.normal_(self.opacity.weight,std=.001);nn.init.constant_(self.opacity.bias,-4.)

    def forward(self,low_rgb):
        x32=self.stem(low_rgb);x16=self.down16(x32);x8=self.down8(x16)
        x=self.up16(F.interpolate(x8,size=x16.shape[-2:],mode='bilinear',align_corners=False))
        x=self.merge16(torch.cat((x,x16),1))
        x=self.up32(F.interpolate(x,size=x32.shape[-2:],mode='bilinear',align_corners=False))
        x=self.merge32(torch.cat((x,x32),1));x=self.native128(self.native64(x))
        return self.rgb(x).sigmoid(),self.opacity(x)


class NeuralEnemyRestorationSR(SemanticEDSRUpscaler):
    def __init__(self,channels=64,blocks=16,scale=4,res_scale=.1,
                 semantic_channels=32,semantic_blocks=4,restoration_channels=48,
                 restoration_bottleneck_blocks=1,frozen_sr_moe=True):
        if frozen_sr_moe is not True:raise ValueError('Neural enemy experiment requires the copied frozen RGB adapter')
        super().__init__(channels,blocks,scale,res_scale,semantic_channels,semantic_blocks)
        self.sr_moe=SRTransformerMoE()
        self.enemy_restoration=FullFrameEnemyDecoder(restoration_channels,restoration_bottleneck_blocks)
        self.config.update(architecture=ARCHITECTURE,restoration_channels=restoration_channels,
            restoration_bottleneck_blocks=restoration_bottleneck_blocks,frozen_sr_moe=True)
        self.configure_training()

    def configure_training(self):
        for name,parameter in self.named_parameters():parameter.requires_grad_(name.startswith(TRAINABLE_PREFIX))
        self.train(True)
        return self

    def train(self,mode=True):
        super().train(mode)
        for name,module in self.named_children():
            if name!='enemy_restoration':module.eval()
        return self

    def forward_with_aux(self,low_rgb,hypothesis=None):
        if(low_rgb.ndim!=4 or low_rgb.shape[1:]!=(3,32,32)or not len(low_rgb)
                or not low_rgb.is_floating_point()or not bool(torch.isfinite(low_rgb).all())):
            raise ValueError('Neural restoration accepts finite native RGB[B,3,32,32] only')
        parameter=next(self.enemy_restoration.parameters())
        low=low_rgb.to(device=parameter.device,dtype=parameter.dtype)
        with torch.no_grad():
            background=SemanticEDSRUpscaler.forward_with_aux(self,low)['image']
            background=background+self.sr_moe(low)['base_rgb_residual'].to(background)
        rgb,logits=self.enemy_restoration(low);alpha=logits.sigmoid()
        proposed=background*(1-alpha)+rgb*alpha;bounded=proposed.clamp(0,1)
        image=bounded.detach()+(proposed-proposed.detach())if self.training else bounded
        return dict(image=image,background=background.detach(),enemy_rgb=rgb,alpha_logits=logits,alpha=alpha,
            inference_rgb_only=True,hypothesis_used=False)

    def forward(self,low_rgb,hypothesis=None):return self.forward_with_aux(low_rgb,hypothesis=hypothesis)['image']


def _hash(values):
    from .renderer_student import _hash_tensors
    return _hash_tensors(values)


def frozen_background_state(model):
    return {name:value for name,value in model.state_dict().items()if not name.startswith(TRAINABLE_PREFIX)}


def neural_restoration_metadata(model):
    if not isinstance(model,NeuralEnemyRestorationSR):raise ValueError('Unknown learned enemy restoration architecture')
    if any('atlas' in name.lower()for name in model.state_dict()):
        raise ValueError('Neural restoration must not contain fixed atlas tensors')
    names=[name for name,p in model.named_parameters()if p.requires_grad]
    if names!=[name for name,_ in model.named_parameters()if name.startswith(TRAINABLE_PREFIX)]:
        raise ValueError('Only the new full-frame enemy decoder may train')
    return dict(architecture=ARCHITECTURE,input_size=32,output_size=128,trainable_prefix=TRAINABLE_PREFIX,
        trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),
        frozen_parameters=sum(p.numel()for p in model.parameters()if not p.requires_grad),
        parameter_tensors=len(list(model.parameters())),trainable_parameter_tensors=len(names),
        frozen_background_sha256=_hash(frozen_background_state(model)),
        atlas_class_used=False,atlas_buffers_present=False,atlas_asset_required=False,
        inference_inputs='RGB32_only',hypothesis_used=False,fullframe_native_subpixels=True,
        direct_RGB_and_alpha_supervision=True,automatic_deployment=False)


def initialize_neural_enemy_restoration(parent,*,seed=173,restoration_channels=48,restoration_bottleneck_blocks=1):
    """Copy only matching frozen tensors from a pinned payload, without building Atlas."""
    config=parent.get('config',{});weights=parent.get('model')
    if(config.get('architecture')!='atlas_enemy_edsr' or config.get('sr_moe_enabled')is not True
            or not isinstance(weights,dict)or any(config.get(key)!=value for key,value in SR_MOE_CONFIG.items())
            or type(parent.get('training_steps'))is not int or parent['training_steps']<0
            or type(seed)is not int):
        raise ValueError('Need the explicit complete frozen-Atlas/five-expert parent payload')
    binding=parent.get('sr_moe_binding')
    if(not isinstance(binding,dict)
            or binding.get('state_sha256')!=_hash({name[len('sr_moe.'):]:value for name,value in weights.items()if name.startswith('sr_moe.')})
            or binding.get('parent_model_sha256')!=_hash({name:value for name,value in weights.items()if not name.startswith('sr_moe.')})):
        raise ValueError('Parent full-model/five-expert tensor binding changed')
    constructor={key:config[key]for key in('channels','blocks','scale','res_scale','semantic_channels','semantic_blocks')}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=NeuralEnemyRestorationSR(**constructor,restoration_channels=restoration_channels,
            restoration_bottleneck_blocks=restoration_bottleneck_blocks)
    target=model.state_dict();copied=[]
    for name,value in target.items():
        if name.startswith(TRAINABLE_PREFIX):continue
        source=weights.get(name)
        if(not torch.is_tensor(source)or source.shape!=value.shape or source.dtype!=value.dtype
                or not bool(torch.isfinite(source).all())):
            raise ValueError('Missing/incompatible frozen background tensor: '+name)
        target[name]=source.detach().cpu().clone();copied.append(name)
    model.load_state_dict(target,strict=True);model.configure_training()
    if not all(torch.equal(model.state_dict()[name],weights[name].detach().cpu())for name in copied):
        raise AssertionError('Frozen background copy is not bit exact')
    report=dict(kind=MIGRATION_KIND,version=1,seed=seed,parent_training_steps=parent['training_steps'],
        source_model_sha256=_hash(weights),copied_tensor_names=copied,copied_tensors=len(copied),
        copied_background_sha256=_hash(frozen_background_state(model)),
        target_config=deepcopy(model.config),new_decoder_parameters=sum(p.numel()for p in model.enemy_restoration.parameters()),
        parent_omitted_tensor_names=sorted(set(weights)-set(copied)),
        frozen_background_bitexact=True,old_Adam_reused=False,new_decoder_optimizer='fresh_Adam',
        atlas_class_used=False,atlas_buffers_copied=False,atlas_asset_read=False,hypothesis_required=False,
        automatic_deployment=False)
    return model,report
