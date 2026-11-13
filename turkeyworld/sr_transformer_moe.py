"""Five RGB-only private Transformer SR experts around an unchanged fixed atlas."""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA='atlas_sr_five_private_transformers_v1'
KIND='atlas_sr_five_transformer_binding_v1'
MIGRATION_KIND='append_atlas_sr_five_transformer_experts_v1'
ROLES=('enemy_detection','sprite_selection_placement','far_walls','near_walls','player_weapon_effects')
CONFIG=dict(sr_moe_enabled=True,sr_moe_role_schema=SCHEMA,sr_moe_experts=5,
    sr_moe_width=64,sr_moe_heads=4,sr_moe_expert_hidden=128,sr_moe_grid=8,sr_moe_rgb_amplitude=1.,
    sr_moe_rgb_resolution=128,sr_moe_visibility_resolution=128,sr_moe_heat_resolution=32)


class SRTransformerMoE(nn.Module):
    """Own five distinct64D token blocks; zero task readouts preserve activation."""
    def __init__(self,width=64,heads=4,expert_hidden=128,grid=8,rgb_amplitude=1.):
        super().__init__()
        if ((width,heads,expert_hidden,grid)!=(64,4,128,8)
                or any(type(v)is not int for v in(width,heads,expert_hidden,grid))
                or type(rgb_amplitude)not in(float,int)or not math.isfinite(rgb_amplitude)
                or rgb_amplitude!=1.):
            raise ValueError('SR five-expert profile requires64D/4heads/FF128/8x8 and full normalized RGB corrections')
        self.width,self.heads,self.expert_hidden,self.grid=width,heads,expert_hidden,grid
        self.rgb_amplitude=float(rgb_amplitude)
        self.input=nn.Conv2d(3,width,4,stride=4)
        self.position=nn.Parameter(torch.randn(1,grid*grid,width)*.02)
        self.input_norm=nn.LayerNorm(width)
        self.experts=nn.ModuleList(nn.TransformerEncoderLayer(width,heads,dim_feedforward=expert_hidden,
            dropout=0.,activation='gelu',batch_first=True,norm_first=True)for _ in ROLES)
        self.router=nn.Linear(width,5)
        self.detection_head=nn.Linear(width,3)
        self.heat_head=nn.Linear(width,16)
        self.geometry_head=nn.Linear(width,12)
        self.visibility_head=nn.Linear(width,16*16)
        self.rgb_heads=nn.ModuleList(nn.Linear(width,3*16*16)for _ in range(3))
        for head in(self.router,self.detection_head,self.heat_head,self.geometry_head,self.visibility_head,*self.rgb_heads):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.last_aux=None

    @staticmethod
    def _pixels(tokens,channels,patch=4):
        pixels=tokens.reshape(-1,8,8,channels,patch,patch)
        return pixels.permute(0,3,1,4,2,5).reshape(-1,channels,8*patch,8*patch)

    def forward(self,low_rgb):
        if (low_rgb.ndim!=4 or low_rgb.shape[1:]!=(3,32,32)or not len(low_rgb)
                or not low_rgb.is_floating_point()or not bool(torch.isfinite(low_rgb).all())):
            raise ValueError('SR experts accept finite nativeRGB[B,3,32,32] only')
        shared=self.input(low_rgb.to(self.input.weight.dtype)).flatten(2).transpose(1,2)
        shared=self.input_norm(shared+self.position)
        logits=self.router(shared).float();gates=logits.softmax(-1)
        features=torch.stack([expert(shared)for expert in self.experts],-2)
        def pooled(index):
            gate=gates[...,index,None].to(features.dtype)
            average=(features[...,index,:]*gate).sum(1)/gate.sum(1).clamp_min(1e-8)
            return average,gate.mean(1)
        detection,detect_gate=pooled(0)
        detection=self.detection_head(detection).float()*detect_gate.float()
        geometry,geometry_gate=pooled(1)
        geometry=self.geometry_head(geometry).float()*geometry_gate.float()
        heat=self._pixels(self.heat_head(features[...,0,:])*gates[...,0,None].to(features.dtype),1).float()
        visible=self._pixels(self.visibility_head(features[...,1,:])*gates[...,1,None].to(features.dtype),1,16).float()
        rgb=[self._pixels(head(features[...,index+2,:])*gates[...,index+2,None].to(features.dtype),3,16).float()
             for index,head in enumerate(self.rgb_heads)]
        residual=sum(rgb).tanh()*self.rgb_amplitude
        result=dict(presence_residual=detection[:,0:1],visual_confirmation_residual=detection[:,1:2],
            guided_confirmation_residual=detection[:,2:3],heat_residual=heat,
            bbox_logit_residual=geometry[:,:4],phase_residual=geometry[:,4:11],player_dead_residual=geometry[:,11:12],
            visibility_residual=visible,
            base_rgb_residual=residual,regional_rgb_residuals=torch.stack(rgb,1),
            router_logits=logits,gates=gates,expert_tokens=features,role_names=ROLES,grid=self.grid,
            inference_inputs='RGB32_only_optional_hint_stays_in_existing_atlas_path',Source_future_inputs=False)
        self.last_aux=result
        return result

    def pop_diagnostics(self):
        result=self.last_aux;self.last_aux=None;return result


def _hash(values):
    from .renderer_student import _hash_tensors
    return _hash_tensors(values)


def _parent_state(model):
    return {name:value for name,value in model.state_dict().items()if not name.startswith('sr_moe.')}


def sr_moe_metadata(model):
    module=getattr(model,'sr_moe',None)
    if module is None:return None
    if not isinstance(module,SRTransformerMoE):raise ValueError('Unknown SR expert adapter')
    return dict(kind=KIND,version=1,role_schema=SCHEMA,roles=list(ROLES),experts=5,
        private_Transformer_blocks_per_expert=1,width=64,heads=4,expert_hidden=128,grid=8,rgb_amplitude=1.,
        rgb_output_resolution=128,visibility_output_resolution=128,heat_output_resolution=32,
        output_sampling='learned_native128_subpixels_without_fixed_RGB_or_visibility_interpolation',
        parameters=sum(p.numel()for p in module.parameters()),parameter_tensors=len(list(module.parameters())),
        state_sha256=_hash(module.state_dict()),parent_model_sha256=_hash(_parent_state(model)),
        atlas_asset_sha256=model.config['atlas_asset_sha256'],atlas_pixels_sha256=model.config['atlas_pixels_sha256'],
        new_paint_enemy=False,Source_future_inputs=False,old_atlas_unchanged=True)


def enable_sr_transformer_moe(model):
    """Append new tensors only, preserving old model/buffers/modes and CPU/CUDA RNG."""
    if model.config.get('architecture')!='atlas_enemy_edsr':
        raise ValueError('Five SR experts require the existing self-contained fixed-atlas model')
    if hasattr(model,'sr_moe'):
        if any(model.config.get(k)!=v for k,v in CONFIG.items()):raise ValueError('SR adapter flag/schema differs')
        return deepcopy(model.sr_moe_migration)
    source_config=deepcopy(model.config);source_sha=_hash(model.state_dict())
    parameter=next(model.parameters());devices=[parameter.device.index]if parameter.is_cuda else[]
    with torch.random.fork_rng(devices=devices):module=SRTransformerMoE()
    model.sr_moe=module.to(device=parameter.device,dtype=parameter.dtype).train(model.training)
    model.config.update(deepcopy(CONFIG))
    report=dict(kind=MIGRATION_KIND,version=1,source_config=source_config,target_config=deepcopy(model.config),
        source_model_sha256=source_sha,added_parameter_names=['sr_moe.'+n for n,_ in module.named_parameters()],
        added_parameter_count=sum(p.numel()for p in module.parameters()),
        old_model_tensors_bitexact=True,zero_output_initialization=True,Source_future_inputs=False,
        atlas_pixels_sha256=model.config['atlas_pixels_sha256'])
    model.sr_moe_migration=deepcopy(report)
    return report


def configure_sr_transformer_moe(model):
    enable_sr_transformer_moe(model)
    # Atlas phase/box/visibility use their existing hard-forward/ST training path.
    model.train().requires_grad_(False);model.sr_moe.train().requires_grad_(True)
    return dict(**sr_moe_metadata(model),old_parameters_frozen=True,atlas_buffers_frozen=True,
        trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad))


def validate_sr_moe_checkpoint(model,checkpoint):
    enabled=model.config.get('sr_moe_enabled',False)
    if not enabled:
        if checkpoint.get('sr_moe_migration')is not None or checkpoint.get('sr_moe_binding')is not None:
            raise ValueError('SR expert metadata requires its explicit enabled architecture')
        return
    report,binding=checkpoint.get('sr_moe_migration'),checkpoint.get('sr_moe_binding')
    actual=sr_moe_metadata(model)
    source_config={k:v for k,v in model.config.items()if k not in CONFIG}
    if (not isinstance(report,dict)or binding!=actual
            or report.get('kind')!=MIGRATION_KIND or report.get('version')!=1
            or report.get('source_config')!=source_config or report.get('target_config')!=model.config
            or report.get('source_model_sha256')!=actual['parent_model_sha256']
            or report.get('atlas_pixels_sha256')!=actual['atlas_pixels_sha256']
            or report.get('added_parameter_names')!=['sr_moe.'+n for n,_ in model.sr_moe.named_parameters()]
            or report.get('added_parameter_count')!=actual['parameters']
            or any(report.get(k)is not v for k,v in dict(old_model_tensors_bitexact=True,
                zero_output_initialization=True,Source_future_inputs=False).items())):
        raise ValueError('SR five-expert migration/binding, frozen parent, atlas or role schema changed')
    model.sr_moe_migration=deepcopy(report)
