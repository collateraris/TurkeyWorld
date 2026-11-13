"""Hybrid SR: learned geometry/phase/occlusion, embedded fixed enemy pixels.

No Source frame, mask, map, atlas loader or reference engine is queried during
inference. Native PNG texels are embedded in checkpoints. Geometry and phase
use hard forward samples with deterministic straight-through training gradients.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .enemy_decoder_sr import EnemyDecoderEDSRUpscaler, project_enemy_hypothesis, sample_box
from .semantic_sr import SemanticEDSRUpscaler

ASSET_SHA256 = '56bec622ff615462971de4d68c982a319c5928f069d6047b93eafd0c210b7ac6'
UNUSED_PAINT_PREFIXES = ('enemy_decoder_condition.', 'enemy_decoder_seed.',
    'enemy_decoder_rgba.', 'enemy_decoder_gain.', 'enemy_decoder_visual_condition.',
    'enemy_decoder_visual_gain.')


def _pixel_hash(frames, sizes):
    digest = hashlib.sha256(frames.detach().cpu().contiguous().numpy().tobytes())
    digest.update(sizes.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_native_atlas(path, expected_sha):
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Fixed enemy atlas file differs from its explicit SHA256')
    with Image.open(path) as source:
        image = source.convert('RGBA')
        xs = [round(i * image.width / 4) for i in range(5)]
        ys = [round(i * image.height / 2) for i in range(3)]
        tiles = [np.asarray(image.crop((xs[i % 4], ys[i // 4], xs[i % 4 + 1], ys[i // 4 + 1]))).copy()
                 for i in range(7)]
    sizes = torch.tensor([[tile.shape[0], tile.shape[1]] for tile in tiles], dtype=torch.int64)
    height, width = sizes.amax(0).tolist()
    frames = torch.zeros(7, 4, height, width, dtype=torch.uint8)
    for index, tile in enumerate(tiles):
        h, w = tile.shape[:2]
        frames[index, :, :h, :w] = torch.from_numpy(tile).permute(2, 0, 1)
    return frames, sizes


class AtlasEnemyEDSRUpscaler(EnemyDecoderEDSRUpscaler):
    """All158 inherited parameter names remain; only a49-value tint head appends."""
    def __init__(self, channels=64, blocks=16, scale=4, res_scale=.1,
                 semantic_channels=32, semantic_blocks=4, enemy_decoder_channels=24,
                 enemy_decoder_stable=True, enemy_decoder_visual_first=True,
                 enemy_decoder_visual_threshold=.9, enemy_decoder_visual_size_max=1.5,
                 enemy_decoder_restoration=True, restoration_blend_scale=1.,
                 atlas_asset_sha256=ASSET_SHA256, atlas_pixels_sha256=None,
                 atlas_buffer_shape=None, atlas_frame_sizes=None, atlas_asset_path=None,
                 atlas_layout='round_4x2_Source_frames0_6',
                 atlas_sampling='nearest_pixel_index_ST_bilinear_gradient',
                 atlas_phase_selection='RGB_hard_argmax_ST_softmax',
                 atlas_visibility='hard_forward_ST_sigmoid'):
        if not (enemy_decoder_stable and enemy_decoder_visual_first and enemy_decoder_restoration):
            raise ValueError('Fixed-atlas SR requires the frozen-parent visual-first restoration profile')
        if (atlas_layout, atlas_sampling, atlas_phase_selection, atlas_visibility) != (
                'round_4x2_Source_frames0_6', 'nearest_pixel_index_ST_bilinear_gradient',
                'RGB_hard_argmax_ST_softmax', 'hard_forward_ST_sigmoid'):
            raise ValueError('Fixed-atlas layout or hard forward inference contract changed')
        if not isinstance(atlas_asset_sha256, str) or len(atlas_asset_sha256) != 64:
            raise ValueError('Fixed enemy atlas requires its original asset SHA256')
        super().__init__(channels, blocks, scale, res_scale, semantic_channels, semantic_blocks,
            enemy_decoder_channels, enemy_decoder_stable, enemy_decoder_visual_first,
            enemy_decoder_visual_threshold, enemy_decoder_visual_size_max,
            enemy_decoder_restoration, restoration_blend_scale)
        # Appended after the complete inherited158-parameter canonical prefix.
        self.atlas_player_dead_head = nn.Linear(enemy_decoder_channels * 2, 1)
        nn.init.zeros_(self.atlas_player_dead_head.weight)
        nn.init.constant_(self.atlas_player_dead_head.bias, -4.)
        embedded = atlas_buffer_shape is not None or atlas_frame_sizes is not None or atlas_pixels_sha256 is not None
        if embedded:
            if (not isinstance(atlas_buffer_shape, (list, tuple)) or len(atlas_buffer_shape) != 4
                    or list(atlas_buffer_shape[:2]) != [7, 4]
                    or any(type(n) is not int or not 1 <= n <= 2048 for n in atlas_buffer_shape[2:])
                    or not isinstance(atlas_frame_sizes, (list, tuple)) or len(atlas_frame_sizes) != 7
                    or not isinstance(atlas_pixels_sha256, str) or len(atlas_pixels_sha256) != 64):
                raise ValueError('Embedded enemy atlas requires its exact shape, native sizes and pixel SHA256')
            sizes = torch.tensor(atlas_frame_sizes, dtype=torch.int64)
            if (sizes.shape != (7, 2) or bool((sizes < 1).any())
                    or bool((sizes > torch.tensor(atlas_buffer_shape[2:])).any())):
                raise ValueError('Embedded enemy atlas native frame sizes exceed its padded buffer')
            frames = torch.zeros(tuple(atlas_buffer_shape), dtype=torch.uint8)
            self._atlas_ready = False
        else:
            path = Path(atlas_asset_path) if atlas_asset_path is not None else Path(__file__).resolve().parents[1] / 'assets/turkey_atlas.png'
            frames, sizes = _load_native_atlas(path, atlas_asset_sha256)
            atlas_pixels_sha256 = _pixel_hash(frames, sizes)
            self._atlas_ready = True
        self.register_buffer('atlas_frames_rgba', frames, persistent=True)
        self.register_buffer('atlas_native_sizes', sizes, persistent=True)
        self.config.update(architecture='atlas_enemy_edsr', atlas_asset_sha256=atlas_asset_sha256,
            atlas_pixels_sha256=atlas_pixels_sha256, atlas_buffer_shape=list(frames.shape),
            atlas_frame_sizes=sizes.tolist(), atlas_layout=atlas_layout, atlas_sampling=atlas_sampling,
            atlas_phase_selection=atlas_phase_selection, atlas_visibility=atlas_visibility)
        for name, parameter in self.named_parameters():
            if name.startswith(UNUSED_PAINT_PREFIXES): parameter.requires_grad_(False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        pixels, sizes = prefix + 'atlas_frames_rgba', prefix + 'atlas_native_sizes'
        if pixels in state_dict and sizes in state_dict:
            value, shape = state_dict[pixels], state_dict[sizes]
            valid = (value.dtype == torch.uint8 and shape.dtype == torch.int64
                and list(value.shape) == self.config['atlas_buffer_shape']
                and shape.tolist() == self.config['atlas_frame_sizes']
                and _pixel_hash(value, shape) == self.config['atlas_pixels_sha256'])
            if not valid:
                error_msgs.append('Embedded enemy atlas shape, native sizes or pixel SHA256 changed')
            self._atlas_ready = bool(valid)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _validate_boxes(self, boxes):
        if not self._atlas_ready:
            raise RuntimeError('Embedded atlas buffers have not passed strict checkpoint loading')
        if (boxes.ndim != 2 or boxes.shape[1] != 4 or not boxes.is_floating_point()
                or boxes.device != self.atlas_frames_rgba.device or not bool(torch.isfinite(boxes).all())):
            raise ValueError('Atlas rendering requires finite full unclipped boxes[B,4] on the atlas device')

    def _warp_native(self, boxes, phase, *, size=128, straight_through=False):
        h, w = self.config['atlas_frame_sizes'][phase]
        texture = self.atlas_frames_rgba[phase:phase+1, :, :h, :w].float() / 255
        # Source uses integer destination pixel indices, not center+0.5.
        axis = torch.arange(size, device=boxes.device, dtype=torch.float32) / size
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        width, height = (boxes[:, 2:] - boxes[:, :2]).unbind(-1)
        u = (xx[None] - boxes[:, 0, None, None]) / width[:, None, None].clamp_min(1e-6)
        v = (yy[None] - boxes[:, 1, None, None]) / height[:, None, None].clamp_min(1e-6)
        inside = ((u >= 0) & (u < 1) & (v >= 0) & (v < 1)
                  & (width > 1e-5)[:, None, None] & (height > 1e-5)[:, None, None])
        ix, iy = (u * w).floor().long().clamp(0, w-1), (v * h).floor().long().clamp(0, h-1)
        index = (iy * w + ix).reshape(len(boxes), -1)
        hard = texture[0].reshape(4, -1)[:, index].permute(1, 0, 2).reshape(len(boxes), 4, size, size)
        hard = hard * inside[:, None].float()
        if not straight_through: return hard
        grid = torch.stack((u * 2-1, v * 2-1), -1)
        with torch.autocast(boxes.device.type, enabled=False):
            soft = F.grid_sample(texture.expand(len(boxes), -1, -1, -1), grid.float(),
                                 mode='bilinear', padding_mode='zeros', align_corners=False)
        return hard.detach() + (soft - soft.detach())

    @staticmethod
    def _dead_color(rgba, bit, *, straight_through=False):
        rgb = rgba[:, :3]
        offset = rgb.new_tensor([70., 8., 5.])[None, :, None, None]
        hard = ((rgb.detach() * 255).round() + offset).div(2).floor() / 255
        soft = (rgb * 255 + offset) / 510
        tint = hard.detach() + (soft-soft.detach()) if straight_through else hard
        return torch.cat((rgb + bit[:, None, None, None] * (tint-rgb), rgba[:, 3:]), 1)

    def render_label_atlas(self, boxes, phase_index, dead_index=None, *, size=128):
        """Fixed-asset label helper; never called with Source labels by forward."""
        self._validate_boxes(boxes)
        phase = phase_index.reshape(-1).to(device=boxes.device)
        if phase.shape != (len(boxes),) or phase.is_floating_point() or bool(((phase < 0) | (phase > 6)).any()):
            raise ValueError('Atlas frame indices must be integer classes0..6')
        result = boxes.new_zeros((len(boxes), 4, size, size), dtype=torch.float32)
        for index in range(7):
            ids = (phase == index).nonzero(as_tuple=False).flatten()
            if len(ids): result[ids] = self._warp_native(boxes[ids].float(), index, size=size)
        if dead_index is not None:
            dead = dead_index.detach().reshape(-1).to(device=boxes.device, dtype=torch.float32)
            if dead.shape != (len(boxes),) or bool(((dead != 0) & (dead != 1)).any()):
                raise ValueError('Atlas dead-tint indices must be binary')
            result = self._dead_color(result, dead)
        return result

    fixed_atlas_frame = render_label_atlas

    def fixed_atlas_rgba(self, boxes, phase_logits, dead_logits=None, *, size=128):
        self._validate_boxes(boxes)
        if phase_logits.shape != (len(boxes), 7) or not bool(torch.isfinite(phase_logits).all()):
            raise ValueError('Atlas phase prediction requires finite NN logits[B,7]')
        if not self.training:
            dead = None if dead_logits is None else (dead_logits.reshape(-1) >= 0)
            return self.render_label_atlas(boxes, phase_logits.argmax(-1), dead, size=size)
        probability = phase_logits.float().softmax(-1)
        hard = F.one_hot(probability.argmax(-1), 7).float()
        phase = hard + (probability-probability.detach())
        frames = torch.stack([self._warp_native(boxes.float(), index, size=size, straight_through=True)
                              for index in range(7)], 1)
        rgba = (frames * phase[:, :, None, None, None]).sum(1)
        if dead_logits is not None:
            probability = dead_logits.float().sigmoid().reshape(-1)
            bit = (probability >= .5).float() + (probability-probability.detach())
            rgba = self._dead_color(rgba, bit, straight_through=True)
        return rgba

    def forward_with_aux(self, low_rgb, hypothesis=None):
        # Explicitly bypass every learned paint/seed/state-condition module.
        result = SemanticEDSRUpscaler.forward_with_aux(self, low_rgb)
        base = result['image']
        values, _, applicable = project_enemy_hypothesis(hypothesis, batch=len(low_rgb), device=low_rgb.device)
        visual = self.enemy_decoder_visual(low_rgb)
        pooled = torch.cat((visual.mean((-2,-1)), visual.amax((-2,-1))), -1)
        heat = self.enemy_decoder_heatmap(visual).float().flatten(1).softmax(-1).reshape(-1,32,32)
        axis = (torch.arange(32, device=low_rgb.device, dtype=torch.float32)+.5)/32
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        heat_center = torch.stack(((heat*xx).sum((-2,-1)), (heat*yy).sum((-2,-1))), -1)
        correction = self.enemy_decoder_box(pooled).float().tanh()
        size = ((.02+self.config['enemy_decoder_visual_size_max']*self.enemy_decoder_size(pooled).float().sigmoid())
                * (2*correction[:,2:]).exp()).clamp(1e-4,5.)
        center = heat_center + .5*size*correction[:,:2]
        visual_box = torch.cat((center-size/2, center+size/2), -1)
        # Optional geometry is projected only from detached *predicted* state.
        state = values[:,:20]
        sine, cosine = F.normalize(state[:,2:4], dim=-1, eps=1e-6).unbind(-1)
        dx, dy = ((state[:,8:10]-state[:,:2])*16).unbind(-1)
        depth, lateral = dx*cosine+dy*sine, -dx*sine+dy*cosine
        projection = 1/(2*math.tan(math.radians(68)/2))
        height = (projection*.93/depth.clamp_min(.05)).clamp_max(5.)
        cx = .5+projection*lateral/depth.clamp_min(.05)
        bottom = .5+projection*.5/depth.clamp_min(.05)
        proposal = torch.stack((cx-height/2,bottom-height,cx+height/2,bottom),-1)
        applicable = applicable & (depth>.12)[:,None]
        def summary(box):
            roi = sample_box(visual, box.clamp(0,1), 8)
            return torch.cat((roi.mean((-2,-1)),roi.amax((-2,-1))),-1)
        guided_summary, visual_summary = summary(proposal), summary(visual_box)
        extent = (proposal[:,2:]-proposal[:,:2]).clamp_min(1e-5)
        delta = .15*self.enemy_decoder_box(guided_summary).float().tanh()*extent.repeat(1,2)
        guided_box = proposal+delta
        guided_center = (guided_box[:,:2]+guided_box[:,2:])/2
        guided_size = (guided_box[:,2:]-guided_box[:,:2]).clamp(1e-4,5.)
        guided_box = torch.cat((guided_center-guided_size/2,guided_center+guided_size/2),-1)
        visual_phase = self.enemy_decoder_animation(visual_summary).float()
        guided_phase = self.enemy_decoder_animation(guided_summary).float()
        dead_logits = self.atlas_player_dead_head(pooled).float()
        visual_rgba = self.fixed_atlas_rgba(visual_box,visual_phase,dead_logits)
        guided_rgba = self.fixed_atlas_rgba(guided_box,guided_phase,dead_logits)
        visible_logits = self.enemy_decoder_visible(visual).float()
        probability = visible_logits.sigmoid()
        hard_visibility = (visible_logits>=0).float()
        visibility = hard_visibility+(probability-probability.detach()) if self.training else hard_visibility
        visual_confirmation = self.enemy_decoder_visual_confirmation(visual_summary).float()
        guided_confirmation = self.enemy_decoder_confirmation(guided_summary).float()
        presence = self.enemy_decoder_global_presence(pooled).float()
        presence_probability = presence.sigmoid()
        agreement = torch.exp(-.5*(((proposal[:,:2]+proposal[:,2:])/2-center)
                          /(.1+extent/2)).square().sum(-1,keepdim=True))
        intersects = (visual_box[:,2:]>0).all(-1,keepdim=True)&(visual_box[:,:2]<1).all(-1,keepdim=True)
        threshold = self.config['enemy_decoder_visual_threshold']
        visual_hard = (presence_probability>=threshold)&(visual_confirmation.sigmoid()>=threshold)&intersects
        guided_hard = (~visual_hard)&(presence_probability>=.5)&applicable&(guided_confirmation>=0)&(agreement>=.05)
        visual_soft = presence_probability*visual_confirmation.sigmoid()*intersects.float()
        guided_soft = (1-visual_soft)*presence_probability*guided_confirmation.sigmoid()*agreement*applicable.float()
        visual_weight = torch.where(visual_hard,torch.ones_like(visual_soft),visual_soft) if self.training else visual_hard.float()
        guided_weight = torch.where(visual_hard,torch.zeros_like(guided_soft),guided_soft) if self.training else guided_hard.float()
        scale = self.restoration_blend_scale
        va = visual_rgba[:,3:]*visibility
        ga = guided_rgba[:,3:]*visibility
        visual_alpha, guided_alpha = va*visual_weight[:,:,None,None]*scale, ga*guided_weight[:,:,None,None]*scale
        proposed = (base.float()+visual_alpha*(visual_rgba[:,:3]-base.float())
                    +guided_alpha*(guided_rgba[:,:3]-base.float()))
        bounded = proposed.clamp(0,1)
        image = bounded.detach()+(proposed-proposed.detach()) if self.training else bounded
        choose_guided = guided_hard
        chosen_box = torch.where(choose_guided,guided_box,visual_box)
        chosen_phase = torch.where(choose_guided,guided_phase,visual_phase)
        chosen_intrinsic = torch.where(choose_guided[:,:,None,None],guided_rgba[:,3:],visual_rgba[:,3:])
        # Compatibility canonical previews only; inference uses native atlas128.
        unit_box = chosen_box.new_tensor([0.,0.,1.,1.])[None].expand(len(low_rgb),-1)
        canonical_visual = self.fixed_atlas_rgba(unit_box,visual_phase,dead_logits,size=32)
        canonical_guided = self.fixed_atlas_rgba(unit_box,guided_phase,dead_logits,size=32)
        result.update(image=image.to(base.dtype),presence_logits=presence,mask_logits_lr=F.avg_pool2d(visible_logits,4),
            enemy_rgba=canonical_guided,visual_rgba=canonical_visual,enemy_bbox=guided_box,visual_bbox=visual_box,
            visual_proposal_bbox=visual_box,enemy_visual_bbox=visual_box,hypothesis_bbox=proposal,
            enemy_applicable=applicable,projection_agreement=agreement,
            enemy_confirmation_logits=guided_confirmation,visual_confirmation_logits=visual_confirmation,
            enemy_animation_logits=guided_phase,visual_animation_logits=visual_phase,
            enemy_animation_index=guided_phase.argmax(-1),visual_animation_index=visual_phase.argmax(-1),
            enemy_visible_logits_hr=visible_logits,visual_visible_logits_hr=visible_logits,
            enemy_alpha_hr=ga,visual_alpha_hr=va,enemy_composite_alpha_hr=guided_alpha,
            visual_composite_alpha_hr=visual_alpha,enemy_gain=image.new_tensor(scale),visual_gain=image.new_tensor(scale),
            visual_route=visual_hard,guided_route=guided_hard,visual_route_weight=visual_weight,guided_route_weight=guided_weight,
            visual_confidence=torch.minimum(presence_probability,visual_confirmation.sigmoid()),
            visual_first=True,restoration=True,restoration_base_rgb=base,restoration_blend_scale=scale,
            guided_raw_warped_rgba=guided_rgba,visual_raw_warped_rgba=visual_rgba,
            atlas_full_bbox=chosen_box,visual_full_bbox=visual_box,guided_full_bbox=guided_box,
            atlas_phase_logits=chosen_phase,atlas_phase_index=chosen_phase.argmax(-1),
            atlas_player_dead_logits=dead_logits,atlas_player_dead_index=(dead_logits[:,0]>=0).long(),
            atlas_visibility_logits_hr=visible_logits,atlas_visible_logits_hr=visible_logits,
            atlas_visibility_probability_hr=probability,atlas_visibility_hard_hr=hard_visibility,
            atlas_intrinsic_alpha_hr=chosen_intrinsic,atlas_effective_alpha_hr=chosen_intrinsic*visibility,
            atlas_composite_alpha_hr=visual_alpha+guided_alpha,
            atlas_asset_sha256=self.config['atlas_asset_sha256'],atlas_pixels_sha256=self.config['atlas_pixels_sha256'],
            fixed_atlas_enemy=True,learned_enemy_paint=False,occlusion_queries='NN_binary_visibility_only',
            phase_source='RGB_NN_logits_only',geometry_source='RGB_NN_full_bbox_optional_predicted_RSSM_proposal',
            visual_animation_source='RGB_only',legacy_RGB_parent_frozen=True)
        result.update(guided_rgba=canonical_guided,guided_bbox=guided_box,
            guided_hypothesis_bbox=proposal,guided_hypothesis_full_bbox=proposal,
            guided_confirmation_logits=guided_confirmation,
            guided_applicable=applicable,guided_alpha_hr=ga,
            guided_animation_logits=guided_phase,guided_animation_index=guided_phase.argmax(-1))
        return result
