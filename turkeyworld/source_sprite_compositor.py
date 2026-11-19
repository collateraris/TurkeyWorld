"""Rasterize the embedded enemy frames using the simulated world's state and map."""
import math

import torch
from torch import nn

from .atlas_enemy_sr import _pixel_hash
from .camera import FOV_DEGREES, camera_basis
from .enemy_animation import enemy_animation_index


def source_sprite_bounds(state, size=128):
    """Integer Source raster rectangle computed in double from the available world state."""
    pose = state.double()
    angle = torch.atan2(pose[:, 2], pose[:, 3])
    dx, dy = ((pose[:, 8:10] - pose[:, :2]) * 16).unbind(-1)
    c, s = angle.cos(), angle.sin()
    forward = dx * c + dy * s
    lateral = -dx * s + dy * c
    projection = size / (2 * math.tan(math.radians(FOV_DEGREES) / 2))
    depth = forward.clamp_min(0.12)
    extent = (projection * 0.93 / depth).clamp_max(size * 5).long()
    cx = (size / 2 + projection * lateral / depth).long()
    bottom = (size // 2 + projection * 0.5 / depth).long()
    left = cx - extent // 2
    return torch.stack((left, bottom - extent, left + extent, bottom), -1), forward


def composite_source_sprite(image, rgb, alpha):
    """Apply the original byte-space alpha blend where the sprite covers the learned scene."""
    a = alpha.float().mul(255).round()
    fg = rgb.float().mul(255).round()
    bg = image.float().mul(255).round()
    mixed = (fg * a + bg * (255 - a)).div(255, rounding_mode="floor") / 255
    return torch.where(a > 0, mixed, image)


def wall_column_depth(state, grid, size=128):
    """Camera-forward distance to the first solid tile at each pixel center (DDA)."""
    if grid.shape != (len(state), 16, 16) or grid.device != state.device:
        raise ValueError("Sprite occlusion requires the current map [B,16,16] on the state device")
    # Double precision preserves crossing order when a ray grazes a corner.
    state = state.double()
    forward, right = camera_basis(state)
    screen = (torch.arange(size, device=state.device, dtype=state.dtype) + 0.5) * (2 / size) - 1.0
    rays = (
        forward[:, None]
        + right[:, None] * (screen * math.tan(math.radians(FOV_DEGREES / 2)))[None, :, None]
    )
    origin = state[:, None, :2] * 16
    tile = origin.floor().long().expand(-1, size, -1)
    step = torch.where(rays >= 0, 1, -1)
    delta = rays.abs().clamp_min(1e-12).reciprocal()
    crossing = torch.where(step > 0, tile + 1 - origin, origin - tile) * delta
    depth = torch.full_like(rays[..., 0], 100.0)
    # Tiles are nonnegative on the closed game map. A dynamic ray can cross
    # at most 16 boundaries of each axis before it exits.
    batch = torch.arange(len(state), device=state.device)[:, None]
    for _ in range(33):
        x_first = crossing[..., 0] < crossing[..., 1]
        axis = torch.stack((x_first, ~x_first), dim=-1)
        distance = torch.where(x_first, crossing[..., 0], crossing[..., 1])
        tile = tile + step * axis
        crossing = crossing + delta * axis
        tx, ty = tile.unbind(-1)
        blocked = (
            (tx < 0)
            | (tx >= 16)
            | (ty < 0)
            | (ty >= 16)
            | (grid[batch, ty.clamp(0, 15), tx.clamp(0, 15)] != 0)
        )
        depth = torch.where(blocked & (distance < depth), distance, depth)
    return depth


class SourceSpriteCompositor(nn.Module):
    """Seven RGBA frames with their original native sizes embedded in the SR checkpoint."""

    def __init__(self, config):
        super().__init__()
        self.sizes = config["atlas_frame_sizes"]
        shape = config["atlas_buffer_shape"]
        self.pixel_hash = config["atlas_pixels_sha256"]
        if len(self.sizes) != 7 or len(shape) != 4 or list(shape[:2]) != [7, 4]:
            raise ValueError("Expected a seven-pose RGBA atlas")
        if any(len(s) != 2 or not (0 < s[0] <= shape[2] and 0 < s[1] <= shape[3]) for s in self.sizes):
            raise ValueError("Invalid native sprite dimensions")
        self.register_buffer("atlas_frames_rgba", torch.zeros(shape, dtype=torch.uint8))
        self.register_buffer("atlas_native_sizes", torch.tensor(self.sizes, dtype=torch.long))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        frames = state_dict.get(prefix + "atlas_frames_rgba")
        sizes = state_dict.get(prefix + "atlas_native_sizes")
        if frames is not None and sizes is not None:
            if (
                frames.dtype != torch.uint8
                or sizes.dtype != torch.int64
                or sizes.tolist() != self.sizes
                or list(frames.shape) != list(self.atlas_frames_rgba.shape)
                or _pixel_hash(frames, sizes) != self.pixel_hash
            ):
                error_msgs.append("Embedded seven-pose enemy atlas fingerprint or dimensions differ")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, image, state, grid, enabled=True):
        n, _, height, width = image.shape
        if (height, width) != (128, 128):
            raise ValueError("Enemy sprites require the 128x128 SR canvas")
        if not enabled:
            return dict(
                sprite_alpha=image.new_zeros(n, 1, 128, 128),
                sprite_rgb=image.new_zeros(n, 3, 128, 128),
                sprite_phase=torch.full((n,), -1, dtype=torch.long, device=image.device),
                sprite_wall_visible=torch.zeros(n, 128, dtype=torch.bool, device=image.device),
            )
        if state is None or grid is None:
            raise ValueError("Enemy sprites require current own state [B,20] and scene map [B,16,16]")
        if state.shape != (n, 20) or state.device != image.device:
            raise ValueError("Enemy sprites require current state [B,20] on the image device")
        phase = enemy_animation_index(state)
        boxes, forward = source_sprite_bounds(state)
        left, top, right, bottom = boxes.unbind(-1)
        size = (right - left).clamp_min(1)
        axis = torch.arange(128, device=image.device)[None, :]
        in_view = (
            (forward > 0.12) & (right > 0) & (left < 128) & (bottom > 0) & (top < 128) & (right > left)
        )
        column_visible = (wall_column_depth(state, grid) > forward[:, None]) & in_view[:, None]
        inside_x = (axis >= left[:, None]) & (axis < right[:, None]) & column_visible
        inside_y = (axis >= top[:, None]) & (axis < bottom[:, None])
        height, width = self.atlas_native_sizes[phase].unbind(-1)
        # Source nearest sampling uses integer offsets, without a +0.5 texel shift.
        x = (
            ((axis - left[:, None]) * width[:, None] // size[:, None])
            .clamp_min(0)
            .minimum(width[:, None] - 1)
        )
        y = (
            ((axis - top[:, None]) * height[:, None] // size[:, None])
            .clamp_min(0)
            .minimum(height[:, None] - 1)
        )
        rgba = self.atlas_frames_rgba.permute(0, 2, 3, 1)[
            phase[:, None, None], y[:, :, None], x[:, None, :]
        ].permute(0, 3, 1, 2)
        rgb = rgba[:, :3].float()
        tint = (rgb + rgb.new_tensor([70, 8, 5])[None, :, None, None]).mul(0.5).floor()
        rgb = torch.where((state[:, 4] <= 0)[:, None, None, None], tint, rgb).div_(255.0)
        alpha = rgba[:, 3:4].float() / 255 * (inside_x[:, None, None, :] & inside_y[:, None, :, None])
        return dict(
            sprite_alpha=alpha, sprite_rgb=rgb, sprite_phase=phase, sprite_wall_visible=column_visible
        )
