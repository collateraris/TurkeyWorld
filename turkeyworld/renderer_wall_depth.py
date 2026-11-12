"""Loss-only Source depth strata for distant walls and thin wall faces.

No renderer or dynamics runtime imports this module. Source poses, maps and
occlusion masks are detached labels, never conditions or corrected predictions.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .renderer_wall_supervision import source_wall_masks, _gun_masks, _per_image
from .engine import Game


DEPTH_BANDS = (
    ('near_0_2', 0., 2.), ('mid_2_5', 2., 5.),
    ('far_5_8', 5., 8.), ('distant_8_12', 8., 12.),
    ('thin_12_plus', 12., math.inf),
)
DEPTH_WALL_PROTOCOL = {
    'name': 'source_native_wall_depth_strata_v1',
    'native_size': 32, 'depth': 'Source camera-forward DDA depth in map tiles',
    'band_edges_tiles': [0., 2., 5., 8., 12.],
    'bands_disjoint': True,
    'horizontal_pair_band': 'farther of its two Source depth columns',
    'normalization': 'per-image region mean, then present images; each band and aggregate remains separate',
    'source_labels_only': True, 'exclude_source_npc_and_weapon_alpha': True,
    'thin_face': 'contiguous same-plane same-material Source wall face, 1 or 2 native columns, depth >= 5',
    'thin_height': 'continuous projected Source wall height <= 2 native pixels',
}


def _source_ray_faces(tilemap, x, y, rx, ry):
    """Return the exact closed-map Game.render DDA depth and wall-face IDs."""
    width, height = tilemap.shape[1], tilemap.shape[0]
    mx = np.full(len(rx), int(x), dtype=np.int32)
    my = np.full(len(rx), int(y), dtype=np.int32)
    sx, sy = np.where(rx >= 0, 1, -1), np.where(ry >= 0, 1, -1)
    ddx = 1 / np.maximum(np.abs(rx), 1e-12)
    ddy = 1 / np.maximum(np.abs(ry), 1e-12)
    tx = np.where(rx >= 0, mx + 1 - x, x - mx) * ddx
    ty = np.where(ry >= 0, my + 1 - y, y - my) * ddy
    hit = np.zeros(len(rx), dtype=bool)
    side = np.zeros(len(rx), dtype=bool)
    depth = np.full(len(rx), 100.)
    material = np.ones(len(rx), dtype=np.int64)
    hit_x, hit_y = mx.copy(), my.copy()
    for _ in range(width + height + 8):
        active = ~hit
        on_x = (tx < ty) & active
        on_y = ~on_x & active
        distance = np.where(on_x, tx, ty)
        mx[on_x] += sx[on_x]; my[on_y] += sy[on_y]
        tx[on_x] += ddx[on_x]; ty[on_y] += ddy[on_y]
        tile = tilemap[np.clip(my, 0, height - 1), np.clip(mx, 0, width - 1)]
        new_hit = active & (tile != 0)
        depth[new_hit] = distance[new_hit]
        side[new_hit] = on_y[new_hit]
        material[new_hit] = tile[new_hit]
        hit_x[new_hit] = mx[new_hit]; hit_y[new_hit] = my[new_hit]
        hit |= new_hit
        if bool(hit.all()):
            break
    if not bool(hit.all()):
        raise ValueError('Source wall depth labels require a closed Source map')
    # Adjacent coplanar tiles with the same material form one visible face.
    plane = np.where(side, hit_y + (sy < 0), hit_x + (sx < 0))
    faces = np.stack((side.astype(np.int64), plane, material), axis=-1)
    return np.maximum(depth, .02), faces


def _thin_face_columns(depth, faces):
    """One/two-column face runs; coplanar adjacent tiles are not false corners."""
    change = np.any(faces[1:] != faces[:-1], axis=-1)
    starts = np.r_[0, np.flatnonzero(change) + 1]
    ends = np.r_[starts[1:], len(depth)]
    thin = np.zeros(len(depth), dtype=bool)
    for first, end in zip(starts, ends):
        if end - first <= 2:
            thin[first:end] = depth[first:end] >= 5.
    return thin, change


def source_wall_depth_labels(grid, states, object_mask):
    """Build exact exposed native32 wall strata without using target RGB.

    Boundary pixels inherit their camera column's wall depth, including the
    sky/floor side of a boundary. Horizontal edge pairs inherit the farther
    depth, so a near wall cannot consume the far-side corner denominator.
    """
    base = source_wall_masks(grid, states, object_mask)  # validates all inputs
    maps = grid.detach().cpu().numpy()
    poses = states.detach().double().cpu().numpy()
    objects = object_mask.detach().cpu().numpy()[:, 0] > .5
    tangent = math.tan(math.radians(68) / 2)
    screen = (np.arange(32, dtype=np.float64) + .5) / 32 * 2 - 1
    depths, heights, corners, thin_faces, thin_heights = [], [], [], [], []
    for tilemap, state, npc in zip(maps, poses, objects):
        x, y = state[:2] * 16
        angle = math.atan2(state[2], state[3])
        c, s = math.cos(angle), math.sin(angle)
        depth, faces = _source_ray_faces(tilemap, x, y,
            c - s * tangent * screen, s + c * tangent * screen)
        height = (32 / (2 * tangent)) / depth
        yy = np.arange(32)[:, None]
        v = (yy - (16 - height[None, :] / 2)) / height[None, :]
        wall = (v >= 0) & (v < 1)
        flash = float(np.clip(state[7], 0, 1)) * Game.FLASH_DURATION
        gun = _gun_masks()[1 if flash > .045 else 2 if flash > 0 else 0]
        exposed = ~(npc | gun)
        thin, changes = _thin_face_columns(depth, faces)
        pairs = changes[None, :] & (wall[:, 1:] | wall[:, :-1])
        pairs &= exposed[:, 1:] & exposed[:, :-1]
        corner = np.zeros((32, 32), dtype=bool)
        corner[:, 1:] |= pairs; corner[:, :-1] |= pairs
        depths.append(depth); heights.append(height)
        corners.append(corner)
        thin_faces.append(wall & thin[None, :] & exposed)
        thin_heights.append(wall & (height[None, :] <= 2.) & exposed)
    def tensor(values):
        return torch.from_numpy(np.stack(values)).to(device=states.device, dtype=torch.float32)
    depth = tensor(depths)[:, None, None, :]
    height = tensor(heights)[:, None, None, :]
    corner = tensor(corners)[:, None]
    thin_face = tensor(thin_faces)[:, None]
    thin_height = tensor(thin_heights)[:, None]
    pair_depth = torch.maximum(depth[..., 1:], depth[..., :-1])
    labels = {'depth': depth, 'projected_height': height,
              'corner': corner, 'thin_face': thin_face, 'thin_height': thin_height,
              'bands': {}, 'aggregates': {}}
    def masks(low, high):
        col = (depth >= low) & (depth < high)
        pair = (pair_depth >= low) & (pair_depth < high)
        return {'wall': base['wall'] * col,
                'boundary': base['boundary'] * col,
                'horizontal': base['horizontal'] * pair,
                'vertical': base['vertical'] * col,
                'corner': corner * col,
                'thin_face': thin_face * col,
                'thin_height': thin_height * col}
    for name, low, high in DEPTH_BANDS:
        labels['bands'][name] = masks(low, high)
    for low in (5., 8., 12.):
        labels['aggregates']['depth_ge_' + str(int(low))] = masks(low, math.inf)
    return labels


def _mean_region(error, mask):
    value, present = _per_image(error, mask.detach().to(error))
    zero = error.sum() * 0
    return value[present].mean() if bool(present.any()) else zero, int(present.sum())


def _region_terms(prediction, target, labels):
    error = (prediction - target).abs()
    result = {}
    for name in ('wall', 'boundary', 'corner', 'thin_face', 'thin_height'):
        value, count = _mean_region(error, labels[name])
        result[name + '_l1'] = value
        result[name + '_images'] = count
        result[name + '_pixels'] = int(labels[name].sum())
    edge_values, edge_present = [], []
    for axis, name in ((-1, 'horizontal'), (-2, 'vertical')):
        error = (prediction.diff(dim=axis) - target.diff(dim=axis)).abs()
        values, present = _per_image(error, labels[name].detach().to(error))
        edge_values.append(values); edge_present.append(present)
    values, present = torch.stack(edge_values), torch.stack(edge_present)
    per_image = (values * present).sum(0) / present.sum(0).clamp_min(1)
    valid = present.any(0)
    result['wall_edge_l1'] = per_image[valid].mean() if bool(valid.any()) else prediction.sum() * 0
    result['wall_edge_images'] = int(valid.sum())
    return result


def depth_banded_wall_terms(prediction, target, labels):
    """Unweighted region terms: do not silently choose production loss weights."""
    if (prediction.shape != target.shape or prediction.shape[1:] != (3, 32, 32)
            or not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all())):
        raise ValueError('Depth-stratified walls require aligned finite native32 RGB')
    target = target.detach().to(prediction)
    return {kind: {name: _region_terms(prediction, target, masks)
                  for name, masks in labels[kind].items()}
            for kind in ('bands', 'aggregates')}


def depth_banded_wall_report(arrays):
    """Plain-JSON same-pose Base/RF/codec references with explicit support."""
    labels = source_wall_depth_labels(arrays['map'], arrays['states'], arrays['mask'])
    result = {'protocol': dict(DEPTH_WALL_PROTOCOL), 'gt_source_poses': True,
              'codec_reference_is_a_proven_lower_bound': False}
    for name, key in (('base', 'base'), ('rf1', 'rgb'), ('codec_reference', 'codec_reference')):
        terms = depth_banded_wall_terms(arrays[key], arrays['source'], labels)
        result[name] = {kind: {band: {field: float(value.detach()) if torch.is_tensor(value) else value
                                            for field, value in row.items()}
                              for band, row in groups.items()}
                        for kind, groups in terms.items()}
    return result
