"""Training-only Source wall projection masks, never renderer conditions."""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import torch

from .engine import Game
from .geometry_targets import _wall_depths


WALL_PROTOCOL = {'name':'source_native_wall_boundary_v1','native_size':32,
    'boundary_l1_weight':.2,'wall_edge_weight':.1,'fov_degrees':68.,
    'source_labels_only':True,'exclude_source_npc_and_weapon_alpha':True,
    'normalization':'per-image region mean, then mean over present images'}


@lru_cache(maxsize=1)
def _gun_masks():
    # Same immutable Source atlas and exact integer-nearest rasterizer. This
    # creates no Source transitions or rendered target RGB and touches no
    # global RNG; Game owns a private generator.
    game = Game(seed=0)
    size = int(32*.67)
    masks = []
    for sprite in game._guns:
        alpha = sprite.copy()
        alpha[...,:3] = 255
        alpha[...,3] = (sprite[...,3] > 0).astype(np.uint8)*255
        canvas = np.zeros((32,32,3),np.uint8)
        Game._blit_rgba(canvas,alpha,16-size//2,32-size,size,size)
        masks.append(canvas[...,0] > 0)
    return tuple(masks)


def source_wall_masks(grid, states, object_mask):
    """Project Source GT geometry in float64 and exclude actual occluders.

    A one-pixel band on either side of the wall/sky/floor boundary supervises
    the boundary location. Edge support also covers interior wall texture and
    corners. NPC/weapon pixels never enter either denominator. Float32 recorded
    poses are decoded as Source.set_state_vector does; no NN pose is corrected.
    """
    if (states.ndim != 2 or states.shape[1] != 20 or grid.shape != (len(states),16,16)
            or object_mask.shape != (len(states),1,32,32)
            or not all(bool(torch.isfinite(v).all()) for v in (grid,states,object_mask))):
        raise ValueError('Source wall labels require finite map16, state20 and native32 NPC masks')
    maps = grid.detach().cpu().numpy()
    poses = states.detach().double().cpu().numpy()
    objects = object_mask.detach().cpu().numpy()[:,0] > .5
    all_wall, all_boundary, all_horizontal, all_vertical = [],[],[],[]
    tangent = math.tan(math.radians(68)/2)
    screen = (np.arange(32,dtype=np.float64)+.5)/32*2-1
    yy = np.arange(32)[:,None]
    for tilemap, state, npc in zip(maps,poses,objects):
        x,y = state[:2]*16
        if not (0 <= x < 16 and 0 <= y < 16) or tilemap[int(y),int(x)] != 0:
            raise ValueError('Source wall labels require a legal GT camera in a walkable tile')
        angle = math.atan2(state[2],state[3])
        c,s = math.cos(angle),math.sin(angle)
        depth = np.maximum(_wall_depths(tilemap,x,y,c-s*tangent*screen,s+c*tangent*screen,100.),.02)
        height = (32/(2*tangent))/depth
        v = (yy-(16-height[None,:]/2))/height[None,:]
        wall = (v >= 0) & (v < 1)
        boundary = np.zeros_like(wall)
        horizontal = wall[:,1:] != wall[:,:-1]
        vertical = wall[1:] != wall[:-1]
        boundary[:,1:] |= horizontal; boundary[:,:-1] |= horizontal
        boundary[1:] |= vertical; boundary[:-1] |= vertical
        flash = float(np.clip(state[7],0,1))*Game.FLASH_DURATION
        gun = _gun_masks()[1 if flash > .045 else 2 if flash > 0 else 0]
        exposed = ~(npc | gun)
        all_wall.append(wall & exposed)
        all_boundary.append(boundary & exposed)
        all_horizontal.append((wall[:,1:] | wall[:,:-1]) & exposed[:,1:] & exposed[:,:-1])
        all_vertical.append((wall[1:] | wall[:-1]) & exposed[1:] & exposed[:-1])
    def tensor(values):
        return torch.from_numpy(np.stack(values)[:,None]).to(device=states.device,dtype=torch.float32)
    return {'wall':tensor(all_wall),'boundary':tensor(all_boundary),
            'horizontal':tensor(all_horizontal),'vertical':tensor(all_vertical)}


def _per_image(error, mask):
    mass = mask.flatten(1).sum(-1)*error.shape[1]
    values = (error*mask).flatten(1).sum(-1)/mass.clamp_min(1)
    return values, mass > 0


def wall_loss_terms(prediction, target, labels):
    if prediction.shape != target.shape or prediction.shape[1:] != (3,32,32):
        raise ValueError('Wall supervision requires aligned native32 RGB')
    target = target.detach().to(prediction)
    zero = prediction.sum()*0
    boundary,present = _per_image((prediction-target).abs(),labels['boundary'].detach().to(prediction))
    boundary_loss = boundary[present].mean() if bool(present.any()) else zero
    edge_values, edge_present = [],[]
    for axis,name in ((-1,'horizontal'),(-2,'vertical')):
        error = (prediction.diff(dim=axis)-target.diff(dim=axis)).abs()
        value,valid = _per_image(error,labels[name].detach().to(prediction))
        edge_values.append(value); edge_present.append(valid)
    values = torch.stack(edge_values)
    valid = torch.stack(edge_present)
    per_image = (values*valid).sum(0)/valid.sum(0).clamp_min(1)
    any_edges = valid.any(0)
    edge_loss = per_image[any_edges].mean() if bool(any_edges.any()) else zero
    return {'boundary_l1':boundary_loss,'wall_edge_l1':edge_loss,
        'boundary_images':int(present.sum()),'wall_edge_images':int(any_edges.sum()),
        'boundary_pixels':int(labels['boundary'].sum()),
        'wall_pixels':int(labels['wall'].sum()),
        'loss':.2*boundary_loss+.1*edge_loss}


def wall_validation_report(arrays):
    labels = source_wall_masks(arrays['map'],arrays['states'],arrays['mask'])
    report = {'protocol':dict(WALL_PROTOCOL),'gt_source_poses':True,
              'codec_reference_is_a_proven_lower_bound':False}
    for name,key in (('base','base'),('rf1','rgb'),('codec_reference','codec_reference')):
        values = wall_loss_terms(arrays[key],arrays['source'],labels)
        report[name] = {key:float(value) if torch.is_tensor(value) else value for key,value in values.items()}
    return report


def wall_preservation_assessment(report, baseline, *, tolerance=.002):
    failures=[]
    for head in ('base','rf1'):
        current,old=report.get(head,{}),baseline.get(head,{})
        for field,count in (('boundary_l1','boundary_images'),('wall_edge_l1','wall_edge_images')):
            a,b=current.get(field),old.get(field)
            if not current.get(count) or current.get(count)!=old.get(count):
                failures.append(head+'_'+field+'_missing_or_changed_support')
            elif any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in (a,b)):
                failures.append(head+'_'+field+'_nonfinite')
            elif a>b+tolerance:
                failures.append(head+'_'+field+'_regressed')
    return {'passed':not failures,'failure_reasons':failures,'absolute_mae_tolerance':tolerance,
            'scope':'Same exposed Source wall labels and present-image denominators'}
