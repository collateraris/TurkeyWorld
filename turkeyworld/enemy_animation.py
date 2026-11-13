"""Torch-only seven-frame animation selector for normalized enemy state20.

This is shared by training labels and predicted-state conditioning. It does
not render, load an atlas, inspect a map or call the reference engine.
"""
import torch

ANIMATION_CLASSES = ('idle','walk_first','walk_second','attack','hurt','death_start','death_late')


def enemy_animation_index(state):
    if (not torch.is_tensor(state) or state.ndim < 1 or state.shape[-1] != 20
            or not state.is_floating_point() or not bool(torch.isfinite(state).all())):
        raise ValueError('Enemy animation requires finite floating normalized state[...,20]')
    # vx/2 and vy/2 are recorded: the Source speed-sum threshold0.05
    # therefore corresponds to0.025 in the normalized representation.
    moving = state[...,10].abs()+state[...,11].abs() > .025
    walk = torch.where(state[...,17] < .5,1,2)
    result = torch.where(moving,walk,torch.zeros_like(walk))
    result = torch.where(state[...,14] > 1./1.2,3,result)
    result = torch.where(state[...,15] > 0,4,result)
    death = torch.where(state[...,16] < .35/.6,5,6)
    return torch.where(state[...,13] < .5,death,result).long()
