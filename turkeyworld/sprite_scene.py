"""Split a predicted scene into the Base background and the SR sprite inputs."""
import torch


def uses_source_sprites(sr):
    return getattr(sr, 'config', {}).get('source_sprite_compositor') is True


def base_scene_state(state, sr, *, navigation_only):
    """For full-atlas SR only, move the render-only NPC slot behind the camera.

    The neural background retains camera, player, weapon cooldown, flash and
    time. The compositor and the next transition use the original state.
    """
    if navigation_only or not uses_source_sprites(sr):
        return state
    background = state.clone()
    # Normalized state stores sin/cos of heading and XY in units of map width.
    behind = state[:, :2] - .25 * state[:, [3, 2]]
    background[:, 8:10] = behind.clamp(0., 1.)
    background[:, 10:19] = 0.
    background[:, 16] = 1.  # Fully elapsed death timer, matching an empty slot.
    return background


def set_sprite_scene(sr, state, grid, *, navigation_only):
    """Bind the current pose and the initial grid for this exact inference batch."""
    if uses_source_sprites(sr):
        if not navigation_only and (state is None or grid is None):
            raise ValueError('Full-atlas SR requires the current predicted state and its initial map')
        if grid is not None and (not torch.is_tensor(grid) or grid.ndim != 3
                                 or (state is not None and len(grid) != len(state))):
            raise ValueError('Sprite map batch must match the current state batch')
        sprite_map = None
        if grid is not None and state is not None:
            sprite_map = grid.detach().to(device=state.device, dtype=torch.long)
        sr.predicted_sprite_map = sprite_map
    sr.enemies_enabled = not navigation_only
    sr.predicted_sprite_state = state.detach() if state is not None else None
