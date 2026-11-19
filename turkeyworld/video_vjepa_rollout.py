"""Offline V-JEPA game frames. The engine supplies the initial scene only."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .dreamer_policy_world import WorldState
from .models import load_world_checkpoint
from .neural_play import EVENT_NAMES
from .record_paired_cuts import action_at, scene_factory
from .record_quiet_chase import tracking_action
from .rssm_combat_protocol import environment_profile
from .small_enemy_data import digest
from .sprite_scene import base_scene_state, set_sprite_scene
from .vjepa_policy_world import VJEPAImaginationAdapter
from .vjepa_sr_feedback import feedback_active, blend_memory
from .workspace_paths import resolve_path


def load_video_world(checkpoint, device='cuda'):
    """Load weights on the CPU; transfer inference modules, never Adam, to the GPU.

    Pass an immutable snapshot (e.g. a hard link to an atomically published trainer
    checkpoint) so receipt and frames refer to precisely the same weights.
    """
    checkpoint = Path(checkpoint).resolve()
    world, saved = load_world_checkpoint(checkpoint, 'cpu')
    step = int(saved['training_steps'])
    settings = dict(saved['vjepa_sr_feedback'])
    del saved
    if settings['Source_future_feedback'] or settings['initial_observation'] != 'generated_Base_SR':
        raise ValueError('Expected autonomous frozen Base/SR feedback')
    adapter = VJEPAImaginationAdapter(world.to(device))
    # FrozenBaseSR authenticates all three feedback assets before loading them.
    r = torch.load(resolve_path(settings['renderer']), weights_only=False, mmap=True, map_location='meta')
    base_step = int(r['base_training_steps'])
    del r
    s = torch.load(resolve_path(settings['sr']), weights_only=False, mmap=True, map_location='meta')
    sr_step = int(s['training_steps'])
    sr_arch = s['config']['architecture']
    del s
    return adapter, dict(step=step, world_step=step, base_step=base_step, sr_step=sr_step,
                         sr_architecture=sr_arch, architecture=world.config.architecture,
                         player_motion_mode=world.config.player_motion_mode,
                         checkpoint_sha256=digest(checkpoint), settings=settings,
                         event_names=list(EVENT_NAMES))


@torch.inference_mode()
def paired_rollout(adapter, scenes, chase, *, action_tapes=None):
    """Yield 180 own-state forecasts in scene order, with the exact action tape.

    A single rendered SR observation initializes game memory. The displayed SR128
    tensor is also the input to the trained visual feedback when its tick is due.
    Source never supplies future poses, frames, sprite masks, actions or memory.
    """
    scenes = list(scenes)
    if not scenes or any(bool(scene['chase']) != chase for scene in scenes):
        raise ValueError('A video batch needs a single environment profile')
    if action_tapes is not None:
        if (action_tapes.shape != (len(scenes), 180, 4) or action_tapes.dtype != np.float32
                or not np.isfinite(action_tapes).all() or action_tapes[:, :, 3].any()):
            raise ValueError("Expected one 180-tick non-firing float32 action tape per scene")
    w, renderer = adapter.world, adapter.feedback_renderer
    device = next(w.parameters()).device
    maps, states = [], []
    for scene in scenes:
        game = scene_factory(scene)(seed=scene['seed'], endless=w.config.endless, enemies_enabled=chase)
        maps.append(game.map.copy())
        states.append(game.state_vector().copy())
    del game
    grid = torch.as_tensor(np.stack(maps), dtype=torch.long, device=device)
    initial = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
    with environment_profile(w, navigation_only=not chase):
        # initialize() replaces this placeholder with the render of the initial state.
        cur = adapter.initialize(initial, grid, initial.new_zeros(len(scenes),3,32,32))
        sr = renderer.sr
        for tick in range(180):
            if action_tapes is None:
                own = cur.state.cpu().numpy()
                action = np.asarray([tracking_action(own[i], s['angle']) if chase else action_at(s,tick)
                                     for i,s in enumerate(scenes)], dtype=np.float32)
            else:
                action = action_tapes[:, tick]
            if action[:,3].any(): raise ValueError('Recording requires quiet controls')
            control = adapter._action(cur, torch.as_tensor(action, device=device))
            adapter._bind_map_context(cur)
            result = w.transition(cur.state, control, cur.map_features, cur.memory)
            state = result['state']
            background = base_scene_state(state, sr, navigation_only=not chase)
            base = renderer.base(background, grid, navigation_only=not chase)
            set_sprite_scene(sr, state, grid, navigation_only=not chase)
            rendered = sr.forward_with_aux(base)
            image = rendered['image']
            alpha = rendered['sprite_alpha']
            if rendered.get('weapon_memory') is not None:
                # Weapon is composited in front of NPC in the final image.
                alpha = alpha*(1-rendered['weapon_memory']['alpha'])
            memory = result['memory']
            active = feedback_active(state, adapter.feedback_settings)
            if bool(active.any()):
                observation = F.interpolate(image, size=(32,32), mode='area')
                memory = blend_memory(w,memory,observation,adapter.feedback_settings,active)
            cur = WorldState(state.float(),memory.flatten(1).float(),cur.map_features,grid)
            adapter._validate_state(cur)
            if image.shape[1:] != (3,128,128) or not torch.isfinite(image).all():
                raise ValueError('SR produced invalid display frame')
            yield dict(action=action, pred_state=state.detach().cpu().numpy(),
                       right=image.clamp(0,1).mul(255).round().to(torch.uint8).permute(0,2,3,1).contiguous().cpu().numpy(),
                       events=result['events'].bool().cpu().numpy(),
                       sprite_visible=(alpha.flatten(1).amax(-1)>.01).cpu().numpy())
