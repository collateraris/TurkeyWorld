"""Continue RGB detail SR with a frozen, patrol-only sprite placement model."""
import torch

from .background_detail_sr import BackgroundDetailSR
from .enemy_animation import enemy_animation_index


def patrol_alpha(alpha, phase, *, enemies_enabled=True, predicted_state=None):
    allowed = phase < 3
    if predicted_state is not None:
        allowed = allowed & (enemy_animation_index(predicted_state) < 3)
        allowed = allowed & (predicted_state[:, 13] >= .5)
    if not enemies_enabled:
        allowed = torch.zeros_like(allowed)
    return alpha * allowed[:, None, None, None].to(alpha)


class PatrolSpriteSR(BackgroundDetailSR):
    def __init__(self, *, sprite_config, **kwargs):
        super().__init__(**kwargs)
        from .models import build_upscaler
        self.sprite_renderer = build_upscaler(sprite_config).eval().requires_grad_(False)
        self.config.update(architecture='patrol_sprite_detail_sr', sprite_config=sprite_config)
        self.enemies_enabled = True
        self.predicted_sprite_state = None

    def forward_with_aux(self, low_rgb, hypothesis=None):
        result = super().forward_with_aux(low_rgb)
        with torch.no_grad():
            sprite = self.sprite_renderer.forward_with_aux(low_rgb)
            alpha = patrol_alpha(sprite['visual_composite_alpha_hr'],
                sprite['visual_animation_index'], enemies_enabled=self.enemies_enabled,
                predicted_state=self.predicted_sprite_state)
            rgb = sprite['visual_raw_warped_rgba'][:, :3]
        result['image'] = (result['image']*(1-alpha)+rgb*alpha).clamp(0, 1)
        result.update(sprite_alpha=alpha, patrol_sprites_only=True,
            combat_animation_sprites=False, sprite_phase=sprite['visual_animation_index'])
        return result
