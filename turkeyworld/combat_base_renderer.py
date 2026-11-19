"""Native RGB32 readout trained on game combat, bound to the deployed Base and its geometry."""
from copy import deepcopy

import torch
from torch import nn

from .base_only_renderer import enable_base_only
from .camera import project_sprite_bounds
from .decoder_rf_chain_bundle import load_decoder_rf_chain_payload
from .renderer_rf5_bundle import parent_bundle_fingerprint
from .renderer_student import _hash_tensors
from .workspace_paths import identity_path


KIND = 'base_only_combat_readout_v1'


class CombatBaseReadout(nn.Module):
    """Learn a per-token RGB correction from Base features, HUD timing and enemy-relative coordinates."""
    def __init__(self, width, hidden=128):
        super().__init__()
        # 18 screen coordinates, 9 event scalars, and 4 relative sprite coordinates.
        self.mlp = nn.Sequential(nn.Linear(width + 31, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 12))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def active(state):
        recent_action = (state[:, 6:8].amax(-1) > 1.e-4) | (state[:, 14:16].amax(-1) > 1.e-4)
        corpse = (state[:, 13] < .5) & (state[:, 18] >= .5)
        return recent_action | corpse

    def forward(self, queries, state, screen, logits):
        projection = project_sprite_bounds(state, image_size=32)
        box = projection['unclipped_boxes'][:, 0, None]
        xy = (screen[..., :2] + 1) * .5
        size = (box[..., 2:] - box[..., :2]).clamp_min(1 / 32)
        relative = (2 * (xy - box[..., :2]) / size - 1).clamp(-3, 3)
        relative = torch.where(projection['sprite_frustum'][:, :, None], relative, 0.)
        offset = torch.cat((relative, relative.abs()), -1)
        events = torch.cat((state[:, 6:8], state[:, 12:18], (state[:, 18:19] > .5)), -1)
        features = torch.cat((queries, screen.expand(len(queries), -1, -1).to(queries),
                              events[:, None].expand(-1, queries.shape[1], -1).to(queries),
                              offset.to(queries)), -1)
        delta = self.mlp(features)
        return torch.where(self.active(state)[:, None, None], logits + delta, logits)


def combat_bundle(parent, readout, *, steps):
    values = {key: value.detach().cpu() for key, value in readout.state_dict().items()}
    return dict(format_version=9, stage='renderer_student', kind=KIND, flow_matching=False,
                solver_steps=0, parent_bundle=parent, parent_bundle_sha256=parent_bundle_fingerprint(parent),
                parent_base_identity=parent['base_student_identity'], local_updates=steps,
                base_training_steps=parent['base_training_steps'] + steps, readout=values,
                readout_sha256=_hash_tensors(values), hidden=readout.mlp[0].out_features)


def load_combat_renderer(teacher, payload, *, checkpoint, require_trained=True, allow_unqualified_codec=False):
    parent = payload.get('parent_bundle')
    steps = payload.get('local_updates')
    if (payload.get('kind') != KIND or payload.get('stage') != 'renderer_student'
            or payload.get('flow_matching') is not False or payload.get('solver_steps') != 0
            or not isinstance(parent, dict) or parent.get('format_version') != 8
            or type(steps) is not int or steps < int(require_trained)
            or payload.get('base_training_steps') != parent.get('base_training_steps', -1) + steps
            or payload.get('parent_base_identity') != parent.get('base_student_identity')
            or payload.get('parent_bundle_sha256') != parent_bundle_fingerprint(parent)):
        raise ValueError('Combat Base bundle identity or training counters differ')
    values = payload.get('readout')
    if (not isinstance(values, dict) or type(payload.get('hidden')) is not int or not 1 <= payload['hidden'] <= 1024
            or not all(isinstance(t, torch.Tensor) and t.dtype == torch.float32 and t.isfinite().all() for t in values.values())
            or payload.get('readout_sha256') != _hash_tensors(values)):
        raise ValueError('Combat Base readout fingerprint or tensors differ')
    # The selected format-8 -> 7 -> 6 chain owns the exact compact Base configuration.
    try:
        width = parent['parent_bundle']['parent_bundle']['base_student_bundle']['config']['width']
    except (KeyError, TypeError) as error:
        raise ValueError('Combat Base parent misses the selected compact head configuration') from error
    if type(width) is not int or not 1 <= width <= 2048:
        raise ValueError('Combat Base parent token width is invalid')
    with torch.random.fork_rng(devices=[]):
        readout = CombatBaseReadout(width, payload['hidden'])
    readout.load_state_dict(values, strict=True)
    load_decoder_rf_chain_payload(teacher, parent, checkpoint=checkpoint, require_trained=require_trained,
                                  allow_unqualified_codec=allow_unqualified_codec)
    head = teacher.base_student
    readout.to(next(head.parameters())).eval().requires_grad_(False)
    head.combat_readout = readout
    enable_base_only(teacher)
    summary = {key: deepcopy(value) for key, value in payload.items() if key not in ('parent_bundle', 'readout')}
    summary.update(checkpoint=identity_path(checkpoint), base_student_identity=_hash_tensors(head.state_dict()),
                   training_steps=payload['base_training_steps'])
    teacher.base_student_metadata = dict(teacher.base_student_metadata, training_steps=payload['base_training_steps'],
                                       head_parameters=sum(p.numel() for p in head.parameters()),
                                       model_sha256=summary['base_student_identity'], combat_readout=payload['readout_sha256'])
    teacher.renderer_student_metadata = summary
    return summary
