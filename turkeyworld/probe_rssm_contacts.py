"""Measure component gradients on a cohort-balanced Source contact batch."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .contact_objectives import (CONTACT_PROTOCOL_V2, contact_gradient_probe,
    prepare_contact_targets, swept_player_wall_loss)
from .models import load_world_checkpoint
from .rssm_contact_revision import ContactAuxiliarySampler
from .rssm_migration import configure_rssm_training
from .staged_curriculum import StagedSequenceData
from .training_pipeline import atomic_json
from .train import state_loss
from .train_rssm_dynamics import episode_names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('world', 'data-root', 'old-data-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--parameter-scope', default='all', choices=('all', 'player_head'))
    parser.add_argument('--latent-mode', default='sample', choices=('sample', 'argmax'))
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.device == 'cuda': torch.cuda.init()
    model, metadata = load_world_checkpoint(args.world, 'cpu')
    step = int(metadata['training_steps'])
    del metadata
    gc.collect()
    model.to(args.device).float()
    configure_rssm_training(model)
    model.core.train(args.latent_mode == 'sample')
    new = StagedSequenceData(args.data_root/'short/rgb/train', args.device, mode='stream', cache_bytes=64*1024**2,
        episode_paths=episode_names(args.data_root, split='train', stage='short'))
    old = StagedSequenceData(args.old_data_root/'short/rgb/train', args.device, mode='stream', cache_bytes=64*1024**2)
    sampler = ContactAuxiliarySampler(new, old)
    batch, selected = sampler.sample({'updates': 0}, np.random.default_rng(173))
    report = contact_gradient_probe(model, batch, contact_protocol=CONTACT_PROTOCOL_V2,
        parameter_scope=args.parameter_scope, seed=173)
    # Measure a separate uninterrupted own-state16-step chunk too: its wall
    # gradients need not have the same scale as Source-conditioned queries.
    parameters = [p for name, p in model.named_parameters() if p.requires_grad
        and (args.parameter_scope == 'all' or 'player_pass_head.' in name)]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if args.device == 'cuda' else []), torch.enable_grad():
        torch.manual_seed(173)
        with torch.no_grad(): fmap = model.encode_map(batch['map']).detach()
        states, target = batch['states'], batch['target_states']
        current = states[:, 0]
        memory = model.initialize_from_image(batch['context_frames'], states[:, :1], fmap)
        path = []
        for offset in range(16):
            result = model.transition(current, batch['actions'][:, offset], fmap, memory)
            current, memory = result['state'], result['memory']
            path.append(current)
        prediction = torch.stack(path, 1)
        supervision = prepare_contact_targets(states[:, :16], target, batch['actions'], contact_protocol=CONTACT_PROTOCOL_V2)
        old = torch.cat([states[:, :1, :2], prediction[:, :-1, :2]], 1)
        wall = swept_player_wall_loss(old, prediction[..., :2], states[:, :16, :2], target[..., :2],
            batch['map'], supervision['sample_weight'], contact_protocol=CONTACT_PROTOCOL_V2)
        coordinate = F.smooth_l1_loss(prediction[..., :2]*16, target[..., :2]*16)
        state_objective = 10*state_loss(prediction.flatten(0, 1), target.flatten(0, 1))
        for name, value, weight in (('autonomous_wall', wall, CONTACT_PROTOCOL_V2['autonomous_wall_weight']),
                ('autonomous_xy', coordinate, 1.), ('autonomous_state', state_objective, 1.)):
            gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            norms = [torch.linalg.vector_norm(g.detach(), dtype=torch.float64) for g in gradients if g is not None]
            norm = float(torch.linalg.vector_norm(torch.stack(norms)))
            report['components'][name] = {'loss': float(value.detach()), 'weight':weight,
                'unweighted_norm':norm, 'weighted_norm':norm*weight}
    report.update(protocol_version=2, phase='source', step=step,
        latent_mode=args.latent_mode,
        source_checkpoint=str(args.world.resolve()), selected_cohorts=selected,
        settings_decision='Measured default weights; retain only if gradient magnitudes are comparable and finite')
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
