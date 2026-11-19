"""Offline distillation of the gameplay V-JEPA visual encoder and recurrent predictor."""
import argparse
import hashlib
import os
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from torch.nn import functional as F

from .distill_renderer import _save
from .models import NeuralWorldModel, WorldModelConfig
from .rssm_combat_protocol import environment_profile
from .training_pipeline import atomic_json


def select_blocks(weights, name, target_depth, depth):
    prefix = f'vjepa_backend.core.{name}.blocks.'
    scores = []
    for index in range(depth):
        outputs = [weights[f'{prefix}{index}.{key}.weight'] for key in ('attention.proj', 'mlp.2')]
        scores.append(sum(w.float().square().mean().item() for w in outputs))
    return sorted(sorted(range(depth), key=scores.__getitem__, reverse=True)[:target_depth])


def build_student(saved, encoder_depth=6, predictor_depth=7):
    cfg = WorldModelConfig(**saved['config'])
    if cfg.architecture != 'vjepa2_ac' or not cfg.vjepa_pooled_memory:
        raise ValueError('Distillation requires pooled V-JEPA')
    cfg = replace(
        cfg,
        vjepa_encoder_depth=encoder_depth,
        vjepa_predictor_depth=predictor_depth,
        vjepa_gradient_checkpointing=False,
    )
    student = NeuralWorldModel(cfg).eval().requires_grad_(False)
    weights = saved['model']
    kept = {
        k: select_blocks(weights, k, n, saved['config'][f'vjepa_{k}_depth'])
        for k, n in [('encoder', encoder_depth), ('predictor', predictor_depth)]
    }
    mapped = {}
    for key in student.state_dict():
        parts = key.split('.')
        if parts[:2] == ['vjepa_backend', 'core'] and len(parts) >= 6 and parts[3] == 'blocks':
            parts[4] = str(kept[parts[2]][int(parts[4])])
        mapped[key] = weights['.'.join(parts)]
    student.load_state_dict(mapped, strict=True)
    student.core.requires_grad_(True)
    # Conditioning is part of the inherited game-state interface.
    student.core.predictor.action_encoder.requires_grad_(False)
    student.core.predictor.state_encoder.requires_grad_(False)
    return student, kept


def dynamics_parameters(model):
    return sum(
        p.numel()
        for k, p in model.named_parameters()
        if not k.startswith(('flow_renderer.', 'vjepa_backend.renderer.'))
        or k.startswith('vjepa_backend.renderer.map_encoder.')
    )


def snapshot(saved, student, kept, step, metrics):
    keys = (
        'format_version',
        'stage',
        'training_steps',
        'state_schema',
        'behavior_version',
        'vjepa_sr_feedback',
        'local_axis_head_migration',
        'quiet_pair_migration',
        'vjepa_spawn_phase_migration',
    )
    return {
        **{k: saved[k] for k in keys if k in saved},
        'config': vars(student.config),
        'model': {k: v.detach().cpu() for k, v in student.state_dict().items()},
        'distillation': {
            'teacher_step': saved['training_steps'],
            'optimizer_steps': step,
            'retained_blocks': kept,
            'metrics': metrics,
            'dynamics_parameters': dynamics_parameters(student),
        },
    }


def make_teacher(saved):
    model = NeuralWorldModel(WorldModelConfig(**saved['config']))
    model.load_state_dict(saved['model'], strict=True)
    return model.eval().requires_grad_(False).to('cuda').half()


@torch.no_grad()
def make_targets(teacher, renderer, dataset, settings, batch_size=16):
    from .vjepa_sr_feedback import feedback_active, blend_memory

    stacks = {}
    device = next(teacher.parameters()).device
    backend = teacher.vjepa_backend
    for nav in (False, True):
        indices = torch.where(dataset['nav'] == nav)[0]
        for clump in indices.split(batch_size):
            if not len(clump):
                continue
            states = dataset['states'][clump].to(device)
            grid = dataset['grid'][clump].to(device)
            actions = dataset['actions'][clump].to(device)
            with environment_profile(teacher, navigation_only=nav):
                fmap = teacher.encode_map(grid)
                # Rebuild actual deployment inputs from a recorded start. Roll all future
                # states from the frozen teacher and the recorded controls.
                state = states[:, 0].float()
                observed = renderer.observation(state, grid, navigation_only=nav)
                encoded = teacher.core.encode_video(observed[:, None])
                memory = backend.initialize_from_latents(encoded, state, fmap)
                recurrent = []
                for tick in range(actions.shape[1]):
                    if tick in (0, 4):
                        recurrent.append(
                            dict(
                                state=state,
                                memory=memory,
                                grid=grid,
                                nav=dataset['nav'][clump].to(device),
                                actions=actions[:, tick : tick + 2],
                                step=torch.full((len(clump),), tick, device=device),
                            )
                        )
                    output = teacher.transition(state, actions[:, tick], fmap, memory)
                    state, memory = output['state'], output['memory']
                    for key in ('latent_prediction', 'target_logits', 'attack_logits'):
                        recurrent[-1].setdefault(key + '_labels', []).append(output[key])
                    mask = feedback_active(state, settings)
                    if mask.any():
                        image = renderer.observation(state, grid, navigation_only=nav)
                        if tick < actions.shape[1] - 1:
                            memory = blend_memory(teacher, memory, image, settings, mask)
                    else:
                        image = observed
                    recurrent[-1].setdefault('feedback', []).append(image)
                    recurrent[-1].setdefault('feedback_mask', []).append(mask)
                for row in recurrent:
                    row.update(rgb=observed, encoded=encoded[:, 0])
                    for key, value in row.items():
                        if isinstance(value, list):
                            value = torch.stack(value[:2], 1)
                        stacks.setdefault(key, []).append(value.detach().cpu())
    return {k: torch.cat(v) for k, v in stacks.items()}


def losses(model, data, indices, settings):
    from .vjepa_sr_feedback import blend_memory

    dev = next(model.parameters()).device
    batch = {k: v[indices].to(dev) for k, v in data.items()}
    nav = bool(batch['nav'][0])
    if not bool((batch['nav'] == nav).all()):
        raise ValueError('Each minibatch uses one environment profile')
    with environment_profile(model, navigation_only=nav), torch.autocast(
        'cuda', dtype=torch.bfloat16
    ):
        fmap = model.encode_map(batch['grid'])
        state = batch['state']
        encoded = model.core.encode_video(batch['rgb'][:, None])
        encoder_loss = F.mse_loss(encoded[:, 0].float(), batch['encoded'].float())
        initial = model.vjepa_backend.initialize_from_latents(encoded, state, fmap)
        memory = torch.where(batch['step'][:, None, None] == 0, initial, batch['memory'])
        latent_loss = state.new_zeros(())
        head_loss = state.new_zeros(())
        for t in range(2):
            output = model.transition(state, batch['actions'][:, t], fmap, memory)
            latent_loss += (
                F.mse_loss(
                    output['latent_prediction'].float(),
                    batch['latent_prediction_labels'][:, t].float(),
                )
                / 2
            )
            # Calibrate the frozen nonlinear readouts as well as the full visual vector.
            head_loss += (
                F.smooth_l1_loss(
                    output['target_logits'].float(), batch['target_logits_labels'][:, t].float()
                )
                / 2
            )
            head_loss += (
                F.smooth_l1_loss(
                    output['attack_logits'].float(), batch['attack_logits_labels'][:, t].float()
                )
                / 2
            )
            state, memory = output['state'], output['memory']
            if t == 0 and batch['feedback_mask'][:, t].any():
                memory = blend_memory(
                    model, memory, batch['feedback'][:, t], settings, batch['feedback_mask'][:, t]
                )
    return encoder_loss, latent_loss, head_loss


@torch.no_grad()
def evaluate(model, data, settings, batch_size):
    totals = np.zeros(3)
    for nav in (False, True):
        for index in torch.where(data['nav'] == nav)[0].split(batch_size):
            if len(index):
                totals += np.array([float(x) for x in losses(model, data, index, settings)]) * len(
                    index
                )
    enc, pred, heads = map(float, totals / len(data['nav']))
    return dict(
        encoder_mse=enc, predictor_mse=pred, head_huber=heads, score=enc + pred + 0.05 * heads
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--teacher', required=True, type=Path)
    ap.add_argument('--cycles', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=19117)
    a = ap.parse_args()
    run_in_output(a)


def run_in_output(a):
    """Keep checkpoints, metrics and split provenance together for one training run."""
    a.output.mkdir(exist_ok=True, parents=True)
    with FileLock(str(a.output / '.distillation.lock'), timeout=0):
        owned = {'init.pt', 'student.pt', 'early.pt', 'final.pt', 'metrics.json', 'split.json',
                 'init.pt.tmp', 'student.pt.tmp'}
        for entry in a.output.iterdir():
            if entry.name in owned or re.fullmatch(r'(early|metrics\.json|split\.json)\.\d+\.tmp', entry.name):
                raise FileExistsError(f'Distillation output already contains {entry.name}: {a.output}')
        _train(a)


def _train(a):
    torch.manual_seed(a.seed)
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = True
    saved = torch.load(a.teacher, mmap=True, weights_only=False, map_location='cpu')
    student, kept = build_student(saved)
    print('student', dynamics_parameters(student), 'kept', kept, flush=True)
    _save(a.output / 'init.pt', snapshot(saved, student, kept, 0, {}))
    from .vjepa_kd_data import load_kd_windows
    from .vjepa_sr_feedback import FrozenBaseSR

    train, hold, manifest = load_kd_windows(a.cycles, a.seed)
    atomic_json(a.output / 'split.json', manifest)
    settings = saved['vjepa_sr_feedback']
    teacher = make_teacher(saved)
    renderer = FrozenBaseSR(settings, 'cuda')
    samples = {}
    for key, data in [('heldout', hold), ('train', train)]:
        samples[key] = make_targets(teacher, renderer, data, settings)
        print(key, 'teacher targets', len(samples[key]['nav']), flush=True)
    del teacher, renderer, train, hold
    torch.cuda.empty_cache()
    student.to('cuda')
    metrics = evaluate(student, samples['heldout'], settings, a.batch)
    print('initial', metrics, flush=True)
    log = [dict(step=0, heldout=metrics)]
    atomic_json(a.output / 'metrics.json', log)
    best = metrics['score']
    best_step = None
    optimizer = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad), lr=a.lr, weight_decay=0.01
    )
    generator = torch.Generator().manual_seed(a.seed)
    pools = {nav: torch.where(samples['train']['nav'] == nav)[0] for nav in (False, True)}
    for step in range(1, a.steps + 1):
        pool = pools[step % 3 == 0]
        index = pool[torch.randint(len(pool), (a.batch,), generator=generator)]
        optimizer.zero_grad(set_to_none=True)
        enc, pred, heads = losses(student, samples['train'], index, settings)
        loss = enc + pred + 0.05 * heads
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite KD loss at {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step in (10, 30, 75) or step % 100 == 0 or step == a.steps:
            metrics = evaluate(student, samples['heldout'], settings, a.batch)
            print('step', step, 'train', float(loss.detach()), 'heldout', metrics, flush=True)
            row = dict(step=step, train_loss=float(loss.detach()), heldout=metrics)
            log.append(row)
            if metrics['score'] < best:
                best = metrics['score']
                _save(a.output / 'student.pt', snapshot(saved, student, kept, step, metrics))
                if best_step is None:
                    temporary = a.output / f'early.{os.getpid()}.tmp'
                    temporary.hardlink_to(a.output / 'student.pt')
                    os.replace(temporary, a.output / 'early.pt')
                best_step = step
                row['saved_best'] = True
            atomic_json(a.output / 'metrics.json', log)
    if best_step is None:
        raise RuntimeError('No student improved held-out distillation loss')
    final = a.output / 'final.pt'
    final.hardlink_to(a.output / 'student.pt')
    with final.open('rb') as f:
        print(
            'selected',
            best_step,
            'heldout_score',
            best,
            'sha256',
            hashlib.file_digest(f, 'sha256').hexdigest(),
        )


if __name__ == '__main__':
    main()
