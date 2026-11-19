"""Compare a combat Base/SR pair with the visuals frozen into a dynamics checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .small_enemy_data import digest
from .training_pipeline import atomic_json
from .vjepa_sr_feedback import FrozenBaseSR

GROUPS = {
    'nav': 'Навигация',
    'quiet': 'Преследование',
    'combat': 'Бой',
    'flash': 'Вспышка',
    'hurt': 'Ранение врага',
    'death': 'Гибель врага',
    'melee': 'Ближняя атака',
}
COMBAT = ('combat', 'flash', 'hurt', 'death', 'melee')
MASKS = ('enemy', 'weapon', 'flash')
SIZES = (32, 128)


def accumulate(totals, key, errors):
    # Each entry in errors is the mean absolute difference over RGB at one pixel.
    total, count = totals.get(key, (0., 0))
    totals[key] = total + errors.sum(dtype=np.float64), count + errors.size


def statistics(totals):
    return {
        key: {'mae': float(total / count) if count else None, 'pixels': count}
        for key, (total, count) in totals.items()
    }


def masks_for(data, rows, size):
    masks = {key: data[f'{key}{size}'][rows].astype(bool) for key in MASKS}
    enemy = masks['enemy']
    masks['background'] = ~(enemy | masks['weapon'] | masks['flash'])
    # Within one pixel of the Source enemy, measure both the visible contour and
    # the immediately surrounding background. Exclude the foreground gun.
    padded = np.pad(enemy, ((0, 0), (1, 1), (1, 1)), mode='edge')
    neighbors = [padded[:, y:y + size, x:x + size] for y in range(3) for x in range(3)]
    masks['enemy_inner_edge'] = enemy & ~np.logical_and.reduce(neighbors)
    masks['enemy_outer_edge'] = masks['background'] & np.logical_or.reduce(neighbors)
    return masks


def render(model, states, maps, navigation):
    low = model.base(states, maps, navigation_only=navigation)
    # Use the production high-resolution path, including its own background pass
    # and map for wall occlusion when evaluating a sprite-compositing checkpoint.
    high = model.image(states, maps, navigation_only=navigation)
    return {
        size: tensor.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        for size, tensor in zip(SIZES, (low, high))
    }


def example_index(data, group, indices):
    # Select using the reference foreground area alone.
    mask = {'flash': 'flash128', 'nav': 'weapon128'}.get(group, 'enemy128')
    areas = data[mask][indices].reshape(len(indices), -1).sum(axis=1)
    visible = np.flatnonzero(areas)
    if not len(visible):
        return indices[0]
    typical = np.median(areas[visible])
    return indices[visible[np.argmin(np.abs(areas[visible] - typical))]]


def save_preview(examples, destination):
    font = ImageFont.truetype('DejaVuSans.ttf', 15)
    side = 152
    sheet = Image.new('RGB', (178 + 6 * side, 59 + len(GROUPS) * (side + 5)), '#172029')
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 8), 'Отложенные кадры • одинаковое состояние игры для обеих моделей',
              font=font, fill='white')
    columns = ('Оригинал 32', 'Base до', 'Base после', 'Оригинал 128', 'SR до', 'SR после')
    for col, title in enumerate(columns):
        draw.text((180 + col * side, 34), title, font=font, fill='#bde3ef')
    for row, (group, label) in enumerate(GROUPS.items()):
        y = 55 + row * (side + 5)
        index, images = examples[group]
        draw.text((14, y + 10), label, font=font, fill='white')
        draw.text((14, y + 34), f'Кадр выборки {index}', font=font, fill='#9bb2c3')
        for col, pixels in enumerate(images):
            tile = Image.fromarray(pixels).resize((side, side), Image.Resampling.NEAREST)
            sheet.paste(tile, (178 + col * side, y))
    sheet.save(destination)


def save_sr_preview(examples, destination):
    font = ImageFont.truetype('DejaVuSans.ttf', 16)
    side = 192
    margin = 168
    sheet = Image.new('RGB', (margin + 3 * (side + 5), 52 + len(examples) * (side + 4)), '#172029')
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(('Оригинал 128×128', 'SR до', 'SR после')):
        draw.text((margin + col * (side + 5), 18), title, font=font, fill='white')
    for row, (group, label) in enumerate(GROUPS.items()):
        y = 48 + row * (side + 4)
        draw.text((12, y + 5), label, font=font, fill='white')
        index, frames = examples[group]
        draw.text((12, y + 30), f'Кадр {index}', font=font, fill='#9bb2c3')
        for col, pixels in enumerate(frames[3:]):
            tile = Image.fromarray(pixels).resize((side, side), Image.Resampling.NEAREST)
            sheet.paste(tile, (margin + col * (side + 5), y))
    sheet.save(destination)


@torch.inference_mode()
def evaluate(args):
    torch.set_num_threads(2)
    parent = torch.load(args.world, mmap=True, weights_only=False, map_location='meta')
    reference = dict(parent['vjepa_sr_feedback'])
    for key, path in (('renderer', args.reference_renderer), ('sr', args.reference_sr)):
        if path is not None:
            reference[key] = str(path.resolve())
            reference[key + '_sha256'] = digest(path)
    candidate = dict(reference)
    for key, path in (('renderer', args.renderer), ('sr', args.sr)):
        candidate[key] = str(path.resolve())
        candidate[key + '_sha256'] = digest(path)
    models = {arm: FrozenBaseSR(config, args.device)
              for arm, config in [('before', reference), ('after', candidate)]}
    keys = ['group', 'states', 'maps', 'rgb32', 'rgb128']
    keys.extend(f'{mask}{size}' for mask in MASKS for size in SIZES)
    with np.load(args.data, allow_pickle=False) as archive:
        data = {key: archive[key] for key in keys}
    groups = {}
    examples = {}
    counts = {}
    identical = {}
    combat_totals = {arm: {} for arm in models}
    for group in GROUPS:
        indices = np.flatnonzero(data['group'] == group)
        if not len(indices):
            raise ValueError(f'No frames in {group} group')
        counts[group] = len(indices)
        example = example_index(data, group, indices)
        totals = {arm: {} for arm in models}
        identical[group] = {str(size): True for size in SIZES}
        for start in range(0, len(indices), args.batch_size):
            rows = indices[start:start + args.batch_size]
            states = torch.as_tensor(data['states'][rows], device=args.device, dtype=torch.float32)
            maps = torch.as_tensor(data['maps'][rows], device=args.device, dtype=torch.long)
            predictions = {arm: render(model, states, maps, group == 'nav')
                           for arm, model in models.items()}
            for size in SIZES:
                target = data[f'rgb{size}'][rows].astype(np.float32) / 255
                masks = masks_for(data, rows, size)
                identical[group][str(size)] &= np.array_equal(
                    predictions['before'][size], predictions['after'][size])
                for arm, frames in predictions.items():
                    errors = np.abs(frames[size] - target).mean(axis=-1)
                    accumulate(totals[arm], f'{size}_rgb', errors)
                    for name, mask in masks.items():
                        accumulate(totals[arm], f'{size}_{name}', errors[mask])
            if example in rows:
                offset = np.flatnonzero(rows == example)[0]
                images = []
                for size in SIZES:
                    images.append(data[f'rgb{size}'][example])
                    for arm in models:
                        images.append(np.rint(predictions[arm][size][offset] * 255).astype(np.uint8))
                examples[group] = int(example), images
        groups[group] = {arm: statistics(metrics) for arm, metrics in totals.items()}
        if group in COMBAT:
            for arm, metrics in totals.items():
                for key, (total, count) in metrics.items():
                    old_total, old_count = combat_totals[arm].get(key, (0., 0))
                    combat_totals[arm][key] = old_total + total, old_count + count
        print(group, counts[group], flush=True)

    macro = {arm: {str(size): float(np.mean([groups[g][arm][f'{size}_rgb']['mae'] for g in COMBAT]))
                   for size in SIZES} for arm in models}
    report = {
        'protocol': 'combat_visuals_paired_source_state_v2',
        'world_step': parent['training_steps'], 'data_file': args.data.name,
        'data_sha256': digest(args.data), 'counts': counts, 'combat_macro_rgb_mae': macro,
        'combat_pooled_pixels': {arm: statistics(metrics) for arm, metrics in combat_totals.items()},
        'groups': groups, 'identical_images': identical,
        'example_indices': {group: value[0] for group, value in examples.items()},
        'reference': reference, 'candidate': candidate,
        'target_condition': 'Matched Source state20 and map on both arms, normal Base/SR image path',
    }
    args.output_dir.mkdir(exist_ok=True, parents=True)
    save_preview(examples, args.output_dir / 'comparison.png')
    if args.reference_sr is not None:
        save_sr_preview(examples, args.output_dir / 'sr_comparison.png')
    atomic_json(args.output_dir / 'paired_visuals.json', report)
    print(macro, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('world', 'renderer', 'sr', 'data', 'output-dir'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--reference-sr', type=Path, help='Override the baseline SR from the world checkpoint.')
    parser.add_argument('--reference-renderer', type=Path, help='Override the baseline Base renderer.')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    evaluate(args)
