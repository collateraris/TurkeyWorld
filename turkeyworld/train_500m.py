"""500M compact V-JEPA2-AC teacher with growing, streamed game data."""
from __future__ import annotations

import argparse
from pathlib import Path

from .train_250m import ExpandedExperiment
from .power_guard import KeepAwake


class MassiveExperiment(ExpandedExperiment):
    PROFILE = '500'
    PARAMETER_TARGET = 500000000

    def __init__(self, args):
        super().__init__(args)
        self.state['dataset_target_transitions'] = 5008192
        self.state['model_parameters'].update(teacher_total=500044741, ema_target_total=239975936)
        self.state['data_growth'] = {'target_base_training_transitions': 5000000,
            'additional_model_visited_transitions': 8192, 'validation_transitions': 16384,
            'append_only': True, 'training_continues_during_collection': True}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('training_runs/vjepa500_20261109'))
    parser.add_argument('--source', type=Path, default=Path('training_runs/vjepa250_20261109/checkpoints/teacher250_world.pt'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pretrain-steps', type=int, default=6000)
    parser.add_argument('--world-repair-steps', type=int, default=3000)
    parser.add_argument('--world-patience', type=int, default=6)
    parser.add_argument('--student-steps', type=int, default=16000)
    parser.add_argument('--repair-steps', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--preview-every', type=int, default=10)
    parser.add_argument('--train-renderer-during-pretrain', action='store_true')
    parser.add_argument('--target-seconds', type=int, default=300)
    parser.add_argument('--cache-mb', type=int, default=512)
    parser.add_argument('--visited-clips', type=int, default=256)
    parser.add_argument('--minimum-train-clips', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--ema', type=float, default=.996)
    parser.add_argument('--context', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--plateau-patience', type=int, default=6)
    parser.add_argument('--plateau-delta', type=float, default=.005)
    parser.add_argument('--quality-passes', type=int, default=3)
    parser.add_argument('--quality-psnr', type=float, default=35.)
    parser.add_argument('--quality-ssim', type=float, default=.98)
    parser.add_argument('--quality-mae', type=float, default=2/255)
    parser.add_argument('--quality-fraction', type=float, default=.95)
    parser.set_defaults(teacher_steps=6000, temporal_updates=400, reflow_pairs=2048,
        validate_every=500, teacher_until_quality=True, teacher_size='500m', aligned_supervision=True,
        adversarial=True, adversarial_weight=.02, adversarial_feature_weight=.05,
        adversarial_warmup=512, adversarial_ramp=512, adversarial_r1_gamma=1.,
        adversarial_r1_interval=16, adversarial_lr=1e-4)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.pretrain_steps, args.world_repair_steps, args.world_patience, args.student_steps,
           args.repair_steps, args.batch, args.cache_mb, args.chunk, args.context, args.horizon,
           args.gradient_accumulation_steps, args.preview_every) < 1:
        parser.error('Counts must be positive')
    experiment = MassiveExperiment(args)
    try:
        with KeepAwake() as awake:
            experiment.state['power_guard'] = {'active': awake.active, 'display_required': False,
                'screen_lock_allowed': True, 'error': awake.error}
            experiment.run()
    except Exception as error:
        experiment.state['status'] = 'failed'
        experiment.update(experiment.state.get('stage', 'initializing'), experiment.state.get('current', 0),
                          experiment.state.get('total'), 'Ошибка обучения500M', error=str(error))
        raise


if __name__ == '__main__':
    main()
