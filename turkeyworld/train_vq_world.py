"""500M teacher CFM and 22.48M student RF using one qualified frozen VQGAN."""
from __future__ import annotations

from pathlib import Path
import os

import torch

from .models import NeuralWorldModel
from .power_guard import KeepAwake
from .train_500m import MassiveExperiment, build_parser
from .training_pipeline import atomic_json
from .vq_world import (STUDENT_VQ_PARAMETERS, TEACHER_VQ_PARAMETERS,
    migrate_teacher_to_vq, student_vq_config, validate_frozen_codec)


class LatentWorldExperiment(MassiveExperiment):
    PROFILE = '500vq'
    STUDENT_PARAMETER_TARGET = STUDENT_VQ_PARAMETERS
    STUDENT_PROFILE_NAME = 'vjepa_vq_rf_22m'

    def __init__(self, args):
        self.codec_info = validate_frozen_codec(args.codec_checkpoint)
        super().__init__(args)
        self.state.update(model='Compact V-JEPA 2-AC 500M / shared VQGAN / latent CFM → RF',
            teacher_size='500m_vqgan', codec_validation=self.codec_info['quality_gate'],
            codec_ready=self.codec_info,
            codec_training={'from_scratch': True, 'pretrained_codec_weights_loaded': False,
                'parameters': self.codec_info['generator_parameters'], 'frozen': True,
                'consecutive_passes': self.codec_info['quality_gate']['consecutive_passes'],
                'required_passes': self.codec_info['quality_gate']['required_passes'],
                'current': self.codec_info['training_steps']},
            pipeline={'shared_codec_for_teacher_and_student': True,
                'codec_checkpoint': str(args.codec_checkpoint.resolve()),
                'codec_fingerprint': self.codec_info['codec_fingerprint']})
        self.state['model_parameters'].update(teacher_total=TEACHER_VQ_PARAMETERS,
            codec_total=self.codec_info['generator_parameters'])
        self.update('initializing', 0, 1, 'Учитель: продолжение динамики и переход к общему VQGAN')

    def make_student_config(self):
        return student_vq_config(self.codec_info['codec_config'])

    def make_student_model(self):
        model = NeuralWorldModel(self.make_student_config()).to(self.device)
        model.flow_renderer.load_codec(self.args.codec_checkpoint)
        return model

    def create_initial_teacher(self, initial):
        return migrate_teacher_to_vq(self.args.source, self.args.codec_checkpoint, initial,
            resume_output=self.root/'checkpoints'/'teacher500vq_world.pt')

    def check_stop(self, model, optimizer, phase, step, metrics):
        control = getattr(self.args, 'control_run_dir', None)
        if control and (control/'stop_training').exists():
            (self.root/'stop_training').touch()
        return super().check_stop(model, optimizer, phase, step, metrics)


def main(argv=None):
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(run_dir=Path('training_runs/vjepa500_vqgan_20261109'),
        source=Path('training_runs/vjepa500_20261109/checkpoints/teacher500_world.pt'),
        teacher_size='500m_vqgan')
    parser.add_argument('--codec-checkpoint', required=True, type=Path)
    parser.add_argument('--control-run-dir', type=Path)
    args = parser.parse_args(argv)
    if min(args.pretrain_steps, args.world_repair_steps, args.world_patience, args.student_steps,
           args.repair_steps, args.batch, args.cache_mb, args.chunk, args.context, args.horizon) < 1:
        parser.error('Counts must be positive')
    experiment = LatentWorldExperiment(args)
    atomic_json(experiment.root/'training.pid', os.getpid())
    atomic_json(experiment.root/'codec_ready.json', experiment.codec_info)
    try:
        with KeepAwake() as awake:
            experiment.state['power_guard'] = {'active': awake.active, 'display_required': False,
                'screen_lock_allowed': True, 'error': awake.error}
            experiment.run()
    except Exception as error:
        experiment.state['status'] = 'failed'
        experiment.update(experiment.state.get('stage', 'initializing'), experiment.state.get('current', 0),
            experiment.state.get('total'), 'Ошибка обучения world model с VQGAN', error=str(error))
        raise


if __name__ == '__main__':
    main()
