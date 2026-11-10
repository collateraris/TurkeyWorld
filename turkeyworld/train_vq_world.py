"""175M/500M teacher CFM and22.48M student RF sharing one frozen VQGAN."""
from __future__ import annotations

from pathlib import Path
import os
import json
import math
import hashlib
import time
from dataclasses import asdict

import torch

from .models import NeuralWorldModel, WorldModelConfig
from .power_guard import KeepAwake
from .train_500m import MassiveExperiment, build_parser
from .training_pipeline import atomic_json, now, replace_file
from .vq_world import (STUDENT_VQ_PARAMETERS, TEACHER_VQ_PARAMETERS,
    migrate_teacher_to_vq, student_vq_config, validate_frozen_codec)


def teacher_vq_config(codec_config, teacher_size='500m_vqgan', spatial_entities=False, spatial_combat=False,
                      game_training_v5=False):
    if teacher_size == '175m_vqgan':
        from .vq_world import teacher175_vq_config
        config = teacher175_vq_config(codec_config)
        config.spatial_entities = bool(spatial_entities)
        config.spatial_combat = bool(spatial_combat)
        config.game_training_v5 = bool(game_training_v5)
        if game_training_v5:
            config.flow_patch_size = 2
            config.flow_block_conditioning = True
        config.__post_init__()
        return config
    if teacher_size != '500m_vqgan':
        raise ValueError('Unknown shared-codec teacher profile')
    from .vjepa2_ac import VJEPA500_PROFILE
    from .vqgan import VQGANConfig
    codec = asdict(codec_config) if isinstance(codec_config, VQGANConfig) else dict(codec_config)
    codec = asdict(VQGANConfig(**codec))
    return WorldModelConfig(kind='teacher', architecture='vjepa2_ac', image_size=codec['image_size'],
        endless=True, renderer_mode='flow_matching', flow_steps=12,
        renderer_codec='vqgan', renderer_codec_config=codec,
        vjepa_patch_size=codec['image_size']//8 if codec['image_size'] <= 128 else 16,
        flow_adapter_width=512, flow_adapter_depth=12, flow_adapter_heads=8,
        spatial_entities=bool(spatial_entities), spatial_combat=bool(spatial_combat),
        game_training_v5=bool(game_training_v5), flow_patch_size=2 if game_training_v5 else 4,
        flow_block_conditioning=bool(game_training_v5), **VJEPA500_PROFILE)


class LatentWorldExperiment(MassiveExperiment):
    PROFILE = '500vq'
    STUDENT_PARAMETER_TARGET = STUDENT_VQ_PARAMETERS
    STUDENT_PROFILE_NAME = 'vjepa_vq_rf_22m'

    def __init__(self, args):
        self.teacher_budget = None
        self.training_subset = None
        self.spatial_entities = bool(getattr(args, 'spatial_entities', False))
        self.spatial_combat = bool(getattr(args, 'spatial_combat', False))
        self.game_training_v5 = bool(getattr(args, 'training_improvements_v5', False))
        if self.spatial_combat and not self.spatial_entities:
            raise ValueError('Spatial combat conditioning requires nine spatial entity states')
        if self.game_training_v5 and not self.spatial_combat:
            raise ValueError('Training v5 requires --spatial-combat and --spatial-entities')
        self.codec_options = {'accept_user_requested': True} if getattr(args, 'accept_current_codec', False) else {}
        validation_options = dict(self.codec_options)
        if hasattr(args, 'image_size'):
            validation_options['expected_image_size'] = args.image_size
        self.codec_info = validate_frozen_codec(args.codec_checkpoint, **validation_options)
        self.image_size = int(getattr(args, 'image_size', self.codec_info['codec_config']['image_size']))
        if self.image_size != self.codec_info['codec_config']['image_size']:
            raise ValueError('Frozen codec image size does not match the requested world resolution')
        args.image_size = self.image_size
        if self.image_size < 128 and not getattr(args, 'from_scratch', False):
            raise ValueError('Native32/64px world training requires --from-scratch in a new run directory')
        if self.image_size != 128:
            from .vq_world import vq_profile_parameter_count
            self.STUDENT_PARAMETER_TARGET = vq_profile_parameter_count(student_vq_config(self.codec_info['codec_config']))
        requested_size = getattr(args, 'teacher_size', '500m_vqgan')
        self.teacher_size = '500m_vqgan' if requested_size == '500m' else requested_size
        if self.teacher_size not in ('175m_vqgan', '500m_vqgan'):
            raise ValueError('Unknown shared-codec teacher profile')
        if self.teacher_size == '175m_vqgan' and not getattr(args, 'from_scratch', False):
            raise ValueError('The175M teacher requires --from-scratch in a new run directory')
        tier = self.teacher_size.split('m_')[0]
        self.PROFILE = tier+'vq'
        self.PARAMETER_TARGET = int(tier)*1_000_000
        args.teacher_size = self.teacher_size
        if getattr(args, 'gan_npc_negatives', False) and (not getattr(args, 'object_aware_rendering', False)
                or not getattr(args, 'adversarial_during_pretrain', False)):
            raise ValueError('NPC GAN negatives require object-aware joint adversarial training')
        if getattr(args, 'adversarial_during_pretrain', False):
            args.adversarial = True
            args.adversarial_weight = args.pretrain_adversarial_weight
            args.adversarial_feature_weight = args.pretrain_adversarial_feature_weight
            args.adversarial_warmup = args.pretrain_adversarial_warmup
            args.adversarial_ramp = args.pretrain_adversarial_ramp
        subset_path = getattr(args, 'training_subset_manifest', None)
        update_budget = getattr(args, 'optimizer_update_budget', None)
        if bool(subset_path) != (update_budget is not None):
            raise ValueError('A fixed subset manifest and optimizer update budget must be supplied together')
        if update_budget is not None and update_budget < 1:
            raise ValueError('Optimizer update budget must be positive')
        if (args.run_dir.resolve()/'teacher_budget.json').exists() and update_budget is None:
            raise ValueError('This run has a bounded teacher segment; resume with its subset and budget options')
        if subset_path:
            from .training_subset import load_training_subset
            self.training_subset = load_training_subset(subset_path, args.run_dir.resolve()/'data'/'rgb'/'train')
        if getattr(args, 'from_scratch', False):
            from .engine import BEHAVIOR_VERSION
            marker = args.run_dir.resolve()/'initialization.json'
            checkpoints = args.run_dir.resolve()/'checkpoints'
            if checkpoints.exists() and any(checkpoints.glob('*.pt')) and not marker.exists():
                raise ValueError('Use a new run directory for from-scratch training; existing weights are preserved')
            if marker.exists():
                previous = json.loads(marker.read_text(encoding='utf-8'))
                expected = {'mode': 'from_scratch_world_and_renderer', 'seed': args.seed,
                    'codec_fingerprint': self.codec_info['codec_fingerprint'], 'behavior_version': BEHAVIOR_VERSION,
                    'pretrained_world_weights_loaded': False, 'pretrained_renderer_weights_loaded': False,
                    'trained_codec_reused': True}
                if previous.get('teacher_size', '500m_vqgan') != self.teacher_size:
                    raise ValueError('Existing initialization marker belongs to a different scratch experiment')
                if previous.get('image_size', 128) != self.image_size:
                    raise ValueError('Existing initialization marker belongs to a different image resolution')
                if self.teacher_size == '175m_vqgan':
                    expected.update(self.teacher_initialization_profile())
                if any(previous.get(key) != value for key, value in expected.items()):
                    raise ValueError('Existing initialization marker belongs to a different scratch experiment')
            args.train_renderer_during_pretrain = True
        super().__init__(args)
        self.state.update(model=f'Compact V-JEPA 2-AC {tier}M / shared VQGAN / latent CFM → RF',
            teacher_size=self.teacher_size, image_size=self.image_size, base_resolution=[self.image_size]*2,
            codec_validation=self.codec_info['quality_gate'],
            codec_ready=self.codec_info,
            codec_training={'from_scratch': True, 'pretrained_codec_weights_loaded': False,
                'parameters': self.codec_info['generator_parameters'], 'frozen': True,
                'accepted_by_user': self.codec_info.get('accepted_by_user', False),
                'quality_gate_passed': self.codec_info['quality_gate'].get('passed', False),
                'consecutive_passes': self.codec_info['quality_gate']['consecutive_passes'],
                'required_passes': self.codec_info['quality_gate']['required_passes'],
                'current': self.codec_info['training_steps']},
            pipeline={'shared_codec_for_teacher_and_student': True,
                'codec_checkpoint': str(args.codec_checkpoint.resolve()),
                'codec_fingerprint': self.codec_info['codec_fingerprint']})
        self.state['model_parameters'].update(teacher_total=self.teacher_parameter_count(),
            codec_total=self.codec_info['generator_parameters'])
        if self.spatial_entities:
            self.state['spatial_conditioning'] = {'enabled': True, 'slots': 9, 'player_slots': 1,
                'enemy_slots': 8, 'coordinates': 'continuous_map_xy', 'local_motion': True,
                'local_map_patch': [3, 3], 'npc_direction': 'observed_velocity_with_validity_flag',
                'runtime_state_source': 'own_predictions', 'source_engine_feedback': False}
        if self.spatial_combat:
            self.state['spatial_conditioning'].update(combat=True, spawn=True,
                flow_participant_tokens=9, health_conditioning=True,
                wall_conditioning=True, map_resolution=[16, 16],
                wall_features=['tile_coordinates', 'tile_bounds', 'surface_type', 'camera_relative_geometry'],
                hit_progress='nominal_HP_derived_per_life_with_ladder_validity')
        if self.game_training_v5:
            self.state['training_improvements'] = {'version': 5, 'ar_xy_offsets': [1, 4, 8, 16],
                'training_wall_penalty': True, 'balanced_attacks': True,
                'spawn': 'eight_learned_anchors_plus_learned_tile_fallback',
                'spawn_normalization': 'actual_respawn_events_across_batch_and_horizon',
                'flow_patch_size': 2, 'flow_image_tokens': (self.codec_info['codec_config']['image_size']//8)**2,
                'flow_block_conditioning': True, 'balanced_foreground_background': True,
                'decoded_endpoint_ssim_edges': True, 'flow_time_bins': 5,
                'behavior_validation_every': 100, 'source_engine_feedback': False}
        # The500M parent estimate must not masquerade as an instantiated EMA.
        self.state['model_parameters'].pop('ema_target_total', None)
        if getattr(args, 'from_scratch', False):
            from .engine import BEHAVIOR_VERSION
            self.state['initialization'] = {'mode': 'from_scratch_world_and_renderer',
                'pretrained_world_weights_loaded': False, 'pretrained_renderer_weights_loaded': False,
                'trained_codec_reused': True, 'seed': args.seed,
                'behavior_version': BEHAVIOR_VERSION,
                'codec_fingerprint': self.codec_info['codec_fingerprint']}
            if self.image_size != 128:
                self.state['initialization']['image_size'] = self.image_size
            if self.teacher_size == '175m_vqgan':
                self.state['initialization'].update(self.teacher_initialization_profile())
            self.state['enemy_policy'] = {'source_deterministic': True, 'version': BEHAVIOR_VERSION,
                'neural_behavior': 'learned_deterministic_transition', 'neural_spawn': 'learned_argmax'}
            self.state['dataset_target_transitions'] = 5000000
            self.state['data_growth']['additional_model_visited_transitions'] = 0
        if self.training_subset is not None:
            selected = self.training_subset
            self.state['training_subset'] = {'manifest_path': str(subset_path.resolve()),
                'fingerprint': selected['fingerprint'], 'frozen': True, 'auto_expand': False,
                'train_clips': selected['count_files'], 'train_transitions': selected['total_transitions'],
                'validation_clips': 512, 'validation_transitions': 16384,
                'behavior_version': selected['behavior_version'], 'source_directory': selected['source_directory'],
                'policy_counts': selected['policy_counts']}
            self.state['dataset_target_transitions'] = selected['total_transitions']
            self.state['data_growth'].update(target_base_training_transitions=selected['total_transitions'],
                append_only=False, auto_expand=False)
            self.state['teacher_until_quality'] = False
            self.state['training_intent'] = 'fit_fixed_subset_before_generalization'
        if getattr(args, 'object_aware_rendering', False):
            if self.training_subset is None or not getattr(args, 'object_mask_directory', None):
                raise ValueError('Object-aware rendering requires the fixed subset and offline NPC mask directory')
            path = args.object_mask_directory.resolve()/'manifest.json'
            masks = json.loads(path.read_text(encoding='utf-8'))
            if (masks.get('status') != 'complete' or masks.get('mask_kind') != 'visible_npc_rgb_difference'
                    or masks.get('training_subset_fingerprint') != self.training_subset['fingerprint']
                    or masks.get('behavior_version') != self.training_subset['behavior_version']
                    or masks.get('source_data_root') != str((self.root/'data').resolve())):
                raise ValueError('Offline NPC mask cache does not match this completed training subset')
            self.state['object_mask_cache'] = {key: masks[key] for key in
                ('mask_kind', 'status', 'total_episodes', 'verified_source_frames', 'training_subset_fingerprint')}
            self.state['object_mask_cache']['path'] = str(args.object_mask_directory.resolve())
            if (getattr(args, 'gan_npc_negatives', False) or self.spatial_combat) and not masks.get('background_frames_available'):
                raise ValueError('NPC GAN negatives require offline cache prepared with --include-backgrounds')
        label = ('Учитель: мир и нейрорендер с нуля; общий обученный VQGAN'
                 if getattr(args, 'from_scratch', False) else 'Учитель: продолжение динамики и переход к общему VQGAN')
        self.update('initializing', 0, 1, label)

    def teacher_parameter_count(self):
        codec = self.codec_info['codec_config']
        if codec.get('image_size', 128) != 128 or getattr(self, 'spatial_entities', False):
            from .vq_world import vq_profile_parameter_count
            return vq_profile_parameter_count(teacher_vq_config(codec, getattr(self, 'teacher_size', '500m_vqgan'),
                spatial_entities=getattr(self, 'spatial_entities', False),
                spatial_combat=getattr(self, 'spatial_combat', False),
                **({'game_training_v5': True} if getattr(self, 'game_training_v5', False) else {})))
        if self.PROFILE == '175vq':
            from .vq_world import TEACHER175_VQ_PARAMETERS
            return TEACHER175_VQ_PARAMETERS
        return TEACHER_VQ_PARAMETERS

    def teacher_initialization_profile(self):
        config = asdict(teacher_vq_config(self.codec_info['codec_config'], self.teacher_size,
                                        spatial_entities=getattr(self, 'spatial_entities', False),
                                        spatial_combat=getattr(self, 'spatial_combat', False),
                                        **({'game_training_v5': True} if getattr(self, 'game_training_v5', False) else {})))
        # Preserve the canonical fingerprint of runs created before this opt-in field.
        if not config.get('spatial_entities'):
            config.pop('spatial_entities', None)
        if not config.get('spatial_combat'):
            config.pop('spatial_combat', None)
        if not config.get('game_training_v5'):
            config.pop('game_training_v5', None)
        if not config.get('flow_block_conditioning'):
            config.pop('flow_block_conditioning', None)
        if config.get('flow_patch_size', 4) == 4:
            config.pop('flow_patch_size', None)
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        result = {'teacher_size': self.teacher_size, 'teacher_parameters': self.teacher_parameter_count(),
                  'teacher_config_fingerprint': fingerprint}
        if getattr(self, 'spatial_entities', False):
            result.update(spatial_entities=True, spatial_entity_count=9)
        if getattr(self, 'spatial_combat', False):
            result['spatial_combat'] = True
        if getattr(self, 'game_training_v5', False):
            result.update(game_training_v5=True, flow_patch_size=2, flow_block_conditioning=True)
        return result

    def training_data_options(self):
        if self.training_subset is None:
            return super().training_data_options()
        options = {'episode_paths': self.training_subset['filenames']}
        if getattr(self.args, 'object_aware_rendering', False):
            options['object_masks_directory'] = self.args.object_mask_directory/'train'
            if getattr(self.args, 'gan_npc_negatives', False) or self.spatial_combat:
                options['counterfactual_backgrounds'] = True
        return options

    def validation_data_options(self):
        if getattr(self.args, 'object_aware_rendering', False):
            return {'object_masks_directory': self.args.object_mask_directory/'val'}
        return super().validation_data_options()

    def update(self, stage, current, total, label, **fields):
        budget = getattr(self, 'teacher_budget', None)
        if budget is not None:
            from .teacher_budget import teacher_budget_progress
            fields['optimizer_budget'] = teacher_budget_progress(budget, self.global_step)
        return super().update(stage, current, total, label, **fields)

    def save(self, model, optimizer, phase, step, metrics, extra=None):
        additions = dict(extra or {})
        if self.training_subset is not None:
            additions['training_subset'] = self.state['training_subset']
        if self.teacher_budget is not None:
            from .teacher_budget import teacher_budget_progress
            additions['optimizer_budget'] = teacher_budget_progress(self.teacher_budget, self.global_step)
        output = super().save(model, optimizer, phase, step, metrics, additions)
        if self.teacher_budget is not None:
            self.teacher_budget.update(checkpoint=str(output), checkpoint_optimizer_step=self.global_step,
                                       updated_at=now())
            atomic_json(self.root/'teacher_budget.json', self.teacher_budget)
        return output

    def run(self):
        from .teacher_budget import TeacherBudgetFinished
        try:
            return super().run()
        except TeacherBudgetFinished:
            return

    def world_until_ready(self, model, data):
        if self.training_subset is None:
            return super().world_until_ready(model, data)
        from .teacher_budget import prepare_teacher_budget, TeacherBudgetFinished
        phase = f'teacher{self.PROFILE}_world'
        checkpoint = self.root/'checkpoints'/f'{phase}.pt'
        phase_step = global_step = 0
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
            phase_step, global_step = int(saved['training_steps']), int(saved.get('global_training_step', 0))
            del saved
        self.global_step = max(self.global_step, global_step)
        self.teacher_budget = prepare_teacher_budget(self.root/'teacher_budget.json', phase=phase,
            current_optimizer_step=global_step, current_phase_step=phase_step,
            limit_updates=self.args.optimizer_update_budget, subset_fingerprint=self.training_subset['fingerprint'])
        if self.teacher_budget['status'] != 'awaiting_user':
            self.fit(model, data, phase, self.teacher_budget['stop_phase_step'], pretrain=True)
            if self.global_step != self.teacher_budget['stop_optimizer_step']:
                raise RuntimeError('Bounded teacher segment did not finish its authorized optimizer updates')
            self.prediction_preview_report(model)
            self.teacher_budget.update(status='awaiting_user', finished_at=now(), checkpoint=str(checkpoint))
            atomic_json(self.root/'teacher_budget.json', self.teacher_budget)
        self.state.update(status='awaiting_user', global_training_step=self.global_step,
            teacher_segment_complete=True, teacher_quality_qualified=False,
            continuation_requires_user=True)
        self.state['renderer_training'] = {**self.state.get('renderer_training', {}),
            'active': False, 'reason': 'optimizer_budget_exhausted'}
        if 'student_plan' in self.state:
            self.state['student_plan']['status'] = 'pending_user_decision'
            atomic_json(self.root/'student_plan.json', self.state['student_plan'])
        self.update(phase, self.teacher_budget['stop_phase_step'], self.teacher_budget['stop_phase_step'],
            'Лимит обучения учителя достигнут; ожидается решение о продолжении', eta_seconds=0,
            message='Фиксированный набор сохранён; расширение данных и следующие этапы автоматически не запускаются')
        raise TeacherBudgetFinished()

    def make_student_config(self):
        return student_vq_config(self.codec_info['codec_config'])

    def make_student_model(self):
        model = NeuralWorldModel(self.make_student_config()).to(self.device)
        model.flow_renderer.load_codec(self.args.codec_checkpoint)
        return model

    def create_initial_teacher(self, initial):
        if getattr(self.args, 'from_scratch', False):
            options = {'teacher_size': getattr(self, 'teacher_size', '500m_vqgan')}
            if getattr(self, 'spatial_entities', False):
                options['spatial_entities'] = True
            if getattr(self, 'spatial_combat', False):
                options['spatial_combat'] = True
            if getattr(self, 'game_training_v5', False):
                options['game_training_v5'] = True
            config = teacher_vq_config(self.codec_info['codec_config'], **options)
            model = NeuralWorldModel(config).eval()
            model.flow_renderer.load_codec(self.args.codec_checkpoint)
            count = sum(parameter.numel() for parameter in model.parameters())
            if count != self.teacher_parameter_count():
                raise ValueError(f'Fresh teacher profile has unexpected size: {count}')
            payload = {'format_version': 1, 'stage': 'world', 'config': asdict(model.config),
                'model': model.state_dict(), 'training_steps': 0, 'global_training_step': 0,
                'initialization': self.state['initialization'], 'behavior_version': self.state['enemy_policy']['version'],
                'pipeline_phase': f'teacher{self.PROFILE}_initial', 'created_at': now(),
                'quality_status': 'Random world and frame generator; frozen locally trained VQGAN reused; unqualified.'}
            initial.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(self.root/'initialization.json', self.state['initialization'])
            temporary = initial.with_suffix('.pt.tmp')
            torch.save(payload, temporary)
            replace_file(temporary, initial)
            return {'from_scratch': True, 'generator_parameters': count,
                'pretrained_world_weights_loaded': False, 'pretrained_renderer_weights_loaded': False,
                'trained_codec_reused': True, 'initial_checkpoint': str(initial)}
        if self.PROFILE != '500vq':
            raise ValueError('The175M teacher requires a fresh initialization')
        return migrate_teacher_to_vq(self.args.source, self.args.codec_checkpoint, initial,
            resume_output=self.root/'checkpoints'/f'teacher{self.PROFILE}_world.pt', **getattr(self, 'codec_options', {}))

    def wait_for_data(self):
        if not getattr(self.args, 'from_scratch', False):
            return super().wait_for_data()
        from .engine import BEHAVIOR_VERSION
        subset = getattr(self, 'training_subset', None)
        while True:
            if (self.root/'stop_training').exists():
                self.state['status'] = 'stopped'
                self.update('data', 0, None, 'Сбор данных остановлен')
                raise SystemExit(0)
            path = self.root/'data'/'status.json'
            if path.exists():
                data = json.loads(path.read_text(encoding='utf-8'))
                if data.get('behavior_version') != BEHAVIOR_VERSION:
                    raise ValueError('Fresh teacher data uses a different enemy behavior version')
                if data.get('status') == 'failed':
                    raise RuntimeError('Deterministic data collection failed; inspect data/status.json')
                self.state['data_preparation'] = data
                groups = data.get('groups', {})
                train = (subset['count_files'] if subset is not None
                         else groups.get('rgb/train', {}).get('files', 0))
                validation = groups.get('rgb/val', {}).get('files', 0)
                minimum = min(self.args.minimum_train_clips, subset['count_files']) if subset is not None else self.args.minimum_train_clips
                if train >= minimum and validation >= 512:
                    return
                self.update('data', train, self.args.minimum_train_clips,
                    'Новые детерминированные данные: подготовка первого обучающего набора',
                    message=f'RGB train: {train}; validation: {validation}/512; сбор полного корпуса продолжается')
            time.sleep(5)

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
        teacher_size='500m_vqgan', target_seconds=120)
    parser.add_argument('--codec-checkpoint', required=True, type=Path)
    parser.add_argument('--sr-checkpoint', type=Path,
        help='Optional trained x4 SR used only for progress-preview rows; never a world-training input')
    parser.add_argument('--image-size', type=int, choices=(32, 64, 128), default=128,
        help='Native source/world/Base/Flow RGB size; the frozen codec must have the same resolution')
    parser.add_argument('--control-run-dir', type=Path)
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--spatial-entities', action='store_true',
        help='Nine explicit map-relative participant tokens and local forward/side motion conditioning')
    parser.add_argument('--spatial-combat', action='store_true',
        help='Per-NPC combat/spawn heads, nine Flow conditions and auxiliary image consequences')
    parser.add_argument('--training-improvements-v5', action='store_true',
        help='AR XY/wall losses, balanced attacks, anchor spawn, patch2/block-conditioned Flow and validation every100')
    parser.add_argument('--motion-curriculum', action='store_true',
        help='Rebalance existing source windows from short movement cases to longer own-prediction rollouts')
    parser.add_argument('--motion-max-horizon', type=int, choices=(1, 4, 8, 16, 29), default=29,
        help='Maximum differentiable source horizon; longer autonomous qualification is evaluated separately')
    parser.add_argument('--teacher-size', choices=('175m_vqgan', '500m_vqgan'), default='500m_vqgan')
    parser.add_argument('--detach-flow-base', action='store_true')
    parser.add_argument('--gan-npc-negatives', action='store_true')
    parser.add_argument('--npc-negative-weight', type=float, default=1.)
    parser.add_argument('--npc-negative-min-fraction', type=float, default=.005)
    parser.add_argument('--object-aware-rendering', action='store_true')
    parser.add_argument('--npc-region-boost', type=float, default=4.)
    parser.add_argument('--object-mask-directory', type=Path)
    parser.add_argument('--adversarial-during-pretrain', action='store_true',
        help='Train sampled RGB and GAN jointly with world/CFM, using checkpointed sampling')
    parser.add_argument('--pretrain-adversarial-weight', type=float, default=.005)
    parser.add_argument('--pretrain-adversarial-feature-weight', type=float, default=.01)
    parser.add_argument('--pretrain-adversarial-warmup', type=int, default=32)
    parser.add_argument('--pretrain-adversarial-ramp', type=int, default=128)
    parser.add_argument('--training-subset-manifest', type=Path,
        help='Immutable manifest of the training clips allowed in this segment')
    parser.add_argument('--optimizer-update-budget', type=int,
        help='Additional teacher optimizer updates, then wait for the user; resumes keep the original limit')
    parser.add_argument('--accept-current-codec', action='store_true',
        help='Use a frozen codec explicitly accepted by the user without claiming its quality gate passed')
    args = parser.parse_args(argv)
    if (not math.isfinite(args.npc_negative_weight) or args.npc_negative_weight <= 0
            or not math.isfinite(args.npc_negative_min_fraction)
            or not 0 <= args.npc_negative_min_fraction <= 1):
        parser.error('NPC-negative weight must be finite/positive and minimum fraction in [0,1]')
    if not math.isfinite(args.npc_region_boost) or args.npc_region_boost < 1:
        parser.error('NPC region boost must be finite and at least one')
    if (not all(math.isfinite(value) for value in (args.pretrain_adversarial_weight, args.pretrain_adversarial_feature_weight))
            or min(args.pretrain_adversarial_weight, args.pretrain_adversarial_feature_weight,
            args.pretrain_adversarial_warmup) < 0 or args.pretrain_adversarial_ramp < 1):
        parser.error('GAN weights/warmup must be nonnegative and ramp must be positive')
    if min(args.pretrain_steps, args.world_repair_steps, args.world_patience, args.student_steps,
           args.repair_steps, args.batch, args.cache_mb, args.chunk, args.context, args.horizon,
           args.gradient_accumulation_steps, args.preview_every) < 1:
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
