"""Expand the trained 2M restorer to 7M; continue until RF-Q VAL256 SSIM>.85."""
from copy import deepcopy
import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

from .conditional_decoder_restorer import (
    conditional_decoder_restorer_metadata, load_conditional_decoder_restorer, _state_sha256)
from .conditional_restorer_7m import expand_conditional_restorer_7m
from .conditional_restorer_goal import conditional_restorer_goal_spec
from .conditional_restorer_objectives import (
    PROTOCOL, configure_conditional_restorer, conditional_restorer_protocol)
from .distill_renderer import _digest
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_combat_data import CombatRendererFrames
from .renderer_small_enemy_data import SmallEnemyBalancedFrames
from .renderer_moe_five_objectives import FiveProfiledFrames
from .renderer_moe_five_phase import original_renderer_configuration
from .renderer_native_phase import _same, restore_renderer_native_rng
from .renderer_student import load_renderer_student, _hash_tensors
from .train_conditional_restorer import frozen_hash
from .train_conditional_restorer_2m import Restorer2MTraining, validate_saved_goal_state
from .training_pipeline import atomic_json


PARAMETERS = 7_000_313
TENSORS = 195


def adopt_restorer_7m_optimizer(expanded, parent, *, lr):
    """Keep both ordered 2M Adam groups and append only65 fresh parameters."""
    prefix = 'flow_renderer.conditional_restorer.'
    if type(lr) not in (float, int) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('7M expansion requires a finite positive new-group learning rate')
    previous = expanded.previous
    old_names = [prefix + name for name, _ in previous.named_parameters()]
    saved = parent.get('optimizer', {})
    step = parent.get('step')
    cumulative = parent.get('total_restorer_updates')
    historical = parent.get('phase', {}).get('parent_restorer_updates')
    if (type(step) is not int or step < 1 or type(historical) is not int or historical < 1
            or type(cumulative) is not int or cumulative != historical + step
            or parent.get('optimizer_parameter_names') != old_names or len(old_names) != 130
            or len(saved.get('param_groups', [])) != 2
            or set(saved.get('state', {})) != set(range(130))
            or saved['param_groups'][0]['params'] != list(range(65))
            or saved['param_groups'][1]['params'] != list(range(65, 130))):
        raise ValueError('7M expansion requires all130 ordered 2M Adam states and cumulative clocks')
    parameters = list(previous.parameters())
    for index, parameter in enumerate(parameters):
        state = saved['state'][index]
        age = state.get('step')
        expected_age = cumulative if index < 65 else step
        if (not torch.is_tensor(age) or age.numel() != 1 or not bool(torch.isfinite(age).all())
                or float(age) != expected_age or any(
                not torch.is_tensor(state.get(key)) or state[key].shape != parameter.shape
                or state[key].dtype != parameter.dtype or not bool(torch.isfinite(state[key]).all())
                for key in ('exp_avg', 'exp_avg_sq'))):
            raise ValueError('7M legacy Adam mixed age, moment shape/dtype or finite state changed')
    groups = [dict(**deepcopy({k: v for k, v in group.items() if k != 'params'}),
                   params=parameters[index*65:(index+1)*65])
              for index, group in enumerate(saved['param_groups'])]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    optimizer.load_state_dict(deepcopy(saved))
    fresh = deepcopy({k: v for k, v in saved['param_groups'][-1].items() if k != 'params'})
    fresh.update(params=list(expanded.expansion.parameters()), lr=float(lr))
    optimizer.add_param_group(fresh)
    result = optimizer.state_dict()
    if (not _same(result['state'], saved['state'])
            or not _same(result['param_groups'][:2], saved['param_groups'])
            or len(result['param_groups'][2]['params']) != 65
            or set(result['param_groups'][2]['params']) & set(result['state'])):
        raise AssertionError('7M optimizer changed inherited Adam or initialized fresh moments')
    return optimizer


class Restorer7MTraining(Restorer2MTraining):
    """Reuse the tested goal, validation, best selection and durable save paths."""
    def __init__(self, args):
        super().__init__(args)
        self.state.update(model='Expanded conditional RGB restorer ~7M', total=None)

    def prepare(self):
        a = self.args
        if a.device == 'cuda':
            torch.cuda.init()
        torch.set_num_threads(a.threads)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.publish(stage_label='Loading the learned 2M restorer and both Adam groups')
        parent = torch.load(a.parent_decoder, map_location='cpu', weights_only=False)
        boundary = torch.load(a.parent_training, map_location='cpu', weights_only=False)
        previous = torch.load(a.restorer_parent, map_location='cpu', weights_only=False)
        generator = torch.load(a.generator_full_checkpoint, map_location='cpu', weights_only=False)
        recipe = torch.load(a.source_recipe_parent, map_location='cpu', weights_only=False)
        inherited = original_renderer_configuration(recipe['configuration'])
        for key in ('world', 'parent_decoder', 'parent_training', 'generator_full_checkpoint', 'source_recipe_parent'):
            if previous['phase']['inputs'].get(key) != _digest(getattr(a, key)):
                raise ValueError('7M expansion changed a frozen parent input: ' + key)
        architecture = previous['phase']['restorer_architecture']
        if (previous['phase']['protocol']['name'] != PROTOCOL
                or architecture['kind'] != 'expanded_conditional_decoder_restorer_v1'
                or architecture['parameters'] != 1997798 or architecture['parameter_tensors'] != 130
                or type(previous.get('total_restorer_updates')) is not int
                or previous['total_restorer_updates'] != previous['phase']['parent_restorer_updates'] + previous['step']):
            raise ValueError('7M continuation requires its complete learned 2M parent and cumulative clock')
        self.parent_updates = previous['total_restorer_updates']
        self.parent_local_updates = previous['step']
        self.source, meta = load_world_checkpoint(a.world, 'cpu')
        if meta['training_steps'] != 5906 or parent['training_steps'] != 30717 or boundary['step'] != 1725:
            raise ValueError('7M expansion changed the frozen world/RF/decoder clocks')
        self.source.eval().requires_grad_(False)
        self.model = deepcopy(self.source)
        load_renderer_student(self.model, a.parent_decoder, allow_unqualified_codec=True)
        self.model.float().to(a.device).eval().requires_grad_(False)
        self.reference = deepcopy(self.model.flow_renderer.codec).eval().requires_grad_(False)
        # The generic strict loader dispatches the pinned nested2M architecture.
        legacy = load_conditional_decoder_restorer(architecture['config'], previous['restorer'],
            metadata={**architecture, 'state_sha256': _state_sha256(previous['restorer'])})
        legacy.to(a.device)
        expanded, expansion = expand_conditional_restorer_7m(legacy, seed=175)
        self.model.flow_renderer.conditional_restorer = expanded
        policy = configure_conditional_restorer(self.model)
        if (policy['trainable_parameters'], policy['trainable_parameter_tensors']) != (PARAMETERS, TENSORS):
            raise ValueError('Unexpected 7M restorer capacity')
        self.names = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        expected_names = ['flow_renderer.conditional_restorer.previous.' +
            name.split('flow_renderer.conditional_restorer.', 1)[1]
            for name in previous['optimizer_parameter_names']]
        expected_names += ['flow_renderer.conditional_restorer.expansion.' + name
                           for name, _ in expanded.expansion.named_parameters()]
        if self.names != expected_names:
            raise ValueError('7M parameter registration changed the canonical old/new group order')
        self.optimizer = adopt_restorer_7m_optimizer(expanded, previous, lr=a.lr)
        self.frozen = frozen_hash(self.model)
        self.reference_hash = _hash_tensors(self.reference.state_dict())
        if (self.frozen != previous['phase']['frozen_fingerprint']
                or self.reference_hash != previous['phase']['reference_codec_sha256']):
            raise AssertionError('7M expansion changed an immutable parent tensor')
        self.publish(stage_label='Restoring unchanged Source data, panels and sampler stream')
        raw = CombatRendererFrames(inherited['combat_data_root'], inherited['appearance_data_root'], 'train', seed=a.seed)
        validation = CombatRendererFrames(inherited['combat_data_root'], inherited['appearance_data_root'], 'val',
                                         seed=a.seed+1, exclude_scene_keys=raw.scene_keys)
        self.train = SmallEnemyBalancedFrames(raw)
        identity = generator['phase']['identity']['parent_identity']
        for key, value in (('train', raw.identity), ('val', validation.identity),
                           ('distance_sampling', self.train.identity['distance_sampling'])):
            if not _same(identity[key], value):
                raise ValueError('7M expansion changed the pinned Source corpus: ' + key)
        self.panel, self.geometry_panel = list(identity['panel']), list(identity['geometry_wall_panel'])
        if (validation.panel_fingerprint(self.panel) != identity['panel_fingerprint']
                or validation.panel_fingerprint(self.geometry_panel) != identity['geometry_wall_fingerprint']):
            raise ValueError('7M expansion changed held-out panels')
        self.data, self.validation_data = FiveProfiledFrames(self.train), FiveProfiledFrames(validation)
        self.preview_ids = []
        for category in ('large', 'partial', 'distant', 'empty_navigation'):
            index = next((i for i in self.panel if validation.rows[i]['category'] == category), None)
            if index is not None and index not in self.preview_ids:
                self.preview_ids.append(index)
        self.preview_ids += [i for i in self.panel if i not in self.preview_ids][:4-len(self.preview_ids)]
        configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
        self.phase = dict(protocol=conditional_restorer_protocol(), configuration=configuration,
            inputs={key: _digest(getattr(a, key)) for key in ('world', 'parent_decoder', 'parent_training',
                'generator_full_checkpoint', 'source_recipe_parent', 'restorer_parent')},
            restorer_architecture=conditional_decoder_restorer_metadata(expanded, include_state_hash=False),
            expansion=expansion, parent_restorer_updates=self.parent_updates,
            parent_restorer_local_updates=self.parent_local_updates,
            generator_training_steps=30717, generator_local_step=1108, base_training_steps=29209,
            decoder_parent_updates=1725, feature_world_step=5906,
            source_train_identity=raw.identity, source_val_identity=validation.identity,
            panel_fingerprint=identity['panel_fingerprint'], geometry_panel_fingerprint=identity['geometry_wall_fingerprint'],
            preview_panel_fingerprint=validation.panel_fingerprint(self.preview_ids),
            frozen_fingerprint=self.frozen, reference_codec_sha256=self.reference_hash,
            restorer_parameter_names=self.names, legacy_Adam_states_preserved=130,
            legacy_Adam_groups_preserved=2, fresh_expansion_Adam_states=0,
            goal=conditional_restorer_goal_spec(), hard_update_budget=None, automatic_active_promotion=False)
        phase_path = self.root/'phase_plan.json'
        if phase_path.exists():
            if json.loads(phase_path.read_text(encoding='utf-8')) != self.phase:
                raise ValueError('Immutable 7M phase, goal or input pins changed')
        else:
            atomic_json(phase_path, self.phase)
        self.plan_sha = _digest(phase_path)
        self.checkpoint, self.step = self.root/'checkpoints/restorer_last.pt', 0
        saved = previous
        if self.checkpoint.exists() and not a.probe:
            saved = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
            if (saved['phase'] != self.phase or saved['plan_sha256'] != self.plan_sha
                    or type(saved.get('step')) is not int or saved['step'] < 0
                    or saved['optimizer_parameter_names'] != self.names
                    or saved.get('total_restorer_updates') != self.parent_updates + saved['step']):
                raise ValueError('7M continuation changed phase, cumulative clocks or parameter names')
            expanded.load_state_dict(saved['restorer'], strict=True)
            self.optimizer.load_state_dict(deepcopy(saved['optimizer']))
            self.step = saved['step']
            self.best, self.goal_history, self.goal_met = validate_saved_goal_state(saved,
                panel_fingerprint=self.phase['panel_fingerprint'], plan_sha256=self.plan_sha,
                feature_world_sha256=self.phase['inputs']['world'], run_dir=self.root)
        self.train.restore_sampler(saved['sampler'])
        if not _same(self.train.sampler_state(), saved['sampler']):
            raise AssertionError('7M continuation failed exact Source sampler restoration')
        restore_renderer_native_rng(saved, device=a.device)
        self.training = dict(protocol=PROTOCOL, restorer_profile='expanded_7m_v1', expanded_restorer=True,
            plan_sha256=self.plan_sha, feature_world_sha256=self.phase['inputs']['world'], feature_world_step=5906,
            parent_decoder_sha256=self.phase['inputs']['parent_decoder'],
            parent_restorer_checkpoint_sha256=self.phase['inputs']['restorer_parent'],
            parent_restorer_updates=self.parent_updates, parent_restorer_local_updates=self.parent_local_updates,
            total_restorer_updates=self.parent_updates+self.step,
            frozen_RF_training_steps=30717, frozen_RF_local_step=1108, base_training_steps=29209,
            frozen_decoder_updates=1725, solver_steps=3, restorer_parameters=PARAMETERS,
            restorer_parameter_tensors=TENSORS, rf_weight=.5, source_weight=.5,
            source_queries_per_update=8, decoder_examples_per_update=16,
            local_step=self.step, local_budget=None, encoder_frozen=True, codebook_frozen=True,
            RF_frozen=True, Base_frozen=True, decoder_frozen=True, post_quant_conv_trainable=False, restorer_only=True,
            inputs=['RF3_quantized_codes', 'decoded_RGB', 'current_state20', 'raw_map16x16'],
            hints_are_predicted_in_game=True, Source_RGB_and_masks_are_labels_only=True,
            preview_panel_fingerprint=self.phase['preview_panel_fingerprint'])
        self.publish(conditional_restorer_training=self.training, current=self.step, global_training_step=self.step,
            renderer_training_steps=30717, base_training_steps=29209,
            validation_panel_fingerprint=self.phase['panel_fingerprint'], ssim_goal=self.goal_status(),
            renderer_training=dict(active=True, rf_active=False, base_active=False, decoder_active=False,
                restorer_active=True, encoder_frozen=True, codebook_frozen=True, gan_active=False))

    def run(self):
        self.prepare()
        if self.args.probe:
            before = deepcopy(self.model.flow_renderer.conditional_restorer.state_dict())
            rows = [self.update() for _ in range(3)]
            self.save(); self.export()
            after = self.model.flow_renderer.conditional_restorer.state_dict()
            proof = dict(passed=any(not torch.equal(before[key], after[key]) for key in before),
                discarded_optimizer_updates=3, production_training_updates=0, rows=rows,
                frozen_assets_exact=True, trainable_parameters=PARAMETERS, trainable_tensors=TENSORS,
                legacy_Adam_states_retained=130, legacy_Adam_groups_retained=2,
                parent_restorer_updates=self.parent_updates,
                seconds_per_update=sum(row['optimizer_update_seconds'] for row in rows)/3,
                peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30 if self.args.device == 'cuda' else 0.)
            atomic_json(self.root/'gpu_warmstart.json', proof)
            self.publish(status='completed', message='Three discarded7M restorer updates passed')
            print(json.dumps(proof), flush=True)
            return
        initial = self.root/'validation_initial.json'
        baseline = json.loads(initial.read_text(encoding='utf-8')) if initial.exists() else None
        if self.step and baseline is None:
            raise ValueError('7M resume requires its initial complete held-out baseline')
        if baseline is None:
            self.publish(stage_label='Measuring preserved 2M output before7M training')
            baseline = self.validate(None)
            atomic_json(initial, baseline)
        self.save()
        self.publish(status='running', stage_label='~7M restorer: train until RF-Q held-out SSIM > 0.85')
        while not self.goal_met:
            if (self.root/'stop_training').exists():
                self.save(); self.publish(status='stopped')
                return
            row = self.update()
            with (self.root/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            self.training.update(local_step=self.step, total_restorer_updates=self.parent_updates+self.step)
            self.save()
            self.publish(current=self.step, global_training_step=self.step, loss=row['train_loss'],
                         latest_metrics=row, conditional_restorer_training=self.training)
            if self.step % 100 == 0:
                self.validate(baseline)
            elif self.step % 10 == 0:
                self.preview()
        self.save()
        self.publish(status='completed', message='Held-out RF-Q + restorer SSIM is strictly above0.85',
                     ssim_goal=self.goal_status())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'world', 'parent-decoder', 'parent-training', 'generator-full-checkpoint',
                 'source-recipe-parent', 'restorer-parent'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--probe', action='store_true')
    a = parser.parse_args(argv)
    a.steps = None
    if a.batch != 8 or not math.isfinite(a.lr) or a.lr <= 0:
        parser.error('7M paired training requires batch8 and a finite positive learning rate')
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve())
    from .vqgan_pipeline import process_alive
    a.run_dir.mkdir(parents=True, exist_ok=True)
    pid = a.run_dir/'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())):
        parser.error('A7M restorer worker is already active')
    pid.write_text(str(os.getpid()), encoding='ascii')
    trainer = Restorer7MTraining(a)
    try:
        with KeepAwake():
            trainer.run()
    except BaseException as error:
        trainer.publish(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
