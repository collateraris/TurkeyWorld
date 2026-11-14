"""Paired decoder-only adaptation on frozen RF3 codes and Source E/Q codes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import os
import random
import time

import numpy as np
import torch

from .decoder_rf_adaptation_objectives import (
    PROTOCOL, TRAINABLE_PREFIXES, configure_decoder_adaptation,
    decoder_rf_adaptation_protocol, decoder_rf_adaptation_loss)
from .decoder_rf_adaptation_validation import (
    evaluate_decoder_adaptation, save_decoder_adaptation_preview)
from .decoder_rf_adaptation_bundle import export_decoder_adaptation
from .distill_renderer import _digest
from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_combat_data import CombatRendererFrames
from .renderer_small_enemy_data import SmallEnemyBalancedFrames
from .renderer_moe_five_objectives import FiveProfiledFrames
from .renderer_moe_five_phase import original_renderer_configuration
from .renderer_native_phase import _same, restore_renderer_native_rng
from .renderer_student import load_renderer_student, _hash_tensors
from .train_renderer_rf5 import durable_save
from .training_pipeline import atomic_json, now


def decoder_state(model):
    return {name: value for name, value in model.flow_renderer.codec.state_dict().items()
            if name.startswith(('decoder.', 'post_quant_conv.'))}


def load_decoder_state(model, state):
    if (not isinstance(state, dict) or set(state) != set(decoder_state(model))
            or any(not torch.is_tensor(value) or not bool(torch.isfinite(value).all())
                   for value in state.values())):
        raise ValueError('Decoder checkpoint must contain exactly the finite decoder/post_quant_conv state')
    for prefix, module in (('decoder.', model.flow_renderer.codec.decoder),
                           ('post_quant_conv.', model.flow_renderer.codec.post_quant_conv)):
        module.load_state_dict({name[len(prefix):]: value for name, value in state.items()
                                if name.startswith(prefix)}, strict=True)


def frozen_hash(model):
    return _hash_tensors({name: value for name, value in model.state_dict().items()
                          if not name.startswith(TRAINABLE_PREFIXES)})


class DecoderTraining:
    def __init__(self, args):
        self.args, self.root = args, args.run_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.state = dict(run_id=self.root.name, status='initializing', stage='decoder_rf_adaptation',
            loss_protocol=PROTOCOL, current=0, global_training_step=0, total=args.steps,
            prediction_every=10, validation_every=100, image_size=32,
            model='Decoder + post_quant_conv; paired frozen RF3 / Source E/Q codes',
            gameplay_qualified=False)

    def publish(self, **values):
        self.state.update(values, updated_at=now(), elapsed_seconds=time.perf_counter() - self.started)
        atomic_json(self.root / 'status.json', self.state)

    def prepare(self):
        a = self.args
        if a.device == 'cuda':
            torch.cuda.init()
        torch.set_num_threads(a.threads)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.publish(stage_label='Loading the immutable frozen RF3 generator')
        self.generator = torch.load(a.generator_checkpoint, map_location='cpu', weights_only=False)
        self.generator_full = torch.load(a.generator_full_checkpoint, map_location='cpu', weights_only=False)
        recipe = torch.load(a.source_recipe_parent, map_location='cpu', weights_only=False)
        inherited = original_renderer_configuration(recipe['configuration'])
        self.source, meta = load_world_checkpoint(a.world, 'cpu')
        if meta['training_steps'] != 5906:
            raise ValueError('Decoder adaptation requires its unchanged feature world5906')
        self.source.eval().requires_grad_(False)
        self.model = deepcopy(self.source)
        load_renderer_student(self.model, a.generator_checkpoint, allow_unqualified_codec=True)
        if (self.model.flow_renderer.steps != 3
                or self.generator['training_steps'] != self.generator_full['renderer_training_steps']
                or not _same(self.model.flow_renderer.state_dict(), self.generator_full['flow'])
                or not _same(self.model.base_student.state_dict(), self.generator_full['base'])):
            raise ValueError('Decoder phase generator differs from the complete stopped RF boundary')
        self.model.float().to(a.device).eval().requires_grad_(False)
        self.reference = deepcopy(self.model.flow_renderer.codec).eval().requires_grad_(False)
        policy = configure_decoder_adaptation(self.model)
        if (policy['trainable_parameters'], policy['trainable_parameter_tensors']) != (528403, 62):
            raise ValueError('Unexpected decoder-only trainable parameter profile')
        self.names = [name for name, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                           lr=a.lr, weight_decay=.01)
        self.frozen = frozen_hash(self.model)
        self.reference_hash = _hash_tensors(self.reference.state_dict())
        self.publish(stage_label='Indexing unchanged Source TRAIN/VAL and native NPC strata')
        raw = CombatRendererFrames(inherited['combat_data_root'], inherited['appearance_data_root'],
                                  'train', seed=a.seed)
        validation = CombatRendererFrames(inherited['combat_data_root'], inherited['appearance_data_root'],
            'val', seed=a.seed + 1, exclude_scene_keys=raw.scene_keys)
        self.train = SmallEnemyBalancedFrames(raw)
        identity = self.generator_full['phase']['identity']['parent_identity']
        for key, value in (('train', raw.identity), ('val', validation.identity),
                           ('distance_sampling', self.train.identity['distance_sampling'])):
            if not _same(identity[key], value):
                raise ValueError('Decoder adaptation changed the pinned Source corpus: ' + key)
        self.panel, self.geometry_panel = list(identity['panel']), list(identity['geometry_wall_panel'])
        if (validation.panel_fingerprint(self.panel) != identity['panel_fingerprint']
                or validation.panel_fingerprint(self.geometry_panel) != identity['geometry_wall_fingerprint']):
            raise ValueError('Decoder adaptation changed its held-out Source panels')
        self.data, self.validation_data = FiveProfiledFrames(self.train), FiveProfiledFrames(validation)
        self.preview_ids = []
        for category in ('large', 'partial', 'distant', 'empty_navigation'):
            index = next((i for i in self.panel if validation.rows[i]['category'] == category), None)
            if index is not None and index not in self.preview_ids:
                self.preview_ids.append(index)
        self.preview_ids += [i for i in self.panel if i not in self.preview_ids][:4-len(self.preview_ids)]
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        self.phase = dict(protocol=decoder_rf_adaptation_protocol(), configuration=config,
            inputs={key: _digest(getattr(a, key)) for key in (
                'world', 'generator_checkpoint', 'generator_full_checkpoint', 'source_recipe_parent')},
            generator_training_steps=self.generator['training_steps'], generator_local_step=self.generator['local_updates'],
            base_training_steps=self.generator['base_training_steps'], feature_world_step=5906,
            source_train_identity=raw.identity, source_val_identity=validation.identity,
            panel_fingerprint=identity['panel_fingerprint'], geometry_panel_fingerprint=identity['geometry_wall_fingerprint'],
            preview_panel_fingerprint=validation.panel_fingerprint(self.preview_ids),
            frozen_fingerprint=self.frozen, reference_codec_sha256=self.reference_hash,
            decoder_parameter_names=self.names, fresh_decoder_Adam=True,
            parent_RF_optimizer_untouched=True, initial_decoder_Adam_states=0)
        phase_path = self.root / 'phase_plan.json'
        if phase_path.exists():
            if json.loads(phase_path.read_text(encoding='utf-8')) != self.phase:
                raise ValueError('Immutable decoder phase configuration or input pins changed')
        else:
            atomic_json(phase_path, self.phase)
        self.plan_sha = _digest(phase_path)
        self.checkpoint = self.root / 'checkpoints/decoder_last.pt'
        self.step, saved = 0, self.generator_full
        if self.checkpoint.exists() and not a.probe:
            saved = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
            if (saved['phase'] != self.phase or saved['plan_sha256'] != self.plan_sha
                    or type(saved.get('step')) is not int or not 0 <= saved['step'] <= a.steps
                    or saved['optimizer_parameter_names'] != self.names):
                raise ValueError('Decoder continuation changed its parent, phase or named optimizer')
            load_decoder_state(self.model, saved['decoder'])
            self.optimizer.load_state_dict(saved['optimizer'])
            self.step = saved['step']
        self.train.restore_sampler(saved['sampler'])
        if not _same(self.train.sampler_state(), saved['sampler']):
            raise AssertionError('Decoder phase Source sampler restoration was not exact')
        restore_renderer_native_rng(saved, device=a.device)
        self.training = dict(protocol=PROTOCOL, plan_sha256=self.plan_sha,
            feature_world_sha256=self.phase['inputs']['world'], feature_world_step=5906,
            frozen_RF_training_steps=self.generator['training_steps'],
            frozen_RF_local_step=self.generator['local_updates'], base_training_steps=29209,
            solver_steps=3, decoder_parameters=528403, decoder_parameter_tensors=62,
            rf_weight=.5, source_weight=.5, source_queries_per_update=a.batch,
            decoder_examples_per_update=2*a.batch, local_step=self.step, local_budget=a.steps,
            encoder_frozen=True, codebook_frozen=True, RF_frozen=True, Base_frozen=True,
            post_quant_conv_trainable=True, preview_panel_fingerprint=self.phase['preview_panel_fingerprint'])
        self.publish(decoder_adaptation_training=self.training, current=self.step, global_training_step=self.step,
            renderer_training_steps=self.generator['training_steps'], base_training_steps=29209,
            validation_panel_fingerprint=self.phase['panel_fingerprint'],
            renderer_training=dict(active=True, rf_active=False, base_active=False,
                decoder_active=True, encoder_frozen=True, codebook_frozen=True, gan_active=False))

    def save(self):
        durable_save(self.checkpoint, dict(decoder=decoder_state(self.model), optimizer=self.optimizer.state_dict(),
            optimizer_parameter_names=self.names, step=self.step, phase=self.phase, plan_sha256=self.plan_sha,
            sampler=self.train.sampler_state(), rng_torch=torch.get_rng_state(), rng_numpy=np.random.get_state(),
            rng_python=random.getstate(), rng_cuda=torch.cuda.get_rng_state_all() if self.args.device=='cuda' else []))

    def update(self):
        self.optimizer.zero_grad(set_to_none=True)
        batch = self.data.batch(self.args.batch)
        started = time.perf_counter()
        value, metrics = decoder_rf_adaptation_loss(self.model, batch)
        value.backward()
        outside = [name for name, p in self.model.named_parameters()
                   if p.grad is not None and not name.startswith(TRAINABLE_PREFIXES)]
        if outside:
            raise AssertionError('A frozen RF/E/Q/Base/world parameter received a gradient')
        norm = float(clip_finite_grad_norm_(self.model.named_parameters(), context='Decoder adaptation'))
        self.optimizer.step(); self.step += 1
        # The full frozen world is measured on initial/first/five/validation boundaries.
        if self.step <= 5 or self.step % 100 == 0:
            if frozen_hash(self.model) != self.frozen or _hash_tensors(self.reference.state_dict()) != self.reference_hash:
                raise AssertionError('Decoder training changed a frozen generator or the reference codec')
        return dict(metrics, step=self.step, loss_protocol=PROTOCOL, plan_sha256=self.plan_sha,
            feature_world_sha256=self.phase['inputs']['world'], gradient_norm=norm,
            optimizer_update_seconds=time.perf_counter()-started, timestamp=now(),
            decoder_refinement_updates=self.step)

    def preview(self, arrays=None):
        if arrays is None:
            _, arrays = evaluate_decoder_adaptation(self.model, self.reference, self.validation_data,
                self.preview_ids, step=self.step, batch_size=4)
        meta = save_decoder_adaptation_preview(arrays, self.root / 'preview.png', self.step)
        meta.update(kind='decoder_rf_adaptation_GT_query', every=10,
            loss_protocol=PROTOCOL, plan_sha256=self.plan_sha,
            feature_world_sha256=self.phase['inputs']['world'],
            validation_panel_fingerprint=self.phase['panel_fingerprint'],
            solver_steps=3, renderer_only=True, world_rollout=False,
            future_source_rgb_in_prior=False, future_source_pose_in_prior=False)
        # Full validation previews and lightweight previews have different frame
        # sets; retain each actual hash and separately bind the fixed VAL panel.
        atomic_json(self.root / 'preview.json', meta)
        self.publish(prediction_preview=meta, preview_url='/preview.png')

    def validate(self, baseline):
        report, arrays = evaluate_decoder_adaptation(self.model, self.reference, self.validation_data,
            self.panel, step=self.step, batch_size=8, geometry_panel=self.geometry_panel, baseline=baseline)
        report.update(plan_sha256=self.plan_sha, feature_world_sha256=self.phase['inputs']['world'])
        atomic_json(self.root / 'validation' / f'step_{self.step:06d}.json', report)
        atomic_json(self.root / 'validation.json', report)
        row = dict(step=self.step, loss_protocol=PROTOCOL, protocol=PROTOCOL,
            plan_sha256=self.plan_sha, feature_world_sha256=self.phase['inputs']['world'], solver_steps=3,
            panel_fingerprint=report['panel_fingerprint'], val_loss=report['val_loss'],
            validation_objective=report['validation_objective'], source_held_out=True,
            renderer_only=True, world_rollout=False)
        with (self.root / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        export_decoder_adaptation(self.model, self.args.generator_checkpoint,
            self.root / 'checkpoints/renderer_for_play.pt', decoder_updates=self.step,
            source_model=self.source, metrics=dict(validation=report),
            provenance=dict(plan_sha256=self.plan_sha))
        self.publish(validation=report, decoder_adaptation_validation=report)
        # Always display the same four declared examples. Their own fingerprint
        # stays distinct from the full256 objective panel fingerprint.
        self.preview()
        return report

    def run(self):
        self.prepare()
        if self.args.probe:
            rows = [self.update() for _ in range(3)]
            self.save()
            proof = dict(passed=True, discarded_optimizer_updates=3, production_training_updates=0,
                rows=rows, frozen_assets_exact=True, trainable_parameters=528403,
                trainable_tensors=62, seconds_per_update=sum(r['optimizer_update_seconds'] for r in rows)/3,
                peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30 if self.args.device=='cuda' else 0.)
            atomic_json(self.root / 'gpu_warmstart.json', proof)
            self.publish(status='completed', message='Three discarded decoder-only GPU updates passed')
            print(json.dumps(dict(probe_passed=True, report=str(self.root/'gpu_warmstart.json'),
                seconds=proof['seconds_per_update'], peak_GiB=proof['peak_cuda_GiB'])), flush=True)
            return
        baseline_path = self.root / 'validation_initial.json'
        baseline = json.loads(baseline_path.read_text(encoding='utf-8')) if baseline_path.exists() else None
        if self.step > 0 and baseline is None:
            raise ValueError('Decoder resume cannot fabricate an initial decoder baseline')
        if baseline is None:
            self.publish(stage_label='Measuring paired old/new decoder before the first optimizer update')
            baseline = self.validate(None)
            atomic_json(baseline_path, baseline)
        self.save()
        self.publish(status='running', stage_label='Decoder only: exact paired RF3/Source-code mixture 50/50')
        while self.step < self.args.steps:
            if (self.root / 'stop_training').exists():
                self.save(); self.publish(status='stopped'); return
            row = self.update()
            with (self.root / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            self.training['local_step'] = self.step
            self.save()
            self.publish(current=self.step, global_training_step=self.step, loss=row['train_loss'],
                latest_metrics=row, decoder_adaptation_training=self.training)
            if self.step % 100 == 0:
                self.validate(baseline)
            elif self.step % 10 == 0:
                self.preview()
        self.save(); self.publish(status='completed', message='Decoder adaptation budget completed; quality is separate')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','world','generator-checkpoint','generator-full-checkpoint','source-recipe-parent'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=173)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--probe',action='store_true')
    a=p.parse_args(argv)
    if a.steps<1 or a.batch!=8 or a.lr<=0:
        p.error('Decoder phase requires positive steps/LR and eight paired Source queries')
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    from .vqgan_pipeline import process_alive
    a.run_dir.mkdir(parents=True,exist_ok=True)
    pid=a.run_dir/'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())):
        p.error('Decoder adaptation worker is already active')
    pid.write_text(str(os.getpid()),encoding='ascii')
    experiment=DecoderTraining(a)
    try:
        with KeepAwake():experiment.run()
    except BaseException as error:
        experiment.publish(status='failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':main()
