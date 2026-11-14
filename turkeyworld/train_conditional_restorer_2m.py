"""Continue a learned restorer at ~2M parameters until held-out RF-Q SSIM>.85."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import os
import random

import numpy as np
import torch

from .conditional_decoder_restorer import (
    ConditionalDecoderRestorer, conditional_decoder_restorer_metadata)
from .conditional_restorer_expansion import expand_conditional_restorer
from .conditional_restorer_goal import (
    PROTOCOL as GOAL_PROTOCOL, conditional_restorer_goal_spec,
    evaluate_restorer_goal, update_restorer_best)
from .conditional_restorer_objectives import PROTOCOL, configure_conditional_restorer, conditional_restorer_protocol
from .conditional_restorer_bundle import export_conditional_restorer
from .distill_renderer import _digest
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_combat_data import CombatRendererFrames
from .renderer_small_enemy_data import SmallEnemyBalancedFrames
from .renderer_moe_five_objectives import FiveProfiledFrames
from .renderer_moe_five_phase import original_renderer_configuration
from .renderer_native_phase import _same, restore_renderer_native_rng
from .renderer_student import load_renderer_student, _hash_tensors
from .train_conditional_restorer import ConditionalRestorerTraining, frozen_hash
from .train_renderer_rf5 import durable_save
from .training_pipeline import atomic_json, now


def adopt_restorer_optimizer(expanded, parent, *, lr):
    """Rename the old65 parameters without resetting any Adam moment or option."""
    prefix = 'flow_renderer.conditional_restorer.'
    expected = [prefix+name for name, _ in expanded.legacy.named_parameters()]
    saved = parent.get('optimizer', {})
    if (parent.get('optimizer_parameter_names') != expected or len(expected) != 65
            or len(saved.get('param_groups', [])) != 1 or len(saved.get('state', {})) != 65
            or type(parent.get('step')) is not int or parent['step'] < 1):
        raise ValueError('Expansion requires every named legacy Adam state from a completed restorer')
    ids = saved['param_groups'][0]['params']
    if ids != list(range(65)) or set(saved['state']) != set(ids):
        raise ValueError('Legacy Adam must retain exact ordered canonical IDs0..64')
    for index, parameter in zip(ids, expanded.legacy.parameters()):
        state = saved['state'][index]
        if (float(state['step']) != parent['step'] or any(
                state[key].shape != parameter.shape or state[key].dtype != parameter.dtype
                or not bool(torch.isfinite(state[key]).all()) for key in ('exp_avg','exp_avg_sq'))):
            raise ValueError('Legacy restorer Adam age, shape, dtype or moments changed')
    optimizer = torch.optim.AdamW(expanded.legacy.parameters(), lr=lr, weight_decay=.01)
    optimizer.load_state_dict(deepcopy(saved))
    if not _same(optimizer.state_dict(), saved):
        raise AssertionError('Expansion changed an existing Adam state or group option')
    optimizer.add_param_group(dict(params=list(expanded.expansion.parameters()), lr=lr, weight_decay=.01))
    after = optimizer.state_dict()
    if (not _same(after['state'], saved['state']) or not _same(after['param_groups'][0],saved['param_groups'][0])
            or len(after['param_groups'][1]['params']) != 65):
        raise AssertionError('Fresh expansion group did not retain the complete old Adam')
    return optimizer


def validate_saved_goal_state(saved, *, panel_fingerprint, plan_sha256, feature_world_sha256, run_dir=None):
    """A resumed worker cannot turn an unmeasured or stale boolean into success."""
    step, history, best, met = (saved.get(key) for key in ('step','goal_history','best','goal_met'))
    if type(step) is not int or step<0 or type(met) is not bool or not isinstance(history,list):
        raise ValueError('Saved goal needs an exact boolean and a measured validation history')
    pins = dict(panel_fingerprint=panel_fingerprint,plan_sha256=plan_sha256,
                feature_world_sha256=feature_world_sha256)
    previous_step, expected_best = -1,None
    for item in history:
        if (not isinstance(item,dict) or type(item.get('step')) is not int
                or not previous_step<item['step']<=step or item['step']%100!=0
                or type(item.get('ssim')) not in (int,float) or not math.isfinite(item['ssim'])
                or not -1<=item['ssim']<=1 or type(item.get('goal_met')) is not bool
                or item['goal_met'] != (item['ssim']>.85)):
            raise ValueError('Saved goal history has invalid measurements or clocks')
        previous_step = item['step']
        if run_dir is not None:
            path = Path(run_dir)/'validation'/f"step_{item['step']:06d}.json"
            if not path.is_file() or item.get('report_sha256') != _digest(path):
                raise ValueError('Saved goal must retain each exact complete validation report')
            report = json.loads(path.read_text(encoding='utf-8'))
            decision = evaluate_restorer_goal(report,**pins)
            if decision['measured_ssim'] != item['ssim'] or decision['goal_met'] != item['goal_met']:
                raise ValueError('Saved goal disagrees with its complete RF-Q measurement')
            expected_best = update_restorer_best(expected_best,report,report_sha256=_digest(path),**pins)
    if history:
        if history[0]['step']!=0 or any(item['goal_met'] for item in history[:-1]) or met!=history[-1]['goal_met']:
            raise ValueError('Saved completion does not follow the first successful full validation')
        maximum = max(item['ssim'] for item in history)
        earliest = next(item['step'] for item in history if item['ssim']==maximum)
        if (not isinstance(best,dict) or best.get('ssim')!=maximum or best.get('step')!=earliest
                or best.get('validation_step')!=earliest or type(best.get('goal_met')) is not bool
                or best['goal_met']!=(maximum>.85)
                or any(best.get(key)!=value for key,value in pins.items())):
            raise ValueError('Saved best must correspond to the earliest maximum measured RF-Q SSIM')
        if expected_best is not None and any(best.get(key)!=value for key,value in expected_best.items() if key!='improved'):
            raise ValueError('Saved best differs from its exact held-out validation record')
    elif met or best is not None or step!=0:
        raise ValueError('A goal without any validation cannot be completed or resumed beyond update0')
    return deepcopy(best),deepcopy(history),met


class Restorer2MTraining(ConditionalRestorerTraining):
    def __init__(self, args):
        super().__init__(args)
        self.state.update(total=None, stop_on_ssim=True, model='Expanded conditional RGB restorer ~2M')
        self.best, self.goal_history, self.goal_met = None, [], False

    def prepare(self):
        a = self.args
        if a.device == 'cuda':
            torch.cuda.init()
        torch.set_num_threads(a.threads)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.publish(stage_label='Loading learned restorer and preserving its Adam state')
        parent = torch.load(a.parent_decoder,map_location='cpu',weights_only=False)
        boundary = torch.load(a.parent_training,map_location='cpu',weights_only=False)
        previous = torch.load(a.restorer_parent,map_location='cpu',weights_only=False)
        generator = torch.load(a.generator_full_checkpoint,map_location='cpu',weights_only=False)
        recipe = torch.load(a.source_recipe_parent,map_location='cpu',weights_only=False)
        inherited = original_renderer_configuration(recipe['configuration'])
        for key in ('world','parent_decoder','parent_training','generator_full_checkpoint','source_recipe_parent'):
            if previous['phase']['inputs'].get(key) != _digest(getattr(a,key)):
                raise ValueError('Expansion changed a frozen parent input: '+key)
        if (previous['phase']['protocol']['name'] != PROTOCOL
                or previous['phase']['restorer_architecture']['parameters'] != 247539):
            raise ValueError('Expansion requires the learned native32 conditional restorer lineage')
        self.parent_updates = previous['step']
        self.source, meta = load_world_checkpoint(a.world,'cpu')
        if meta['training_steps'] != 5906 or parent['training_steps'] != 30717 or boundary['step'] != 1725:
            raise ValueError('Expansion changed the frozen world/RF/decoder clocks')
        self.source.eval().requires_grad_(False)
        self.model = deepcopy(self.source)
        load_renderer_student(self.model,a.parent_decoder,allow_unqualified_codec=True)
        self.model.float().to(a.device).eval().requires_grad_(False)
        self.reference = deepcopy(self.model.flow_renderer.codec).eval().requires_grad_(False)
        legacy = ConditionalDecoderRestorer(seed=a.seed).to(a.device)
        legacy.load_state_dict(previous['restorer'],strict=True)
        if not all(bool(torch.isfinite(v).all()) for v in legacy.state_dict().values()):
            raise ValueError('Parent restorer contains nonfinite weights')
        expanded, expansion = expand_conditional_restorer(legacy,seed=174)
        self.model.flow_renderer.conditional_restorer = expanded
        policy = configure_conditional_restorer(self.model)
        if (policy['trainable_parameters'],policy['trainable_parameter_tensors']) != (1997798,130):
            raise ValueError('Unexpected expanded restorer profile')
        self.names = [name for name,p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = adopt_restorer_optimizer(expanded,previous,lr=a.lr)
        self.frozen,self.reference_hash = frozen_hash(self.model),_hash_tensors(self.reference.state_dict())
        if self.frozen != previous['phase']['frozen_fingerprint']:
            raise AssertionError('Expansion changed an immutable model parameter or buffer')
        self.publish(stage_label='Restoring the same Source corpus, held-out panels and sampling clock')
        raw = CombatRendererFrames(inherited['combat_data_root'],inherited['appearance_data_root'],'train',seed=a.seed)
        val = CombatRendererFrames(inherited['combat_data_root'],inherited['appearance_data_root'],'val',
                                  seed=a.seed+1,exclude_scene_keys=raw.scene_keys)
        self.train = SmallEnemyBalancedFrames(raw)
        identity = generator['phase']['identity']['parent_identity']
        for key,value in (('train',raw.identity),('val',val.identity),
                          ('distance_sampling',self.train.identity['distance_sampling'])):
            if not _same(identity[key],value):
                raise ValueError('Expansion changed the pinned Source corpus: '+key)
        self.panel,self.geometry_panel = list(identity['panel']),list(identity['geometry_wall_panel'])
        if (val.panel_fingerprint(self.panel) != identity['panel_fingerprint']
                or val.panel_fingerprint(self.geometry_panel) != identity['geometry_wall_fingerprint']):
            raise ValueError('Expansion changed held-out panels')
        self.data,self.validation_data = FiveProfiledFrames(self.train),FiveProfiledFrames(val)
        self.preview_ids = []
        for category in ('large','partial','distant','empty_navigation'):
            index = next((i for i in self.panel if val.rows[i]['category']==category),None)
            if index is not None and index not in self.preview_ids:
                self.preview_ids.append(index)
        self.preview_ids += [i for i in self.panel if i not in self.preview_ids][:4-len(self.preview_ids)]
        config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        self.phase = dict(protocol=conditional_restorer_protocol(),configuration=config,
            inputs={key:_digest(getattr(a,key)) for key in ('world','parent_decoder','parent_training',
                'generator_full_checkpoint','source_recipe_parent','restorer_parent')},
            restorer_architecture=conditional_decoder_restorer_metadata(expanded,include_state_hash=False),
            expansion=expansion,parent_restorer_updates=self.parent_updates,
            generator_training_steps=30717,generator_local_step=1108,base_training_steps=29209,
            decoder_parent_updates=1725,feature_world_step=5906,
            source_train_identity=raw.identity,source_val_identity=val.identity,
            panel_fingerprint=identity['panel_fingerprint'],geometry_panel_fingerprint=identity['geometry_wall_fingerprint'],
            preview_panel_fingerprint=val.panel_fingerprint(self.preview_ids),
            frozen_fingerprint=self.frozen,reference_codec_sha256=self.reference_hash,
            restorer_parameter_names=self.names,legacy_Adam_states_preserved=65,fresh_expansion_Adam_states=0,
            goal=conditional_restorer_goal_spec(),hard_update_budget=None,automatic_active_promotion=False)
        path = self.root/'phase_plan.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != self.phase:
                raise ValueError('Immutable expanded phase, goal or input pins changed')
        else:
            atomic_json(path,self.phase)
        self.plan_sha = _digest(path)
        self.checkpoint,self.step = self.root/'checkpoints/restorer_last.pt',0
        saved = previous
        if self.checkpoint.exists() and not a.probe:
            saved = torch.load(self.checkpoint,map_location='cpu',weights_only=False)
            if (saved['phase'] != self.phase or saved['plan_sha256'] != self.plan_sha
                    or type(saved.get('step')) is not int or saved['step'] < 0
                    or saved['optimizer_parameter_names'] != self.names):
                raise ValueError('Expanded continuation changed its phase or optimizer names')
            expanded.load_state_dict(saved['restorer'],strict=True)
            self.optimizer.load_state_dict(saved['optimizer'])
            self.step = saved['step']
            self.best,self.goal_history,self.goal_met = validate_saved_goal_state(saved,
                panel_fingerprint=self.phase['panel_fingerprint'],plan_sha256=self.plan_sha,
                feature_world_sha256=self.phase['inputs']['world'],run_dir=self.root)
        self.train.restore_sampler(saved['sampler'])
        if not _same(self.train.sampler_state(),saved['sampler']):
            raise AssertionError('Expanded phase failed exact Source sampler restoration')
        restore_renderer_native_rng(saved,device=a.device)
        self.training = dict(protocol=PROTOCOL,restorer_profile='expanded_2m_v1',expanded_restorer=True,
            plan_sha256=self.plan_sha,feature_world_sha256=self.phase['inputs']['world'],feature_world_step=5906,
            parent_decoder_sha256=self.phase['inputs']['parent_decoder'],
            parent_restorer_checkpoint_sha256=self.phase['inputs']['restorer_parent'],
            parent_restorer_updates=self.parent_updates,total_restorer_updates=self.parent_updates+self.step,
            frozen_RF_training_steps=30717,frozen_RF_local_step=1108,base_training_steps=29209,
            frozen_decoder_updates=1725,solver_steps=3,restorer_parameters=1997798,restorer_parameter_tensors=130,
            rf_weight=.5,source_weight=.5,source_queries_per_update=8,decoder_examples_per_update=16,
            local_step=self.step,local_budget=None,encoder_frozen=True,codebook_frozen=True,
            RF_frozen=True,Base_frozen=True,decoder_frozen=True,post_quant_conv_trainable=False,restorer_only=True,
            inputs=['RF3_quantized_codes','decoded_RGB','current_state20','raw_map16x16'],
            hints_are_predicted_in_game=True,Source_RGB_and_masks_are_labels_only=True,
            preview_panel_fingerprint=self.phase['preview_panel_fingerprint'])
        self.publish(conditional_restorer_training=self.training,current=self.step,global_training_step=self.step,
            renderer_training_steps=30717,base_training_steps=29209,
            validation_panel_fingerprint=self.phase['panel_fingerprint'],ssim_goal=self.goal_status(),
            renderer_training=dict(active=True,rf_active=False,base_active=False,decoder_active=False,
                restorer_active=True,encoder_frozen=True,codebook_frozen=True,gan_active=False))

    def goal_status(self):
        latest = self.goal_history[-1] if self.goal_history else {}
        return dict(protocol=GOAL_PROTOCOL,target=.85,comparison='strict_greater_than',
            metric='models.rf_new.summary.ssim_mean',frames=256,panel_fingerprint=self.phase['panel_fingerprint'],
            current_ssim=latest.get('ssim'),best_ssim=self.best.get('ssim') if self.best else None,
            best_step=self.best.get('step') if self.best else None,validation_step=latest.get('step'),
            goal_met=self.goal_met)

    def payload(self):
        return dict(restorer=self.model.flow_renderer.conditional_restorer.state_dict(),
            optimizer=self.optimizer.state_dict(),optimizer_parameter_names=self.names,step=self.step,
            total_restorer_updates=self.parent_updates+self.step,phase=self.phase,plan_sha256=self.plan_sha,
            sampler=self.train.sampler_state(),rng_torch=torch.get_rng_state(),rng_numpy=np.random.get_state(),
            rng_python=random.getstate(),rng_cuda=torch.cuda.get_rng_state_all() if self.args.device=='cuda' else [],
            best=self.best,goal_met=self.goal_met,goal_history=self.goal_history)

    def save(self):
        durable_save(self.checkpoint,self.payload())

    def update(self):
        row = super().update()
        row.update(parent_restorer_updates=self.parent_updates,total_restorer_updates=self.parent_updates+self.step,
                   restorer_updates=self.parent_updates+self.step)
        return row

    def export(self,report=None,path=None):
        return export_conditional_restorer(self.model,self.args.parent_decoder,
            path or self.root/'checkpoints/renderer_for_play.pt',restorer_updates=self.parent_updates+self.step,
            source_model=self.source,metrics=dict(validation=report or {}),provenance=dict(plan_sha256=self.plan_sha,
                parent_restorer_checkpoint_sha256=self.phase['inputs']['restorer_parent'],
                parent_restorer_updates=self.parent_updates,expansion_updates=self.step))

    def validate(self,baseline):
        report = super().validate(baseline)
        pins = dict(panel_fingerprint=self.phase['panel_fingerprint'],plan_sha256=self.plan_sha,
                    feature_world_sha256=self.phase['inputs']['world'])
        decision = evaluate_restorer_goal(report,**pins)
        self.best = update_restorer_best(self.best,report,**pins)
        self.goal_met = decision['goal_met']
        self.goal_history.append(dict(step=self.step,ssim=decision['measured_ssim'],goal_met=self.goal_met))
        report['ssim_goal'] = self.goal_status()
        atomic_json(self.root/'validation'/f'step_{self.step:06d}.json',report)
        atomic_json(self.root/'validation.json',report)
        report_sha = _digest(self.root/'validation'/f'step_{self.step:06d}.json')
        self.goal_history[-1]['report_sha256'] = report_sha
        if self.best['improved']:
            self.best['report_sha256'] = report_sha
            atomic_json(self.root/'best.json',self.best)
            durable_save(self.root/'checkpoints/restorer_best.pt',self.payload())
            self.export(report,self.root/'checkpoints/renderer_best.pt')
        self.save()
        self.publish(validation=report,conditional_restorer_validation=report,ssim_goal=self.goal_status())
        if self.goal_met:
            atomic_json(self.root/'goal_reached.json',dict(decision=decision,best=self.best,step=self.step,timestamp=now()))
        return report

    def run(self):
        self.prepare()
        if self.args.probe:
            before = deepcopy(self.model.flow_renderer.conditional_restorer.state_dict())
            rows = [self.update() for _ in range(3)]
            self.save(); self.export()
            after = self.model.flow_renderer.conditional_restorer.state_dict()
            proof = dict(passed=any(not torch.equal(before[k],after[k]) for k in before),
                discarded_optimizer_updates=3,production_training_updates=0,rows=rows,frozen_assets_exact=True,
                trainable_parameters=1997798,trainable_tensors=130,legacy_Adam_retained=True,
                parent_restorer_updates=self.parent_updates,
                seconds_per_update=sum(row['optimizer_update_seconds'] for row in rows)/3,
                peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30 if self.args.device=='cuda' else 0.)
            atomic_json(self.root/'gpu_warmstart.json',proof)
            self.publish(status='completed',message='Three discarded expanded-restorer GPU updates passed')
            print(json.dumps(proof),flush=True); return
        initial = self.root/'validation_initial.json'
        baseline = json.loads(initial.read_text(encoding='utf-8')) if initial.exists() else None
        if self.step and baseline is None:
            raise ValueError('Expanded resume requires the recorded initial held-out baseline')
        if baseline is None:
            self.publish(stage_label='Measuring preserved learned restorer before expanding training')
            baseline = self.validate(None)
            atomic_json(initial,baseline)
        self.save()
        self.publish(status='running',stage_label='~2M restorer: train until RF-Q held-out SSIM > 0.85')
        while not self.goal_met:
            if (self.root/'stop_training').exists():
                self.save(); self.publish(status='stopped'); return
            row = self.update()
            with (self.root/'metrics.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            self.training.update(local_step=self.step,total_restorer_updates=self.parent_updates+self.step)
            self.save()
            self.publish(current=self.step,global_training_step=self.step,loss=row['train_loss'],latest_metrics=row,
                         conditional_restorer_training=self.training)
            if self.step%100==0:
                self.validate(baseline)
            elif self.step%10==0:
                self.preview()
        self.save()
        self.publish(status='completed',message='Held-out RF-Q + restorer SSIM is strictly above 0.85',
                     ssim_goal=self.goal_status())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','world','parent-decoder','parent-training','generator-full-checkpoint',
                 'source-recipe-parent','restorer-parent'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--seed',type=int,default=173)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--probe',action='store_true')
    a = parser.parse_args(argv)
    a.steps = None
    if a.batch!=8 or a.lr<=0:
        parser.error('Expanded paired training requires batch8 and a positive learning rate')
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    from .vqgan_pipeline import process_alive
    a.run_dir.mkdir(parents=True,exist_ok=True)
    pid = a.run_dir/'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())):
        parser.error('An expanded restorer worker is already active')
    pid.write_text(str(os.getpid()),encoding='ascii')
    trainer = Restorer2MTraining(a)
    try:
        with KeepAwake():trainer.run()
    except BaseException as error:
        trainer.publish(status='failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':
    main()
