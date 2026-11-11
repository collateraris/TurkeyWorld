"""Independent Source-supervised one-enemy compact Base/RF1 training."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .distill_renderer import _digest,_save
from .gradient_safety import clip_finite_grad_norm_
from .models import load_world_checkpoint
from .power_guard import KeepAwake
from .renderer_enemy_data import EnemyFrames
from .renderer_enemy_objectives import enemy_refinement_loss
from .renderer_enemy_validation import evaluate_enemy_renderer,save_enemy_preview
from .renderer_refinement import configure_refinement,frozen_fingerprint,export_refined_bundle
from .renderer_student import load_renderer_student,_hash_tensors
from .training_pipeline import atomic_json,now


class EnemyRendererTraining:
    def __init__(self,args):
        self.args=args;self.root=Path(args.run_dir).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.started=time.perf_counter()
        self.state={'run_id':self.root.name,'stage':'renderer_enemy_refinement','status':'initializing',
            'model':'One-NPC Source → compact Base + RF1; VQGAN/RSSM/map frozen',
            'current':0,'global_training_step':0,'total':args.steps,'started_at':now(),'image_size':32,
            'prediction_every':args.preview_every,'validation_every':args.validate_every,
            'loss_protocol':'one_enemy_source_balanced_renderer_v1','scope':'Fixed Source poses; not autonomous game qualification',
            'world_training':False,'gameplay_qualified':False}
        if getattr(args,'source_latent_weight',0.)>0:
            self.state.update(loss_protocol='one_enemy_source_balanced_renderer_latent_v2',
                source_latent_weight=args.source_latent_weight,
                source_latent_target='frozen_current_compactE(Source); loss labels only')
        if getattr(args,'combat_data_root',None):
            self.state.update(loss_protocol='one_enemy_combat_navigation_renderer_v1',
                model='Source combat + enemy appearance + detailed navigation → compact Base / RF1',
                source_mixture={'combat':.5,'appearance':.25,'navigation':.25},gan_active=False)
        if getattr(args,'couple_base_flow',False) or getattr(args,'base_regression_weight',0)>0:
            self.state.update(loss_protocol='one_enemy_source_coupled_base_rf_guard_v3',
                flow_base_detached=not getattr(args,'couple_base_flow',False),
                base_regression_weight=getattr(args,'base_regression_weight',0.))
        self.state['base_preservation_required']=getattr(args,'require_base_preservation',False)
        if getattr(args,'distance_balance',False):
            self.state.update(loss_protocol='one_enemy_distance_source_crop_v4',
                source_mixture={'near':.25,'mid':.25,'far':.25,'original_replay':.25},
                distance_mixture={'near':.25,'mid':.25,'far':.25,'original_replay':.25},
                object_crop_weight=getattr(args,'object_crop_weight',0.))
        if getattr(args,'until_enemy_quality',False):
            self.state.update(total=None,training_until_enemy_quality=True,eta_seconds=None,
                scope='Experimental NPC-appearance target; not full Source/gameplay qualification')
    def publish(self,**values):
        self.state.update(values,updated_at=now(),elapsed_seconds=time.perf_counter()-self.started)
        atomic_json(self.root/'status.json',self.state)

    def run(self):
        args=self.args;device=torch.device(args.device)
        if device.type=='cuda':torch.cuda.init()
        torch.set_num_threads(args.threads)
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        source,meta=load_world_checkpoint(args.world,'cpu');source_step=meta['training_steps'];del meta
        source.eval().requires_grad_(False)
        working,meta=load_world_checkpoint(args.world,'cpu');del meta;gc.collect()
        initial=load_renderer_student(working,args.renderer_checkpoint,
            allow_unqualified_codec=args.allow_unqualified_codec)
        initial_renderer_steps=initial['training_steps']
        initial_base_steps=working.base_student_bundle['training_steps']
        parent_adam=None
        if getattr(args,'parent_adam_checkpoint',None):
            parent_adam=torch.load(args.parent_adam_checkpoint,map_location='cpu',weights_only=False)
            initial_renderer_steps-=parent_adam['step'];initial_base_steps-=parent_adam['step']
            if min(initial_renderer_steps,initial_base_steps)<0:raise ValueError('Parent cumulative renderer counters changed')
        original_base = None
        if getattr(args,'pixel_residual',False):
            from .base_student import enable_base_pixel_residual
            original_base=working.base_student
            working.base_student=enable_base_pixel_residual(original_base)
        working.float().to(device)
        policy=configure_refinement(working)
        if policy['base_trainable']>1000000 or policy['flow_trainable']>7000000 or policy['codec_parameters']>1000000:
            raise ValueError('User compact renderer parameter budgets changed')
        optimizer=torch.optim.AdamW([
            {'params':working.base_student.parameters(),'lr':args.base_lr},
            {'params':[p for p in working.flow_renderer.parameters() if p.requires_grad],'lr':args.lr}],weight_decay=.01)
        frozen=frozen_fingerprint(working)
        codec_identity=_hash_tensors(working.flow_renderer.codec.state_dict())
        codec_ids={id(p) for p in working.flow_renderer.codec.parameters()}
        if any(id(p) in codec_ids for group in optimizer.param_groups for p in group['params']):
            raise AssertionError('Codec entered enemy renderer optimizer')
        frozen_sr_asset=None
        if getattr(args,'frozen_sr_checkpoint',None):
            sr=torch.load(args.frozen_sr_checkpoint,map_location='cpu',weights_only=False)
            if (sr.get('stage')!='sr'or sr.get('input_size')!=32 or sr.get('output_size')!=128
                    or not isinstance(sr.get('training_steps'),int)or sr['training_steps']<1):
                raise ValueError('Ordered renderer stage requires its completed native32-to128 SR asset')
            frozen_sr_asset={'path':str(args.frozen_sr_checkpoint),'sha256':_digest(args.frozen_sr_checkpoint),
                'training_steps':sr['training_steps'],'small_enemy_stage_gate':sr.get('small_enemy_stage_gate'),
                'model_loaded_for_renderer_training':False,'used_as_training_input_or_loss':False,
                'immutable_asset_binding_only':True,'joint_SR_quality_evaluated':False}
            del sr
            self.publish(frozen_sr_asset=frozen_sr_asset)
        self.publish(stage_label='Indexing genuine one-enemy Source frames',
            source_world_step=source_step,model_parameters=policy,solver_steps=1,
            renderer_training={'active':True,'base_active':True,'codec_frozen':True,'codec_all_buffers_frozen':True,
                'rssm_frozen':True,'map_frozen':True,'geometry_frozen':True,'sr_frozen':True,'gan_active':False,
                'flow_base_detached':not getattr(args,'couple_base_flow',False)})
        focus_weights=({'large':.45,'partial':.15,'distant':.05,'corpse':.05,'medium':.05,
            'empty_navigation':.125,'empty_hidden':.125} if getattr(args,'large_focus',False) else None)
        small_mode=bool(getattr(args,'small_data_root',None))
        if small_mode:
            from .small_enemy_data import SmallEnemyData
            train=SmallEnemyData(args.small_data_root,'train',seed=args.seed)
            validation=SmallEnemyData(args.small_data_root,'val',seed=args.seed+1,exclude_scene_keys=train.scene_keys)
            self.state.update(loss_protocol='one_enemy_tiny_Source_codec_adaptation_v1',
                source_mixture={'tiny':.75,'navigation':.0625,'near':.0625,'corpse':.0625,'hp0':.0625},
                codec_adaptation=True)
        elif getattr(args,'combat_data_root',None):
            from .renderer_combat_data import CombatRendererFrames
            if focus_weights is not None:raise ValueError('Combat mixture has its own explicit quotas; large-focus is incompatible')
            train=CombatRendererFrames(args.combat_data_root,args.appearance_data_root,'train',seed=args.seed)
            validation=CombatRendererFrames(args.combat_data_root,args.appearance_data_root,'val',
                seed=args.seed+1,exclude_scene_keys=train.scene_keys)
        else:
            train=EnemyFrames(args.data_root,args.empty_data_root,'train',seed=args.seed,weights=focus_weights)
            validation=EnemyFrames(args.data_root,args.empty_data_root,'val',seed=args.seed+1,exclude_scene_keys=train.scene_keys)
        if ({r['seed'] for r in train.files}&{r['seed'] for r in validation.files}
                or {r['sha256'] for r in train.files}&{r['sha256'] for r in validation.files}):
            raise ValueError('Enemy Source train/val identities overlap')
        panel=validation.fixed_panel(args.validation_frames,seed=args.seed)
        distance_validation=None
        preserve_validation=preserve_baseline=None
        if getattr(args,'preserve_distance_panel',None):
            from .renderer_distance_data import SourceDistancePanel
            preserve_validation=SourceDistancePanel(args.preserve_distance_panel)
            preserve_baseline=json.loads(args.preserve_distance_baseline.read_text(encoding='utf-8'))['distance_validation']
            if preserve_validation.panel_fingerprint(list(range(len(preserve_validation.rows))))!=preserve_baseline['panel_fingerprint']:
                raise ValueError('Historical distance panel identity changed')
        if getattr(args,'distance_balance',False):
            from .renderer_distance_data import DistanceBalancedFrames,SourceDistancePanel
            if not getattr(args,'combat_data_root',None) or not getattr(args,'distance_panel',None):
                raise ValueError('Distance phase needs the parent combat mixture and immutable held-out panel')
            train=DistanceBalancedFrames(train)
            distance_validation=SourceDistancePanel(args.distance_panel)
            if any(bytes.fromhex(row['scene_key']) in train.scene_keys for row in distance_validation.records):
                raise ValueError('Distance VAL pose overlaps a TRAIN pose')
        hp0_validation=hp0_panel=None
        if getattr(args,'hp0_validation',False):
            if not getattr(args,'combat_data_root',None):raise ValueError('Supplemental HP0 needs the explicit combat Source corpus')
            from .codec_battle_data import CodecBattleFrames
            stress_train=CodecBattleFrames(args.combat_data_root,args.appearance_data_root,'train',seed=args.seed)
            hp0_validation=CodecBattleFrames(args.combat_data_root,args.appearance_data_root,'val',seed=args.seed+1,
                exclude_render_keys=stress_train.render_keys)
            hp0_panel=list(hp0_validation.terminal_indices)
            if len(hp0_panel)<18:raise ValueError('Require at least18 genuine held-out HP0 targets')
            del stress_train;gc.collect()
        warmup_panel=[]
        if getattr(args,'base_warmup_steps',0):
            groups={}
            for index in train.buckets['large']:
                row=train.rows[index]
                if not train.files[row['file']]['dynamic']:groups.setdefault(row['file'],[]).append(index)
            count=getattr(args,'warmup_large_frames',16)
            if len(groups)<count:raise ValueError('Need distinct static LARGE TRAIN files for the explicit Base overfit phase')
            rng=np.random.default_rng(args.seed+991)
            for file_index in rng.choice(sorted(groups),count,replace=False):
                warmup_panel.append(int(rng.choice(groups[int(file_index)])))
        if original_base is not None:
            from .renderer_enemy_objectives import enemy_conditions
            from .renderer_refinement import refinement_forward
            conditions=enemy_conditions(working,validation.batch(4,indices=panel[:4]))
            original_base.to(device).eval().requires_grad_(False)
            with torch.no_grad():
                old_base=original_base(conditions['state'],conditions['features'],conditions['geometry'])
                new_base,residual=working.base_student.forward_with_residual(conditions['state'],conditions['features'],conditions['geometry'])
                flow=working.flow_renderer
                endpoint=flow.latent_sample(old_base,conditions['state'],noise=flow.fixed_noise,map_condition=conditions['map'])
                old_rgb=flow.codec.decode(flow.codec.quantize(flow.denormalize_latents(endpoint))['quantized'])
                _,new_rgb,_=refinement_forward(working,conditions)
            if not (torch.equal(old_base,new_base) and torch.equal(old_rgb,new_rgb) and not residual.any()):
                raise AssertionError('Zero pixel residual changed the parent renderer function')
            migration={'kind':'zero_initialized_post_codec_pixel_residual','parent_renderer':str(args.renderer_checkpoint),
                'parent_renderer_sha256':_digest(args.renderer_checkpoint),'new_parameters':1740,
                'base_parameters':policy['base_trainable'],'base_rgb_bitexact_at_initialization':True,
                'final_rgb_bitexact_at_initialization':True,'codec_frozen':True,'amplitude':.5}
            atomic_json(self.root/'pixel_residual_migration.json',migration)
            self.publish(pixel_residual_migration=migration)
            del original_base,conditions;gc.collect()
        identity={'world_sha256':_digest(args.world),'renderer_sha256':_digest(args.renderer_checkpoint),
            'train':(train.parent.identity if distance_validation is not None else train.identity),
            'val':validation.identity,'panel':panel,
            'panel_fingerprint':validation.panel_fingerprint(panel),'codec_sha256':codec_identity,
            'warmup_train_panel':warmup_panel}
        if hp0_validation is not None:
            identity.update(hp0_validation=hp0_validation.identity,hp0_panel=hp0_panel,
                hp0_panel_fingerprint=hp0_validation.panel_fingerprint(hp0_panel))
        if distance_validation is not None:
            identity.update(distance_sampling=train.identity['distance_sampling'],
                distance_panel_fingerprint=distance_validation.panel_fingerprint(list(range(len(distance_validation.rows)))))
        if preserve_validation is not None:
            identity.update(preserve_distance_panel_sha256=_digest(args.preserve_distance_panel),
                preserve_distance_baseline_sha256=_digest(args.preserve_distance_baseline))
        if parent_adam is not None:identity['parent_adam_sha256']=_digest(args.parent_adam_checkpoint)
        if frozen_sr_asset is not None:identity['frozen_sr_asset']=frozen_sr_asset
        base_anchor=None
        if getattr(args,'base_anchor_checkpoint',None):
            from .renderer_preservation_validation import validate_base_anchor_identity
            anchor=torch.load(args.base_anchor_checkpoint,map_location='cpu',weights_only=False)
            validate_base_anchor_identity(anchor['identity'],identity)
            base_anchor=anchor['report'];del anchor
            identity['base_anchor_sha256']=_digest(args.base_anchor_checkpoint)
        optional_defaults={'combat_data_root':None,'appearance_data_root':None,'couple_base_flow':False,
            'base_regression_weight':0.,'require_base_preservation':False,'record_sample_indices':False,
            'hp0_validation':False,'base_anchor_checkpoint':None,'distance_balance':False,
            'distance_panel':None,'object_crop_weight':0.,'source_training_checkpoint':None,
            'small_data_root':None,'parent_adam_checkpoint':None}
        optional_defaults.update(preserve_distance_panel=None,preserve_distance_baseline=None)
        optional_defaults['frozen_sr_checkpoint']=None
        configuration={key:str(value.resolve()) if isinstance(value,Path) else value for key,value in vars(args).items()
            if key not in optional_defaults or value!=optional_defaults[key]}
        config_path=self.root/'experiment_config.json'
        if config_path.exists():
            old=json.loads(config_path.read_text(encoding='utf-8'))
            if ({k:v for k,v in old.items() if k!='steps'}!={k:v for k,v in configuration.items() if k!='steps'}
                    or args.steps<old['steps']):raise ValueError('Enemy refinement settings changed or budget shrank')
            if distance_validation is not None and args.steps>old['steps']:
                extension=self.root/f'budget_extension_{old["steps"]:06d}_{args.steps:06d}.json'
                if not extension.exists():raise ValueError('Distance budget extension requires an immutable explicit ledger')
                ledger=json.loads(extension.read_text(encoding='utf-8'))
                if (ledger.get('previous_budget')!=old['steps'] or ledger.get('new_budget')!=args.steps
                        or ledger.get('start_step')!=old['steps']
                        or ledger.get('source_checkpoint_sha256')!=_digest(self.root/'checkpoints/renderer_enemy_last.pt')
                        or ledger.get('phase_plan_sha256')!=_digest(self.root/'distance_phase_plan.json')):
                    raise ValueError('Distance extension checkpoint/boundary identity changed')
        atomic_json(config_path,configuration)
        atomic_json(self.root/'source_identity.json',identity)
        self.publish(data_preparation={'status':'complete','train_frames':len(train.rows),'val_frames':len(validation.rows),
            'train_categories':train.identity['category_counts'],'val_categories':validation.identity['category_counts'],
            'excluded_val_frames':getattr(validation,'excluded_frames',{}),
            'excluded_val_scene_keys':getattr(validation,'excluded_scene_keys',0),
            'train_fingerprint':train.identity['fingerprint'],'validation_fingerprint':validation.identity['fingerprint']},
            validation_panel_fingerprint=identity['panel_fingerprint'])
        checkpoint=self.root/'checkpoints/renderer_enemy_last.pt'
        best_path=self.root/'checkpoints/renderer_enemy_best.pt'
        baseline_path=self.root/'parent_renderer_baseline.pt'
        step,best,streak,enemy_streak=0,None,0,0
        new_phase=False
        phase_start=0
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if (saved['identity']!=identity or saved['frozen_fingerprint']!=frozen
                    or {k:v for k,v in saved['configuration'].items() if k!='steps'}!={k:v for k,v in configuration.items() if k!='steps'}
                    or args.steps<saved['configuration']['steps']):
                raise ValueError('Enemy refinement checkpoint/data/protocol identity changed')
            working.base_student.load_state_dict(saved['base'],strict=True)
            working.flow_renderer.load_state_dict(saved['flow'],strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            step,best,streak=saved['step'],saved['best'],saved['streak']
            enemy_streak=saved.get('enemy_streak',0)
            train.restore_sampler(saved['sampler'])
            torch.set_rng_state(saved['rng_torch'].cpu());np.random.set_state(saved['rng_numpy']);random.setstate(saved['rng_python'])
            if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in saved['rng_cuda']])
            del saved;gc.collect()
            if distance_validation is not None:
                phase_path=self.root/'distance_phase_plan.json'
                if not phase_path.exists():raise ValueError('Distance resume requires its immutable phase plan')
                phase=json.loads(phase_path.read_text(encoding='utf-8'))
                if (phase['source_checkpoint_sha256']!=_digest(args.source_training_checkpoint)
                        or phase['identity']!=identity):raise ValueError('Distance phase plan changed')
                phase_start=phase['start_step']
            if parent_adam is not None:phase_start=parent_adam['step']
        elif parent_adam is not None:
            if not small_mode:raise ValueError('Parent Adam adoption requires the explicit tiny Source protocol')
            from .small_enemy_renderer_init import restore_small_renderer_parent
            proof=restore_small_renderer_parent(parent_adam,working,optimizer)
            step=phase_start=parent_adam['step'];new_phase=True
            train.rng.bit_generator.state=parent_adam['sampler'].get('parent',parent_adam['sampler'])['rng']
            torch.set_rng_state(parent_adam['rng_torch'].cpu());np.random.set_state(parent_adam['rng_numpy']);random.setstate(parent_adam['rng_python'])
            if device.type=='cuda':torch.cuda.set_rng_state_all([value.cpu()for value in parent_adam['rng_cuda']])
            proof.update(start_step=step,new_source_sampler_draws=0,old_source_sampler=parent_adam['sampler'],
                source_checkpoint_sha256=_digest(args.parent_adam_checkpoint),global_rng_restored=True,
                old_counters_preserved=True,new_dataset_sampler_is_explicit_phase=True)
            atomic_json(self.root/'tiny_renderer_adam_migration.json',proof)
        elif getattr(args,'source_training_checkpoint',None):
            if distance_validation is None:raise ValueError('Cross-run optimizer adoption requires the explicit distance phase')
            from .renderer_distance_training import restore_distance_parent
            saved=torch.load(args.source_training_checkpoint,map_location='cpu',weights_only=False)
            phase=restore_distance_parent(saved,working,optimizer,train,identity,configuration,frozen)
            step=phase_start=phase['start_step'];new_phase=True
            torch.set_rng_state(saved['rng_torch'].cpu());np.random.set_state(saved['rng_numpy']);random.setstate(saved['rng_python'])
            if device.type=='cuda':torch.cuda.set_rng_state_all([value.cpu()for value in saved['rng_cuda']])
            phase.update(status='initialized',source_checkpoint=str(args.source_training_checkpoint),
                source_checkpoint_sha256=_digest(args.source_training_checkpoint),identity=identity,
                rng_restored=True,phase_budget=args.steps-step,
                codec_all_parameters_buffers_bitexact=_hash_tensors(working.flow_renderer.codec.state_dict())==codec_identity)
            if not phase['codec_all_parameters_buffers_bitexact']:raise AssertionError('Distance parent codec changed')
            phase_path=self.root/'distance_phase_plan.json'
            if phase_path.exists():
                if json.loads(phase_path.read_text(encoding='utf-8'))!=phase:
                    raise ValueError('Immutable distance phase initialization changed')
            else:atomic_json(phase_path,phase)
            self.publish(distance_phase=phase,current=step,global_training_step=step)
            del saved;gc.collect()
        def save():
            _save(checkpoint,{'base':working.base_student.state_dict(),'flow':working.flow_renderer.state_dict(),
                'optimizer':optimizer.state_dict(),'step':step,'best':best,'streak':streak,
                'enemy_streak':enemy_streak,
                'configuration':configuration,'identity':identity,'frozen_fingerprint':frozen,
                'sampler':train.sampler_state(),'rng_torch':torch.get_rng_state(),
                'rng_numpy':np.random.get_state(),'rng_python':random.getstate(),
                'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []})
            if distance_validation is not None:
                audit={'protocol':'visible_source_distance_v1','global_step':step,'start_step':phase_start,
                    'phase_updates':step-phase_start,'parent_draws_preserved':train.parent_draws,
                    'new_phase_draws':train.phase_draws,'quota_draws':dict(train.distance_counts),
                    'actual_source_distance_draws':dict(train.actual_distance_counts),
                    'new_replay_pool_draws':{key:count-phase['parent_mix_counts'][key]
                        for key,count in train.parent.mix_counts.items()},
                    'visible_definition':'Source HP>0,alive,exact NPC pixel mask nonempty',
                    'source_pixels_resized_for_sampling':False,'validation_used_for_updates':False}
                atomic_json(self.root/'source_sampling_audit.json',audit)
        def verify_frozen():
            if (frozen_fingerprint(working)!=frozen or _hash_tensors(working.flow_renderer.codec.state_dict())!=codec_identity
                    or working.flow_renderer.codec.training or working.flow_renderer.codec.quantize.training):
                raise AssertionError('Frozen codec/dynamics/map/geometry parameters or buffers changed')
            if frozen_sr_asset is not None and _digest(args.frozen_sr_checkpoint)!=frozen_sr_asset['sha256']:
                raise AssertionError('Selected ordered-stage SR asset changed')
            self.publish(frozen_verification={'step':step,'all_frozen_parameters_buffers_bitexact':True,
                'codec_all_parameters_buffers_bitexact':True,'codec_optimizer_excluded':True,'codec_sha256':codec_identity})
        def evaluate_full(at,reference=None):
            report,arrays=evaluate_enemy_renderer(working,validation,panel,step=at,batch_size=args.validation_batch,
                baseline=reference,require_base_preservation=getattr(args,'require_base_preservation',False),base_anchor=base_anchor)
            if frozen_sr_asset is not None:report['frozen_sr_asset']=frozen_sr_asset
            if small_mode:
                from .small_enemy_validation import small_enemy_cohorts,relative_small_enemy_gate
                report['small_enemy_cohorts']={name:small_enemy_cohorts(arrays[key],arrays['source'],
                    arrays['mask'],arrays['states'])for name,key in (('base','base'),('rf1','rgb'))}
                gate=(relative_small_enemy_gate(report,reference,predictions=('base','rf1'))
                    if reference is not None else {'passed':False,'failure_reasons':['initial_reference_only']})
                report['small_enemy_stage_gate']=gate
                if reference is not None:
                    report['legacy_other_panel_selection']=report['experimental_selection']
                    report['experimental_selection']={'validated_improvement':gate['passed'],
                        'failure_reasons':gate['failure_reasons'],'scope':gate['scope']}
                report['selection_score']=sum(report['small_enemy_cohorts'][name]['tiny']['foreground_mae']
                    +.25*report['small_enemy_cohorts'][name]['overall']['mae_mean']for name in ('base','rf1'))
            if hp0_validation is not None:
                extra,extra_arrays=evaluate_enemy_renderer(working,hp0_validation,hp0_panel,step=at,
                    batch_size=args.validation_batch,baseline=(reference or {}).get('terminal_supplement'),
                    require_base_preservation=getattr(args,'require_base_preservation',False),preservation_cohorts=('hp0',),
                    base_anchor=(base_anchor or {}).get('terminal_supplement'))
                report['terminal_supplement']=extra
                report['terminal_supplement_fingerprint']=identity['hp0_panel_fingerprint']
                if reference is None:
                    _save(self.root/'parent_hp0_baseline.pt',{'identity':identity['hp0_panel_fingerprint'],
                        'panel':hp0_panel,'report':extra,'arrays':extra_arrays})
                if getattr(args,'require_base_preservation',False):
                    report['passed']=report['passed'] and extra['passed']
                    if reference is not None and not extra['base_preservation']['passed']:
                        report['experimental_selection']['validated_improvement']=False
                        report['experimental_selection']['failure_reasons'].append('hp0_base_preservation_release_gate')
                del extra_arrays
            if preserve_validation is not None:
                from .small_enemy_validation import distance_preservation_gate
                guard,guard_arrays=evaluate_enemy_renderer(working,preserve_validation,
                    list(range(len(preserve_validation.rows))),step=at,batch_size=args.validation_batch)
                report['historical_distance_validation']=guard
                preserved=distance_preservation_gate(guard,preserve_baseline)
                report['historical_distance_preservation']=preserved
                if reference is not None and not preserved['passed']:
                    report['small_enemy_stage_gate']['passed']=False
                    report['small_enemy_stage_gate']['failure_reasons']+=preserved['failure_reasons']
                    report['experimental_selection']['validated_improvement']=False
                    report['experimental_selection']['failure_reasons']+=preserved['failure_reasons']
                del guard_arrays
            if distance_validation is not None:
                extra,extra_arrays=evaluate_enemy_renderer(working,distance_validation,
                    list(range(len(distance_validation.rows))),step=at,batch_size=args.validation_batch)
                report['distance_validation']=extra
                if reference is not None:
                    from .renderer_distance_training import distance_candidate_assessment
                    comparison=distance_candidate_assessment(extra,reference['distance_validation'])
                    report['distance_improvement']=comparison
                    if not comparison['passed']:
                        report['experimental_selection']['validated_improvement']=False
                        report['experimental_selection']['failure_reasons'].append('distance_bands_not_all_improved')
                atomic_json(self.root/'distance_validation'/f'step_{at:06d}.json',extra)
                if at==phase_start:
                    np.savez_compressed(self.root/'distance_baseline_images.npz',
                        **{key:value.numpy()for key,value in extra_arrays.items()if torch.is_tensor(value)})
                del extra_arrays
            report['release_passed']=bool(report['base_preservation']['passed']
                and (hp0_validation is None or report['terminal_supplement']['base_preservation']['passed']))
            return report,arrays
        if baseline_path.exists():
            baseline_saved=torch.load(baseline_path,map_location='cpu',weights_only=False)
            if baseline_saved['identity']!=identity:raise ValueError('Immutable parent baseline changed')
            baseline=baseline_saved['report'];del baseline_saved
        else:
            if step and not new_phase:raise ValueError('Resume requires the original parent renderer baseline')
            self.publish(stage_label='Initial held-out enemy/empty renderer validation')
            baseline,arrays=evaluate_full(step)
            verify_frozen()
            _save(baseline_path,{'identity':identity,'report':baseline,'arrays':arrays})
            atomic_json(self.root/'validation_initial.json',baseline);save_enemy_preview(arrays,self.root/'preview.png',step)
            self.publish(validation_initial=baseline,validation=baseline)
        if new_phase:
            verify_frozen();save()
        preview_ids=[]
        for category in (('tiny','near','corpse','navigation')if small_mode else('large','partial','corpse','empty_navigation')):
            matches=[i for i in panel if validation.rows[i]['category']==category]
            if matches:preview_ids.append(matches[0])
        first_step=step+1
        until_enemy=getattr(args,'until_enemy_quality',False)
        while (until_enemy or step<args.steps) and (True if small_mode else enemy_streak<2 if until_enemy else streak<2):
            if (self.root/'stop_training').exists():
                save();self.publish(status='stopped',current=step,global_training_step=step);return
            step+=1;configure_refinement(working);optimizer.zero_grad(set_to_none=True)
            base_only=step<=getattr(args,'base_warmup_steps',0)
            if base_only:
                ids=train.rng.choice(warmup_panel,args.batch,replace=True).tolist()
                batch=train.batch(args.batch,indices=ids)
            else:batch=train.batch(args.batch)
            loss,metrics=enemy_refinement_loss(working,batch,
                ssim_weight=args.ssim_weight,edge_weight=args.edge_weight,contrast_weight=args.contrast_weight,
                base_only=base_only,latent_weight=getattr(args,'source_latent_weight',0.),
                detach_base=not getattr(args,'couple_base_flow',False),
                base_regression_weight=getattr(args,'base_regression_weight',0.),
                object_crop_weight=getattr(args,'object_crop_weight',0.))
            if getattr(args,'record_sample_indices',False):
                metrics['source_batch_indices']=batch['row_index'].tolist()
                metrics['source_sampler_draws']=train.draws
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Invalid enemy renderer loss')
            loss.backward()
            if getattr(args,'separate_gradient_clips',False):
                base_norm=float(clip_finite_grad_norm_(working.base_student.named_parameters(),context=f'Enemy Base update{step}'))
                flow_norm=float(clip_finite_grad_norm_(((n,p)for n,p in working.flow_renderer.named_parameters()if p.requires_grad),
                    context=f'Enemy RF update{step}'))
                metrics.update(base_gradient_norm=base_norm,rf_gradient_norm=flow_norm,
                    gradient_norm=(base_norm**2+flow_norm**2)**.5,gradient_clipping='separate_base_rf')
            else:metrics['gradient_norm']=float(clip_finite_grad_norm_(working.named_parameters(),context=f'Enemy renderer update{step}'))
            optimizer.step()
            if step==first_step:verify_frozen()
            report=None
            if step%args.validate_every==0 or step==args.steps or step==getattr(args,'base_warmup_steps',0):
                report,arrays=evaluate_full(step,baseline)
                verify_frozen();report['frozen_components_bitexact']=True
                eligible=report['experimental_selection']['validated_improvement']
                target_met=eligible and all(report[name]['visible_enemy']['foreground_mae']<=.08
                    and report[name]['visible_enemy']['closer_to_source_than_without_npc_rate']>=.75 for name in ('base','rf1'))
                report['enemy_appearance_target']={'passed':target_met,'npc_roi_mae_max':.08,
                    'presence_rate_min':.75,'requires_empty_regression_guards':True,
                    'scope':'Experimental appearance only, not full Source quality or gameplay'}
                enemy_streak=enemy_streak+1 if target_met else 0
                if eligible and (best is None or report['selection_score']<best['score']):
                    best={'step':step,'score':report['selection_score']}
                    _save(best_path,{'base':working.base_student.state_dict(),'flow':working.flow_renderer.state_dict(),**best})
                streak=streak+1 if report['passed'] else 0
                atomic_json(self.root/'validation.json',report)
                atomic_json(self.root/'validation'/f'step_{step:06d}.json',report)
                if distance_validation is not None:
                    _save(self.root/'checkpoints'/f'distance_step_{step:06d}.pt',
                        {'base':working.base_student.state_dict(),'flow':working.flow_renderer.state_dict(),
                         'step':step,'identity':identity,'frozen_fingerprint':frozen,
                         'source_world':str(args.world),'source_initial_renderer':str(args.renderer_checkpoint),
                         'source_quality_passed':report['passed'],
                         'experimental_selection':report['experimental_selection'],
                         'automatic_active_promotion':False})
                save_enemy_preview(arrays,self.root/'preview.png',step)
            elif step%args.preview_every==0:
                _,arrays=evaluate_enemy_renderer(working,validation,preview_ids,step=step,batch_size=args.validation_batch)
                save_enemy_preview(arrays,self.root/'preview.png',step)
            if step%args.preview_every==0 or report is not None:
                atomic_json(self.root/'preview.json',{'step':step,'every':args.preview_every,'kind':'Source_enemy_GT_query',
                    'source_held_out':True,'renderer_only':True,'world_rollout':False,'codec_frozen':True,
                    'feature_policy':'static_zero128_or_dynamic_frozen_past_prior','source_quality_gate_evaluated':report is not None})
            record={'step':step,'global_training_step':step,'stage':'renderer_enemy_refinement','timestamp':now(),
                'loss_protocol':self.state['loss_protocol'],**metrics}
            if report is not None:record['source_validation']=report
            with (self.root/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
            self.publish(status='running',stage_label='Base TRAIN16 overfit warmup; held-out data untouched' if base_only else
                'Joint Source combat / appearance / detailed navigation; codec and RSSM frozen' if getattr(args,'combat_data_root',None) else 'Joint NPC-large curriculum with frozen codec',
                current=step,global_training_step=step,
                progress=None if until_enemy else (step-phase_start)/max(args.steps-phase_start,1),loss=metrics['train_loss'],
                latest_metrics=metrics,validation=report or self.state.get('validation'),selected_best=best,
                sampler_draws=train.draws,sampler_case_counts=train.case_counts,
                sampler_pool_counts=getattr(train,'mix_counts',None),sampler_cohort_counts=getattr(train,'cohort_counts',None),
                distance_sampler_counts=getattr(train,'distance_counts',None),
                actual_distance_sample_counts=getattr(train,'actual_distance_counts',None),
                phase_start_step=phase_start,phase_updates=step-phase_start,
                component_phase='base_large_train_overfit' if base_only else 'joint_large_enemy',
                base_warmup_steps=getattr(args,'base_warmup_steps',0),warmup_frames=len(warmup_panel),
                warmup_draws=min(step,getattr(args,'base_warmup_steps',0))*args.batch,
                enemy_target_streak=enemy_streak,enemy_target_required_passes=2,
                control_milestone=args.steps,eta_seconds=None if until_enemy else
                    (time.perf_counter()-self.started)/max(step-first_step+1,1)*(args.steps-step))
            if step%args.save_every==0 or report is not None or step==first_step:save()
        save()
        if best is None:
            if small_mode and self.state.get('validation'):
                atomic_json(self.root/'validation_selected.json',self.state['validation'])
            self.publish(status='completed',current=step,experimental_candidate_validated=False,
                source_quality_passed=False,message='Training completed; no candidate passed NPC-improvement and empty-regression guards',
                analysis_checkpoint=str(checkpoint),release_passed=False,
                active_renderer_changed=False);return
        selected=torch.load(best_path,map_location='cpu',weights_only=False)
        working.base_student.load_state_dict(selected['base'],strict=True);working.flow_renderer.load_state_dict(selected['flow'],strict=True)
        report,arrays=evaluate_full(selected['step'],baseline)
        verify_frozen()
        if not report['experimental_selection']['validated_improvement']:
            raise AssertionError('Selected enemy candidate no longer passes its held-out guards')
        exported=export_refined_bundle(source,working,self.root/'checkpoints',updates=selected['step'],report=report,
            initial_renderer_steps=initial_renderer_steps,initial_base_steps=initial_base_steps,
            provenance={'kind':'one_enemy_Source_GT_renderer_refinement','source_identity':identity,
                'feature_policies':['static_zero128','dynamic_frozen_past_prior'],'codec_frozen':True,
                'dynamics_frozen':True,'automatic_active_promotion':False})
        atomic_json(self.root/'validation_selected.json',report);save_enemy_preview(arrays,self.root/'preview.png',selected['step'])
        self.publish(status='completed',current=step,selected_step=selected['step'],validation=report,
            renderer_export=exported,source_quality_passed=report['passed'],experimental_candidate_validated=True,
            release_passed=report.get('release_passed'),active_renderer_changed=False,gameplay_qualified=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('world','renderer-checkpoint','data-root','empty-data-root','run-dir'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--combat-data-root',type=Path,help='Opt-in Source combat/appearance/navigation mixture')
    parser.add_argument('--appearance-data-root',type=Path,help='Static one-NPC Source snapshots for the optional mixture')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--allow-unqualified-codec',action='store_true')
    parser.add_argument('--pixel-residual',action='store_true')
    parser.add_argument('--large-focus',action='store_true')
    parser.add_argument('--separate-gradient-clips',action='store_true')
    parser.add_argument('--couple-base-flow',action='store_true',help='Opt-in RF gradient through the frozen encoder into Base')
    parser.add_argument('--base-regression-weight',type=float,default=0.,help='Opt-in per-image foreground/background RF-vs-Base hinge')
    parser.add_argument('--require-base-preservation',action='store_true',help='Require both held-out RF>=Base and Base>=fixed-parent guards')
    parser.add_argument('--record-sample-indices',action='store_true',help='Record exact training order for paired controlled pilots')
    parser.add_argument('--hp0-validation',action='store_true',help='Keep the original panel and add at least18 render-condition-disjoint HP0 queries')
    parser.add_argument('--base-anchor-checkpoint',type=Path,help='Preserve the original frozen-parent Base reference across an explicit codec revision')
    parser.add_argument('--until-enemy-quality',action='store_true')
    parser.add_argument('--distance-balance',action='store_true',help='75% equal live visible Source distance bands +25% parent replay')
    parser.add_argument('--distance-panel',type=Path,help='Immutable independent Source VAL192 near/mid/far panel')
    parser.add_argument('--source-training-checkpoint',type=Path,help='Explicit distance phase inherits completed Base/RF Adam/RNG/clocks')
    parser.add_argument('--object-crop-weight',type=float,default=0.,help='Native Source foreground crop SSIM weight; no target resize')
    parser.add_argument('--small-data-root',type=Path,help='Explicit true Source32/128 tiny-enemy corpus')
    parser.add_argument('--parent-adam-checkpoint',type=Path,help='Preserve completed compatible Base/RF Adam after fixed-Q codec revision')
    parser.add_argument('--preserve-distance-panel',type=Path)
    parser.add_argument('--preserve-distance-baseline',type=Path)
    parser.add_argument('--frozen-sr-checkpoint',type=Path,help='Pin the completed ordered-stage SR asset; never optimized or used as Source input')
    parser.add_argument('--base-warmup-steps',type=int,default=0)
    parser.add_argument('--warmup-large-frames',type=int,default=16)
    for name,default in (('steps',2000),('batch',8),('validation-frames',256),('validation-batch',8),
                         ('validate-every',100),('preview-every',10),('save-every',10),('seed',173),('threads',4)):
        parser.add_argument('--'+name,type=int,default=default)
    for name,default in (('lr',.0001),('base-lr',.0001),('ssim-weight',.2),('edge-weight',.1),('contrast-weight',.1)):
        parser.add_argument('--'+name,type=float,default=default)
    parser.add_argument('--source-latent-weight',type=float,default=0.,
        help='Optional RF endpoint supervision from frozen compactE(Source) labels; never an input')
    args=parser.parse_args(argv)
    if bool(args.combat_data_root)!=bool(args.appearance_data_root):
        parser.error('Combat and appearance data roots must be supplied together')
    if bool(args.preserve_distance_panel)!=bool(args.preserve_distance_baseline):
        parser.error('Historical distance preservation requires both immutable panel and baseline')
    if min(args.steps,args.batch,args.validation_frames,args.validation_batch,args.validate_every,args.preview_every,args.save_every,args.threads)<=0:
        parser.error('Positive renderer budgets and batch sizes required')
    if min(args.lr,args.base_lr)<=0 or min(args.ssim_weight,args.edge_weight,args.contrast_weight,args.source_latent_weight,args.base_regression_weight,args.object_crop_weight)<0:
        parser.error('Positive rates and nonnegative loss weights required')
    if args.base_warmup_steps<0 or args.warmup_large_frames<1:
        parser.error('Warmup steps must be nonnegative and frame count positive')
    args=argparse.Namespace(**{k:v.resolve() if isinstance(v,Path) else v for k,v in vars(args).items()})
    from .vqgan_pipeline import process_alive
    args.run_dir.mkdir(parents=True,exist_ok=True);pid=args.run_dir/'training.pid'
    if pid.exists() and process_alive(int(pid.read_text())):parser.error('Enemy renderer worker is already active')
    atomic_json(pid,__import__('os').getpid())
    experiment=EnemyRendererTraining(args);experiment.publish()
    try:
        with KeepAwake():experiment.run()
    except BaseException as error:
        experiment.publish(status='failed',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()

