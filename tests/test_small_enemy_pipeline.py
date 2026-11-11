"""Independent ordered-pipeline audit: fake children, no GPU or real process."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from turkeyworld.train_small_enemy_pipeline import (SmallEnemyPipeline, ORDER,
    next_stage, stage_can_advance, immutable_json, verified_live_child,
    milestone_complete, validate_stage_artifact)


def reports(stage, *, improved=True):
    names={'codec':('source_reconstruction','full_rf1'),'sr':('clean','codec'),'renderer':('base','rf1')}[stage]
    frames={'overall':512,'tiny':384,'navigation':37,'near':36,'corpse':37,'hp0':18}
    baseline={'step':100,'panel_fingerprint':'fixed512','source_held_out':True,'source_quality_passed':False,
        'gameplay_qualified':False,'small_enemy_cohorts':{name:{cohort:{'frames':count,'foreground_mae':.2,
            'mae_mean':.03,'ssim_mean':.8}for cohort,count in frames.items()}for name in names}}
    current=copy.deepcopy(baseline);current['small_enemy_stage_gate']={'passed':True}
    for name in names:
        current['small_enemy_cohorts'][name]['tiny']['foreground_mae']=.19 if improved else .2
    return current,baseline


class SmallEnemyPipelineTests(unittest.TestCase):
    @staticmethod
    def finish_stage(run,stage,report,budget=2000):
        (run/'status.json').write_text(json.dumps({'status':'completed','current':budget+(8000 if stage=='renderer'else 0)}))
        folder=run/'checkpoints';folder.mkdir(exist_ok=True)
        names={'codec':('codec_student.pt','distilled_vqgan_codec'),
            'sr':('sr_for_game.pt','sr'),'renderer':('renderer_student.pt','renderer_student')}
        name,kind=names[stage]
        payload={'stage':kind,'training_steps':budget,'model':{'private_test_weight':torch.arange(2)}}
        if stage=='codec':payload.update(quality_gate=report,frozen=True,fixed_codebook=True)
        elif stage=='sr':payload['validation']=report
        else:payload.update(metrics={'source_validation':report},solver='euler',solver_steps=1)
        torch.save(payload,folder/name)

    def fixture(self,root):
        parents=root/'parents';parents.mkdir()
        world=parents/'world.pt';torch.save({'config':{'navigation_only':False},'model':{'unchanged':torch.arange(3)}},world)
        assets={}
        for name in ['renderer_parent','renderer_navigation_parent','codec_parent','sr_parent','renderer_adam_parent']:
            assets[name]=parents/(name+'.pt');assets[name].write_bytes(('immutable_'+name).encode())
        data=root/'data';data.mkdir();(data/'manifest.json').write_text(json.dumps({'status':'complete','fingerprint':'fixed_source'}))
        return SimpleNamespace(run_dir=root/'pipeline',data_root=data,world=world,device='cpu',**assets)

    def test_strict_stage_order_recomputes_heldout_gate_and_never_uses_absolute_quality_as_relative_pass(self):
        self.assertEqual(tuple(ORDER),('codec','sr','renderer'))
        for previous,expected in [([], 'codec'),(['codec'],'sr'),(['codec','sr'],'renderer'),(list(ORDER),None)]:
            self.assertEqual(next_stage(previous),expected)
        for invalid in [['sr'],['codec','renderer'],['codec','codec'],list(ORDER)+['codec']]:
            with self.assertRaises(ValueError):next_stage(invalid)
        for stage in ORDER:
            good,baseline=reports(stage);self.assertTrue(stage_can_advance(stage,good,baseline))
            self.assertFalse(good['source_quality_passed']);self.assertFalse(good['gameplay_qualified'])
            primary={'codec':'source_reconstruction','sr':'clean','renderer':'rf1'}[stage]
            for mutate in ['saved_false','tiny_worse','missing_guard','nonfinite','changed_panel','not_heldout']:
                bad=copy.deepcopy(good)
                if mutate=='saved_false':bad['small_enemy_stage_gate']['passed']=False
                elif mutate=='tiny_worse':bad['small_enemy_cohorts'][primary]['tiny']['foreground_mae']=.2
                elif mutate=='missing_guard':del bad['small_enemy_cohorts'][primary]['hp0']
                elif mutate=='nonfinite':bad['small_enemy_cohorts'][primary]['near']['mae_mean']=float('nan')
                elif mutate=='changed_panel':bad['panel_fingerprint']='other'
                else:bad['source_held_out']=False
                self.assertFalse(stage_can_advance(stage,bad,baseline),(stage,mutate))
        codec,baseline=reports('codec');codec['small_enemy_cohorts']['full_rf1']['tiny']['foreground_mae']=.21
        self.assertFalse(stage_can_advance('codec',codec,baseline))

    def test_immutable_assets_and_json_bindings_reject_changed_bytes_without_overwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.fixture(root);controller=SmallEnemyPipeline(args)
            asset_dir=controller.assets();record=(controller.root/'parent_assets.json').read_bytes()
            nav=torch.load(asset_dir/'world_navigation.pt',map_location='cpu',weights_only=False)
            original=torch.load(args.world,map_location='cpu',weights_only=False)
            self.assertTrue(nav['config']['navigation_only']);self.assertFalse(original['config']['navigation_only'])
            self.assertTrue(torch.equal(nav['model']['unchanged'],original['model']['unchanged']))
            controller.assets();self.assertEqual((controller.root/'parent_assets.json').read_bytes(),record)
            saved=(asset_dir/'codec_parent.pt').read_bytes();args.codec_parent.write_bytes(b'changed_source')
            with self.assertRaises(ValueError):controller.assets()
            self.assertEqual((asset_dir/'codec_parent.pt').read_bytes(),saved)
            path=root/'immutable.json';immutable_json(path,{'fixed':1});before=path.read_bytes();stamp=path.stat().st_mtime_ns
            immutable_json(path,{'fixed':1});self.assertEqual(path.stat().st_mtime_ns,stamp)
            with self.assertRaises(ValueError):immutable_json(path,{'fixed':2})
            self.assertEqual(path.read_bytes(),before)

    def test_failed_codec_extends_without_advancing_and_resumes_the_durable_budget_then_qualifies_in_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.fixture(root);controller=SmallEnemyPipeline(args);calls=[]
            game=root/'game_assets';game.write_bytes(b'unchanged_game_bundle')
            def command(self,stage,budget,assets):return ['fake_worker',stage,str(budget)]
            def child(argv,**kwargs):
                stage,budget=argv[1],int(argv[2]);calls.append((stage,budget))
                run=controller.root/stage;current,baseline=reports(stage,improved=not(stage=='codec'and budget==2000))
                (run/'validation_initial.json').write_text(json.dumps(baseline))
                (run/'validation_selected.json').write_text(json.dumps(current))
                self.finish_stage(run,stage,current,budget)
                return SimpleNamespace(pid=1000+len(calls),returncode=0,poll=lambda:0)
            class InterruptRestart(Exception):pass
            first=True
            def interrupted(argv,**kwargs):
                nonlocal first
                if not first:raise InterruptRestart()
                first=False;return child(argv,**kwargs)
            with patch.object(SmallEnemyPipeline,'command',command),patch('turkeyworld.train_small_enemy_pipeline.verified_live_child',return_value=None), \
                    patch('turkeyworld.train_small_enemy_pipeline.process_alive',return_value=False), \
                    patch('turkeyworld.train_small_enemy_pipeline.subprocess.Popen',side_effect=interrupted), \
                    patch('turkeyworld.train_small_enemy_pipeline.time.sleep',side_effect=AssertionError('No real waits')):
                with self.assertRaises(InterruptRestart):controller.run()
            self.assertEqual(calls,[('codec',2000)]);extension=controller.root/'codec/extension_002000_003000.json'
            before=extension.read_bytes();self.assertEqual(json.loads((controller.root/'codec/milestone.json').read_text())['budget'],3000)
            resumed=SmallEnemyPipeline(args)
            with patch.object(SmallEnemyPipeline,'command',command),patch('turkeyworld.train_small_enemy_pipeline.verified_live_child',return_value=None), \
                    patch('turkeyworld.train_small_enemy_pipeline.process_alive',return_value=False), \
                    patch('turkeyworld.train_small_enemy_pipeline.subprocess.Popen',side_effect=child), \
                    patch('turkeyworld.train_small_enemy_pipeline.time.sleep',side_effect=AssertionError('No real waits')):
                resumed.run()
            self.assertEqual(calls,[('codec',2000),('codec',3000),('sr',2000),('renderer',2000)])
            self.assertEqual(extension.read_bytes(),before);self.assertEqual(resumed.state['completed_stages'],list(ORDER))
            self.assertEqual(resumed.state['status'],'completed');self.assertFalse(resumed.state['automatic_deployment'])
            self.assertFalse(resumed.state['gameplay_qualified']);self.assertEqual(game.read_bytes(),b'unchanged_game_bundle')

    def test_stop_before_start_never_spawns_and_live_process_identity_is_verified_without_os_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.fixture(root);controller=SmallEnemyPipeline(args)
            (controller.root/'stop_pipeline').touch()
            with patch.object(controller,'assets',side_effect=AssertionError('No assets after stop')), \
                    patch('turkeyworld.train_small_enemy_pipeline.subprocess.Popen',side_effect=AssertionError('No stopped worker launch')):
                controller.run()
            self.assertEqual(controller.state['status'],'stopped')
            launch=root/'launch.json';argv=['python','-m','fake_stage','--run-dir','fixed']
            launch.write_text(json.dumps({'pid':12345,'argv':argv}))
            with patch('turkeyworld.train_small_enemy_pipeline.process_alive',return_value=True), \
                    patch('psutil.Process',return_value=SimpleNamespace(cmdline=lambda:argv)):
                self.assertEqual(verified_live_child(launch,argv),12345)
            with patch('turkeyworld.train_small_enemy_pipeline.process_alive',return_value=True), \
                    patch('psutil.Process',return_value=SimpleNamespace(cmdline=lambda:['python','-m','other'])):
                with self.assertRaises(ValueError):verified_live_child(launch,argv)
            with patch('turkeyworld.train_small_enemy_pipeline.process_alive',return_value=True):
                with self.assertRaises(ValueError):verified_live_child(launch,argv+['changed'])
            with patch('turkeyworld.train_small_enemy_pipeline.process_alive',return_value=False):self.assertIsNone(verified_live_child(launch,argv))

    def test_parent_restart_adopts_verified_running_codec_without_duplicate_launch_or_premature_sr(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.fixture(root);controller=SmallEnemyPipeline(args)
            codec=controller.root/'codec';codec.mkdir();alive={'value':True};spawned=[]
            command=lambda self,stage,budget,assets:['fake_worker',stage,str(budget)]
            argv=['fake_worker','codec','2000'];(codec/'launch.json').write_text(json.dumps({'pid':12345,'argv':argv,'budget':2000}))
            def tick(seconds):
                self.assertEqual(spawned,[],'SR must wait until the adopted codec actually exits and passes')
                good,old=reports('codec');(codec/'validation_initial.json').write_text(json.dumps(old))
                (codec/'validation_selected.json').write_text(json.dumps(good));self.finish_stage(codec,'codec',good)
                alive['value']=False
            def child(argv,**kwargs):
                self.assertFalse(alive['value']);stage=argv[1];spawned.append(stage)
                report,baseline=reports(stage);run=controller.root/stage
                (run/'validation_initial.json').write_text(json.dumps(baseline));(run/'validation_selected.json').write_text(json.dumps(report))
                self.finish_stage(run,stage,report)
                return SimpleNamespace(pid=2000+len(spawned),returncode=0,poll=lambda:0)
            with patch.object(SmallEnemyPipeline,'command',command), \
                    patch('turkeyworld.train_small_enemy_pipeline.process_alive',side_effect=lambda pid:alive['value']if pid==12345 else False), \
                    patch('psutil.Process',return_value=SimpleNamespace(cmdline=lambda:argv)), \
                    patch('turkeyworld.train_small_enemy_pipeline.time.sleep',side_effect=tick), \
                    patch('turkeyworld.train_small_enemy_pipeline.subprocess.Popen',side_effect=child):
                controller.run()
            self.assertEqual(spawned,['sr','renderer']);self.assertEqual(controller.state['completed_stages'],list(ORDER))
            self.assertFalse(controller.state['automatic_deployment']);self.assertFalse(controller.state['gameplay_qualified'])

    def test_early_passing_selected_codec_still_waits_for_its_live_milestone_before_sr(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.fixture(root);controller=SmallEnemyPipeline(args)
            codec=controller.root/'codec';codec.mkdir();alive={'value':True};spawned=[];waits=[]
            command=lambda self,stage,budget,assets:['fake_worker',stage,str(budget)]
            argv=['fake_worker','codec','2000']
            (codec/'launch.json').write_text(json.dumps({'pid':12345,'argv':argv,'budget':2000}))
            good,old=reports('codec')
            (codec/'validation_initial.json').write_text(json.dumps(old))
            (codec/'validation_selected.json').write_text(json.dumps(good))
            (codec/'status.json').write_text(json.dumps({'status':'running','current':1700,'total':2000}))
            def tick(seconds):
                self.assertEqual(spawned,[],'An early passing export cannot bypass a live codec milestone')
                waits.append(seconds);alive['value']=False
                self.finish_stage(codec,'codec',good)
            def child(argv,**kwargs):
                self.assertFalse(alive['value'],'Do not run SR concurrently with the still-running codec')
                stage=argv[1];spawned.append(stage);run=controller.root/stage
                report,baseline=reports(stage)
                (run/'validation_initial.json').write_text(json.dumps(baseline))
                (run/'validation_selected.json').write_text(json.dumps(report))
                self.finish_stage(run,stage,report)
                return SimpleNamespace(pid=2000+len(spawned),returncode=0,poll=lambda:0)
            with patch.object(SmallEnemyPipeline,'command',command), \
                    patch('turkeyworld.train_small_enemy_pipeline.process_alive',side_effect=lambda pid:alive['value']if pid==12345 else False), \
                    patch('psutil.Process',return_value=SimpleNamespace(cmdline=lambda:argv)), \
                    patch('turkeyworld.train_small_enemy_pipeline.time.sleep',side_effect=tick), \
                    patch('turkeyworld.train_small_enemy_pipeline.subprocess.Popen',side_effect=child):
                controller.run()
            self.assertTrue(waits);self.assertEqual(spawned,['sr','renderer'])
            self.assertEqual(controller.state['completed_stages'],list(ORDER))

    def test_durable_milestone_and_matching_selected_artifact_are_both_required(self):
        with tempfile.TemporaryDirectory()as folder:
            run=Path(folder);report,_=reports('codec')
            self.assertFalse(milestone_complete('codec',run,2000))
            (run/'status.json').write_text(json.dumps({'status':'running','current':2000}))
            self.assertFalse(milestone_complete('codec',run,2000))
            (run/'status.json').write_text(json.dumps({'status':'completed','current':1999}))
            self.assertFalse(milestone_complete('codec',run,2000))
            (run/'status.json').write_text(json.dumps({'status':'completed','current':2000}))
            self.assertTrue(milestone_complete('codec',run,2000))
            with self.assertRaises(ValueError):validate_stage_artifact('codec',run,report)
            self.finish_stage(run,'codec',report)
            self.assertTrue(validate_stage_artifact('codec',run,report)['sha256'])
            changed=copy.deepcopy(report);changed['panel_fingerprint']='wrong'
            with self.assertRaises(ValueError):validate_stage_artifact('codec',run,changed)
            (run/'launch.json').write_text(json.dumps({'pid':12345}))
            with patch('turkeyworld.train_small_enemy_pipeline.process_alive',return_value=True):
                self.assertFalse(milestone_complete('codec',run,2000))


if __name__=='__main__':unittest.main()
