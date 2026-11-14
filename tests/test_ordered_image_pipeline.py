from argparse import Namespace
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from turkeyworld.ordered_image_pipeline import(
    OrderedImagePipeline,MonitorMirror,stage_complete,validated_prefix,ORDER,PROTOCOL,read,sha)


class FakePart:
    def __init__(self,value):self.value=value
    def state_dict(self):return {'weight':torch.tensor([float(self.value)])}


class FakeWorld:
    def __init__(self,base=2,flow=3):self.base_student=FakePart(base);self.flow_renderer=FakePart(flow)


class OrderedImagePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.base=self.root/'base';self.base.mkdir()
        (self.base/'budget_3000_plan.json').write_text('{"total_updates":3000}')
        for name in('world.pt','recipe.pt','decoder.pt','sr.pt'):(self.root/name).write_bytes(name.encode())
        source=self.root/'source';source.mkdir();monitor=self.root/'monitor';monitor.mkdir()
        self.args=Namespace(run_dir=self.root/'chain',base_run=self.base,world=self.root/'world.pt',
            source_recipe_parent=self.root/'recipe.pt',decoder_parent=self.root/'decoder.pt',
            sr_parent=self.root/'sr.pt',sr_source_root=source,monitor_control_dir=monitor,device='cpu',poll_seconds=.01)
        self.pipe=OrderedImagePipeline(self.args)

    def tearDown(self):self.temp.cleanup()

    def test_exact_budgets_and_no_skipped_stage_or_changed_resume_plan(self):
        self.assertTrue(stage_complete(dict(status='completed',current=3000,total=3000,quality_passed=False)))
        for state in(dict(status='completed',current=2999,total=3000),dict(status='stopped',current=3000,total=3000),
                     dict(status='completed',current=3001,total=3000),dict(status='completed',current=True,total=3000)):
            self.assertFalse(stage_complete(state))
        stages=deepcopy(self.pipe.state['stages']);stages['rf3']['status']='completed'
        with self.assertRaisesRegex(ValueError,'skipped'):validated_prefix(stages)
        self.pipe.store();altered=deepcopy(self.args);altered.device='cuda'
        with self.assertRaisesRegex(ValueError,'plan changed'):OrderedImagePipeline(altered)

    def seal_base_fixture(self,*,mismatch=False):
        run=self.base;checkpoints=run/'checkpoints';checkpoints.mkdir()
        final=dict(step=3000,base=FakePart(9).state_dict(),flow=FakePart(3).state_dict())
        selected=dict(step=200,base=FakePart(2).state_dict(),flow=FakePart(3).state_dict(),optimizer={'age':200})
        export=dict(format_version=6,local_updates=200)
        torch.save(final,checkpoints/'base_last.pt');torch.save(selected,checkpoints/'base_best.pt')
        torch.save(export,checkpoints/'base_best_for_preview.pt')
        validation=run/'validation';validation.mkdir()
        (validation/'step_000200.json').write_text('{"step":200,"source_held_out":true,"quality_passed":false}')
        (run/'best.json').write_text(json.dumps(dict(step=200,ssim=.7,report_sha256=sha(validation/'step_000200.json'))))
        with patch('turkeyworld.models.load_world_checkpoint',return_value=(FakeWorld(base=1 if mismatch else 2),{})),\
             patch('turkeyworld.renderer_student.load_renderer_student',return_value={'format_version':6}):
            return self.pipe.seal('base')

    def test_handoff_selects_entire_best200_not_latest3000_or_quality_gate(self):
        handoff=self.seal_base_fixture()
        self.assertEqual(handoff['completed_updates'],3000);self.assertEqual(handoff['selected_step'],200)
        saved=torch.load(handoff['files']['training']['path'],map_location='cpu',weights_only=False)
        self.assertEqual(saved['optimizer']['age'],200);self.assertEqual(saved['base']['weight'].item(),2)
        self.assertEqual(self.pipe.state['stages']['base']['current'],3000)
        argv=self.pipe.command('rf3');self.assertIn('turkeyworld.train_renderer_rf_chain',argv)
        self.assertEqual(argv[argv.index('--steps')+1],'3000')
        self.assertEqual(argv[argv.index('--parent-checkpoint')+1],handoff['files']['training']['path'])
        self.assertIn('--probe',self.pipe.command('rf3',probe=True))
        Path(handoff['files']['training']['path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.pipe.command('rf3')

    def test_mixed_best_export_weights_are_rejected_before_next_stage(self):
        with self.assertRaisesRegex(ValueError,'Base weights'):self.seal_base_fixture(mismatch=True)
        self.assertNotEqual(self.pipe.state['stages']['base']['status'],'completed')

    def test_monitor_never_mixes_stage_history_or_partial_lines_or_preview(self):
        mirror=MonitorMirror(self.pipe.root);one=self.root/'one';two=self.root/'two';one.mkdir();two.mkdir()
        (one/'metrics.jsonl').write_bytes(b'{"step":1}\n{"step":')
        self.assertIsNone(mirror.update(one))
        self.assertEqual((self.pipe.root/'metrics.jsonl').read_bytes(),b'{"step":1}\n')
        with(one/'metrics.jsonl').open('ab')as stream:stream.write(b'2}\n')
        (one/'preview.png').write_bytes(b'actual png bytes')
        (one/'preview.json').write_text('{"step":2,"source_held_out":true}')
        os.utime(one/'preview.png',ns=(1_000_000_000,1_000_000_000))
        os.utime(one/'preview.json',ns=(2_000_000_000,2_000_000_000))
        image=mirror.update(one);self.assertEqual(image['step'],2)
        self.assertEqual((self.pipe.root/'metrics.jsonl').read_bytes(),b'{"step":1}\n{"step":2}\n')
        (two/'metrics.jsonl').write_bytes(b'{"step":0}\n')
        self.assertIsNone(mirror.update(two));self.assertFalse((self.pipe.root/'preview.png').exists())
        self.assertEqual((self.pipe.root/'metrics.jsonl').read_bytes(),b'{"step":0}\n')

    def test_failed_or_stopped_child_cannot_advance_and_fixed_budget_can(self):
        (self.base/'status.json').write_text(json.dumps(dict(status='failed',current=100,total=3000,error='bad data')))
        with self.assertRaisesRegex(RuntimeError,'bad data'):self.pipe.wait('base',None)
        self.assertNotEqual(self.pipe.state['stages']['base']['status'],'completed')
        (self.base/'status.json').write_text(json.dumps(dict(status='completed',current=3000,total=3000,quality_passed=False)))
        self.assertTrue(self.pipe.wait('base',None))
        (self.base/'status.json').write_text(json.dumps(dict(status='stopped',current=123,total=3000)))
        self.assertFalse(self.pipe.wait('base',None));self.assertEqual(self.pipe.state['status'],'stopped')

    def test_automatic_order_preflights_and_failure_stop_future_stages(self):
        launched=[]
        def launch(key,*,probe=False):launched.append((key,probe));return 123
        def seal(key):self.pipe.state['stages'][key].update(status='completed',current=3000)
        with patch.object(self.pipe,'launch',side_effect=launch),patch.object(self.pipe,'wait',return_value=True),\
             patch.object(self.pipe,'seal',side_effect=seal),patch.object(self.pipe,'publish'):
            self.pipe.run()
        self.assertEqual(launched,[('base',False),('rf3',True),('rf3',False),('decoder',True),('decoder',False),('sr',True),('sr',False)])
        self.assertEqual(self.pipe.state['status'],'completed')
        self.assertEqual(sum(row['current']for row in self.pipe.state['stages'].values()),12000)
        # A new supervisor with a failed RF preflight must not start decoder/SR.
        self.pipe.state['stages']={k:dict(status='queued',current=0)for k in ORDER};launched.clear()
        def wait(key,pid,*,probe=False):
            if probe:raise RuntimeError('bad RF GPU preflight')
            return True
        with patch.object(self.pipe,'launch',side_effect=launch),patch.object(self.pipe,'wait',side_effect=wait),\
             patch.object(self.pipe,'seal',side_effect=seal),patch.object(self.pipe,'publish'),\
             self.assertRaisesRegex(RuntimeError,'bad RF GPU'):
            self.pipe.run()
        self.assertEqual(launched,[('base',False),('rf3',True)])
        self.assertEqual(self.pipe.state['status'],'failed')


if __name__=='__main__':unittest.main()
