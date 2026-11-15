"""Optional cycle telemetry must never cancel a real motion checkpoint boundary."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.train_rssm_motion_only import MotionOnlyTraining


@dataclass
class TinyConfig:
    architecture:str='dreamer_rssm'
    navigation_only:bool=False


class TinyWorld(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor([1.,2.]));self.config=TinyConfig()


class MotionCycleTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.cycle=self.root/'cycle'/'status.json';self.cycle.parent.mkdir()
        self.cached=dict(protocol='dreamer_motion_Source_world_policy_cycle_v1',status='running',
            phase='world_training',round_index=3,world_updates=178)
        self.cycle.write_text(json.dumps(self.cached),encoding='utf-8')
        worker=MotionOnlyTraining.__new__(MotionOnlyTraining)
        worker.root=self.root/'motion';worker.root.mkdir();worker.args=SimpleNamespace(cycle_status=self.cycle)
        worker.device=torch.device('cpu');worker.rng=np.random.default_rng(173);worker.plan_sha='a'*64
        worker.checkpoint=worker.root/'checkpoints/rssm_world.pt';worker._optimizer_parameter_names=['weight']
        worker.state=dict(status='running',global_training_step=7591,current=7591,loss=.25,
            latest_metrics=dict(step=7591,moe_player_loss=.1,moe_enemy_loss=.15),
            rssm_training=dict(motion_only_revision=dict(updates=718)),dreamer_cycle=deepcopy(self.cached),
            quality_status='Held-out quality is measured separately',combat_migration=dict(source_step=6873))
        self.worker=worker;self.model=TinyWorld();self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3)
        self.optimizer.zero_grad();self.model.weight.square().sum().backward();self.optimizer.step()
        self.enterContext(patch('turkeyworld.dreamer_motion_cycle.time.sleep',return_value=None))

    def locked_cycle(self,*,transient=0):
        original=Path.read_text;attempts=[]
        def read_text(path,*args,**kwargs):
            if Path(path)==self.cycle:
                attempts.append(1)
                if len(attempts)>64:raise AssertionError('Optional telemetry retries must be bounded')
                if not transient or len(attempts)<=transient:
                    raise PermissionError(13,'Windows sharing lock on optional cycle status',str(path))
            return original(path,*args,**kwargs)
        return patch.object(Path,'read_text',autospec=True,side_effect=read_text),attempts

    def warning(self):
        warning=self.worker.state.get('cycle_telemetry_warning')
        self.assertTrue(warning,'An optional read failure needs a visible nonfatal warning')
        return warning

    def test_transient_cycle_read_recovers_new_telemetry_and_publishes_current_training_state(self):
        newer=dict(self.cached,world_updates=179);self.cycle.write_text(json.dumps(newer),encoding='utf-8')
        mocked,attempts=self.locked_cycle(transient=2)
        with mocked:self.worker.publish()
        self.assertEqual(len(attempts),3);self.assertEqual(self.worker.state['dreamer_cycle'],newer)
        published=json.loads((self.worker.root/'status.json').read_text(encoding='utf-8'))
        self.assertEqual(published['global_training_step'],7591);self.assertEqual(published['status'],'running')
        self.assertEqual(published['dreamer_cycle'],newer)

    def test_persistent_optional_lock_preserves_last_good_cycle_and_actual_training_progress(self):
        self.worker.state.update(global_training_step=7592,current=7592,loss=.125)
        mocked,attempts=self.locked_cycle()
        with mocked:self.worker.publish()
        self.assertGreater(len(attempts),1);self.assertLessEqual(len(attempts),64)
        self.assertEqual(self.worker.state['dreamer_cycle'],self.cached);self.assertNotIn('error',self.worker.state)
        self.assertIn('PermissionError',str(self.warning()))
        published=json.loads((self.worker.root/'status.json').read_text(encoding='utf-8'))
        self.assertEqual((published['status'],published['global_training_step'],published['loss']),('running',7592,.125))
        self.assertEqual(published['dreamer_cycle'],self.cached);self.assertTrue(published['cycle_telemetry_warning'])

    def test_actual_adam_checkpoint_save_completes_while_optional_cycle_json_is_locked(self):
        self.worker.save(self.model,self.optimizer,{})
        self.optimizer.zero_grad();self.model.weight.square().sum().backward();self.optimizer.step()
        self.worker.state.update(global_training_step=7592,current=7592,loss=.125)
        mocked,attempts=self.locked_cycle()
        with mocked:self.worker.save(self.model,self.optimizer,{})
        self.assertGreater(len(attempts),1);self.warning()
        saved=torch.load(self.worker.checkpoint,map_location='cpu',weights_only=False)
        self.assertEqual((saved['training_steps'],saved['global_training_step']),(7592,7592))
        self.assertEqual(saved['run_state']['global_training_step'],7592)
        self.assertTrue(torch.equal(saved['model']['weight'],self.model.weight.detach()))
        live=self.optimizer.state_dict()
        for key in('step','exp_avg','exp_avg_sq'):
            self.assertTrue(torch.equal(saved['optimizer']['state'][0][key],live['state'][0][key]),key)
        self.assertEqual(saved['optimizer_parameter_names'],['weight'])
        self.assertEqual(json.loads((self.worker.root/'status.json').read_text())['global_training_step'],7592)

    def test_malformed_or_nondictionary_optional_json_never_replaces_cached_cycle(self):
        for content in('{not valid JSON','[]','null','"wrong shape"','42','{}','{"protocol":"wrong"}'):
            self.cycle.write_text(content,encoding='utf-8')
            with self.subTest(content=content):
                self.worker.publish();self.warning()
                self.assertEqual(self.worker.state['dreamer_cycle'],self.cached)
                published=json.loads((self.worker.root/'status.json').read_text())
                self.assertEqual(published['status'],'running');self.assertEqual(published['dreamer_cycle'],self.cached)

    def test_good_cycle_read_after_lock_clears_stale_warning_without_resetting_training(self):
        mocked,_=self.locked_cycle()
        with mocked:self.worker.publish()
        self.warning();newer=dict(self.cached,world_updates=179);self.cycle.write_text(json.dumps(newer),encoding='utf-8')
        self.worker.state.update(global_training_step=7592,current=7592);self.worker.publish()
        self.assertNotIn('cycle_telemetry_warning',self.worker.state)
        self.assertEqual(self.worker.state['dreamer_cycle'],newer)
        published=json.loads((self.worker.root/'status.json').read_text())
        self.assertEqual(published['global_training_step'],7592);self.assertNotIn('cycle_telemetry_warning',published)

    def test_missing_optional_cycle_file_warns_but_keeps_publishing_last_good_cycle(self):
        self.cycle.unlink();self.worker.publish()
        self.warning();self.assertEqual(self.worker.state['dreamer_cycle'],self.cached)
        self.assertEqual(json.loads((self.worker.root/'status.json').read_text())['status'],'running')

    def test_first_optional_read_failure_without_cached_cycle_still_publishes_real_state(self):
        self.worker.state.pop('dreamer_cycle');mocked,_=self.locked_cycle()
        with mocked:self.worker.publish()
        self.warning();published=json.loads((self.worker.root/'status.json').read_text())
        self.assertEqual(published['global_training_step'],7591);self.assertNotIn('dreamer_cycle',published)

    def test_authoritative_status_write_error_is_not_swallowed_as_optional_telemetry(self):
        with patch('turkeyworld.train_rssm_dynamics.atomic_json',side_effect=PermissionError(13,'Own status is unavailable')):
            with self.assertRaises(PermissionError):self.worker.publish()
        self.assertEqual(self.worker.state['dreamer_cycle'],self.cached)

    def test_authoritative_checkpoint_replace_failure_raises_and_retains_last_good_checkpoint(self):
        self.worker.save(self.model,self.optimizer,{});before=self.worker.checkpoint.read_bytes()
        self.worker.state.update(global_training_step=7592,current=7592)
        with patch('turkeyworld.distill_renderer.replace_file',side_effect=PermissionError(13,'Own checkpoint is locked')):
            with self.assertRaises(PermissionError):self.worker.save(self.model,self.optimizer,{})
        self.assertEqual(self.worker.checkpoint.read_bytes(),before)
        saved=torch.load(self.worker.checkpoint,map_location='cpu',weights_only=False)
        self.assertEqual(saved['training_steps'],7591)

    def test_unconfigured_optional_mirror_does_not_block_ordinary_status_publication(self):
        self.worker.args.cycle_status=None;self.worker.publish()
        self.assertEqual(json.loads((self.worker.root/'status.json').read_text())['global_training_step'],7591)


if __name__=='__main__':unittest.main()
