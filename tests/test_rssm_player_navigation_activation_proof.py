"""The read-only activation proof must detect one changed tensor or old RNG."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from turkeyworld.rssm_player_navigation_revision import (
    PROTOCOL,FULL_HORIZON_PROTOCOL,digest,file_sha,player_navigation_settings,
)


class PlayerActivationProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'training_runs/diagnostic_tools/verify_player_navigation_revision_20261114.py'
        spec = importlib.util.spec_from_file_location('player_navigation_activation_proof', path)
        cls.proof = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.proof)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.child, self.path = (self.root / name for name in ('source.pt', 'child.pt', 'plan.json'))
        self.settings = player_navigation_settings()
        gate = dict(updates=1166, current_horizon=450, final_horizon_goal=dict(streak=0),
                    samplers=dict(old=dict(draws=777, rng=dict(state=42))))
        old = dict(global_training_step=6518, rssm_training=dict(moe_training=gate,
                   navigation_replay_revision=dict(draws=64, rng=dict(state=43))),
                   loss=3.95, latest_metrics=dict(train_loss=3.95), objective_history=[dict(old=True)],
                   rssm_spawn_conditioning_training=dict(current_step=6518))
        source = dict(run_state=old, global_training_step=6518,
            model={str(i): torch.tensor([i], dtype=torch.float32) for i in range(1599)},
            optimizer=dict(state={i:dict(step=torch.tensor(1.), exp_avg=torch.tensor([.1]),
                                        exp_avg_sq=torch.tensor([.01])) for i in range(442)},
                           param_groups=[dict(params=list(range(i, 442, 7)), lr=1e-4) for i in range(7)]),
            optimizer_parameter_names=[str(i) for i in range(442)], config=dict(memory=5232),
            state_schema=dict(size=20), rssm_spawn_conditioning_migration=dict(pinned=True),
            rng_state=np.random.default_rng(173).bit_generator.state,
            torch_rng_state=torch.get_rng_state(), cuda_rng_state=[])
        torch.save(source, self.source)
        identity = dict(protocol='contact_test', cohorts=dict(both_free=4))
        self.plan = dict(kind=PROTOCOL,version=1,settings=self.settings, contact_sampler_identity=identity,
            source_checkpoint=str(self.source.resolve()), source_checkpoint_sha256=file_sha(self.source),
            source_optimizer_step=6518, source_gate_sha256=digest(gate),
            source_H450_goal_sha256=digest(gate['final_horizon_goal']))
        self.path.write_text(json.dumps(self.plan), encoding='utf-8')
        child = deepcopy(source); state = child['run_state']
        state['training_protocol'] = dict(tbptt=64)
        state['active_objective_formula_revision'] = PROTOCOL
        identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False,
                                                separators=(',', ':')).encode()).hexdigest()
        contact = dict(draws=0, round_robin=0, cohort_draws=dict(both_free=0),
            seed=self.settings['contact']['seed'], identity_sha256=identity_sha,
            rng=np.random.default_rng(self.settings['contact']['seed']).bit_generator.state)
        state['rssm_training']['player_navigation_revision'] = dict(
            protocol=PROTOCOL, sha256=file_sha(self.path), settings=deepcopy(self.settings),
            source_optimizer_step=6518, last_optimizer_step=6518, updates=0,
            contact_draws=0, contact_sampler_state=contact,
            contact_sampler_identity_sha256=digest(identity), baseline_complete=False)
        state['objective_history'].append(dict(kind=PROTOCOL, previous_TOTAL_comparable=False,
            source_loss=old['loss'], source_latest_metrics=deepcopy(old['latest_metrics'])))
        self.payload = child
        torch.save(child, self.child)

    def test_complete_boundary_passes_without_writes(self):
        before = [file_sha(path) for path in (self.source, self.child, self.path)]
        value = self.proof.verify(self.source, self.child, self.path)
        self.assertTrue(value['passed'], value['checks'])
        self.assertEqual((value['model_tensor_count'], value['Adam_state_count'], value['Adam_group_count']), (1599, 442, 7))
        self.assertFalse(value['production_files_mutated'])
        self.assertEqual(before, [file_sha(path) for path in (self.source, self.child, self.path)])
        json.dumps(value, allow_nan=False)

    def test_changed_model_old_sampler_or_rng_cannot_pass(self):
        for name, mutate, check in (
            ('model', lambda p:p['model']['17'].add_(1), 'model_all_tensors_bitexact'),
            ('sampler', lambda p:p['run_state']['rssm_training']['moe_training']['samplers']['old'].update(draws=778),
             'old_MoE_gate_and_sampler_contents_preserved'),
            ('rng', lambda p:p['rng_state']['state'].update(state=42), 'numpy_training_RNG_bitexact'),
        ):
            value = deepcopy(self.payload); mutate(value); torch.save(value, self.child)
            with self.subTest(name=name):
                result = self.proof.verify(self.source, self.child, self.path)
                self.assertFalse(result['passed'])
                self.assertFalse(result['checks'][check])

    def test_explicit_full450_smaller_batch_preserves_raw_source_gate(self):
        from turkeyworld.rssm_moe_batch_revision import MICRO32_PROTOCOL,MICRO16_SINGLE_PROTOCOL
        source=torch.load(self.source,map_location='cpu',weights_only=False)
        raw=source['run_state']['rssm_training']['moe_training']
        raw.update(batch_protocol=MICRO32_PROTOCOL,batch_revision=dict(sha256='a'*64))
        torch.save(source,self.source)
        child=deepcopy(source);state=child['run_state'];gate=state['rssm_training']['moe_training']
        gate.update(batch_protocol=MICRO16_SINGLE_PROTOCOL,batch_revision=dict(
            sha256='b'*64,source_checkpoint_sha256=file_sha(self.source),
            source_gate=deepcopy(raw),source_gate_sha256=digest(raw),
            previous_batch_revision_sha256='a'*64,updates=0,main_examples=0,main_transitions=0))
        settings=player_navigation_settings(tbptt=450,microbatch_size=16)
        plan=deepcopy(self.plan)
        plan.update(kind=FULL_HORIZON_PROTOCOL,version=2,settings=settings,
            source_checkpoint_sha256=file_sha(self.source),source_gate_sha256=digest(gate),
            source_H450_goal_sha256=digest(raw['final_horizon_goal']),
            source_pre_batch_gate_sha256=digest(raw),batch_revision_sha256='b'*64)
        self.path.write_text(json.dumps(plan),encoding='utf-8')
        state['training_protocol']=dict(tbptt=450)
        state['active_objective_formula_revision']=FULL_HORIZON_PROTOCOL
        revision=deepcopy(self.payload['run_state']['rssm_training']['player_navigation_revision'])
        revision.update(protocol=FULL_HORIZON_PROTOCOL,settings=settings,sha256=file_sha(self.path))
        state['rssm_training']['player_navigation_revision']=revision
        state['objective_history'].append(dict(kind=FULL_HORIZON_PROTOCOL,previous_TOTAL_comparable=False,
            source_loss=source['run_state']['loss'],source_latest_metrics=deepcopy(source['run_state']['latest_metrics'])))
        torch.save(child,self.child)
        result=self.proof.verify(self.source,self.child,self.path)
        self.assertTrue(result['passed'],result['checks'])
        self.assertEqual(result['new_TBPTT'],450)
        self.assertTrue(result['checks']['explicit_B32_to_B16_lineage'])


if __name__ == '__main__': unittest.main()
