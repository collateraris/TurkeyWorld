import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from turkeyworld.train_sr_combat import clean_quality_gate, resume_revisions, QUALITY_MAE


def report(ssim=.92,mae=2.4/255):
    return dict(step=123,source_held_out=True,world_rollout=False,input_kind='clean_source_lr',
        frames=384,panel_fingerprint='fixed384',summary=dict(ssim_mean=ssim,mae_mean=mae))


class SRCleanQualityTests(unittest.TestCase):
    def test_both_user_criteria_are_strict_and_mae_is_in255_units(self):
        valid=clean_quality_gate(report())
        self.assertTrue(valid['target_met']);self.assertAlmostEqual(valid['current_mean_mae_255'],2.4)
        for ssim,mae in ((.91,2.4/255),(.92,QUALITY_MAE),(.90,.001),(.99,.02)):
            self.assertFalse(clean_quality_gate(report(ssim,mae))['passed'])
        self.assertEqual(valid['thresholds']['ssim_operator'],'>')
        self.assertEqual(valid['thresholds']['mae_operator'],'<')

    def test_missing_nonfinite_wrong_domain_or_wrong_panel_size_never_pass(self):
        for value in (None,{},dict(summary=None),report(float('nan'),.001),report(.95,float('inf')),
                      report(True,.001),report(1.1,.001),report(.99,-.01)):
            self.assertFalse(clean_quality_gate(value)['passed'])
        for key,value in (('frames',383),('source_held_out',False),('input_kind','codec_reconstruction')):
            bad=report();bad[key]=value;self.assertFalse(clean_quality_gate(bad)['passed'])
        bad=report();del bad['summary']['mae_mean'];self.assertFalse(clean_quality_gate(bad)['passed'])

    def test_quality_resume_above10000_bypasses_old_budget_without_changing_adam_or_rng(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'boundary.pt';source.write_bytes(b'exact8672fullcheckpoint')
            old=root/'budget10000.json';old.write_bytes(b'immutable old budget ledger')
            sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            plan=dict(reason='explicit_user_request',from_completed_updates=8672,fixed_max_updates=None,
                validation_every=100,preview_every=100,training_plan_sha256='original',panel_fingerprint='fixed384',
                source_checkpoint=str(source),source_checkpoint_sha256=sha(source),
                previous_budget_plan=str(old),previous_budget_plan_sha256=sha(old),
                criteria=dict(input_kind='clean_source_lr',frames=384,ssim_gt=.91,mae_lt=QUALITY_MAE,
                    mae_255_lt=2.5,required_consecutive_passes=1))
            path=root/'quality.json';path.write_text(json.dumps(plan))
            saved=dict(refinement_updates=12001,model={'weight':torch.tensor([3.])},
                optimizer={'state':{0:{'step':torch.tensor(12001.),'exp_avg':torch.tensor([.2])}}},
                run_state=dict(total=None,training_until_quality=True,
                    training_continuation={'plan_sha256':'old-plan'},quality_revision={'plan_sha256':sha(path)}))
            before=copy.deepcopy(saved);rng=torch.get_rng_state().clone()
            with patch('turkeyworld.train_sr_combat.validate_continuation',side_effect=AssertionError('Old capped plan must be bypassed')):
                continuation,quality=resume_revisions(saved,steps=10000,continuation_plan=old,
                    quality_revision_plan=path,training_plan_sha256='original',panel_fingerprint='fixed384',resume_sha256='new12001')
            self.assertEqual(continuation,{'plan_sha256':'old-plan'})
            self.assertIsNone(quality['fixed_max_updates'])
            self.assertEqual(saved['refinement_updates'],12001)
            self.assertTrue(torch.equal(saved['model']['weight'],before['model']['weight']))
            self.assertTrue(torch.equal(saved['optimizer']['state'][0]['exp_avg'],before['optimizer']['state'][0]['exp_avg']))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            with self.assertRaises(ValueError):
                resume_revisions(saved,steps=10000,continuation_plan=old,quality_revision_plan=path,
                    training_plan_sha256='original',panel_fingerprint='different',resume_sha256='new12001')


if __name__=='__main__':unittest.main()
