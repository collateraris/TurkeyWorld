from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_train_base_geometry as support
from turkeyworld.base_geometry_budget import base_budget_revision_plan,bind_base_budget_revision


class BaseBudgetTests(unittest.TestCase):
    def source(self):
        return dict(step=600,base_training_steps=29809,plan_sha256='a'*64,
            phase=dict(protocol=dict(name='base_expanded_geometry_v1'),configuration=dict(steps=2000)),
            optimizer={'preserved':1},sampler={'PCG':'same'},best={'step':300,'ssim':.75})

    def test_exact_source_and_pins_extend_TOTAL_without_mutating_any_old_training_record(self):
        saved=self.source();before=deepcopy(saved)
        plan=base_budget_revision_plan(saved,source_checkpoint='immutable.pt',source_checkpoint_sha256='b'*64)
        record=bind_base_budget_revision(saved,plan,path=Path('budget.json'),sha256='c'*64,loaded_checkpoint_sha256='b'*64)
        self.assertEqual(saved,before);self.assertEqual(record['total_updates'],3000)
        resumed=deepcopy(saved);resumed.update(step=2500,base_budget_revision=record)
        self.assertEqual(bind_base_budget_revision(resumed,plan,path=Path('budget.json'),sha256='c'*64,
            loaded_checkpoint_sha256='d'*64),record)
        for mutate in (lambda p:p.update(total_updates=5000),lambda p:p.update(optimizer_reset=True),
                       lambda p:p.update(phase_plan_sha256='e'*64),lambda p:p.update(source_step=601)):
            bad=deepcopy(plan);mutate(bad)
            with self.assertRaises(ValueError):bind_base_budget_revision(saved,bad,path='budget.json',sha256='c'*64,
                loaded_checkpoint_sha256='b'*64)
        with self.assertRaises(ValueError):bind_base_budget_revision(saved,plan,path='budget.json',sha256='c'*64,
            loaded_checkpoint_sha256='d'*64)
        with self.assertRaises(ValueError):bind_base_budget_revision(resumed,plan,path='budget.json',sha256='e'*64,
            loaded_checkpoint_sha256='d'*64)

    def test_actual_driver_passes2000_and_stops_exactly3000_counting_old_updates(self):
        with tempfile.TemporaryDirectory()as root:
            trainer=support.Loop(root,start=1999,steps=2000,scores={s:.75 for s in range(2000,3001,100)})
            trainer.effective_steps=3000;trainer.state['total']=3000
            with patch('turkeyworld.train_base_geometry.evaluate_base_geometry',side_effect=support.evaluator(trainer)):
                trainer.run()
            self.assertEqual(trainer.updates,list(range(2000,3001)))
            self.assertEqual(trainer.step,3000);self.assertEqual(trainer.state['status'],'completed')
            self.assertFalse(trainer.state['gameplay_qualified'])
            self.assertEqual(trainer.state['latest_metrics']['renderer_training_steps'],30717)


if __name__=='__main__':unittest.main()
