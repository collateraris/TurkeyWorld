from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.train_rssm_moe import RSSMMoETraining


@dataclass
class Config:
    navigation_only: bool = False
    rssm_five_role_enabled: bool = True


class RSSMFiveTrainingTests(unittest.TestCase):
    def test_inherited_full_checkpoint_save_retains_five_lineage_and_original_named_optimizer(self):
        model = torch.nn.Linear(2, 2); model.config = Config()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model(torch.ones(1, 2)).square().mean().backward(); optimizer.step()
        trainer = RSSMMoETraining.__new__(RSSMMoETraining)
        trainer.state = dict(global_training_step=5939, rssm_training={'moe_training': {'current_horizon': 208}},
                             combat_migration={'old': True}, quality_status='experimental')
        trainer.rng = np.random.default_rng(173); trainer.device = torch.device('cpu')
        trainer.plan_sha = 'old_moe_plan'; trainer._optimizer_parameter_names = ['weight', 'bias']
        trainer.publish = lambda: None
        report = {'kind': 'append_five_role_transformer_fusion', 'source_step': 5938}
        metadata = {'rssm_five_migration': report, 'combat_rssm_migration': {'old': 'combat'}}
        captured = {}
        with tempfile.TemporaryDirectory() as directory:
            trainer.checkpoint = Path(directory) / 'world.pt'
            with patch('turkeyworld.train_rssm_combat._save', side_effect=lambda path, payload: captured.update(payload)):
                trainer.save(model, optimizer, metadata)
        self.assertEqual(captured['rssm_five_migration'], report)
        self.assertEqual(captured['combat_rssm_migration'], metadata['combat_rssm_migration'])
        self.assertEqual(captured['config']['rssm_five_role_enabled'], True)
        self.assertEqual(captured['optimizer_parameter_names'], ['weight', 'bias'])
        self.assertEqual(captured['training_steps'], 5939)
        self.assertEqual(captured['run_state']['rssm_training']['moe_training']['current_horizon'], 208)
        self.assertEqual(set(captured['optimizer']['state']), {0, 1})


if __name__ == '__main__': unittest.main()
