"""Real CPU SGD and exact resume of the visual-first continuation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from turkeyworld.enemy_decoder_migration import prepare_visual_first_continuation
from turkeyworld.models import build_upscaler
from turkeyworld.train_semantic_sr import run
from turkeyworld.visual_first_objectives import PROTOCOL
import test_enemy_decoder_training as guided_training
import test_semantic_sr_training as semantic_training


class VisualPairs(guided_training.GuidedPairs):
    def batch(self,size,*,indices=None,rng=None):
        result=super().batch(size,indices=indices,rng=rng)
        result['cohort']=[('near'if i==0 else'mid'if i<4 else'far'if i<6 else'guard')
            for i in result['row_index'].tolist()]
        result['stratum']=['occluded'if i>=6 else'17_64'for i in result['row_index'].tolist()]
        return result


def evaluate(model,data,panel,**kwargs):
    report,preview=guided_training.guided_evaluate(model,data,panel,**kwargs)
    if model.config.get('enemy_decoder_visual_first'):
        report['kind']='visual_first_enemy_decoder_super_resolution_v2'
    return report,preview


def invoke(args):
    with patch('turkeyworld.enemy_decoder_data.EnemyDecoderData',VisualPairs), \
            patch('turkeyworld.semantic_sr_validation.evaluate_semantic_sr',evaluate), \
            patch('turkeyworld.semantic_sr_validation.save_semantic_sr_preview',semantic_training.preview):
        run(args)


class VisualFirstTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_two_real_updates_resume_exactly_preserve_active_sampler_and_parent_Adam(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);helper=guided_training.EnemyDecoderTrainingTests()
            with patch.object(semantic_training,'build_upscaler',
                    side_effect=lambda config:build_upscaler({**config,'blocks':16})):
                stable_initial,selected,_,_=helper.migrated(root,stable=True)
            hints=root/'hypotheses'
            stable_args=helper.args(root/'stable_parent',stable_initial,selected,2,hints)
            stable_args.batch=8;invoke(stable_args)
            source_path=root/'stable_parent/checkpoints/sr_last.pt'
            source=torch.load(source_path,map_location='cpu',weights_only=False)
            self.assertEqual(source['sampler']['draws'],16)
            self.assertEqual(len(source['optimizer']['state']),149)
            initial=root/'visual_first_initial.pt'
            report=prepare_visual_first_continuation(source_path,initial,seed=173)
            migrated=torch.load(initial,map_location='cpu',weights_only=False)
            self.assertTrue(report['sampler_preserved'])
            self.assertEqual(migrated['sampler'],source['sampler'])
            self.assertEqual(migrated['visual_first_updates'],0)
            self.assertEqual(migrated['parent_training_steps'],source['training_steps'])
            self.assertEqual(migrated['parent_decoder_updates'],source['decoder_updates'])
            for key in ('rng_numpy','rng_python'):
                self.assertEqual(migrated[key],source[key])
            torch.testing.assert_close(migrated['rng_torch'],source['rng_torch'],rtol=0,atol=0)
            for name,value in source['model'].items():
                torch.testing.assert_close(value,migrated['model'][name],rtol=0,atol=0)
            for index,row in source['optimizer']['state'].items():
                for name,value in row.items():
                    torch.testing.assert_close(value,migrated['optimizer']['state'][index][name],rtol=0,atol=0)

            def args(run_dir,updates):
                value=helper.args(run_dir,initial,selected,updates,hints)
                value.visual_first=True;value.batch=8
                return value
            invoke(args(root/'continuous',2))
            invoke(args(root/'resumed',1));invoke(args(root/'resumed',2))
            a=torch.load(root/'continuous/checkpoints/sr_last.pt',map_location='cpu',weights_only=False)
            b=torch.load(root/'resumed/checkpoints/sr_last.pt',map_location='cpu',weights_only=False)
            self.assertEqual((a['semantic_updates'],a['decoder_updates'],a['visual_first_updates']),(2,2,2))
            self.assertEqual(a['training_steps'],source['training_steps']+2)
            self.assertEqual(a['parent_decoder_updates'],2)
            self.assertEqual(a['sampler']['draws'],source['sampler']['draws']+16)
            self.assertEqual(a['parent_sampler'],source['sampler'])
            for key in ('optimizer_parameter_names','sampler','rng_numpy','rng_python'):
                self.assertEqual(a[key],b[key])
            torch.testing.assert_close(a['rng_torch'],b['rng_torch'],rtol=0,atol=0)
            self.assertEqual(a['optimizer']['param_groups'],b['optimizer']['param_groups'])
            for name,value in a['model'].items():
                torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
            for index,row in a['optimizer']['state'].items():
                for name,value in row.items():
                    torch.testing.assert_close(value,b['optimizer']['state'][index][name],rtol=0,atol=0)

            frozen=report['frozen_parent_parameter_names'];self.assertEqual(len(frozen),102)
            for name in frozen:
                torch.testing.assert_close(source['model'][name],a['model'][name],rtol=0,atol=0)
            source_ids={name:index for names,group in zip(source['optimizer_parameter_names'],source['optimizer']['param_groups'])
                for name,index in zip(names,group['params'])}
            for name in frozen:
                index=source_ids[name]
                for key,value in source['optimizer']['state'][index].items():
                    torch.testing.assert_close(value,a['optimizer']['state'][index][key],rtol=0,atol=0)
            for prefix in ('enemy_decoder_visual_condition.','enemy_decoder_rgba.'):
                self.assertTrue(any(not torch.equal(value,a['model'][name])for name,value in migrated['model'].items()
                    if name.startswith(prefix)),prefix)
            rows=[json.loads(line)for line in(root/'resumed/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['step']for r in rows],[1,2])
            for row in rows:
                self.assertEqual(row['objective_protocol'],PROTOCOL)
                self.assertEqual(row['hypotheses_deliberately_dropped'],4)
                self.assertEqual(row['mismatched_predicted_hypotheses'],2)
                self.assertEqual(row['actual_hypotheses_available'],4)
                self.assertEqual(row['input_domain_counts'],{'npr':6,'clean':1,'codec':1})
                self.assertEqual(row['vision_canonical_frames'],4)
                self.assertFalse(row['future_Source_states_used_as_input'])
                self.assertTrue(row['RSSM_hint_optional'])
                self.assertTrue(row['visual_can_render_without_hint'])
            self.assertGreater(sum(r['vision_missing_hint_visible_frames']for r in rows),0)
            status=json.loads((root/'resumed/status.json').read_text())
            self.assertEqual(status['semantic_sr_training']['protocol'],'visual_first_enemy_decoder_super_resolution_v2')
            self.assertFalse(status['gameplay_qualified']);self.assertFalse(status['automatic_deployment'])


if __name__=='__main__':unittest.main()
