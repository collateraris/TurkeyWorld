from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from turkeyworld.cache_enemy_decoder_hypotheses import build_hypothesis_cache
from turkeyworld.cache_sr_neural_pairs import build_cache
from turkeyworld.enemy_decoder_data import EnemyDecoderData,hr_box_and_visibility
from turkeyworld.enemy_animation import enemy_animation_index
from turkeyworld.small_enemy_data import digest
from turkeyworld.sr_neural_pairs_data import collect_source_pairs
from test_cache_sr_neural_pairs import FrozenRenderer,identity
from test_sr_neural_pairs_data import source_snapshots


def paired_fixture(root):
    source=Path(root)/'source'
    for split in ('train','val'):
        source_snapshots(source,split)
    source_pairs=Path(root)/'source_pairs'
    collect_source_pairs(source_pairs,[source],train_count=8,val_count=8,workers=1,shard_size=3)
    paired=Path(root)/'paired'
    build_cache(source_pairs,paired,FrozenRenderer(),cache_identity=identity(),batch_size=2)
    return paired


class EnemyDecoderDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_hr_box_visibility_shape_and_empty_zero(self):
        masks=torch.zeros(3,1,128,128)
        masks[0,0,10:20,30:50]=1; masks[1,0,127,127]=1
        boxes,visible=hr_box_and_visibility(masks)
        self.assertEqual(boxes.shape,(3,4));self.assertEqual(visible.shape,(3,1))
        self.assertEqual(visible.dtype,torch.bool)
        self.assertTrue(torch.equal(boxes[0],torch.tensor([30,10,50,20])/128))
        self.assertTrue(torch.equal(boxes[1],torch.tensor([127,127,128,128])/128))
        self.assertTrue(torch.equal(boxes[2],torch.zeros(4)))
        self.assertEqual(visible[:,0].tolist(),[True,True,False])

    def test_immutable_sidecar_reader_static_fallback_labels_and_actual_balancing_resume(self):
        with TemporaryDirectory()as directory:
            paired=paired_fixture(directory);base_manifest=(paired/'manifest.json').read_bytes()
            source_shas={str(p):digest(p)for p in paired.rglob('*.npz')}
            hypotheses=Path(directory)/'hypotheses'
            build_hypothesis_cache(paired,hypotheses,FrozenRenderer(),world_sha256='frozen-world',batch_size=2)
            train=EnemyDecoderData(paired,'train',hypotheses_root=hypotheses)
            val=EnemyDecoderData(paired,'val',hypotheses_root=hypotheses)
            self.assertEqual(len(val.fixed_panel()),8)
            before=train.sampler_state();base_before=train.base.sampler_state()
            explicit=train.batch(8,indices=list(range(8)))
            self.assertEqual(train.sampler_state(),before);self.assertEqual(train.base.sampler_state(),base_before)
            self.assertEqual(explicit['hypothesis'].shape,(8,21));self.assertFalse(explicit['hypothesis'].any())
            self.assertEqual(explicit['visible_labels'].shape,(8,1))
            self.assertTrue(torch.equal(explicit['animation_label'],enemy_animation_index(explicit['state'])))
            self.assertTrue(torch.equal(explicit['visible_labels'],explicit['mask_hr'].bool().flatten(1).any(1)[:,None]))
            self.assertTrue(torch.equal(explicit['bbox_hr'][~explicit['visible_labels'][:,0]],
                torch.zeros((int((~explicit['visible_labels'][:,0]).sum()),4))))
            state=train.sampler_state();first=train.batch(120)['row_index'].tolist()
            self.assertEqual(train.case_counts,{'visible':60,'empty':60})
            self.assertEqual(train.subcounts['visible'],{'near':20,'mid':20,'far':20})
            self.assertEqual({key:sum(values.values())for key,values in train.animation_subcounts.items()},
                {'near':20,'mid':20,'far':20})
            train.restore_sampler(state);self.assertEqual(first,train.batch(120)['row_index'].tolist())
            self.assertEqual(train.base.sampler_state(),base_before)
            self.assertEqual(base_manifest,(paired/'manifest.json').read_bytes())
            self.assertTrue(all(digest(p)==sha for p,sha in source_shas.items()))
            json.dumps(train.identity,allow_nan=False);json.dumps(train.sampler_state(),allow_nan=False)
            bad=deepcopy(train.sampler_state());bad['subcounts']['visible']['near']+=1
            with self.assertRaisesRegex(ValueError,'strata counters'):train.restore_sampler(bad)
            with self.assertRaisesRegex(ValueError,'identity'):
                train.restore_sampler(train.base.sampler_state())
            bad=deepcopy(train.sampler_state())
            key=next(iter(bad['animation_subcounts']['near']))
            bad['animation_subcounts']['near'][key]+=1
            with self.assertRaisesRegex(ValueError,'animation counters'):train.restore_sampler(bad)

    def test_sidecar_cannot_be_rebound_to_another_base_manifest(self):
        with TemporaryDirectory()as directory:
            paired=paired_fixture(directory);hypotheses=Path(directory)/'hypotheses'
            build_hypothesis_cache(paired,hypotheses,FrozenRenderer(),world_sha256='frozen-world',batch_size=2)
            manifest=paired/'manifest.json';value=json.loads(manifest.read_text());value['unrelated_phase']='different'
            manifest.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'exact immutable paired corpus'):
                EnemyDecoderData(paired,'train',hypotheses_root=hypotheses)


if __name__=='__main__':unittest.main()
