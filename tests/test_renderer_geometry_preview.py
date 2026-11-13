"""Geometry-only previews support empty navigation and fixed stratum order."""
import copy
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from test_renderer_enemy_objectives import tiny_model
from turkeyworld.engine import Game
from turkeyworld.renderer_geometry_phase import geometry_preview_arrays,save_geometry_preview,GEOMETRY_PREVIEW_COLUMNS
from turkeyworld.renderer_native_phase import _same


class GeometryPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_four_actual_empty_source_queries_no_npc_metrics_and_no_mutations(self):
        values=[]
        for index in range(4):
            game=Game(seed=index,enemies_enabled=False,endless=True)
            previous=game.state_vector();before=game.render(size=32,overlays=False)
            action=np.array([.1,.05,.2,0],np.float32)
            game.step(action,dt=1/30)
            rgb=game.render(size=32,overlays=False)
            values.append(dict(state=game.state_vector(),map=game.map.copy(),target=rgb,background=rgb,
                object_mask=np.zeros((32,32),np.float32),previous_state=previous,previous_rgb=before,action=action))
        batch={}
        for key in values[0]:
            value=torch.from_numpy(np.stack([row[key]for row in values]))
            if key in('target','background','previous_rgb'):value=value.permute(0,3,1,2).float()/255
            elif key=='object_mask':value=value[:,None]
            else:value=value.long()if key=='map'else value.float()
            batch[key]=value
        batch['dynamic']=torch.tensor([False,True,False,True])
        class Data:
            split='val'
            rng=np.random.default_rng(173)
            draws=71
            def batch(self,n,indices):return {key:value[indices]for key,value in batch.items()}
        data=Data();model=tiny_model();model.base_student.train()
        saved={k:v.clone()for k,v in model.state_dict().items()}
        modes=[m.training for m in model.modules()]
        rng=(torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate(),copy.deepcopy(data.rng.bit_generator.state))
        with patch('turkeyworld.engine.Game.step',side_effect=AssertionError('No source simulation')), \
                patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No source rendering')), \
                patch('turkeyworld.renderer_enemy_validation.evaluate_enemy_renderer',side_effect=AssertionError('No NPC evaluator')):
            arrays=geometry_preview_arrays(model,data,[0,1,2,3])
        self.assertEqual(set(arrays),{'source','base','rgb','codec_reference'})
        self.assertTrue(torch.equal(arrays['source'],batch['target']))
        self.assertTrue(_same(saved,model.state_dict()));self.assertEqual(modes,[m.training for m in model.modules()])
        self.assertTrue(torch.equal(torch.get_rng_state(),rng[0]));self.assertTrue(_same(np.random.get_state(),rng[1]))
        self.assertEqual(random.getstate(),rng[2]);self.assertEqual(data.rng.bit_generator.state,rng[3]);self.assertEqual(data.draws,71)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_png_columns_preserve_pinned_order_without_npc_reselection_and_mirror_exact(self):
        panel=[13,99,70,20]; representatives=[99,20,70,13]
        receipt=dict(split='val',preview_indices=representatives,panel_fingerprint='a'*64,
            stratum_indices={key:[index]for key,index in zip(GEOMETRY_PREVIEW_COLUMNS,representatives)})
        colors={99:30,20:80,70:150,13:220}
        images=torch.stack([torch.full((3,32,32),colors[index]/255)for index in panel])
        arrays={key:images.clone()for key in ('source','base','rgb','codec_reference')}
        with TemporaryDirectory()as folder:
            path=Path(folder)/'preview_geometry.png';mirror=Path(folder)/'preview.png'
            metadata=save_geometry_preview(arrays,path,6010,panel=panel,receipt=receipt,mirror=mirror)
            self.assertEqual(path.read_bytes(),mirror.read_bytes())
            image=np.asarray(Image.open(path))
            self.assertEqual(image.shape,(740,640,3))
            for column,index in enumerate(representatives):
                self.assertEqual(image[105,column*160+80].tolist(),[colors[index]]*3)
            self.assertEqual(metadata['columns'],list(GEOMETRY_PREVIEW_COLUMNS))
            self.assertEqual(metadata['source_query_indices'],representatives)
            self.assertEqual(metadata['geometry_wall_fingerprint'],'a'*64)
            self.assertFalse(metadata['full_validation']);self.assertFalse(metadata['source_quality_gate_evaluated'])
            bad=copy.deepcopy(receipt);bad['preview_indices']=list(reversed(representatives))
            with self.assertRaises(ValueError):save_geometry_preview(arrays,path,6010,panel=panel,receipt=bad)


if __name__=='__main__':unittest.main()
