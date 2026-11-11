"""CPU private-copy integration: new Q-compatible E/D, old Adam, new Source pool."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.train_renderer_enemy import EnemyRendererTraining
from turkeyworld.renderer_student import _hash_tensors
from test_renderer_enemy_objectives import tiny_model,batch as source_batch


class TinySource:
    def __init__(self,root,split,seed=173,**kwargs):
        self.split=split;self.rng=np.random.default_rng(seed);self.draws=0
        self.rows=[{'category':name}for name in ('tiny','near','corpse','navigation','hp0')]
        self.files=[{'path':str(Path(root)/split/f'{i}.npz'),'seed':i+(0 if split=='train'else 100),
            'sha256':split+str(i)}for i in range(5)]
        self.identity={'fingerprint':split,'category_counts':dict.fromkeys((r['category']for r in self.rows),1)}
        self.scene_keys={split};self.case_counts=dict.fromkeys((r['category']for r in self.rows),0)
    def fixed_panel(self,count,seed=173):return list(range(count))
    def panel_fingerprint(self,ids):return self.split+str(ids)
    def batch(self,n,indices=None):
        if indices is None:
            indices=self.rng.integers(5,size=n).tolist();self.draws+=n
            for i in indices:self.case_counts[self.rows[i]['category']]+=1
        values=source_batch(tuple(False for _ in range(n)))
        values['object_mask'].zero_()
        for j,i in enumerate(indices):
            values['state'][j,:2]=.2;values['state'][j,4]=1
            values['state'][j,8:10]=torch.tensor([.4,.2]);values['state'][j,12:14]=torch.tensor([.6,1.])
            if i==0:values['object_mask'][j,:,10,10]=1
            if i==1:values['object_mask'][j,:,8:20,8:20]=1;values['state'][j,8]=.2+1/16
            if i==2:values['object_mask'][j,:,8:12,8:12]=1;values['state'][j,12:14]=0
            if i==3:values['state'][j,12:14]=0
            if i==4:values['state'][j,4]=0
        values['row_index']=torch.tensor(indices)
        return values
    def sampler_state(self):return {'rng':self.rng.bit_generator.state,'draws':self.draws,'case_counts':self.case_counts.copy()}
    def restore_sampler(self,saved):
        self.rng.bit_generator.state=saved['rng'];self.draws=saved['draws'];self.case_counts=saved['case_counts'].copy()


class SmallRendererTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_real_tiny_loop_restores8000_adam_without_overwriting_new_codec_and_counts_once(self):
        with TemporaryDirectory()as temporary:
            root=Path(temporary);template=tiny_model()
            optimizer=torch.optim.AdamW([{'params':template.base_student.parameters(),'lr':1e-4},
                {'params':[p for p in template.flow_renderer.parameters()if p.requires_grad],'lr':1e-4}],weight_decay=.01)
            sum(p.square().sum()for g in optimizer.param_groups for p in g['params']).backward();optimizer.step()
            for entry in optimizer.state.values():entry['step'].fill_(8000)
            saved={'base':copy.deepcopy(template.base_student.state_dict()),'flow':copy.deepcopy(template.flow_renderer.state_dict()),
                'optimizer':copy.deepcopy(optimizer.state_dict()),'step':8000,
                'sampler':{'parent':{'rng':np.random.default_rng(24).bit_generator.state}},
                'rng_torch':torch.get_rng_state(),'rng_numpy':np.random.get_state(),'rng_python':__import__('random').getstate(),'rng_cuda':[]}
            parent=root/'parent.pt';torch.save(saved,parent)
            # Only the compact E/D changes; Q and all non-codec assets remain exact.
            with torch.no_grad():next(template.flow_renderer.codec.decoder.parameters()).add_(.001)
            newcodec=_hash_tensors(template.flow_renderer.codec.state_dict())
            world=root/'world.pt';initial=root/'initial.pt';world.write_bytes(b'world');initial.write_bytes(b'renderer')
            args=SimpleNamespace(run_dir=root/'run',world=world,renderer_checkpoint=initial,
                data_root=root,empty_data_root=root,small_data_root=root/'small',parent_adam_checkpoint=parent,
                device='cpu',seed=173,threads=2,allow_unqualified_codec=True,steps=8001,batch=2,
                lr=1e-4,base_lr=1e-4,validation_frames=5,validation_batch=5,validate_every=100,preview_every=100,
                save_every=10,ssim_weight=.2,edge_weight=.1,contrast_weight=.1)
            captured={}
            def load(*args,**kwargs):return copy.deepcopy(template),{'training_steps':4250}
            def renderer(model,*args,**kwargs):
                model.base_student_bundle={'training_steps':17200}
                return {'training_steps':17600}
            def export(source,working,directory,**kwargs):captured.update(kwargs);return {'test_export':True}
            with patch('turkeyworld.train_renderer_enemy.load_world_checkpoint',side_effect=load), \
                    patch('turkeyworld.train_renderer_enemy.load_renderer_student',side_effect=renderer), \
                    patch('turkeyworld.small_enemy_data.SmallEnemyData',TinySource), \
                    patch('turkeyworld.small_enemy_validation.relative_small_enemy_gate',return_value={
                        'passed':True,'failure_reasons':[],'scope':'isolated_control_flow_test'}), \
                    patch('turkeyworld.train_renderer_enemy.export_refined_bundle',side_effect=export):
                EnemyRendererTraining(args).run()
            final=torch.load(args.run_dir/'checkpoints/renderer_enemy_last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(final['step'],8001);self.assertEqual(final['sampler']['draws'],2)
            self.assertEqual({int(v['step'])for v in final['optimizer']['state'].values()},{8001})
            self.assertEqual(_hash_tensors({key.removeprefix('codec.'):value for key,value in final['flow'].items()if key.startswith('codec.')}),newcodec)
            self.assertEqual(captured['initial_renderer_steps'],9600)
            self.assertEqual(captured['initial_base_steps'],9200)
            self.assertEqual(captured['updates'],8001)
            proof=json.loads((args.run_dir/'tiny_renderer_adam_migration.json').read_text())
            self.assertTrue(proof['adam_all_moments_options_bitexact']);self.assertTrue(proof['new_codec_retained'])
            self.assertTrue((args.run_dir/'validation_selected.json').exists())


if __name__=='__main__':unittest.main()
