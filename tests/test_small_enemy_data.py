from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.engine import Game,BEHAVIOR_VERSION
from turkeyworld.small_enemy_data import collect,SmallEnemyData,_observe,inventory,replay_selected,digest
from turkeyworld.codec_enemy_objectives import native_object_crop_terms,codec_enemy_loss
from test_codec_enemy_objectives import fixture,same_state


def source_snapshots(root,split):
    game=Game(seed=173 if split=='train' else 274,endless=True)
    rng=np.random.default_rng(173 if split=='train' else 274)
    arrays={k:[]for k in ('states','frames','object_masks','background_frames')};snapshots=[]
    for kind in ['tiny']*48+['near','corpse','hp0','navigation']*4:
        for _ in range(1000):
            state=game.get_state();nav=kind=='navigation';state['enemies_enabled']=not nav
            player_x=2.5 if split=='train' else 13.5;y=2.5
            distance=float(rng.uniform(6,10))if kind=='tiny' else 1.
            enemy_x=player_x+distance if split=='train'else player_x-distance
            state['player'].update(x=player_x,y=y,angle=(0. if split=='train' else np.pi)+float(rng.uniform(-.08,.08)),
                hp=0. if kind=='hp0' else 100.,shot_flash=0.,cooldown=0.)
            state['time']=float(rng.uniform(0,120))
            state['enemies'][0].update(x=enemy_x,y=y,hp=0. if kind in ('corpse','navigation') else 60.,
                alive=kind not in ('corpse','navigation'),death_timer=.6 if nav else 0.,hurt_timer=0.,vx=0.,vy=0.)
            game.set_state(state);low,_,bg,_,mask,_=_observe(game)
            if kind!='tiny'or 1<=int(mask.sum())<=16:break
        else:raise AssertionError('No genuine visible tiny Source fixture')
        for name,value in dict(states=game.state_vector(),frames=low,object_masks=mask,background_frames=bg).items():arrays[name].append(value)
        snapshots.append(game.get_state())
    path=Path(root)/'static_enemy/snapshots'/split/'episode_173.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**{k:np.asarray(v)for k,v in arrays.items()},map=game.map,seed=game.seed,
        split=split,behavior_version=BEHAVIOR_VERSION,snapshot_states=json.dumps(snapshots),
        world_transition_compatible=False,stage='navigation')
    return path


class SmallEnemyDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_real_source_replay_true128_small_masks_split_guards_and_reader_no_engine(self):
        with TemporaryDirectory()as folder:
            source=Path(folder)/'source';train=source_snapshots(source,'train');val=source_snapshots(source,'val')
            before=[digest(train),digest(val)]
            output=Path(folder)/'pairs';report=collect(output,[source],train_count=16,val_count=16,workers=1)
            self.assertEqual(report['status'],'complete')
            for split in ('train','val'):
                self.assertEqual(report['groups'][split]['cohorts']['tiny'],12)
                self.assertEqual(report['groups'][split]['examples'],16)
                for category in ('near','corpse','hp0','navigation'):self.assertEqual(report['groups'][split]['cohorts'][category],1)
            data=SmallEnemyData(output,'val');panel=data.fixed_panel(16)
            with patch.object(Game,'step',side_effect=AssertionError('Reader must not simulateSource')), \
                    patch.object(Game,'render',side_effect=AssertionError('Reader must not renderSource')):
                batch=data.batch(16,indices=panel)
            self.assertEqual(batch['target'].shape,(16,3,32,32));self.assertEqual(batch['hr'].shape,(16,3,128,128))
            self.assertFalse(batch['dynamic'].any());self.assertEqual(float(batch['previous_rgb'].abs().sum()),0.)
            for row in report['records']:
                if row['cohort']=='tiny':self.assertTrue(1<=row['mask_area']<=16 and row['npc_alive'])
            train_hash={r['rgb_sha256']for r in report['records']if r['split']=='train'}
            self.assertFalse(train_hash&{r['rgb_sha256']for r in report['records']if r['split']=='val'})
            self.assertEqual(before,[digest(train),digest(val)])
            raw=(output/'manifest.json').read_bytes();collect(output,[source],train_count=16,val_count=16,workers=1)
            self.assertEqual(raw,(output/'manifest.json').read_bytes())
            saved=deepcopy(data.sampler_state());left=data.batch(16)['row_index'];data.restore_sampler(saved)
            self.assertTrue(torch.equal(left,data.batch(16)['row_index']))

    def test_replay_rejects_forged_source_native_rgb_instead_of_synthesizing_target(self):
        with TemporaryDirectory()as folder:
            path=source_snapshots(Path(folder)/'source','train')
            found=inventory([Path(folder)/'source'],'train');row=found['rows'][0]
            with np.load(path)as z:fields={k:z[k]for k in z.files}
            fields['frames'][row['frame'],0,0,0]^=1
            # Alter the matching background too, so this is a replay check,
            # not merely a mask-consistency shape check.
            fields['background_frames'][row['frame'],0,0,0]^=1
            np.savez_compressed(path,**fields)
            source={**found['files'][0],'sha256':digest(path)}
            row={**row,'sample_index':0}
            with self.assertRaisesRegex(ValueError,'bitexact'):
                replay_selected({'source':source,'rows':[row],'split':'train','output':folder,'fingerprint':'test'})

    def test_one_pixel_native_crop_has_gradients_and_empty_foreground_is_finite(self):
        prediction=torch.zeros(2,3,32,32,requires_grad=True)
        target=torch.zeros_like(prediction,requires_grad=True);mask=torch.zeros(2,1,32,32,requires_grad=True)
        with torch.no_grad():target[0,:,16,16]=1.;mask[0,0,16,16]=1.
        l1,edge,count=native_object_crop_terms(prediction,target,mask)
        self.assertEqual(count,1);(l1+.1*edge).backward()
        self.assertGreater(float(prediction.grad[0,:,16,16].abs().sum()),0.)
        self.assertIsNone(target.grad);self.assertIsNone(mask.grad)
        p=torch.zeros(1,3,32,32,requires_grad=True)
        a,b,count=native_object_crop_terms(p,torch.zeros_like(p),torch.zeros(1,1,32,32))
        self.assertEqual(count,0);self.assertEqual(float(a.detach()+b.detach()),0.)
        (a+b).backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_crop_optin_real_codec_preserves_all_quantizer_buffers_and_legacy_arithmetic(self):
        student,_,_,batch,refs=fixture();student.requires_grad_(True).train()
        q={k:v.clone()for k,v in student.quantize.state_dict().items()}
        old,old_metrics=codec_enemy_loss(student,batch,refs)
        same,same_metrics=codec_enemy_loss(student,batch,refs,object_crop_weight=0.)
        self.assertTrue(torch.equal(old,same));self.assertEqual(old_metrics,same_metrics)
        loss,metrics=codec_enemy_loss(student,batch,refs,object_crop_weight=.1)
        loss.backward();self.assertTrue(metrics['object_crop_active']);self.assertFalse(metrics['object_crop_resize'])
        self.assertGreater(float(student.decoder.conv_out.weight.grad.abs().sum()),0.)
        self.assertTrue(same_state(q,student.quantize))
        self.assertTrue(all(v.grad is None for v in (*batch.values(),*refs.values())))


if __name__=='__main__':unittest.main()
