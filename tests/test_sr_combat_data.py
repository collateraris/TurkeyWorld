import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from turkeyworld.engine import BEHAVIOR_VERSION, Game
from turkeyworld.sr_combat_data import replay_pairs, SRCombatPairData


def source_episode(path, split='val'):
    game=Game(seed=551,endless=True,enemies_enabled=False)
    initial=json.dumps(game.get_state());states=[];frames=[];high=[];actions=[];events=[]
    for index in range(5):
        states.append(game.state_vector());frames.append(game.render(size=32,overlays=False))
        high.append(game.render(size=128,overlays=False))
        if index<4:
            action=np.array([.5,.2,.3,0],np.float32);event=game.step(action)
            actions.append(action);events.append(Game.event_vector(event))
    np.savez_compressed(path,initial_state=np.asarray(initial),states=np.asarray(states),frames=np.asarray(frames),
        actions=np.asarray(actions),events=np.asarray(events),map=game.map,seed=np.asarray(551),split=np.asarray(split),
        stage=np.asarray('navigation'),behavior_version=np.asarray(BEHAVIOR_VERSION))
    return frames,high,states


class SRCombatDataTests(unittest.TestCase):
    def test_true_native_pair_replay_and_target_only_loader(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.npz';low,high,states=source_episode(source)
            job=dict(source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
                rows=[dict(frame=1,cohort='navigation'),dict(frame=3,cohort='navigation')],output=str(root/'paired'),fingerprint='test')
            records=replay_pairs(job)
            self.assertEqual(len(records),2)
            for row in records:
                with np.load(root/'paired'/row['relative_path']) as pair:
                    frame=row['source_frame']
                    np.testing.assert_array_equal(pair['frames'],np.stack(low[frame-1:frame+1]))
                    np.testing.assert_array_equal(pair['hr_frames'],np.stack(high[frame-1:frame+1]))
                    np.testing.assert_array_equal(pair['states'],np.stack(states[frame-1:frame+1]))
                    self.assertTrue(bool(pair['paired_source_rgb']))
                    self.assertFalse(np.array_equal(pair['hr_frames'][1],np.repeat(np.repeat(pair['frames'][1],4,0),4,1)))
            data=SRCombatPairData(root/'paired/rgb/val',image_size=32,cache_bytes=2*1024**2)
            panel=data.fixed_panel();self.assertEqual([frame for _,frame in panel],[1,1])
            batch=data.batch(2,indices=panel)
            self.assertEqual(tuple(batch['lr'].shape),(2,3,32,32));self.assertEqual(tuple(batch['hr'].shape),(2,3,128,128))
            self.assertEqual(tuple(batch['hr_object_mask'].shape),(2,1,128,128))
            self.assertFalse(bool(batch['hr_object_mask'].any()))

    def test_source_lr_tampering_is_rejected_before_any_pair_is_written(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.npz';source_episode(source)
            with np.load(source) as saved:values={key:saved[key] for key in saved.files}
            values['frames'][1,0,0,0]^=1;np.savez_compressed(source,**values)
            job=dict(source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
                rows=[dict(frame=1,cohort='navigation')],output=str(root/'out'),fingerprint='test')
            with self.assertRaisesRegex(ValueError,'native32 RGB'):replay_pairs(job)
            self.assertFalse(list((root/'out').rglob('*.npz')))


if __name__=='__main__':unittest.main()
