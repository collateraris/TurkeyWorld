from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.engine import Game, BEHAVIOR_VERSION
from turkeyworld.spawn import SPAWN_ANCHORS, decode_spawn_positions
from turkeyworld.structured_state import structured_transition
from turkeyworld.rssm_spawn_curriculum import (source_recovery_labels,event_windows,
    RecoveryCombatSampler,evaluate_spawn_events)
from turkeyworld.staged_curriculum import StagedSequenceData


def make_source_files(directory, *, split='train', initial_death_timer=2.99):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    for phase in range(9):
        game=Game(seed=phase,endless=True,enemies_enabled=True)
        initial=game.get_state()
        if phase<8:
            anchor=np.asarray(SPAWN_ANCHORS[phase])+0.5
            cells=np.argwhere(game.map==0)[:,::-1]+.5
            point=max(cells,key=lambda point:np.linalg.norm(point-anchor))
            initial['player'].update(x=float(point[0]),y=float(point[1]))
            initial['time']=phase*3+.1
            initial['enemies'][0].update(alive=False,hp=0.,death_timer=initial_death_timer)
        else:
            game.map=np.zeros((16,16),np.uint8);game.map[[0,-1],:]=1;game.map[:,[0,-1]]=1;game.map[2,3]=1
            initial['player'].update(x=2.7,y=2.7,angle=0.)
            initial['enemies'][0].update(x=3.35,y=3.29,alive=True,hp=60.,attack_cooldown=0.)
        game.set_state(initial)
        def observe():
            image=game.render(size=32,overlays=False);enemies=game.enemies
            game.enemies=[]
            try:background=game.render(size=32,overlays=False)
            finally:game.enemies=enemies
            return image,background
        image,background=observe()
        states=[game.state_vector()];frames=[image];backgrounds=[background]
        actions=[];events=[];hits=[];attacks=[]
        for _ in range(32):
            action=np.zeros(4,np.float32);event=game.step(action,dt=1/30)
            image,background=observe();states.append(game.state_vector());frames.append(image);backgrounds.append(background)
            actions.append(action);events.append(Game.event_vector(event));hits.append(event.get('hit_id',-1))
            attacks.append([float(0 in event.get('attacker_ids',[]))])
        path=directory/f'episode_{phase}.npz'
        np.savez_compressed(path,states=np.stack(states),frames=np.stack(frames),actions=np.stack(actions),
            events=np.stack(events),hit_ids=np.asarray(hits,np.int64),attack_flags=np.asarray(attacks,np.float32),
            map=game.map,dt=np.asarray(1/30,np.float32),split=np.asarray(split),seed=phase,
            format_version=4,behavior_version=BEHAVIOR_VERSION,background_frames=np.stack(backgrounds),
            object_masks=np.any(np.stack(frames)!=np.stack(backgrounds),axis=-1))
        paths.append(path)
    return paths


class KnownSpawnModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(()))
        self.config=SimpleNamespace(navigation_only=False,simulation_hz=30,endless=True,game_training_v5=False)
    def encode_map(self,grid):self.grid=grid;return grid[:,None].float()
    def initialize_from_image(self,rgb,state,fmap):return state.new_zeros(len(state),2)
    def transition(self,state,action,fmap,memory):
        logits=state.new_zeros(len(state),1,256)
        positions,valid=decode_spawn_positions(logits,state,self.grid==0,1/30,deterministic=True)
        tiles=(positions*16).long().clamp(0,15);labels=tiles[:,:,0]+16*tiles[:,:,1]
        logits.scatter_(-1,labels[:,:,None],10.)
        result=structured_transition(state,state,action,state.new_tensor([[-8.,8.]]).expand(len(state),-1),
            state.new_full((len(state),1),-8.),dt=1/30,endless=True,spawn_positions=positions,spawn_valid=valid)
        return {**result,'spawn_logits':logits,'spawn_positions':positions,'spawn_valid':valid}


class SpawnCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_actual_source_covers_eight_anchors_and_target_windows_include_terminal_boundary(self):
        with TemporaryDirectory() as folder:
            paths=make_source_files(Path(folder)/'train')
            anchors=[]
            for phase,path in enumerate(paths[:8]):
                with np.load(path) as saved:labels=source_recovery_labels(saved['states'],saved['attack_flags'],float(saved['dt']))
                self.assertEqual(np.flatnonzero(labels['spawn']).tolist(),[0])
                anchors.append(int(labels['anchor'][0]))
                self.assertEqual(int(labels['clock_phase'][0]),phase)
            self.assertEqual(anchors,list(range(8)))
        self.assertEqual(event_windows([17],19,2,16).tolist(),[1])
        for context in (1,2):
            for event in range(18):
                for start in event_windows([event],19,context,16):
                    self.assertLessEqual(start+context-1,event)
                    self.assertLess(event,start+context-1+16)

    def test_exact_4_4_8_quotas_anchor_fairness_and_saved_rng_resume(self):
        with TemporaryDirectory() as folder:
            paths=make_source_files(Path(folder)/'train')
            data=SimpleNamespace(split='train',directory=paths[0].parent,episodes=[{'path':p}for p in paths])
            sampler=RecoveryCombatSampler(data);rng=np.random.default_rng(173)
            sampler.sample(16,rng);saved=deepcopy(sampler.state_dict());rng_saved=deepcopy(rng.bit_generator.state)
            first=sampler.sample(16,rng)
            self.assertEqual(set(sampler.anchor_counts.values()),{1})
            self.assertEqual(sampler.last_sample['quota'],{'spawn':4,'attack_negative':4,'ordinary_combat':8})
            restored=RecoveryCombatSampler(data);restored.restore_state(saved)
            rng2=np.random.default_rng();rng2.bit_generator.state=rng_saved
            np.testing.assert_array_equal(first,restored.sample(16,rng2))
            self.assertEqual(sampler.state_dict(),restored.state_dict())
            bad=deepcopy(saved);bad['draws']+=1
            with self.assertRaises(ValueError):restored.restore_state(bad)
            data.split='val'
            with self.assertRaises(ValueError):RecoveryCombatSampler(data)

    def test_expanded_validation_measures_every_actual_event_and_all_anchors_without_engine(self):
        with TemporaryDirectory() as folder:
            directory=Path(folder)/'val';make_source_files(directory,split='val')
            data=StagedSequenceData(directory,'cpu',mode='stream')
            with patch.object(Game,'step',side_effect=AssertionError('No runtime Source engine')), \
                    patch.object(Game,'render',side_effect=AssertionError('No Source renderer')):
                report=evaluate_spawn_events(KnownSpawnModel(),data,Path(folder)/'report',batch_size=3)
            self.assertEqual(report['events'],8);self.assertEqual(report['missing_anchor_classes'],[])
            self.assertEqual(set(report['per_anchor']),set(map(str,range(8))))
            self.assertEqual(report['overall']['metrics']['spawn_decoded_top1_tile_accuracy']['denominator'],8)
            self.assertEqual(report['overall']['metrics']['spawn_decoded_top1_tile_accuracy']['mean'],1.)
            self.assertFalse(report['source_future_feedback']);self.assertEqual(report['runtime_engine_calls'],0)


if __name__=='__main__':unittest.main()
