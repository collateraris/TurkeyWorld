"""Explicit appended TRAIN replay for serial Source/world/policy cycles."""
import json
from pathlib import Path

import torch

from .dreamer_policy_data import PolicyReplay,file_sha,digest
from .workspace_paths import resolve_path,identity_path


class PolicyCycleData:
    def __init__(self,receipt_path,*,device,horizon,seed,previous=None):
        self.path=resolve_path(receipt_path)
        self.receipt=json.loads(self.path.read_text(encoding='utf-8'))
        self.receipt_sha=file_sha(self.path)
        if(self.receipt.get('split')!='train'or self.receipt.get('validation_unchanged')is not True):
            raise ValueError('Policy cycle replay must be a declared TRAIN-only Source collection')
        for row in self.receipt['episodes']:
            if row['split']!='train'or file_sha(row['path'])!=row['sha256']:
                raise ValueError('Collected policy replay bytes or split changed')
        directories=[resolve_path(self.receipt[key])for key in('nav_train_directory','enemy_train_directory')]
        declared={resolve_path(row['path'])for row in self.receipt['episodes']}
        actual={path.resolve()for directory in directories for path in directory.glob('*.npz')}
        if actual!=declared:raise ValueError('Cycle replay files differ from the immutable declared receipt')
        self.data=PolicyReplay(directories,device=device,horizon=horizon,
            seed=seed+int(digest(self.receipt)[:8],16))
        if previous is not None and previous['receipt_sha256']==self.receipt_sha:
            self.data.load_state_dict(previous['replay'])

    def merge(self,batch,per_stream):
        half=max(1,per_stream//2)
        extra=self.data.batch(half,balanced=False)
        target=torch.cat((torch.arange(half),torch.arange(per_stream,per_stream+half))).to(batch['states'].device)
        for key,value in batch.items():
            if key in extra:
                updated=value.clone()
                updated.index_copy_(0,target,extra[key])
                batch[key]=updated
        return batch

    def state_dict(self):
        return dict(receipt_path=identity_path(self.path),receipt_sha256=self.receipt_sha,
            corpus_identity_sha256=digest(self.receipt),replay=self.data.state_dict())
