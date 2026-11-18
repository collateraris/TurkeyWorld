"""CPU, player-only ablation separating learned contact decisions from head initialization."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .learned_axis_collision import LocalAxisCollisionHead, head_digest
from .structured_state import structured_transition
from .trajectory_objectives import circle_wall_penalty
from .training_pipeline import atomic_json
from .validate_nav_contacts import nav_contacts, manifest


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--head',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    eps=nav_contacts();s=torch.from_numpy(np.stack([e['states'] for e in eps]))
    acts=torch.from_numpy(np.stack([e['actions'] for e in eps]));grid=torch.from_numpy(np.stack([e['grid'] for e in eps]))
    out={'panel_sha256':manifest(eps)['fingerprint'],'player_only':True,'steps':180,'Source_future_feedback':False,
         'notes':'Identical player-only hard command integration in both arms; no latent backbone or renderer. Learned head weights vs original initialization.'}
    net=LocalAxisCollisionHead();pack=torch.load(a.head,map_location='cpu',weights_only=False)
    for arm in ('untrained','trained'):
        if arm=='trained':net.load_state_dict(pack['model'])
        st=s[:,3].clone();px=[]
        for tick in range(180):
            proposed=st.clone()
            proposed[:,:2]+=net(st,acts[:,tick],grid)['raw_player_delta']
            st=structured_transition(st,proposed,acts[:,tick],st.new_zeros(len(eps),2),
                                     st.new_zeros(len(eps),1),endless=True,respawn_enabled=False)['state']
            px.append(st[:,:2])
        pred=torch.stack(px,dim=1);e=((pred-s[:,4:,:2])*16).norm(dim=-1)
        w=(circle_wall_penalty(pred*16,grid,.20)>0).sum(1)
        out[arm]={'head_sha256':head_digest(net),'groups':{}}
        for kind in ('strafe_wall','corner_exit','all'):
            mask=[kind=='all' or ep['kind']==kind for ep in eps]
            out[arm]['groups'][kind]={'H180':e[mask,-1].mean().item(),'path_mean':e[mask].mean().item(),
                                       'mean_wall_ticks':w[mask].float().mean().item()}
    atomic_json(a.output,out);print(out)
if __name__=='__main__':main()
