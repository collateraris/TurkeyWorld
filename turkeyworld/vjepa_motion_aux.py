"""Short-prefix player supervision; Source futures are labels only."""
import torch
from torch.nn import functional as F
from .structured_state import player_command_delta

PREVIOUS_REVISION = 'balanced_motion_short16_contact_v2'
REVISION = 'available_motion_blocks16_contact_v3'

def commands(states, actions):
    return player_command_delta(states.reshape(-1,20), actions.reshape(-1,4)).reshape(*actions.shape[:-1],2)

def window_kind(states, actions, targets):
    n=min(32,len(actions)); previous=states[:n]
    cmd=commands(previous,actions[:n]); delta=targets[:n,:2]-previous[:,:2]
    requested=cmd.abs()>1e-5
    active=requested.any(-1)
    if active.float().mean()<.25:return 'idle'
    blocked=(requested & (delta.abs()<cmd.abs()*.25)).sum()/requested.sum().clamp_min(1)
    return 'blocked' if blocked>.25 else 'moving'

def player_auxiliary(prediction, target, initial, actions, length=16, source_initial=None):
    n=min(length,prediction.shape[1]);pred=prediction[:,:n];truth=target[:,:n]
    own=torch.cat((initial[:,None],pred[:,:-1]),1)
    source=torch.cat(((initial if source_initial is None else source_initial)[:,None],truth[:,:-1]),1)
    delta=pred[...,:2]-own[...,:2];label=truth[...,:2]-source[...,:2]
    velocity=F.smooth_l1_loss(delta*480,label*480)
    command=commands(source,actions[:,:n]).detach()
    mask=command.abs()>1e-5
    # Continuous pass fraction supervises both blocked and free axes. No hard
    # collision decisions and no Source states are fed into the live rollout.
    scale=command.abs().clamp_min(.001)
    passed=delta*command.sign()/scale
    expected=(label*command.sign()/scale).clamp(0,1)
    contact=(F.smooth_l1_loss(passed,expected,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
    return velocity,contact


def segmented_player_auxiliary(prediction,target,initial,actions,block_size=16):
    """Every disjoint block, including the final4 ticks; retain full BPTT.

    Own and Source previous states remain separate at block boundaries.
    Source futures are labels only. No teacher reset occurs in the rollout.
    """
    total=prediction.shape[1];velocity=prediction.sum()*0;contact=velocity;blocks=[]
    for start in range(0,total,block_size):
        end=min(start+block_size,total)
        own_initial=initial if start==0 else prediction[:,start-1]
        source_initial=initial if start==0 else target[:,start-1]
        v,c=player_auxiliary(prediction[:,start:end],target[:,start:end],own_initial,actions[:,start:end],length=end-start,source_initial=source_initial)
        weight=(end-start)/total;velocity=velocity+weight*v;contact=contact+weight*c
        blocks.append((start+1,end,v,c))
    return velocity,contact,blocks
