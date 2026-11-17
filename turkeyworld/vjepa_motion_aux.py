"""Short-prefix player supervision; Source futures are labels only."""
import torch
from torch.nn import functional as F
from .structured_state import player_command_delta

REVISION = 'balanced_motion_short16_contact_v2'

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

def player_auxiliary(prediction, target, initial, actions, length=16):
    n=min(length,prediction.shape[1]);pred=prediction[:,:n];truth=target[:,:n]
    own=torch.cat((initial[:,None],pred[:,:-1]),1)
    source=torch.cat((initial[:,None],truth[:,:-1]),1)
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
