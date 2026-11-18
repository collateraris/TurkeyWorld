"""Learned local player-axis pass logits for the 30 Hz V-JEPA navigation profile.

The static tile map is already an input to the world. Each wall in a 3x3 crop
votes on the proposed sub-cell position with a shared trainable classifier.
The footprint / pass boundary is learned from examples, not a runtime physics
call. There is no player radius, collision predicate, or future pose input.
"""
import hashlib
import torch
from torch import nn
from .structured_state import player_command_delta

MODE = 'learned_axis_collision'
MODE_NAV = MODE + '_nav'
PREFIX = 'vjepa_backend.local_axis_head.'
PROTOCOL = 'learned_local_tile_axis_pass_v1'


class LocalAxisCollisionHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Generic polynomial features of the two unsigned tile-edge coordinates.
        # Tie reflections and axis permutations; the same weights apply to every
        # tile material. The intercept, terms and corner curvature are learned.
        self.score = nn.Linear(8,1)
        with torch.no_grad():
            self.score.weight.zero_()
            self.score.weight[0,0] = 0.001  # monotone ordering only
            self.score.bias.zero_()
        self.register_buffer('_offsets', torch.tensor([[i,j] for j in (-1,0,1) for i in (-1,0,1)]), persistent=False)

    def _pass_logit(self, xy_tiles, grid):
        cell = xy_tiles.floor().long()[:,None,:]+self._offsets
        x,y = cell.unbind(-1)
        raw = grid[torch.arange(grid.shape[0],device=grid.device)[:,None],y.clamp(0,15),x.clamp(0,15)]
        wall = (raw != 0) | (x<0) | (x>=16) | (y<0) | (y>=16)
        relative = (xy_tiles[:,None,:]-(cell+.5)).abs()-.5
        hi,lo = relative.amax(-1),relative.amin(-1)
        p,q = hi.relu()*4,lo.relu()*4
        features = torch.stack((hi,lo,p,q,p.square(),q.square(),p*q,lo.abs()),-1)
        tile = 512.*self.score(features.to(self.score.weight.dtype)).squeeze(-1).float()
        # For an empty crop all walls are absent: no motion restriction.
        return tile.masked_fill(~wall,80.).amin(dim=1)

    def forward(self,state,action,grid,dt=1/30):
        if grid.ndim==2:
            grid=grid[None]
        if grid.shape[1:]!=(16,16) or len(grid) not in (1,len(state)):
            raise ValueError('Local player motion requires the aligned 16x16 static map')
        grid=grid.expand(len(state),-1,-1)
        xy=state[:,:2].float()*16
        command=player_command_delta(state.float(),action.float(),dt)
        delta=command*16
        px=self._pass_logit(torch.stack((xy[:,0]+delta[:,0],xy[:,1]),-1),grid)
        x_pass=(px>=0).to(xy.dtype)
        py=self._pass_logit(torch.stack((xy[:,0]+delta[:,0]*x_pass,xy[:,1]+delta[:,1]),-1),grid)
        logits=torch.stack((px,py),-1)
        fraction=(logits>=0).to(xy.dtype)
        return dict(player_pass_logits=logits,player_pass_fraction=fraction,player_command_delta=command,
                    raw_player_delta=command*fraction)


def head_digest(head):
    h=hashlib.sha256()
    for n,t in sorted(head.state_dict().items()):
        h.update(n.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def with_local_axis_head(payload, head, *, nav_only=False):
    """Append trained XY head; retain all original weights, Adam slots and clocks.

    The regular world optimizer has no parameters of this head. A separate
    supervised optimizer trains the head; the published copy stays frozen.
    """
    cfg=payload['config']
    mode=MODE_NAV if nav_only else MODE
    if cfg.get('architecture')!='vjepa2_ac' or cfg.get('player_motion_mode','delta')!='delta':
        raise ValueError('Only migrate V-JEPA delta checkpoints; never replace an existing head')
    if not cfg.get('endless') or cfg.get('simulation_hz',30)!=30 or cfg.get('transition_mode')!='event_v1':
        raise ValueError('Local head requires the 30 Hz endless event_v1 profile')
    weights={PREFIX+k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    if any(n.startswith(PREFIX) for n in payload['model']):raise ValueError('Checkpoint already includes local head weights')
    if not all(torch.isfinite(t).all() for t in weights.values()):raise ValueError('Nonfinite learned head')
    proof=dict(protocol=PROTOCOL,source_step=payload['training_steps'],source_mode='delta',target_mode=mode,
               head_sha256=head_digest(head),added_weight_keys=sorted(weights),
               retained_original_weight_keys=len(payload['model']),parent_adam_unchanged=True,
               parent_rng_and_clocks_unchanged=True,no_reference_future=True)
    return {**payload,'config':{**cfg,'player_motion_mode':mode},'model':{**payload['model'],**weights},
            'local_axis_head_migration':proof}

def verified_migration(model, saved):
    """Verify sidecar identity before accepting an interface-compatible refresh."""
    proof=saved.get('local_axis_head_migration',{})
    mode=model.config.player_motion_mode
    if mode not in (MODE_NAV,MODE) or model.config.architecture!='vjepa2_ac':return False
    head=getattr(model.vjepa_backend,'local_axis_head',None)
    if not isinstance(head,LocalAxisCollisionHead) or not isinstance(proof,dict):return False
    added=sorted(PREFIX+k for k in head.state_dict())
    return (proof.get('protocol')==PROTOCOL and proof.get('source_mode')=='delta'
            and proof.get('target_mode')==mode and proof.get('head_sha256')==head_digest(head)
            and proof.get('added_weight_keys')==added and proof.get('no_reference_future') is True
            and proof.get('parent_adam_unchanged') is True and proof.get('parent_rng_and_clocks_unchanged') is True
            and proof.get('retained_original_weight_keys')==len(saved['model'])-len(added)
            and type(proof.get('source_step')) is int and proof['source_step']<=saved['training_steps'])


def compatible_policy_refresh(previous,model,saved):
    current=model.config
    return (previous.player_motion_mode=='delta' and previous.architecture=='vjepa2_ac'
            and all(v==getattr(current,k) for k,v in vars(previous).items() if k!='player_motion_mode')
            and verified_migration(model,saved))


def verify_frozen_resume(model,saved,frozen,current_hash):
    """At initial attach, compare ALL legacy frozen weights, and the new head separately."""
    from .renderer_student import _hash_tensors
    if model.config.player_motion_mode=='learned_axis_quiet_pair':
        from .quiet_pair_profile import verify_frozen_resume as verify_quiet
        return verify_quiet(model,saved,frozen,current_hash)
    old=saved.get('vjepa_sr_training',{}).get('frozen_parameters_sha256')
    if model.config.player_motion_mode not in (MODE,MODE_NAV):
        return not old or old==current_hash
    if not verified_migration(model,saved):raise ValueError('Learned player head does not match migration receipt')
    if old==current_hash:return True
    if (saved['training_steps']==saved['local_axis_head_migration']['source_step'] and old
        and old==_hash_tensors({k:v for k,v in frozen.items() if not k.startswith(PREFIX)})):
        return True
    raise ValueError('Original frozen dynamics/renderer changed during learned player head resume')
