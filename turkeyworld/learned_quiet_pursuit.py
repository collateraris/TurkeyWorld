"""Learned quiet chase: raw occupancy + own actor poses; no Source or runtime physics.

The route module is a min-plus network: shared learned per-tile traversal costs,
32 cardinal message-passing layers, learned tie preferences. A separate gate
chooses player-directed vs adjacent-cell motion using sampled occupancy features.
The NPC contact classifier has its own weights (not the player's footprint).
"""
import torch
from torch import nn
from torch.nn import functional as F
from .learned_axis_collision import LocalAxisCollisionHead


class QuietPursuitHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge = nn.Linear(1, 1)
        self.rank = nn.Parameter(torch.zeros(4))
        self.visibility = nn.Linear(3, 1)
        self.stop_model = nn.Linear(1, 1)
        self.raw_speed = nn.Parameter(torch.zeros(()))
        self.contact = LocalAxisCollisionHead()
        for layer in (self.edge, self.visibility, self.stop_model):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.register_buffer("dirs", torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]]), persistent=False)
        self.register_buffer("samples", torch.linspace(0, 1, 129), persistent=False)
        # Inference buffers only. They are invalidated on any map or learned edge-cost change.
        self._cache = None

    def visibility_features(self, npc, player, grid):
        grid = grid.expand(len(npc), -1, -1)
        t = self.samples[None, :, None]
        samples = npc[:, None] + (player - npc)[:, None] * t  # map tile coordinates
        uv = (samples / 8 - 1)[:, :, None, :]
        texture = (grid != 0).float()[:, None]
        # Both channels are observations on the supplied static map (no ray cast).
        point = F.grid_sample(texture, uv, mode="nearest", padding_mode="border", align_corners=False)[
            :, 0, :, 0
        ]
        soft = F.grid_sample(texture, uv, mode="bilinear", padding_mode="border", align_corners=False)[
            :, 0, :, 0
        ]
        return torch.stack((point.amax(1), soft.amax(1), point.mean(1)), dim=-1)

    def _apply(self, fn, *args, **kwargs):
        # Nonpersistent caches must never keep tensors on an obsolete device.
        self._cache = None
        return super()._apply(fn, *args, **kwargs)

    def distances(self, grid):
        # [goal, row, col]. Generic local message passing, no navigation field from engine.
        device = grid.device
        baseline = grid.float()
        edge = (
            F.softplus(
                self.edge((baseline != 0).to(self.edge.weight.dtype)[..., None]).squeeze(-1).float() * 512.0
            )
            + 0.01
        )
        if not self.training and self._cache is not None:
            g, c, values = self._cache
            if g.device == grid.device and torch.equal(g, grid) and torch.equal(c, edge.detach()):
                return values
        z = torch.arange(256, device=device).view(16, 16)
        goal = torch.arange(256, device=device)[:, None, None] == z[None]
        dp = torch.where(goal, 0.0, 1e4)
        for _ in range(32):
            q = F.pad(dp + edge[None], (1, 1, 1, 1), value=1e4)
            dp = torch.minimum(
                dp,
                torch.minimum(
                    torch.minimum(q[:, 1:-1, 2:], q[:, 1:-1, :-2]),
                    torch.minimum(q[:, 2:, 1:-1], q[:, :-2, 1:-1]),
                ),
            )
        values = dp + edge[None]
        if not self.training:
            self._cache = (grid.detach().clone(), edge.detach().clone(), values.detach())
        return values

    def route_components(self, npc, player, grid):
        b = len(npc)
        if grid.ndim == 2:
            grid = grid[None]
        if grid.shape[1:] != (16, 16) or len(grid) not in (1, b):
            raise ValueError("Expected static 16x16 grid")
        # All common-map streams share the same small routing cache. Different
        # maps in one batch are still evaluated independently.
        if len(grid) > 1 and not torch.equal(grid, grid[0:1].expand_as(grid)):
            return torch.cat(
                [self.route_components(npc[i : i + 1], player[i : i + 1], grid[i : i + 1]) for i in range(b)]
            )
        v = self.distances(grid[0])
        p = player.floor().long().clamp(0, 15)
        n = npc.floor().long()[:, None, :] + self.dirs
        g = p[:, 1] * 16 + p[:, 0]
        costs = v[g[:, None], n[..., 1].clamp(0, 15), n[..., 0].clamp(0, 15)]
        logits = -costs + (self.rank * 128.0).tanh() * 0.2
        return logits.masked_fill(((n < 0) | (n >= 16)).any(-1), -1e4)

    def forward(self, state, player_next_xy, grid):
        if grid.ndim == 2:
            grid = grid[None]
        if player_next_xy.shape != (len(state), 2):
            raise ValueError("Expected normalized next own player XY")
        npc = state[:, 8:10].float() * 16
        player = player_next_xy.float() * 16
        rel = player - npc
        route = self.route_components(npc, player, grid).argmax(-1)
        wp = npc.floor() + 0.5 + self.dirs[route]
        sight = (
            self.visibility(self.visibility_features(npc, player, grid).to(self.visibility.weight.dtype))
            .squeeze(-1)
            .float()
            * 256.0
        )
        target = torch.where((sight >= 0)[:, None], rel, wp - npc)
        move = (
            self.stop_model(rel.norm(dim=-1, keepdim=True).to(self.stop_model.weight.dtype)).float() * 256.0
            > 0
        )
        speed = F.softplus(self.raw_speed.float() * 128.0)
        d = F.normalize(target, dim=-1) * speed / 30 * move
        grid = grid.expand(len(npc), -1, -1)
        x = npc[:, 0] + d[:, 0]
        okx = self.contact._pass_logit(torch.stack((x, npc[:, 1]), -1), grid) >= 0
        dx = d[:, 0] * okx
        oky = self.contact._pass_logit(torch.stack((npc[:, 0] + dx, npc[:, 1] + d[:, 1]), -1), grid) >= 0
        return torch.stack((dx, d[:, 1] * oky), -1) / 16
