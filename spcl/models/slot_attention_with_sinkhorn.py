from torch import nn
import torch
import numpy as np
from torch.nn import init
import torch.nn.functional as F

def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0)
  grid = grid.astype(np.float32)
  return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed(nn.Module):
  """Adds soft positional embedding with learnable projection."""

  def __init__(self, hidden_size, resolution):
    """Builds the soft position embedding layer.

    Args:
      hidden_size: Size of input feature dimension.
      resolution: Tuple of integers specifying width and height of grid.
    """
    super().__init__()
    self.dense = nn.Conv2d(4, hidden_size, 1)
    self.grid = torch.from_numpy(build_grid(resolution)).cuda()
    self.grid = self.grid.permute(0, -1, 1, 2)
    self.reset_params()

  def forward(self, inputs):
      return inputs + self.dense(self.grid)

  def reset_params(self):
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              init.kaiming_normal_(m.weight, mode='fan_out')
              if m.bias is not None:
                  init.constant_(m.bias, 0)
          elif isinstance(m, nn.BatchNorm2d):
              init.constant_(m.weight, 1)
              init.constant_(m.bias, 0)
          elif isinstance(m, nn.BatchNorm1d):
              init.constant_(m.weight, 1)
              init.constant_(m.bias, 0)
          elif isinstance(m, nn.Linear):
              init.normal_(m.weight, std=0.001)
              if m.bias is not None:
                  init.constant_(m.bias, 0)


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        # self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
        self.slots = nn.Parameter(torch.randn(1, num_slots, hidden_dim))

        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, hidden_dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(hidden_dim)
        self.norm_pre_ff = nn.LayerNorm(hidden_dim)

        self.reset_params()

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots


        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_sigma.expand(b, n_s, -1)
        # slots = torch.normal(mu, sigma)
        slots = self.slots.repeat(b, 1, 1)
        d = slots.shape[-1]

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(1):#self.iters
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale #b, k, h*w
            Q = dots.reshape(b, -1)
            Q = Q.softmax(dim=-1) / Q.sum(-1, keepdim=True)
            Q = Q.reshape(b, n_s, -1)
            for _ in range(3):
                u = Q.sum(-1, keepdim=True)
                u_ = Q.sum(1, keepdim=True)
                Q *= torch.ones_like(u)*n_s / u
                Q *= torch.ones_like(u_)*n / u_


            attn_ = Q
            attn = attn_ / attn_.sum(dim=1, keepdim=True)

            # select = attn_.sum(-1)
            # tmp, select = select.sort(descending=True)

            updates = torch.einsum('bjd,bij->bid', v, attn) # b, k, d

            # slots = self.gru(
            #     updates.reshape(-1, d),
            #     slots_prev.reshape(-1, d)
            # )
            #
            # slots = slots.reshape(b, -1, d)
            # slots = slots + self.mlp(self.norm_pre_ff(updates))
            slots = updates
            out = slots.reshape(b, -1) / n_s

        return out

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
