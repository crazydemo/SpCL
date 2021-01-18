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
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=2048):
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
        self.to_v1 = nn.Linear(dim, hidden_dim)
        self.to_v2 = nn.Linear(dim, hidden_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim*8)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(hidden_dim)
        self.norm_pre_ff = nn.LayerNorm(hidden_dim)

        self.reset_params()

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        slots = self.slots.repeat(b, 1, 1)
        d = slots.shape[-1]

        inputs = self.norm_input(inputs)
        k, v1, v2 = self.to_k(inputs), self.to_v1(inputs), self.to_v2(inputs)

        for _ in range(1):#self.iters
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots_qk = torch.einsum('bid,bjd->bij', q, k) * self.scale #b, k, h*w

            dots_kv1 = torch.einsum('bid,bjd->bij', k, v1) * self.scale  # b, h*w, h*w

            dots_qk = dots_qk.repeat([1,16,1])
            # print(dots_qk.size())
            # print("dots_qk", dots_qk.size())
            attn = dots_qk + dots_kv1

            attn = attn.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            # print("attn", attn.size())
            updates = torch.einsum('bij,bid->bjd', attn, v2) # b, hw, d
            # print("updates", updates.size())
            # updates = updates.permute(0, 2, 1, 3)
            # b, hw, k, d = updates.size()
            # print(b, hw, k, d)
            # updates = updates.mean(1)
            # updates = self.mlp(updates)

        return updates
        # return inputs

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
