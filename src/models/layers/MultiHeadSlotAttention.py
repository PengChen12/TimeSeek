
from __future__ import annotations

import torch
from torch import einsum, nn
from torch.nn import init
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

class MultiHeadSlotAttention(Module):
    def __init__(
        self,
        num_slots,
        dim,
        heads = 4,
        dim_head = 64,
        iters = 3,
        eps = 1e-8,
        hidden_dim = 128,
        temperature = None
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)

        dim_inner = dim_head * heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.to_q = nn.Linear(dim, dim_inner)
        self.to_k = nn.Linear(dim, dim_inner)
        self.to_v = nn.Linear(dim, dim_inner)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = nn.Linear(dim_inner, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.norm_pre_ff = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        self.temperature = temperature
        self.pred_keep_slot = nn.Linear(dim, 2, bias = False)

    def forward(
        self,
        inputs,
        num_slots: int | None = None
    ):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = repeat(self.slots_mu, '1 1 d -> b s d', b = b, s = n_s)
        sigma = repeat(self.slots_logsigma.exp(), '1 1 d -> b s d', b = b, s = n_s)

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

        inputs = self.norm_input(inputs)        

        k, v = self.to_k(inputs), self.to_v(inputs)
        k, v = map(self.split_heads, (k, v))

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)

            q = self.to_q(slots)
            q = self.split_heads(q)

            dots = einsum('... i d, ... j d -> ... i j', q, k) * self.scale

            attn = dots.softmax(dim = -2)
            attn = F.normalize(attn + self.eps, p = 1, dim = -1)

            updates = einsum('... j d, ... i j -> ... i d', v, attn)
            updates = self.merge_heads(updates)
            updates = self.combine_heads(updates)

            updates, packed_shape = pack([updates], '* d')
            slots_prev, _ = pack([slots_prev], '* d')

            slots = self.gru(updates, slots_prev)

            slots, = unpack(slots, packed_shape, '* d')
            slots = slots + self.mlp(self.norm_pre_ff(slots))


        keep_slot_logits = self.pred_keep_slot(slots)
        keep_slots = torch.nn.functional.gumbel_softmax(keep_slot_logits, dim=-1, tau=self.temperature, hard=True)
        keep_slots = keep_slots[..., -1]  # Float["batch num_slots"] of {0., 1.}
        # Select only the slots that are kept
        keep_slots_expanded = keep_slots.unsqueeze(-1).expand_as(slots)
        slots = slots * keep_slots_expanded

        dots = torch.einsum('bid,bjd->bij', slots, inputs) * self.scale   #[ B x NS x C]
        attn = torch.nn.functional.gumbel_softmax(dots, dim = -2, hard=True)    #[ B x NS x C]
        # print(attn[0,:,:])
        attn = attn.permute(0,2,1)   #[ B x C x NS]
        corr_token = torch.einsum('bjd,bij->bid', slots, attn)  #[ B x C x D]

        return  corr_token, keep_slots.sum()
