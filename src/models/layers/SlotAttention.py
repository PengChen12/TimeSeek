__all__ = ['SlotAttention']
import torch
from torch import nn
from torch.nn import init

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        # self.to_q1 = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.pred_keep_slot = nn.Linear(dim, 2, bias = False)

    def forward(self, inputs, num_slots = None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs) #[ B x C x D]

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots) #[ B x NS x D]

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale      #[ B x NS x C]
            attn = dots.softmax(dim=-2) + self.eps      #[ B x NS x C]

            attn = attn / attn.sum(dim=-1, keepdim=True)   #[ B x NS x C]
            updates = torch.einsum('bjd,bij->bid', v, attn)     #[ B x NS x D]

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))




        slots = self.norm_slots(slots)
        q = self.to_q(slots)    #[ B x NS x D]
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale   #[ B x NS x C]
        attn = torch.nn.functional.gumbel_softmax(dots, dim = -2, hard=True)    #[ B x NS x C]
        # print(attn[0,:,:])
        attn = attn.permute(0,2,1)   #[ B x C x NS]
        corr_token = torch.einsum('bjd,bij->bid', slots, attn)  #[ B x C x D]

        return corr_token