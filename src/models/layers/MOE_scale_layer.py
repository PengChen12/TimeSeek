from torch import nn
import torch
import torch.nn.functional as F
from PIL import Image
from math import sqrt
import numpy as np

# from layers.layer import *
from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
# from torchvision import models
# from PIL import Image
from math import sqrt
import numpy as np
import math
from torch import einsum
import torch
from torch import nn
from torch.nn import init
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange


class MultiHeadSlotAttention(nn.Module):
    def __init__(
            self,
            num_slots,
            dim,
            heads=4,
            dim_head=64,
            iters=3,
            eps=1e-8,
            hidden_dim=128,
            temperature=1
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

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)

        dim_inner = dim_head * heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)

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
        self.pred_keep_slot = nn.Linear(dim, 2, bias=False)

    def forward(
            self,
            inputs
    ):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        #n_s = num_slots if num_slots is not None else self.num_slots
        n_s = self.num_slots

        mu = repeat(self.slots_mu, '1 1 d -> b s d', b=b, s=n_s)
        sigma = repeat(self.slots_logsigma.exp(), '1 1 d -> b s d', b=b, s=n_s)

        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)

        inputs = self.norm_input(inputs)

        k, v = self.to_k(inputs), self.to_v(inputs)
        k, v = map(self.split_heads, (k, v))

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)

            q = self.to_q(slots)
            q = self.split_heads(q)

            dots = einsum('... i d, ... j d -> ... i j', q, k) * self.scale

            attn = dots.softmax(dim=-2)
            attn = F.normalize(attn + self.eps, p=1, dim=-1)

            updates = einsum('... j d, ... i j -> ... i d', v, attn)
            updates = self.merge_heads(updates)
            updates = self.combine_heads(updates)

            updates, packed_shape = pack([updates], '* d')
            slots_prev, _ = pack([slots_prev], '* d')

            slots = self.gru(updates, slots_prev)

            slots, = unpack(slots, packed_shape, '* d')
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        keep_slot_logits = self.pred_keep_slot(slots)
        #print(keep_slot_logits)
        keep_slots = torch.nn.functional.gumbel_softmax(keep_slot_logits, dim=-1, tau=self.temperature, hard=True)
        keep_slots = keep_slots[..., -1]  # Float["batch num_slots"] of {0., 1.}
        # Select only the slots that are kept
        keep_slots_expanded = keep_slots.unsqueeze(-1).expand_as(slots)
        slots = slots * keep_slots_expanded

        dots = torch.einsum('bid,bjd->bij', slots, inputs) * self.scale  # [ B x NS x C]
        attn = torch.nn.functional.gumbel_softmax(dots, dim=-2, hard=True)  # [ B x NS x C]
        # print(attn[0,:,:])
        attn = attn.permute(0, 2, 1)  # [ B x C x NS]
        corr_token = torch.einsum('bjd,bij->bid', slots, attn)  # [ B x C x D]

        return corr_token


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=1, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        ##共性知识
        self.slots = nn.Parameter(torch.randn(1, num_slots, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.temperature = 1
        self.pred_keep_slot = nn.Linear(dim, 2, bias=False)

    def forward(self, inputs, num_slots=None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)
        slots = slots + self.slots

        inputs1 = self.norm_input(inputs)
        k, v = self.to_k(inputs1), self.to_v(inputs1)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps

            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        keep_slot_logits = self.pred_keep_slot(slots)
        # print(keep_slot_logits)
        keep_slots = torch.nn.functional.gumbel_softmax(keep_slot_logits, dim=-1, tau=self.temperature, hard=True)
        keep_slots = keep_slots[..., -1]  # Float["batch num_slots"] of {0., 1.}
        # Select only the slots that are kept
        keep_slots_expanded = keep_slots.unsqueeze(-1).expand_as(slots)
        slots = slots * keep_slots_expanded

        dots = torch.einsum('bid,bjd->bij', slots, inputs) * self.scale  # [ B x NS x C]
        attn = torch.nn.functional.gumbel_softmax(dots, dim=-2, hard=True)  # [ B x NS x C]
        # print(attn[0,:,:])
        attn = attn.permute(0, 2, 1)  # [ B x C x NS]
        corr_token = torch.einsum('bjd,bij->bid', slots, attn)  # [ B x C x D]
        corr_token += inputs

        return corr_token



class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                nn.ReLU(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src, prev=None, key_padding_mask=None, attn_mask=None):

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q, K=None, V=None, prev=None,
                key_padding_mask=None, attn_mask=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class FullAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()

class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)

        return self.out_projection(out)



class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(int(seg_num), factor, d_model), requires_grad=True)

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        self.norm3 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        self.norm4 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        # self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        dim_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')


        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    return nn.Parameter(W_pos, requires_grad=learn_pe)



class Temporal_Layers(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, e_layers):
        super(Temporal_Layers, self).__init__()

        self.ts_d_model = d_model
        self.n_heads = n_heads
        self.ts_d_ff = d_ff
        self.ts_layers =e_layers

        self.temporal_layers = nn.ModuleList(
            [TSTEncoderLayer(self.ts_d_model, n_heads=self.n_heads, d_k=None, d_v=None,
                             d_ff=self.ts_d_ff, norm='BatchNorm', attn_dropout=0.1, dropout=0.2, res_attention=False, pre_norm=False, store_attn=False)
             for i in range(self.ts_layers)])

    def forward(self, hiddens):
        '''
        x: bs,  nvar, T
        hiddens: [bs, nvar, num_patch, d_model
        '''
        if hiddens.shape[0] == 0:
            return hiddens
        temporal_x = torch.reshape(hiddens, (hiddens.shape[0]*hiddens.shape[1],hiddens.shape[2],hiddens.shape[3]))
        for layer in self.temporal_layers:
            temporal_x = layer(temporal_x)
        temporal_x = torch.reshape(temporal_x, (hiddens.shape[0], hiddens.shape[1], hiddens.shape[2], hiddens.shape[3]))
        return temporal_x



class Channel_Layers(nn.Module):
    def __init__(self, patch_num, d_model, n_heads, d_ff, e_layers, num_slots=4):
        super(Channel_Layers, self).__init__()
        self.patch_num = patch_num
        self.ts_d_model = d_model
        self.ts_d_ff = d_ff
        self.ts_layers =2
        self.n_heads = n_heads
        factor = 10
        # self.channel_layers = nn.ModuleList(
        #     [MultiHeadSlotAttention(num_slots=4, heads=4, dim=d_model, hidden_dim=d_ff) for i in range(self.ts_layers)]
        # )

        self.channel_layers = nn.ModuleList(
            [SlotAttention(num_slots=num_slots, dim=d_model, hidden_dim=d_ff) for i in range(self.ts_layers)]
        )

        # self.channel_layers = nn.ModuleList(
        #     [TwoStageAttentionLayer(self.patch_num, factor, self.ts_d_model, self.n_heads, self.ts_d_ff) for i in range(self.ts_layers)]
        # )

    def forward(self, hiddens):
        '''
        x : bs, c, T
        '''
        if hiddens.shape[0] == 0:
            return hiddens

        ##input shape: b x c x patch_num x d_model
        b,c, n, d = hiddens.shape
        for layer in self.channel_layers:
            # print(hiddens.shape)
            # print(hiddens[:,:,1:,:].reshape(b*(n-1), c, d).shape)
            # print(layer(hiddens[:,:,1:,:].reshape(b*(n-1), c, d)).shape)
            hiddens[:,:,1:,:] = layer(hiddens[:,:,1:,:].reshape(b*(n-1), c, d)).reshape(b, c, n-1, d)
        # hiddens = self.up_to_patch(hiddens.permute(0,1,3,2)).permute(0,1,3,2)
        return hiddens




class Channel_Layers_attention(nn.Module):
    def __init__(self, patch_num, d_model, n_heads, d_ff, e_layers, num_slots=4):
        super(Channel_Layers_attention, self).__init__()
        self.patch_num = patch_num
        self.ts_d_model = d_model
        self.ts_d_ff = d_ff
        self.ts_layers =2
        self.n_heads = n_heads
        factor = 10
        # self.channel_layers = nn.ModuleList(
        #     [MultiHeadSlotAttention(num_slots=4, heads=4, dim=d_model, hidden_dim=d_ff) for i in range(self.ts_layers)]
        # )
        # self.channel_layers = nn.ModuleList(
        #     [SlotAttention(num_slots=num_slots, dim=d_model, hidden_dim=d_ff) for i in range(self.ts_layers)]
        # )
        self.channel_layers = nn.ModuleList(
            [TwoStageAttentionLayer(self.patch_num, factor, self.ts_d_model, self.n_heads, self.ts_d_ff) for i in range(self.ts_layers)]
        )

    def forward(self, hiddens):
        '''
        x : bs, c, T
        '''
        if hiddens.shape[0] == 0:
            return hiddens

        ##input shape: b x c x patch_num x d_model
        b,c, n, d = hiddens.shape
        for layer in self.channel_layers:
            # print(hiddens.shape)
            # print(hiddens[:,:,1:,:].reshape(b*(n-1), c, d).shape)
            # print(layer(hiddens[:,:,1:,:].reshape(b*(n-1), c, d)).shape)
            # 'b ts_d seg_num d_model
            hiddens = layer(hiddens)
            # hiddens[:,:,1:,:] = layer(hiddens[:,:,1:,:].reshape(b*(n-1), c, d)).reshape(b, c, n-1, d)
        # hiddens = self.up_to_patch(hiddens.permute(0,1,3,2)).permute(0,1,3,2)
        return hiddens

class Mix_layer(nn.Module):
    def __init__(self, patch_num, d_model, n_heads, d_ff, e_layers, num_slots=4):
        super(Mix_layer, self).__init__()
        self.patch_num = patch_num
        self.ts_d_model = d_model
        self.ts_d_ff = d_ff
        self.ts_layers = e_layers
        self.n_heads = n_heads
        self.channel_layers = nn.ModuleList(
            [SlotAttention(num_slots=num_slots, dim=d_model, hidden_dim=d_ff) for i in range(self.ts_layers)]
        )
        self.temporal_layers = nn.ModuleList(
            [TSTEncoderLayer(self.ts_d_model, n_heads=self.n_heads, d_k=None, d_v=None,
                             d_ff=self.ts_d_ff, norm='BatchNorm', attn_dropout=0.1, dropout=0.2, res_attention=False,
                             pre_norm=False, store_attn=False)
             for i in range(self.ts_layers)])


    def forward(self, hiddens):
        b, c, n, d = hiddens.shape
        for l in range(self.ts_layers):
            hiddens = torch.reshape(hiddens,
                                       (b*c, n, d))
            hiddens = self.temporal_layers[l](hiddens)
            hiddens = torch.reshape(hiddens,
                                       (b, c, n, d))
            hiddens[:,:,1:,:] = self.channel_layers[l](hiddens[:, :, 1:, :].reshape(b * (n - 1), c, d)).reshape(b, c, n - 1, d)
        return hiddens




##MOE
def top_p_sampling_batched_all_sequence(logits, top_p=0.9, temperature=1.0):
    """
    Apply Top-p sampling to every element in the sequence for each item in the batch.
    Returns the selected token indices and the corresponding threshold indices.

    :param logits: Logits from a language model with shape (sequence length, batch size, L)
    :param top_p: Cumulative probability threshold (float)
    :param temperature: Sampling temperature (float)
    :return: Tuple of tensors (selected token indices, threshold indices) for each position in each sequence in the batch
    """
    # Apply temperature
    logits = logits / temperature
    # print(logits)

    # Convert logits to probabilities
    # probabilities = torch.softmax(logits, dim=-1)
    # Sort probabilities and their indices in descending order
    sorted_probs, sorted_indices = torch.sort(logits, descending=True)
    # print(sorted_probs)
    # print(sorted_indices)


    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > top_p

    # mask = sorted_probs > top_p
    # print(mask)

    # Find the threshold indices
    threshold_indices = mask.long().argmax(dim=-1)
    threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
    # print(threshold_mask)

    mask = mask & ~threshold_mask
    mask = ~mask
    # print("--------------")
    # print(mask)

    sorted_indices = torch.where(mask, -1, sorted_indices)
    sorted_probs = torch.where(mask, 0.0, sorted_probs)
    return sorted_probs, sorted_indices


class MOE_layers(nn.Module):
    """
    Routes input to one of N MLP "experts"
    """

    def __init__(self, patch_num, d_model, n_heads, d_ff, e_layers, num_slots=4, channel_key=1):
        super(MOE_layers, self).__init__()
        self.top_p_threshold = 0.5
        self.num_experts = 2
        self.router = torch.nn.Linear(d_model, 2)
        self.temporal_module = Temporal_Layers(d_model, n_heads, d_ff, e_layers)
        self.channel_module = Channel_Layers(patch_num, d_model, n_heads, d_ff, e_layers, num_slots)
        self.channel_key = channel_key


    def forward(self, hidden_states):
        '''
        x: [bs, C, T]
        '''

        # bs, C, T = x.shape
        bs = hidden_states.shape[0]
        nvars = hidden_states.shape[1]

        # print(hidden_states.shape)

        if nvars != 1 and self.channel_key==1:
            temporal_hidden = torch.mean(hidden_states, dim=1)
            temporal_hidden = torch.mean(temporal_hidden, dim=1)

            route = self.router(temporal_hidden)
            route = route.unsqueeze(1)
            route = torch.nn.functional.softmax(route, dim=2)
            topk_weights, topk_ind = top_p_sampling_batched_all_sequence(route, self.top_p_threshold)
            # print(topk_weights)
            # print(topk_ind)
            output_total = torch.zeros_like(hidden_states).to(hidden_states)
            topk_weights = topk_weights.view(-1, topk_weights.size(2))
            topk_ind = topk_ind.view(-1, topk_ind.size(2))

            ###这里进行负载 balance loss计算
            P_e = topk_weights.sum(dim=0) / bs  # [num_experts]
            # 计算 Load Balance Loss
            balance_loss = self.num_experts * (P_e.pow(2).sum())

            ##交叉熵计算
            log_routing_weight = torch.log(topk_weights + 1e-9)  # 避免 log(0)
            # 2. 按交叉熵公式计算
            dynamic_loss = -torch.sum(topk_weights * log_routing_weight, dim=-1).mean()

            sample_ind, expert_ind = torch.where(topk_ind == 0)
            hidden = hidden_states[sample_ind, :, :]
            # print(hidden.shape[0])
            expert_output = self.channel_module(hidden)
            # print(expert_output.shape)
            # print(topk_weights[sample_ind, expert_ind].shape)
            output_total[sample_ind] = topk_weights[sample_ind, expert_ind].unsqueeze(1).unsqueeze(2).unsqueeze(3)*expert_output

            # sample_ind_1, expert_ind_1 = torch.where(topk_ind == 1)
            # output_total[sample_ind_1] += hidden_states[sample_ind_1, :, :]

            # #output_total = self.temporal_module(hidden_states)
            output_total = output_total + self.temporal_module(hidden_states)
        else:
            balance_loss = 0
            dynamic_loss = 0
            output_total = self.temporal_module(hidden_states)

        return output_total, balance_loss, dynamic_loss



class layers(nn.Module):
    """
    Routes input to one of N MLP "experts"
    """

    def __init__(self, patch_num, d_model, n_heads, d_ff, e_layers, num_slots=20, channel_key=1):
        super(layers, self).__init__()
        self.top_p_threshold = 0.5
        self.num_experts = 2
        self.router = torch.nn.Linear(d_model, 2)
        self.mixer_module = Mix_layer(patch_num, d_model, n_heads, d_ff, e_layers, num_slots)


    def forward(self, hidden_states):
        '''
        x: [bs, C, T]
        '''
        output = self.mixer_module(hidden_states)
        balance_loss = 0
        dynamic_loss = 0
        return output, balance_loss, dynamic_loss






class MOE_layers_without_router(nn.Module):
    """
    Routes input to one of N MLP "experts"
    """

    def __init__(self, patch_num, d_model, n_heads, d_ff, e_layers, num_slots=4, channel_key=1):
        super(MOE_layers_without_router, self).__init__()
        self.top_p_threshold = 0.5
        self.num_experts = 2
        self.router = torch.nn.Linear(d_model, 2)
        self.temporal_module = Temporal_Layers(d_model, n_heads, d_ff, e_layers)
        self.channel_module = Channel_Layers(patch_num, d_model, n_heads, d_ff, e_layers, num_slots)
        self.channel_key = channel_key


    def forward(self, hidden_states):
        '''
        x: [bs, C, T]
        '''

        output_total = self.temporal_module(hidden_states) + self.channel_module(hidden_states)

        balance_loss = 0
        dynamic_loss = 0


        return output_total, balance_loss, dynamic_loss



class MOE_layers_attention(nn.Module):
    """
    Routes input to one of N MLP "experts"
    """

    def __init__(self, patch_num, d_model, n_heads, d_ff, e_layers, num_slots=4, channel_key=1):
        super(MOE_layers_attention, self).__init__()
        self.top_p_threshold = 0.5
        self.num_experts = 2
        self.router = torch.nn.Linear(d_model, 2)
        self.temporal_module = Temporal_Layers(d_model, n_heads, d_ff, e_layers)
        self.channel_module = Channel_Layers_attention(patch_num, d_model, n_heads, d_ff, e_layers, num_slots)
        self.channel_key = channel_key


    def forward(self, hidden_states):
        '''
        x: [bs, C, T]
        '''

        output_total = self.temporal_module(hidden_states) + self.channel_module(hidden_states)

        balance_loss = 0
        dynamic_loss = 0


        return output_total, balance_loss, dynamic_loss

