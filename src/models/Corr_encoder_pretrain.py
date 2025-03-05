
__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from ..models.layers.pos_encoding import *
from ..models.layers.basics import *
from ..models.layers.attention import *
from src.models.layers.decoder import Decoder

import torch.fft as fft
import math
import numpy as np
from einops import reduce, rearrange, repeat
from src.callback.decompose import st_decompose

            
# Cell
# Cell
class PatchTST(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int, mask_mode:str = 'patch',mask_nums:int = 3,
                 e_layers:int=3, d_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, codebook_size=128, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='sincos', learn_pe:bool=False, head_dropout = 0, 
                 head_type = "prediction", individual = False,
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):

        super().__init__()
        # Basic
        
        #        
        self.codebook_size=codebook_size

        # Encoder
        self.corr_module = Corr_module(n_vars=c_in, patch_len=patch_len, num_patch=num_patch, n_layers=3, d_model=d_model, 
                                       n_heads=n_heads, d_ff=d_ff, codebook_size=codebook_size, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                       act=act, pre_norm=pre_norm, res_attention=res_attention, store_attn=store_attn, pe=pe,learn_pe=learn_pe)


    def forward(self, z):                             
        """
        z_masked : tensor [bs x num_patch x n_vars x patch_len x mask_nums]
        z_orginal : tensor [bs x num_patch x n_vars x patch_len]
        """   
        if self.codebook_size !=0:

            z_d, z_q ,z,z_q_g= self.corr_module(z)
            return z_d, z_q ,z,z_q_g

        else:

            z_d, z_q = self.corr_module(z)
            return z_d, z_q


class Corr_module(nn.Module):
    def __init__(self, n_vars:int, patch_len:int, num_patch:int,n_layers:int=3, d_model=128, 
                 n_heads=16, d_ff:int=256, codebook_size=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., 
                 act:str="gelu", res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='sincos', learn_pe:bool=False):

        super().__init__()

        # Embedding
        self.corr_embedding = nn.Linear(patch_len, d_model)
        self.maxpool=nn.MaxPool1d(kernel_size=num_patch)

        # Codebook
        self.codebook_size=codebook_size
        if self.codebook_size!=0:
            self.vq_embedding = nn.Embedding(self.codebook_size, d_model)
            self.vq_embedding.weight.data.uniform_(-1.0 / d_model, 1.0 / d_model)
        
        # Position Embedding
        self.corr_pos = positional_encoding(pe, learn_pe, 3000, d_model)
        self.corr_drop_out = nn.Dropout(dropout)

        # Encoder
        self.corr_encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)
        
        self.proj_1 = ProjectionHead(input_dims=d_model, output_dims=128, hidden_dims=192)
        self.proj_2 = ProjectionHead(input_dims=d_model, output_dims=128, hidden_dims=192)
        
        self.n_vars = n_vars
        self.num_patch = num_patch
        self.d_model  = d_model

    def channel_mask(self, input, mask_ratio=0.4):
        bs, n_vars, d_model = input.shape
        mask_num=int(mask_ratio*n_vars)

        ranmdom_matrix = torch.rand(bs,n_vars)
        mask_matrix = torch.ones(bs, n_vars).to('cuda')
        mask_index = torch.topk(ranmdom_matrix, k=mask_num, dim=1, largest=True).indices.to('cuda')
        mask_matrix.scatter_(1, mask_index, 0)
        # print(mask_matrix)
        mask_matrix = mask_matrix.unsqueeze(-1).expand(-1, -1, d_model)
        return torch.mul(input, mask_matrix)
    
    def channel_partial_mask(self, input, mask_ratio=0.4):
        bs, num_patch, n_vars, d_model = input.shape
        mask_num=int(mask_ratio*num_patch)

        ranmdom_matrix = torch.rand(bs,num_patch,n_vars)
        mask_matrix = torch.ones(bs, num_patch,n_vars).to('cuda')
        mask_index = torch.topk(ranmdom_matrix, k=mask_num, dim=1, largest=True).indices.to('cuda')
        mask_matrix.scatter_(1, mask_index, 0)
        # print(mask_matrix)
        mask_matrix = mask_matrix.unsqueeze(-1).expand(-1, -1, -1, d_model)
        return torch.mul(input, mask_matrix)

    def forward(self, z):                             
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """   
        bs, num_patch, n_vars, patch_len = z.shape
        z = self.corr_embedding(z)  # [bs x num_patch x n_vars x d_model]
        z = self.corr_drop_out(z + self.corr_pos[:n_vars, :])

        z_d=self.channel_partial_mask(input=z, mask_ratio=0.4)  # [bs x num_patch x n_vars x d_model]
        z_d=self.maxpool(z_d.permute(0,2,3,1).reshape(-1,self.d_model,num_patch)).reshape(bs,n_vars,self.d_model) # [bs x n_vars x d_model]

        z=self.maxpool(z.permute(0,2,3,1).reshape(-1,self.d_model,num_patch)).reshape(bs,n_vars,self.d_model) # [bs x n_vars x d_model]
        
        if self.codebook_size!=0:
            embedding = self.vq_embedding.weight.data
            B, N, D = z.shape
            K, _ = embedding.shape
            embedding_broadcast = embedding.reshape(1, K, 1, D)
            z_broadcast = z.reshape(B, 1, N, D)
            distance = torch.sum((embedding_broadcast - z_broadcast) ** 2, 3)
            # make C to the second dim
            # z_token_q = self.vq_embedding(torch.argmin(distance, 1))
            z_q_g = torch.mean(self.vq_embedding(torch.topk( distance, k=1, dim=1, largest=False)[1]),dim=1)
            # stop gradient
            z_q = z + (z_q_g - z).detach()
        else:
            z_q=z

        # z_d=self.channel_mask(input=z, mask_ratio=0.4)  # [bs x n_vars x d_model]
        # encoder 
        z_d = self.corr_encoder(z_d)  # [bs x n_vars x d_model]

        # pretrain
        z_q = self.proj_1(z_q)
        z_d = self.proj_2(z_d)

        if self.codebook_size!=0:
            return z_d, z_q, z, z_q_g
        else:
            return z_d, z_q


class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
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


    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            # attention_mask = causal_attention_mask(src.shape[1]).to(src.device)
            # src2, attn = self.self_attn(src, src, src, attn_mask=attention_mask)
            src2, attn = self.self_attn(src, src, src)
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


def causal_attention_mask(seq_length):
    """
    创建一个因果注意力掩码。掩码中的每个位置 (i, j) 
    表示在计算第i个位置的attention时, 第j个位置是否可以被看见。
    如果j <= i, 这个位置被设为1(可见), 否则设为0(不可见)。
    
    Args:
        seq_length (int): 序列的长度
    
    Returns:
        torch.Tensor: 因果注意力掩码，大小为 (seq_length, seq_length)
    """
    mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
    return mask


class ProjectionHead(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=128):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        # projection head for finetune
        self.proj_head = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )

        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        bs, n_vars, d_model = x.shape
        x = self.repr_dropout(self.proj_head(x.reshape(bs*n_vars, d_model)))
        x = x.reshape(bs, n_vars, self.output_dims)

        return x