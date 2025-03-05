
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
from src.models.layers.decoder_cnn import Decoder

import torch.fft as fft
import math
import numpy as np
from einops import reduce, rearrange, repeat
from src.callback.decompose import st_decompose

            
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
                 e_layers:int=3, d_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='sincos', learn_pe:bool=False, head_dropout = 0, 
                 head_type = "prediction", individual = False,
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):

        super().__init__()
        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'

        # Basic
        self.num_patch = num_patch
        self.target_dim=target_dim
        self.out_patch_num = math.ceil(target_dim / patch_len)

        # Embedding
        self.embedding = nn.Linear(patch_len, d_model)
        self.decoder_embedding = nn.Parameter(torch.randn(1, 1, 1, d_model),requires_grad=True)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, 1, d_model),requires_grad=True)

        # Position Embedding
        self.pos = positional_encoding(pe, learn_pe, 1 + num_patch + self.out_patch_num, d_model)
        self.drop_out = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=e_layers, 
                                    store_attn=store_attn)
        
        # Decoder
        self.decoder = Decoder(d_layers, patch_len=patch_len, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)

        # Head
        self.n_vars = c_in
        self.head_type = head_type
        self.mask_mode = mask_mode
        self.mask_nums = mask_nums
        self.d_model  = d_model

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = decoder_PredictHead(d_model, patch_len, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)

    def decoder_predict(self, bs, n_vars, dec_cross, out_patch_num, pos):
        """
        dec_cross: tensor [bs x  n_vars x num_patch x d_model]
        """
        dec_in = self.decoder_embedding.expand(bs, n_vars, out_patch_num, -1)
        dec_in = dec_in + pos[-out_patch_num:,:]
        decoder_output = self.decoder(dec_in, dec_cross)
        decoder_output = decoder_output.transpose(2,3)

        return decoder_output


    def forward(self, z):                             
        """
        z_masked : tensor [bs x num_patch x n_vars x patch_len x mask_nums]
        z_orginal : tensor [bs x num_patch x n_vars x patch_len]
        """   

        z_masked, z_original, out_patch_num = z

        bs, num_patch, n_vars, patch_len, mask_nums = z_masked.shape 
        z_masked = z_masked.permute(4,0,1,2,3).reshape(bs*mask_nums, num_patch, n_vars, patch_len)
        z = torch.cat((z_original, z_masked), dim=0) 

        pos = PositionalEncoding(1 + num_patch + out_patch_num, self.d_model, normalize=True).to(z.device)

        # tokenizer
        cls_tokens = self.cls_embedding.expand(bs*(mask_nums + 1), n_vars, -1, -1)
        z = self.embedding(z).permute(0,2,1,3) # [bs*(1+mask_nums) x n_vars x num_patch x d_model]
        z = torch.cat((cls_tokens, z), dim=2)  # [bs*(1+mask_nums) x n_vars x (1 + num_patch) x d_model]
        z = self.drop_out(z + pos[:1 + num_patch, :])

        # encoder 
        z = torch.reshape(z, (-1, 1 + num_patch, self.d_model)) # [bs*(1+mask_nums)*n_vars x num_patch x d_model]
        z = self.encoder(z)
        z = torch.reshape(z, (-1, n_vars, 1 + num_patch, self.d_model)) # [bs*(1+mask_nums), n_vars x num_patch x d_model]
        z_masked = z[bs:,:,:,:] # [bs*mask_nums, n_vars x num_patch x d_model]
        z_original = z[0:bs,:,:,:] # [bs, n_vars x num_patch x d_model]

        # aggregation
        if self.mask_mode == "multi" or self.mask_mode == "freq_multi" or self.mask_mode == "freq_multi_rio":
            z_masked = z_masked.reshape(self.mask_nums, -1, n_vars, num_patch + 1, self.d_model,).mean(0).transpose(2,3)     # z_masked: [bs x nvars x d_model x num_patch]    

        # decoder_prediction
        z_predict = self.decoder_predict(bs, n_vars, z_original[:,:,:,:], out_patch_num, pos)
        z_predict = self.head(z_predict)  
        z_predict = z_predict.permute(0,1,3,2) # [bs x num_patch x patch_len x n_vars]
        z_predict = z_predict.reshape(z_predict.shape[0],-1,z_predict.shape[3])

        # recontruction
        z_reconstruct = self.head(z_masked[:,:,:,1:])

        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain

        return z_reconstruct, z_predict


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x


class decoder_PredictHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,3,1)                  # [bs x num_patch x patch_len x nvars]
        return x.reshape(x.shape[0],-1,x.shape[3])


class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, n_embedding=32,
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.n_embedding=n_embedding         

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)     


        #(
        #discrete encoding: projection of feature vectors onto a discrete vector space
        if self.n_embedding!=0:
            self.W_D = nn.Linear(patch_len, d_model)
            self.vq_embedding = nn.Embedding(self.n_embedding, d_model)
            self.vq_embedding.weight.data.uniform_(-1.0 / d_model,
                                                1.0 / d_model)

        # if self.n_embedding!=0:
        #     self.W_D_s = nn.Linear(patch_len, d_model)
        #     self.W_D_t = nn.Linear(patch_len, d_model)
        #     self.W_D_r = nn.Linear(patch_len, d_model)
        #     self.vq_embedding_s = nn.Embedding(self.n_embedding, d_model)
        #     self.vq_embedding_s.weight.data.uniform_(-1.0 / d_model,
        #                                         1.0 / d_model)
        #     self.vq_embedding_t = nn.Embedding(self.n_embedding, d_model)
        #     self.vq_embedding_t.weight.data.uniform_(-1.0 / d_model,
        #                                         1.0 / d_model)
            
        #)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)

    def forward(self, x) -> Tensor:       
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, num_patch, n_vars, patch_len = x.shape      # x: [bs x num_patch x nvars x d_model]
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars): 
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x1 = torch.stack(x_out, dim=2)
        else:
            x1 = self.W_P(x)                                                    
        x1 = x1.transpose(1,2)
        #(
        # if self.n_embedding!=0:
        #     decompose = st_decompose(kernel_size=3)

        #     #svq
        #     trend, season, res = decompose(x.transpose(2, 3).reshape([x.shape[0],x.shape[1]*x.shape[3],x.shape[2]]))
        #     trend=trend.reshape(x.shape[0],x.shape[1],x.shape[3],x.shape[2]).transpose(2, 3)
        #     season=season.reshape(x.shape[0],x.shape[1],x.shape[3],x.shape[2]).transpose(2, 3)
        #     res=res.reshape(x.shape[0],x.shape[1],x.shape[3],x.shape[2]).transpose(2, 3)

        #     #cvq
        #     trend, season, res = decompose(x.transpose(1, 2).reshape([x.shape[0],x.shape[2],x.shape[1]*x.shape[3]]))
        #     trend=trend.reshape(x.shape[0],x.shape[2],x.shape[1],x.shape[3]).transpose(1, 2)
        #     season=season.reshape(x.shape[0],x.shape[2],x.shape[1],x.shape[3]).transpose(1, 2)
        #     res=res.reshape(x.shape[0],x.shape[2],x.shape[1],x.shape[3]).transpose(1, 2)

        #     season=self.W_D_s(season).transpose(1, 3)
        #     trend=self.W_D_t(trend).transpose(1, 3)
        #     res=self.W_D_r(res).transpose(1, 3)
        #     x2=season+trend+res
        #     embedding_s = self.vq_embedding_s.weight.data
        #     embedding_t = self.vq_embedding_t.weight.data
        #     N, C, H, W = x2.shape
        #     K, _ = embedding_s.shape
        #     embedding_s_broadcast = embedding_s.reshape(1, K, C, 1, 1)
        #     embedding_t_broadcast = embedding_t.reshape(1, K, C, 1, 1)
        #     season_broadcast = season.reshape(N, 1, C, H, W)
        #     trend_broadcast = trend.reshape(N, 1, C, H, W)
        #     distance_s = torch.sum((embedding_s_broadcast - season_broadcast) ** 2, 2)
        #     distance_t = torch.sum((embedding_t_broadcast - trend_broadcast) ** 2, 2)
        #     nearest_neighbor_s = torch.argmin(distance_s, 1)
        #     nearest_neighbor_t = torch.argmin(distance_t, 1)
        #     xq_s = self.vq_embedding_s(nearest_neighbor_s).permute(0, 3, 1, 2)
        #     xq_t = self.vq_embedding_t(nearest_neighbor_t).permute(0, 3, 1, 2)
        #     xq=xq_s+xq_t+res

        if self.n_embedding!=0:
            x2 = self.W_D(x)
            x2 = x2.transpose(1, 3)
            embedding = self.vq_embedding.weight.data
            N, C, H, W = x2.shape
            K, _ = embedding.shape
            embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
            x2_broadcast = x2.reshape(N, 1, C, H, W)
            distance = torch.sum((embedding_broadcast - x2_broadcast) ** 2, 2)
            
            nearest_neighbor = torch.argmin(distance, 1)
            xq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)

            
            # soft_vq=3
            # ds,nearest_neighbors=torch.topk(distance, soft_vq, dim=1, largest = False)
            # xq =self.vq_embedding(nearest_neighbors[:,0,:,:])*torch.unsqueeze((torch.sum(ds,dim=1)-ds[:,0,:,:]),3)
            # for i in range(1,soft_vq):
            #     xq += self.vq_embedding(nearest_neighbors[:,i,:,:])*torch.unsqueeze((torch.sum(ds,dim=1)-ds[:,i,:,:]),3)
            # xq=(xq/((soft_vq-1)*torch.unsqueeze(torch.sum(ds,dim=1),3))).permute(0, 3, 1, 2)

            # make C to the second dim
            x2 = x2.transpose(1, 3)
            x2 = x2.transpose(1, 2)
            xq = xq.transpose(1, 3)
            xq = xq.transpose(1, 2)
            # stop gradient
            decoder_input = x2 + (xq - x2).detach()

            u = torch.reshape(decoder_input+x1, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
            u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]

            # Encoder
            z = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]
            z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]
            z = z.permute(0,1,3,2)                                                # z: [bs x nvars x d_model x num_patch]

            return z, x2, xq
        #)  
        else: 
                                                  # x: [bs x nvars x num_patch x d_model]        
            u = torch.reshape(x1, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
            u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]
            
            # Encoder
            z = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]
            z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]
            z = z.permute(0,1,3,2)                                                 # z: [bs x nvars x d_model x num_patch]

            return z
    
    
# Cell
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



