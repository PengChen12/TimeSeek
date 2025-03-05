
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
import torch.nn.functional as F
import math
import numpy as np
from einops import reduce, rearrange, repeat
from src.callback.decompose import st_decompose

import torchvision.models as models
from torchvision.transforms import Resize
from torchvision import transforms
from PIL import Image

            
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
                 head_type = "prediction", individual = False, img_size = 64,
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):

        super().__init__()
        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'

        # Basic
        self.num_patch = num_patch
        self.target_dim=target_dim
        self.img_size = img_size
        self.out_patch_num = math.ceil(target_dim / patch_len)
        self.resize = Resize([img_size,img_size], interpolation=Image.BILINEAR)
        self.toPIL = transforms.ToPILImage()
        self.flatten = nn.Flatten(start_dim=1)

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
        
        self.image_encoder = ImageEncoder()

        # Decoder
        self.decoder = Decoder(d_layers, patch_len=patch_len, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)

        # Head
        self.n_vars = c_in
        self.head_type = head_type
        self.mask_mode = mask_mode
        self.mask_nums = mask_nums
        self.d_model  = d_model

        # prejection head
        # self.projection_img = ProjectionHead(input_dims=d_model, output_dims=int(d_model/4), hidden_dims=int(d_model/2))
        # self.projection_cls = ProjectionHead(input_dims=d_model, output_dims=int(d_model/4), hidden_dims=int(d_model/2))

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = decoder_PredictHead(d_model, patch_len, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)

    def decoder_predict(self, bs, n_vars, dec_cross):
        """
        dec_cross: tensor [bs x  n_vars x num_patch x d_model]
        """
        # dec_in = self.decoder_embedding.expand(bs, n_vars, self.out_patch_num, -1)
        dec_in = dec_cross[:,:,-1,:].unsqueeze(2).expand(-1,-1,self.out_patch_num,-1)
        

        dec_in = dec_in + self.pos[-self.out_patch_num:,:]
        decoder_output = self.decoder(dec_in, dec_cross)
        decoder_output = decoder_output.transpose(2,3)

        return decoder_output
    
    def ts_resize(self, x):
        '''
        input: [bs, n_vars, num_patch, patch_len]
        output: [bs*n_vars, 3, 224, 224]
        '''
        bs, num_patch, n_vars, patch_len = x.shape
        seq_len = num_patch * patch_len

        # flatten
        x = x.permute(0,1,3,2).reshape(bs, seq_len, n_vars)
        # find period
        # periods, _ = FFT_for_Period(x, k=3)
        periods = [4,24,48]

        # # padding
        # x = self.padding(x, periods[0])
        # # channel independent
        # x = x.permute(0,2,1).reshape(bs*n_vars, -1)
        # # resize
        # x_2d = x.reshape(-1, x.shape[-1]//periods[0], periods[0]).unsqueeze(1) 
        # x_resize = self.resize(x_2d).expand(-1,3,-1,-1)

        x_list = []
        for p in periods:
            x_in = self.padding(x, p)
            x_in = x_in.permute(0,2,1).reshape(bs*n_vars, -1)
            x_2d = x_in.reshape(-1, x_in.shape[-1]//p, p).unsqueeze(1)
            x_resize = self.resize(x_2d).expand(-1,3,-1,-1).unsqueeze(0)
            x_list.append(x_resize) 
        x_resize = torch.cat(x_list, dim=0)

        # x_list = []
        # for p in periods:
        #     x_in = self.padding(x, p)
        #     x_in = x_in.permute(0,2,1).reshape(bs*n_vars, -1)
        #     x_list.append(self.resize(x_in.reshape(-1, x_in.shape[-1]//p, p).unsqueeze(1)))
        # x_resize = torch.cat(x_list, dim=1)
        # pic = self.toPIL(x_resize[0][0])
        # pic.save('/home/Decoder_version_2/visualization/C_top_3.jpg')

        return x_resize
    
    def padding(self, x, periods):
        bs, seq_len, n_vars = x.shape
        if seq_len % periods != 0:
            padding_left = periods - seq_len%periods
            padding = torch.zeros([bs, padding_left, n_vars]).to(x.device)
            return torch.cat([padding, x], dim = 1)
        else:
            return x

    def forward(self, z):                             
        """
        z_masked : tensor [bs x num_patch x n_vars x patch_len x mask_nums]
        z_orginal : tensor [bs x num_patch x n_vars x patch_len]
        """   

        z_masked,z_original = z
        bs, num_patch, n_vars, patch_len, mask_nums = z_masked.shape 
        z_masked = z_masked.permute(4,0,1,2,3).reshape(bs*mask_nums, num_patch, n_vars, patch_len)
        z = torch.cat((z_original, z_masked), dim=0) 

        # x_2d
        # z_2d = self.ts_resize(z_original).reshape(3*bs*n_vars, 3, self.img_size, self.img_size)
        # # img_tokens = self.image_encoder(z_2d).reshape(-1, bs, n_vars, self.d_model).permute(1,2,0,3)  # [bs x n_vars x 3 x d_model]
        # img_tokens = self.image_encoder(z_2d).reshape(-1, bs, n_vars, self.d_model).permute(1,2,3,0)  # [bs x n_vars x d_model x 3]
        # # img_tokens = img_tokens.mean(3)
        # img_tokens = F.max_pool1d(
        #     img_tokens.reshape(bs*n_vars*self.d_model, 3),
        #     kernel_size = 3,
        # ).reshape(bs, n_vars, self.d_model) # [bs x n_vars x d_model]

        # img_tokens_cat = img_tokens.unsqueeze(0).expand((mask_nums + 1), -1, -1, -1, -1).reshape(bs*(mask_nums + 1), n_vars, -1, self.d_model)

        # tokenizer
        cls_tokens = self.cls_embedding.expand(bs*(mask_nums + 1), n_vars, -1, -1)
        z = self.embedding(z).permute(0,2,1,3) # [bs*(1+mask_nums) x n_vars x num_patch x d_model]
        z = torch.cat((cls_tokens, z), dim=2)  # [bs*(1+mask_nums) x n_vars x (1 + num_patch) x d_model]
        z = self.drop_out(z + self.pos[:1 + self.num_patch, :])

        # encoder 
        z = torch.reshape(z, (-1, 1 + self.num_patch, self.d_model)) # [bs*(1+mask_nums)*n_vars x num_patch x d_model]
        z = self.encoder(z)
        z = torch.reshape(z, (-1, n_vars, 1 + self.num_patch, self.d_model)) # [bs*(1+mask_nums), n_vars x num_patch x d_model]
        z_masked = z[bs:,:,:,:] # [bs*mask_nums, n_vars x num_patch x d_model]
        z_original = z[0: bs,:,:,:] # [bs, n_vars x num_patch x d_model]
        cls_tokens_original = z_original[:,:,0,:]

        # aggregation
        if self.mask_mode == "multi" or self.mask_mode == "freq_multi":
            z_masked = z_masked.reshape(self.mask_nums, -1, n_vars, num_patch+1, self.d_model,).mean(0).transpose(2,3)     # z_masked: [bs x nvars x d_model x num_patch]    

        # decoder_prediction
        z_predict = self.decoder_predict(bs, n_vars, z_original)
        z_predict = self.head(z_predict) 
        z_predict = z_predict.permute(0,1,3,2) # [bs x num_patch x patch_len x n_vars]
        z_predict = z_predict.reshape(z_predict.shape[0],-1,z_predict.shape[3])
        z_predict = z_predict[:,:self.target_dim, :]  

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
        self.linear = ProjectionHead(n_vars*d_model, n_classes, 128)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        bs, n_vars, d_model, num_patch = x.shape       
        
        x = x[:,:,:,0]# only consider the First item in the sequence, x: bs x nvars x d_model
        # x = torch.mean(x, dim = -1)

        # x = torch.mean(x, dim = -1)
        # x = x.reshape(bs*n_vars, d_model, -1)
        # x = torch.max_pool1d(x, num_patch).squeeze(-1)
        # x = x.reshape(bs, n_vars, d_model)


        x = x.reshape(bs, n_vars*d_model)
        x = self.flatten(x)         # x: bs x nvars * d_model
        y = self.linear(x)         # y: bs x n_classes
        return y

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
        

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1.调用模型
        self.encoder = models.resnet18(pretrained=True)
        # 2.提取fc层中固定的参数
        fc_features = self.encoder.fc.in_features
        # 3.修改输出的类别为10
        self.encoder.fc = nn.Identity()
    
    def forward(self, x):
        return self.encoder(x)

# class ImageEncoder(nn.Module):
#     def __init__(self, out_dim, finetune_type='none'):
#         super().__init__()
#         # 1.调用模型
#         # ResNet = models.resnet18(pretrained=True)
#         self.encoder = models.vit_b_16(weights = models.ViT_B_16_Weights.DEFAULT)
#         # 2.改变输出维度
#         self.encoder.heads[0] = nn.Linear(self.encoder.heads[0].in_features, out_dim)
#         # # 3.冻结参数
#         # if finetune_type != 'full':
#         #     for n, param in self.encoder.named_parameters():
#         #         if 'ln' == finetune_type:
#         #             param.requires_grad = 'norm' in n
#         #         elif 'bias' == finetune_type:
#         #             param.requires_grad = 'bias' in n
#         #         elif 'none' == finetune_type:
#         #             param.requires_grad = False
#         #         elif 'mlp' in finetune_type:
#         #             param.requires_grad = '.mlp.' in n
#         #         elif 'attn' in finetune_type:
#         #             param.requires_grad = '.attn.' in n
    
#     def forward(self, x):
#         '''
#         input: [bs, c, h, w]
#         output: [bs, out_dim]
#         '''
#         return self.encoder(x)



