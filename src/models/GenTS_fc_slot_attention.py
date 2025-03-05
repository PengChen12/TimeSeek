__all__ = ['GenTS']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from ..models.layers.pos_encoding import *
from ..models.layers.basics import *
from ..models.layers.attention import *

from src.models.layers.decoder_cnn import Decoder
from src.models.layers.MOE_scale_layer import MOE_layers, layers, MOE_layers_attention

import torch.fft as fft
import math
import numpy as np
from einops import reduce, rearrange, repeat
from src.callback.decompose import st_decompose
from timm.models.layers import trunc_normal_

from src.models.layers.cka import CudaCKA


# Cell
class GenTS(nn.Module):
    """
    Output dimension:
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """

    def __init__(self, c_in: int, target_dim: int, patch_len: int, stride: int, num_patch: int,
                 mask_mode: str = 'patch', mask_nums: int = 3,
                 e_layers: int = 3, d_layers: int = 3, d_model=128, n_heads=16, shared_embedding=True, d_ff: int = 256,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.4, dropout: float = 0., act: str = "gelu",
                 res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'sincos', learn_pe: bool = False, head_dropout=0,
                 head_type="prediction", individual=False, channel_key=1, num_slots=4, is_finetune=0,
                 y_range: Optional[tuple] = None, verbose: bool = False, **kwargs):

        super().__init__()
        assert head_type in ['pretrain', 'prediction', 'regression',
                             'classification'], 'head type should be either pretrain, prediction, or regression'

        # Basic
        self.num_patch = num_patch
        self.target_dim = target_dim ###预测长度
        self.out_patch_num = math.ceil(target_dim / patch_len)
        self.is_finetune = is_finetune
        if self.is_finetune == 0:
            self.target_patch_len = 48
        else:
            self.target_patch_len = patch_len

        # self.target_patch_len=48
        self.stride =patch_len

        self.scale_conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)
        self.scale_conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=4, padding=0)

        # Embedding
        self.embedding = nn.Linear(self.target_patch_len, d_model)
        self.decoder_embedding = nn.Parameter(torch.randn(1, 1, 1, d_model), requires_grad=True)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, 1, d_model), requires_grad=True)
        self.sep_embedding = nn.Parameter(torch.randn(1, 1, 1, d_model), requires_grad=True)

        # Position Embedding
        self.pos = positional_encoding(pe, learn_pe, 1 + num_patch + self.out_patch_num, d_model)
        self.drop_out = nn.Dropout(dropout)

        # Encoder
        ##这里可能不需要输入长度，直接去除掉
        self.encoder_layer_scale2 = MOE_layers_attention(self.num_patch/4, d_model, n_heads, d_ff, e_layers, num_slots, channel_key) ##最粗尺度
        self.encoder_layer_scale1 = MOE_layers_attention(self.num_patch/2, d_model, n_heads, d_ff, e_layers, num_slots, channel_key) ##中间尺度
        self.encoder_layer_scale0 = MOE_layers_attention(self.num_patch, d_model, n_heads, d_ff, e_layers, num_slots, channel_key) ##最细尺度

        self.gatelayer1 = GateLayer_new(d_model)
        self.gatelayer2 = GateLayer_new(d_model)

        ##1D反卷积
        self.scale_convtranspose1 = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=2, stride=2)
        self.scale_convtranspose2 = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=2, stride=2)


        # Decoder
        ##这个不需要进行修改
        self.decoder_linear = nn.Linear(d_model, d_model)
        self.decoder = Decoder(d_layers, patch_len=patch_len, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                               attn_dropout=attn_dropout, dropout=dropout)


        # Head
        self.n_vars = c_in
        self.head_type = head_type
        self.mask_mode = mask_mode
        self.mask_nums = mask_nums
        self.d_model = d_model
        self.patch_len = patch_len

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len,
                                     head_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = decoder_PredictHead(d_model, self.patch_len, self.target_patch_len, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)


    def get_dynamic_weights(self, n_preds):
        """
        Generate dynamic weights for the replicated tokens. This example uses a linearly decreasing weight.
        You can modify this to use other schemes like exponential decay, sine/cosine, etc.
        """
        # Linearly decreasing weights from 1.0 to 0.5 (as an example)
        weights = torch.linspace(1.0, 0.5, n_preds)
        return weights

    def decoder_predict(self, bs, n_vars, dec_cross, enc_out):
        """
        dec_cross: tensor [bs x  n_vars x num_patch x d_model]
        """

        dec_in = self.decoder_linear(dec_cross[:, :, -1, :]).unsqueeze(2).expand(-1, -1, self.out_patch_num, -1)
        weights = self.get_dynamic_weights(self.out_patch_num).to(dec_in.device)
        dec_in = dec_in * weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        decoder_output = self.decoder(dec_in, enc_out)
        decoder_output = decoder_output.transpose(2, 3)

        return decoder_output


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_patch(self, xb, patch_len, stride):
        """
        xb: [bs x seq_len x n_vars]
        """
        seq_len = xb.shape[1]
        num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
        tgt_len = patch_len + stride * (num_patch - 1)
        s_begin = seq_len - tgt_len

        xb = xb[:, s_begin:, :]  # xb: [bs x tgt_len x nvars]
        xb = xb.unfold(dimension=1, size=patch_len, step=stride)  # xb: [bs x num_patch x n_vars x patch_len]
        return xb, num_patch

    def forward(self, z):
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """
        '''
        z: tensor [bs x length x n_vars]
        '''
        # padding
        bs, l, nvars = z.shape
        ##降采样
        z2 = self.scale_conv2(z.reshape(bs * nvars, 1, l)).reshape(bs, -1, nvars)
        z1 = self.scale_conv1(z.reshape(bs * nvars, 1, l)).reshape(bs, -1, nvars)
        if z.shape[1] < self.patch_len:
            padding = torch.zeros([z.shape[0], self.patch_len - z.shape[1], z.shape[2]]).to(z.device)
            z = torch.cat((z, padding), dim=1)

        z, num_patch = self.create_patch(z, self.patch_len, self.stride)  # xb: [bs x seq_len x n_vars]
        z1, num_patch1 = self.create_patch(z1, self.patch_len, self.stride)
        z2, num_patch2 = self.create_patch(z2, self.patch_len, self.stride)


        if self.is_finetune == 0: ##zero-shot
            z = resize(z, self.target_patch_len)
            z1 = resize(z1, self.target_patch_len)
            z2 = resize(z2, self.target_patch_len)

        # z = resize(z, self.target_patch_len)
        # z1 = resize(z1, self.target_patch_len)
        # z2 = resize(z2, self.target_patch_len)


        bs, num_patch, n_vars, patch_len = z.shape


        # tokenizer
        cls_tokens = self.cls_embedding.expand(bs, n_vars, -1, -1)
        z2 = self.embedding(z2).permute(0,2,1,3)
        z2 = torch.cat((cls_tokens, z2), dim=2)
        z2 = self.drop_out(z2 + self.pos[:1 + int(self.num_patch//4), :])
        z2, b_l_2, d_l_2 = self.encoder_layer_scale2(z2)

        z2 = F.interpolate(z2[:,:,1:,:], size=(self.num_patch//2, self.d_model), mode='bilinear',
                                     align_corners=False)
        # z2 = self.scale_convtranspose2(z2[:,:,1:,:].reshape(bs*nvars, self.d_model, -1)).reshape(bs, nvars, -1, self.d_model)


        z1 = self.embedding(z1).permute(0, 2, 1, 3)
        c2 = z2.shape[2]
        z1 = z1.clone()
        z1[:,:,:c2,:] = self.gatelayer1(z2, z1[:,:,:c2,:])

        mean2 = torch.mean(z1, dim=2).unsqueeze(2).expand(-1,-1,c2,-1)
        z1 = z1 - mean2
        z1= torch.cat((cls_tokens, z1), dim=2)
        z1 = self.drop_out(z1 + self.pos[:1 + int(self.num_patch // 2), :])
        z1, b_l_1, d_l_1 = self.encoder_layer_scale1(z1)

        z1[:,:,1:,:] = z1[:,:,1:,:] + mean2

        z1= F.interpolate(z1[:, :, 1:, :], size=(self.num_patch, self.d_model), mode='bilinear',
                           align_corners=False)
        # z1 = self.scale_convtranspose1(z1[:, :, 1:, :].reshape(bs * nvars, self.d_model, -1)).reshape(bs, nvars, -1,
        #                                                                                               self.d_model)

        z = self.embedding(z).permute(0, 2, 1, 3)  # [bs x n_vars x num_patch x d_model]
        c1 = z1.shape[2]
        z = z.clone()
        z[:, :, :c1, :] = self.gatelayer2(z1, z[:,:,:c1, :])



        ##计算均值
        mean1 = torch.mean(z, dim=2).unsqueeze(2).expand(-1,-1,c1,-1)
        z = z - mean1
        z = torch.cat((cls_tokens, z), dim=2)  # [bs x n_vars x (1 + num_patch) x d_model]
        z = self.drop_out(z + self.pos[:1 + self.num_patch, :])

        z, b_l, d_l = self.encoder_layer_scale0(z)
        z[:,:,1:,:] = z[:,:,1:,:] + mean1
        z = torch.reshape(z, (-1, n_vars, 1 + self.num_patch, self.d_model))  # [bs, n_vars x num_patch x d_model]

        # decoder
        z = self.decoder_predict(bs, n_vars, z[:, :, :, :], z)


        z = self.head(z[:, :, :, :])
        z = z[:, :self.target_dim, :]

        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return z


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:, :, :, -1]  # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)  # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)  # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:, :, :, -1]  # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)  # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)  # y: bs x n_classes
        return y


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model * num_patch

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
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * num_patch]
                z = self.linears[i](z)  # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)  # x: [bs x nvars x (d_model * num_patch)]
            x = self.dropout(x)
            x = self.linear(x)  # x: [bs x nvars x forecast_len]
        return x.transpose(2, 1)  # [bs x forecast_len x nvars]


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

        x = x.transpose(2, 3)  # [bs x nvars x num_patch x d_model]
        x = self.linear(self.dropout(x))  # [bs x nvars x num_patch x patch_len]
        x = x.permute(0, 2, 1, 3)  # [bs x num_patch x nvars x patch_len]
        return x


class decoder_PredictHead(nn.Module):
    def __init__(self, d_model, patch_len, target_patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, target_patch_len)
        self.patch_len = patch_len

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2, 3)  # [bs x nvars x num_patch x d_model]
        x = self.linear(self.dropout(x))  # [bs x nvars x num_patch x patch_len]
        x = resize(x, target_patch_len=self.patch_len)
        x = x.permute(0, 2, 3, 1)  # [bs x num_patch x  x patch_len x nvars]
        return x.reshape(x.shape[0], -1, x.shape[3])


class GateLayer(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        gate_value = self.gate(x)
        #print(gate_value.shape)
        #print(gate_value[0,0,:,0])
        return gate_value.sigmoid() * x


class GateLayer_new(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(2*dim, 1)

    def forward(self, x, y):
        A = torch.cat([x,y], dim=-1)
        gate_value = self.gate(A)
        return gate_value.sigmoid() * x + y



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


def resize(x, target_patch_len):
    '''
    x: tensor [bs x num_patch x n_vars x patch_len]]
    '''
    bs, num_patch, n_vars, patch_len = x.shape
    x = x.reshape(bs * num_patch, n_vars, patch_len)
    x = F.interpolate(x, size=target_patch_len, mode='linear', align_corners=False)
    return x.reshape(bs, num_patch, n_vars, target_patch_len)


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)