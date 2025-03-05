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
from src.models.layers.MOE_scale_layer import MOE_layers

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

    def __init__(self, c_in: int, target_dim: int, patch_len: int, stride: int, target_patch_len: int, num_patch: int,
                 mask_mode: str = 'patch', mask_nums: int = 3,
                 e_layers: int = 3, d_layers: int = 3, d_model=128, n_heads=16, d_ff: int = 256, img_size=64,
                 norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu",
                 res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'sincos', learn_pe: bool = False, head_dropout=0,
                 head_type="prediction", y_range: Optional[tuple] = None):

        super().__init__()
        assert head_type in ['pretrain', 'prediction', 'regression',
                             'classification'], 'head type should be either pretrain, prediction, or regression'

        # Basic
        self.num_patch = num_patch
        self.target_dim = target_dim
        self.out_patch_num = math.ceil(target_dim / patch_len)
        self.target_patch_len = target_patch_len

        self.scale_conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)
        self.scale_conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=4, padding=0)

        # Embedding
        self.embedding = nn.Linear(self.target_patch_len, d_model)
        # self.decoder_embedding = nn.Parameter(torch.randn(1, 1, 1, d_model),requires_grad=True)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, 1, d_model), requires_grad=True)

        # Position Embedding
        self.pos = positional_encoding(pe, learn_pe, 1 + num_patch + self.out_patch_num, d_model)
        self.drop_out = nn.Dropout(dropout)

        # Encoder
        ##这里可能不需要输入长度，直接去除掉
        self.encoder_layer_scale2 = MOE_layers(self.num_patch / 4, d_model, n_heads, d_ff, e_layers, num_slots=10)  ##最粗尺度
        self.encoder_layer_scale1 = MOE_layers(self.num_patch / 2, d_model, n_heads, d_ff, e_layers, num_slots=10)  ##中间尺度
        self.encoder_layer_scale0 = MOE_layers(self.num_patch, d_model, n_heads, d_ff, e_layers, num_slots=10)  ##最细尺度

        self.gatelayer1 = GateLayer_new(d_model)
        self.gatelayer2 = GateLayer_new(d_model)

        ##1D反卷积
        self.scale_convtranspose1 = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=2,
                                                       stride=2)
        self.scale_convtranspose2 = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=2,
                                                       stride=2)


        # Decoder
        self.decoder_linear = nn.Linear(d_model, d_model)
        self.decoder = Decoder(d_layers, patch_len=patch_len, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                               dropout=dropout)

        # Head
        self.n_vars = c_in
        self.head_type = head_type
        self.mask_mode = mask_mode
        self.mask_nums = mask_nums
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = patch_len

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, self.target_patch_len,
                                     head_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = decoder_PredictHead(d_model, self.target_patch_len, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)

    def decoder_predict(self, bs, n_vars, dec_cross):
        """
        dec_cross: tensor [bs x  n_vars x num_patch x d_model]
        """
        # dec_in = self.decoder_embedding.expand(bs, n_vars, self.out_patch_num, -1)
        dec_in = self.decoder_linear(dec_cross[:, :, -1, :]).unsqueeze(2).expand(-1, -1, self.out_patch_num, -1)
        weights = self.get_dynamic_weights(self.out_patch_num).to(dec_in.device)
        dec_in = dec_in * weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # dec_in = dec_in + self.pos[-self.out_patch_num:,:]
        decoder_output = self.decoder(dec_in, dec_cross)
        decoder_output = decoder_output.transpose(2, 3)

        return decoder_output

    def get_dynamic_weights(self, n_preds):
        """
        Generate dynamic weights for the replicated tokens. This example uses a linearly decreasing weight.
        You can modify this to use other schemes like exponential decay, sine/cosine, etc.
        """
        # Linearly decreasing weights from 1.0 to 0.5 (as an example)
        weights = torch.linspace(1.0, 0.5, n_preds)
        return weights
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
        z_masked : tensor [bs x num_patch x n_vars x patch_len x mask_nums]
        z_orginal : tensor [bs x num_patch x n_vars x patch_len]

        x: tensor [bs x   T x nvars]
        z_masked: tensor [ bs x num_patch x n_vars x patch_len]
        z_patch: tensor [bs x num_patch x n_vars x patch_len]
        """

        x, z_masked, z_patch = z
        bs, l, nvars = x.shape
        z_new = torch.cat([z_masked, z_patch],dim=0)
        bs_new, num_patch, nvars, patch_len = z_new.shape
        z2 = self.scale_conv2(x.reshape(bs * nvars, 1, l)).reshape(bs, -1, nvars)
        z1 = self.scale_conv1(x.reshape(bs * nvars, 1, l)).reshape(bs, -1, nvars)
        if z1.shape[1] < self.patch_len:
            padding = torch.zeros([z1.shape[0], self.patch_len - z1.shape[1], z1.shape[2]]).to(z1.device)
            z1 = torch.cat((z1, padding), dim=1)
        if z2.shape[1] < self.patch_len:
            padding = torch.zeros([z2.shape[0], self.patch_len - z2.shape[1], z2.shape[2]]).to(z2.device)
            z2 = torch.cat((z2, padding), dim=1)

        z1, num_patch1 = self.create_patch(z1, self.patch_len, self.stride)
        z2, num_patch2 = self.create_patch(z2, self.patch_len, self.stride)

        cls_tokens = self.cls_embedding.expand(bs, nvars, -1, -1)
        cls_tokens_all = self.cls_embedding.expand(bs_new, nvars, -1, -1)
        if z_new.shape[1] < 4:
            #z_masked = self.embedding(z_masked).permute(0, 2, 1, 3)  # [bs x n_vars x num_patch x d_model]
            z_new = self.embedding(z_new).permute(0,2,1,3)
            mean0 = torch.mean(z_new[bs:,:,:,:], dim=2).unsqueeze(2).expand(-1, -1, z_new.shape[2], -1)
            mean0 = torch.cat([mean0, mean0], dim=0)
            z_new = z_new - mean0
        else:
            if z1.shape[1] < 2:
                z1 = self.embedding(z1).permute(0, 2, 1, 3)
                mean1 = torch.mean(z1, dim=2).unsqueeze(2).expand(-1, -1, z1.shape[2], -1)
                z1 = z1 - mean1
            else:
                z2 = self.embedding(z2).permute(0, 2, 1, 3)
                z2 = torch.cat((cls_tokens, z2), dim=2)
                z2 = self.drop_out(z2 + self.pos[:z2.shape[2], :])
                z2 = self.drop_out(z2)
                z2, b_l_2, d_l_2 = self.encoder_layer_scale2(z2)
                # z2 = self.scale_convtranspose2(z2[:, :, 1:, :].reshape(bs * nvars, self.d_model, -1)).reshape(bs, nvars,
                #                                                                                               -1,
                #                                                                                               self.d_model)
                z2 = F.interpolate(z2[:, :, 1:, :], size=(self.num_patch // 2, self.d_model), mode='bilinear',
                                   align_corners=False)

                # z2 = self.gatelayer(z2)
                # z1 = self.embedding(z1).permute(0, 2, 1, 3)
                # c2 = z2.shape[2]
                # c1 = z1.shape[2]
                # if c2 < c1:
                #     z2 = torch.cat([z2, z2[:, :, -1, :].unsqueeze(2)], dim=2)
                # z1 = z1 + 0.25 * z2[:, :, :c1, :]

                z1 = self.embedding(z1).permute(0, 2, 1, 3)
                c2 = z2.shape[2]
                c1 = z1.shape[2]
                if c2 < c1:
                    z2 = torch.cat([z2, z2[:, :, -1, :].unsqueeze(2)], dim=2)
                z1 = self.gatelayer1(z2[:,:,:c1,:], z1)


                mean1 = torch.mean(z1, dim=2).unsqueeze(2).expand(-1, -1, c2, -1)
                z1 = z1 - mean1

            z1 = torch.cat((cls_tokens, z1), dim=2)

            z1 = self.drop_out(z1 + self.pos[:z1.shape[2], :])
            z1, b_l_1, d_l_1 = self.encoder_layer_scale1(z1)
            z1[:, :, 1:, :] = z1[:, :, 1:, :] + mean1

            # z1 = self.scale_convtranspose1(z1[:, :, 1:, :].reshape(bs * nvars, self.d_model, -1)).reshape(bs, nvars, -1,
            #                                                                                               self.d_model)
            z1 = F.interpolate(z1[:, :, 1:, :], size=(self.num_patch, self.d_model), mode='bilinear',
                               align_corners=False)

            # z1 = self.gatelayer(z1)
            # z_new = self.embedding(z_new).permute(0, 2, 1, 3)  # [bs x n_vars x num_patch x d_model]
            # c = z_new.shape[2]
            # if z1.shape[2] < c:
            #     z1 = torch.cat([z1, z1[:, :, -1, :].unsqueeze(2)], dim=2)
            # z_new = z_new + 0.25 * torch.cat([z1[:, :, :c, :], z1[:, :, :c, :]], dim=0)

            z_new = self.embedding(z_new).permute(0, 2, 1, 3)  # [bs x n_vars x num_patch x d_model]
            c = z_new.shape[2]
            if z1.shape[2] < c:
                z1 = torch.cat([z1, z1[:, :, -1, :].unsqueeze(2)], dim=2)
            z_new = self.gatelayer2(torch.cat([z1[:, :, :c, :], z1[:, :, :c, :]], dim=0), z_new)



            #print(torch.mean(z_new[bs:, :, :, :], dim=2).unsqueeze(2).expand(-1,-1, c, -1).shape)
            mean0 = torch.mean(z_new[bs:, :, :, :], dim=2).unsqueeze(2).expand(-1, -1, c, -1)
            mean0 = torch.cat([mean0, mean0], dim=0)
            z_new = z_new - mean0


        z_new = torch.cat((cls_tokens_all, z_new), dim=2)  # [bs x n_vars x (1 + num_patch) x d_model]
        z_new = self.drop_out(z_new + self.pos[:1 + self.num_patch, :])
        z_new, b_l, d_l = self.encoder_layer_scale0(z_new)
        z_new[:, :, 1:, :] = z_new[:, :, 1:, :] + mean0

        ##这里进行拆分
        z_masked = z_new[:bs, :, :, :]
        z_patch = z_new[bs:, :, :, :]
        z_masked = torch.reshape(z_masked, (-1, nvars, 1 + self.num_patch, self.d_model))  # [bs, n_vars x num_patch x d_model]
        z_patch = torch.reshape(z_patch, (-1, nvars, 1 + self.num_patch, self.d_model))

        # decoder_prediction
        z_predict = self.decoder_predict(bs, nvars, z_patch)
        z_predict = self.head(z_predict, patch_len)
        z_predict = z_predict.permute(0, 1, 3, 2)  # [bs x num_patch x patch_len x n_vars]
        z_predict = z_predict.reshape(z_predict.shape[0], -1, z_predict.shape[3])
        z_predict = z_predict[:, :self.target_dim, :]


        # recontruction
        z_reconstruct = self.head(z_masked[:, :, 1:, :].permute(0,1,3,2), patch_len)


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

    def forward(self, x, patch_len):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2, 3)  # [bs x nvars x num_patch x d_model]
        # print("---------")
        # print(x.shape)
        x = self.linear(self.dropout(x))  # [bs x nvars x num_patch x patch_len]
        # x = resize(x, patch_len)
        x = x.permute(0, 2, 1, 3)  # [bs x num_patch x nvars x patch_len]
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

        x = x.transpose(2, 3)  # [bs x nvars x num_patch x d_model]
        x = self.linear(self.dropout(x))  # [bs x nvars x num_patch x patch_len]
        x = x.permute(0, 2, 3, 1)  # [bs x num_patch x patch_len x nvars]
        return x.reshape(x.shape[0], -1, x.shape[3])




def resize(x, target_patch_len):
    '''
    x: tensor [bs x num_patch x n_vars x patch_len]]
    '''
    bs, num_patch, n_vars, patch_len = x.shape
    x = x.reshape(bs * num_patch, n_vars, patch_len)
    x = F.interpolate(x, size=target_patch_len, mode='linear')
    return x.reshape(bs, num_patch, n_vars, target_patch_len)


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
