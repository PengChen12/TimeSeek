__all__ = ['TimeSeek']

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
from src.models.layers.decoder_orginal import Decoder
from src.models.layers.MOE_scale_layer import MOE_layers

import torch.fft as fft
import math
import numpy as np
from einops import reduce, rearrange, repeat
from src.callback.decompose import st_decompose

import torchvision.models as models
from torchvision.transforms import Resize
from torchvision import transforms
from PIL import Image


class ImageEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.encoder = models.resnet18(pretrained=True)
        fc_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(fc_features, out_dim)

    def forward(self, x):
        return self.encoder(x)


# Cell
class TimeSeek(nn.Module):
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
                 norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu",
                 res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'sincos', learn_pe: bool = False, head_dropout=0,
                 head_type="prediction", individual=False, channel_key=1,
                 y_range: Optional[tuple] = None, verbose: bool = False, **kwargs):

        super().__init__()
        assert head_type in ['pretrain', 'prediction', 'regression',
                             'classification'], 'head type should be either pretrain, prediction, or regression'

        # Basic
        self.num_patch = num_patch
        self.target_dim = target_dim
        self.out_patch_num = math.ceil(target_dim / patch_len)
        self.resize = Resize([224, 224], interpolation=Image.BILINEAR)
        self.toPIL = transforms.ToPILImage()

        self.scale_conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)
        self.scale_conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=4, padding=0)

        self.padding_patch_layer = nn.ReplicationPad1d((0, patch_len))

        # Embedding
        self.embedding = nn.Linear(patch_len, d_model)
        self.decoder_embedding = nn.Parameter(torch.randn(1, 1, 1, d_model), requires_grad=True)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, 1, d_model), requires_grad=True)

        # Position Embedding
        self.pos = positional_encoding(pe, learn_pe, 1 + num_patch, d_model)
        self.drop_out = nn.Dropout(dropout)
        num_slots=10

        # Encoder
        # self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
        #                           pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=e_layers,
        #                           store_attn=store_attn)
        self.encoder_layer_scale2 = MOE_layers(self.num_patch / 4, d_model, n_heads, d_ff, e_layers, num_slots, channel_key)  ##最粗尺度
        self.encoder_layer_scale1 = MOE_layers(self.num_patch / 2, d_model, n_heads, d_ff, e_layers, num_slots, channel_key)  ##中间尺度
        self.encoder_layer_scale0 = MOE_layers(self.num_patch, d_model, n_heads, d_ff, e_layers, num_slots, channel_key)  ##最细尺度
        self.image_encoder = ImageEncoder(d_model)

        self.gatelayer1 = GateLayer_new(d_model)
        self.gatelayer2 = GateLayer_new(d_model)

        ##1D
        self.scale_convtranspose1 = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=2,
                                                       stride=2)
        self.scale_convtranspose2 = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=2,
                                                       stride=2)

        # Head
        self.n_vars = c_in
        self.head_type = head_type
        self.mask_mode = mask_mode
        self.mask_nums = mask_nums
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = patch_len

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len,
                                     head_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = decoder_PredictHead(d_model, patch_len, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)

    def ts_resize(self, x):
        '''
        input: [bs, n_vars, num_patch, patch_len]
        output: [bs*n_vars, 3, 224, 224]
        '''
        bs, num_patch, n_vars, patch_len = x.shape
        seq_len = num_patch * patch_len

        # flatten
        x = x.permute(0, 1, 3, 2).reshape(bs, seq_len, n_vars)
        # find period
        periods, _ = FFT_for_Period(x, k=3)
        periods = [24]
        # padding
        x = self.padding(x, periods[0])
        # channel independent
        x = x.permute(0, 2, 1).reshape(bs * n_vars, -1)
        # resize
        x_2d = x.reshape(-1, x.shape[-1] // periods[0], periods[0]).unsqueeze(1)
        x_resize = self.resize(x_2d).expand(-1, 3, -1, -1)
        pic = self.toPIL(x_resize[0])
        # pic.save('/home/Decoder_version_2/visualization/BeetleFly_top_2.jpg')

        return x_resize

    def padding(self, x, periods):
        bs, seq_len, n_vars = x.shape
        if seq_len % periods != 0:
            padding_left = periods - seq_len % periods
            padding = torch.zeros([bs, padding_left, n_vars]).to(x.device)
            return torch.cat([padding, x], dim=1)
        else:
            return x

    def decoder_predict(self, bs, n_vars, dec_cross):
        """
        dec_cross: tensor [bs x  n_vars x num_patch x d_model]
        """
        dec_in = self.decoder_embedding.expand(bs, n_vars, self.out_patch_num, -1)
        dec_in = dec_in + self.pos[-self.out_patch_num:, :]
        decoder_output = self.decoder(dec_in, dec_cross, self.out_patch_num)
        decoder_output = decoder_output.transpose(2, 3)

        return decoder_output

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

        bs, l, nvars = z.shape
        ##降采样
        z2 = self.scale_conv2(z.reshape(bs * nvars, 1, l)).reshape(bs, -1, nvars)
        z1 = self.scale_conv1(z.reshape(bs * nvars, 1, l)).reshape(bs, -1, nvars)

        if z.shape[1] < self.patch_len:
            padding = torch.zeros([z.shape[0], self.patch_len - z.shape[1], z.shape[2]]).to(z.device)
            z = torch.cat((z, padding), dim=1)
        if z1.shape[1] < self.patch_len:
            padding = torch.zeros([z1.shape[0], self.patch_len - z1.shape[1], z1.shape[2]]).to(z1.device)
            z1 = torch.cat((z1, padding), dim=1)
        if z2.shape[1] < self.patch_len:
            padding = torch.zeros([z2.shape[0], self.patch_len - z2.shape[1], z2.shape[2]]).to(z2.device)
            z2 = torch.cat((z2, padding), dim=1)

        z, num_patch = self.create_patch(z, self.patch_len, self.stride)  # xb: [bs x seq_len x n_vars]
        z1, num_patch1 = self.create_patch(z1, self.patch_len, self.stride)
        z2, num_patch2 = self.create_patch(z2, self.patch_len, self.stride)


        bs, num_patch, n_vars, patch_len = z.shape

        # x_2d
        z_2d = self.ts_resize(z)
        img_tokens = self.image_encoder(z_2d).reshape(bs, nvars, -1).unsqueeze(2)
        cls_tokens = self.cls_embedding.expand(bs, n_vars, -1, -1)

        bs, num_patch, n_vars, patch_len = z.shape
        if z.shape[1]<24:
            z = self.embedding(z).permute(0, 2, 1, 3)  # [bs x n_vars x num_patch x d_model]
            mean0 = torch.mean(z, dim=2).unsqueeze(2).expand(-1, -1, z.shape[2], -1)
            z = z - mean0
        else:
            if z1.shape[1]<12:
                z1 = self.embedding(z1).permute(0,2,1,3)
                mean1 = torch.mean(z1, dim=2).unsqueeze(2).expand(-1, -1, z1.shape[2], -1)
                z1 = z1 - mean1
            else:
                z2 = self.embedding(z2).permute(0, 2, 1, 3)
                z2 = torch.cat((cls_tokens, z2), dim=2)
                z2 = self.drop_out(z2 + self.pos[:z2.shape[2], :])
                z2 = self.drop_out(z2)
                z2, b_l_2, d_l_2 = self.encoder_layer_scale2(z2)
                z2 = F.interpolate(z2[:, :, 1:, :], size=(self.num_patch // 2, self.d_model), mode='bilinear',
                                   align_corners=False)

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

            z1 = F.interpolate(z1[:, :, 1:, :], size=(self.num_patch, self.d_model), mode='bilinear',
                               align_corners=False)

            z = self.embedding(z).permute(0, 2, 1, 3)  # [bs x n_vars x num_patch x d_model]
            c = z.shape[2]
            c1 = z1.shape[2]
            if z1.shape[2] < z.shape[2]:
                z1 = torch.cat([z1, z1[:, :, -1, :].unsqueeze(2)], dim=2)
            z = self.gatelayer2(z1[:, :, :c, :], z)

            mean0 = torch.mean(z, dim=2).unsqueeze(2).expand(-1, -1, c, -1)
            z = z - mean0


        z = torch.cat((img_tokens, z), dim=2)  # [bs x n_vars x (1 + num_patch) x d_model]
        z = self.drop_out(z + self.pos[:1 + self.num_patch, :])
        z, b_l, d_l = self.encoder_layer_scale0(z)
        z[:, :, 1:, :] = z[:, :, 1:, :] + mean0
        z = torch.reshape(z, (-1, n_vars, 1 + self.num_patch, self.d_model))  # [bs, n_vars x num_patch x d_model]


        # predict
        z = self.head(z.transpose(2, 3))

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
        x = x[:, :, :, 0]  # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)  # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)  # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = ProjectionHead(n_vars * d_model, n_classes, 128)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        bs, n_vars, d_model, num_patch = x.shape

        x = x[:, :, :, 0]  # only consider the First item in the sequence, x: bs x nvars x d_model
        # x = torch.mean(x, dim = -1)

        # x = torch.mean(x, dim = -1)
        # x = x.reshape(bs*n_vars, d_model, -1)
        # x = torch.max_pool1d(x, num_patch).squeeze(-1)
        # x = x.reshape(bs, n_vars, d_model)

        x = x.reshape(bs, n_vars * d_model)
        x = self.flatten(x)  # x: bs x nvars * d_model
        y = self.linear(x)  # y: bs x n_classes
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
        x = self.repr_dropout(self.proj_head(x))
        return x


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



class GateLayer_new(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(2*dim, 1)

    def forward(self, x, y):
        A = torch.cat([x,y], dim=-1)
        gate_value = self.gate(A)
        return gate_value.sigmoid() * x + y