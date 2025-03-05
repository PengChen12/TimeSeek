import torch 
from torch import nn
from .attention import *
from .pos_encoding import *
from .basics import *
import random

class DecoderLayer(nn.Module):
    def __init__(self, patch_len, d_model, n_heads, d_ff=None, attn_dropout = 0.2, dropout=0.5, norm="BatchNorm"):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model, n_heads, res_attention=False, attn_dropout=attn_dropout)
        self.cross_attention = MultiheadAttention(d_model, n_heads, attn_dropout=attn_dropout)
        
        if 'batch' in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm3 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)


        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.ReLU(),
                                nn.Linear(d_ff, d_model))
        
        self.p = 0.1

    def forward(self, x, cross, output_num_patch):
        batch, n_vars, num_patch, d_model = x.shape
        x = x.reshape(batch*n_vars, num_patch, d_model)
        cross = cross.reshape(batch*n_vars, -1, d_model)

        x_cross , _ = self.cross_attention(x, cross, cross)
        x_cross = self.norm1(x_cross) + x

        # Randomly replace elements with a probability p
        # replace_mask = torch.rand(x.shape).to(x.device) < self.p
        # x_cross = torch.where(replace_mask, x, x_cross)
        
        attention_mask = causal_attention_mask(output_num_patch).to(x.device)
        x_attn , _= self.self_attention(x_cross, attn_mask=attention_mask) 
        x_attn = self.dropout(self.norm2(x_attn)) + x_cross

        x_ff = self.MLP1(x_cross)
        x_ff = self.norm3(x_ff) + x_cross

        x_ff = x_ff.reshape(batch, n_vars, num_patch, d_model)

        return x_ff


class Decoder(nn.Module):
    def __init__(self, d_layers, patch_len, d_model, n_heads, d_ff=None, attn_dropout=0.2, dropout=0.1):
        super(Decoder, self).__init__()

        self.decoder_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decoder_layers.append(DecoderLayer(patch_len, d_model, n_heads, d_ff, attn_dropout, dropout))

    def forward(self, x, cross, output_num_patch):
        output = x
        for layer in self.decoder_layers:
            output = layer(x, cross, output_num_patch)
        return output


        
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


