
import torch
import torch.nn as nn
from .core import Callback
from src.models.layers.revin import RevIN
from sklearn.preprocessing import StandardScaler

import pandas as pd

class RevInCB(Callback):
    def __init__(self, num_features: int, eps=1e-5, 
                        affine:bool=False, denorm:bool=True, pretrain:bool=False):
        """        
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param denorm: if True, the output will be de-normalized

        This callback only works with affine=False.
        if affine=True, the learnable affine_weights and affine_bias are not learnt
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.denorm = denorm
        self.revin = RevIN(num_features, eps, affine)
        self.pretrain = pretrain
        # df_raw = pd.read_csv('/home/bigmodel/Decoder_version_1/data/predict_datasets/ETTh1.csv')
        # border1s = [0 * 30 * 24]
        # border2s = [12 * 30 * 24]
        # cols_data = df_raw.columns[1:]
        # df_data = df_raw[cols_data]
        # train_data = df_data[border1s[0]:border2s[0]]
        # self.scaler = StandardScaler()
        # self.scaler.fit(train_data.values)
        # self.mean = torch.Tensor(self.scaler.mean_)
        # self.std = torch.Tensor(self.scaler.scale_)
    

    def before_forward(self): 
        self.revin_norm()
    def after_forward(self): 
        if self.denorm: self.revin_denorm() 
        
    def revin_norm(self):
        xb_revin = self.revin(self.xb, 'norm')      # xb_revin: [bs x seq_len x nvars]
        self.learner.xb = xb_revin

    def revin_denorm(self):
        pred = self.revin(self.pred, 'denorm', self.pretrain)      # pred: [bs x target_window x nvars]

        # self.mean = self.mean.to(pred.device)
        # self.std = self.std.to(pred.device)
        # bs, T, C = pred.shape
        
        # pred = pred.reshape(bs*T, C)
        # pred = (pred-self.mean)/self.std
        # pred = pred.reshape(bs, T, C)
        
        # yb = self.learner.yb
        # yb = yb.reshape(bs*T, C)
        # yb = (yb-self.mean)/self.std

        # self.learner.yb = yb.reshape(bs, T, C)
        self.learner.pred = pred


    

