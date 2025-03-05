import random
import torch
from torch import nn
from .core import Callback
from .decompose import st_decompose

# Cell
class PatchCB_decompose(Callback):

    def __init__(self, patch_len, stride ):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): 
        self.set_patch()
       
    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        nan_mask = ~self.xb.isnan() 
        x[~nan_mask] = 0
        x = x.float()
        xb_patch, num_patch = create_patch(x, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        # learner get the transformed input
        self.learner.xb = xb_patch                              # xb_patch: [bs x num_patch x n_vars x patch_len]
    
    def decompose(self):
        bs, seq_len, n_vars = self.xb.shape
        st_decompose_inner = st_decompose()
        trend, season, res = st_decompose_inner(self.xb.shape) # bs x t x d
        mix = torch.cat([trend,season,res], dim=0) # bs*3 x t x d
        return mix

class PatchCB(Callback):

    def __init__(self, patch_len, stride ):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): 
        self.learner.loss_func = self._loss 
        self.set_patch()
       
    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        nan_mask = ~self.xb.isnan() 
        self.xb[~nan_mask] = 0
        x = self.xb.float()

        # padding
        if x.shape[1] < self.patch_len:
            padding = torch.zeros([x.shape[0], self.patch_len - x.shape[1], x.shape[2]]).to(x.device)
            x = torch.cat((x, padding), dim=1)
            
        xb_patch, num_patch = create_patch(x, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        # learner get the transformed input
        self.learner.xb = xb_patch                              # xb_patch: [bs x num_patch x n_vars x patch_len]
        self.learner.yb = xb_patch, self.learner.yb 
        self.mask = torch.ones([self.learner.xb.shape[0],self.learner.xb.shape[1],self.learner.xb.shape[2]], device=self.xb.device)
        self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
        
    def _loss(self, preds, target):        
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        """
        if preds.shape != target.shape:
            preds,_ = create_patch(preds,self.patch_len,self.stride)
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.mask).sum() / self.mask.sum()
        return loss

class PatchCB_AR(Callback):

    def __init__(self, patch_len, stride ):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): 
        self.set_patch()
       
    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        nan_mask = ~self.xb.isnan() 
        self.xb[~nan_mask] = 0
        x = self.xb.float()

        # padding
        x = self.padding(x)
        y = self.padding(self.yb, order=False)
            
        xb_patch, num_patch = create_patch(x, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        yb_patch, num_patch = create_patch(y, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        # learner get the transformed input
        self.learner.xb = xb_patch, yb_patch                              # xb_patch: [bs x num_patch x n_vars x patch_len]
    
    def padding(self, x, order=True):
        padding_len = x.shape[1] % self.patch_len
        padding = torch.zeros([x.shape[0], int(padding_len), x.shape[2]]).to(x.device)
        if order:
            x = torch.cat((padding, x), dim=1)
        else:
            x = torch.cat((x, padding), dim=1)
        return x
    
class PatchCB_ad(Callback):

    def __init__(self, patch_len, stride ):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): 
        self.set_patch()
       
    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        nan_mask = ~self.xb.isnan() 
        self.xb[~nan_mask] = 0
        x = self.xb.float()
        xb_patch, num_patch = create_patch(x, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        # learner get the transformed input
        self.learner.xb = xb_patch                              # xb_patch: [bs x num_patch x n_vars x patch_len]
        self.learner.yb = xb_patch                              # xb_patch: [bs x num_patch x n_vars x patch_len]




class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio,
                 maks_mode="patch",
                 mask_nums:int = 3,
                 threshold_ratio_list=[0.05,0.3,0.4,0.5],
                mask_when_pred:bool=False):
        """
        Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.
        Args:
            patch_len:        patch length
            stride:           stride
            mask_ratio:       mask ratio
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio        
        self.mask_mode = maks_mode
        self.mask_nums = mask_nums
        self.threshold_ratio_list=threshold_ratio_list

    def before_fit(self):
        # overwrite the predefined loss function
        self.learner.loss_func = self._loss        
        device = self.learner.device       
 
    def before_forward(self): 
        if self.mask_mode == "patch":
            self.patch_masking()
        elif self.mask_mode == "point":
            self.point_masking()
        elif self.mask_mode == 'multi':
            self.multi_patch_masking()
        elif self.mask_mode == 'freq':
            self.freq_masking()
        elif self.mask_mode == 'freq_multi':
            self.freq_multi_masking()
        else:
            self.learner.xb, num_patch = create_patch(self.xb, self.patch_len, self.stride)
            self.learner.yb, num_patch = create_patch(self.yb, self.patch_len, self.stride)
            self.mask = torch.ones([self.learner.yb.shape[0],self.learner.yb.shape[1],self.learner.yb.shape[2]], device=self.xb.device)
            self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]

        
    def patch_masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        if self.learner.channel_num!=0:
            xb=self.channel_chosing(self.learner.channel_num)
            xb_patch, num_patch = create_patch(xb, self.patch_len, self.stride)    # xb_patch: [bs x num_patch x n_vars x patch_len]
        else:
            xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb_patch: [bs x num_patch x n_vars x patch_len]
        xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)   # xb_mask: [bs x num_patch x n_vars x patch_len]
        self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
        self.learner.xb = xb_mask       # learner.xb: masked 4D tensor   
        self.learner.yb = xb_patch      # learner.yb: non-masked 4d tensor

    def channel_chosing(self, num_channel):
        B, T, C = self.xb.shape
        if C > num_channel:
            numbers = list(range(0,C))
            selected_numbers = random.sample(numbers, num_channel)
            return self.xb[:,:,selected_numbers]
        else:
            return self.xb
 
    def point_masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        bs, L, n_var = self.xb.shape
        num_patch = (max(L, self.patch_len)-self.patch_len) // self.stride + 1
        x_ = torch.empty(bs, num_patch, 1, self.patch_len).to(self.device)
        x_mask_ = torch.empty(bs, num_patch, 1, self.patch_len).to(self.device)
        mask_ = torch.empty(bs, num_patch, 1).to(self.device)
        for i in range(n_var):
             x = self.xb[:,:,i].unsqueeze(-1)
             xb_patch, num_patch = create_patch(x, self.patch_len, self.stride)
             xb_mask, _, mask, _ = random_masking(xb_patch, self.mask_ratio)
             x_ = torch.cat((x_, xb_patch), dim=2)
             x_mask_ = torch.cat((x_mask_, xb_mask), dim=2)
             mask = mask.bool()
             mask_ = torch.cat((mask_, mask), dim=2)
        
        # remove empty 
        x_ = x_[:, :, 1:, :]  # 移除维度 2 上的空维度
        x_mask_ = x_mask_[:, :, 1:, :]  # 移除维度 2 上的空维度
        mask_ = mask_[:, :, 1:]

        self.learner.xb = x_mask_       # learner.xb: masked 4D tensor    
        self.learner.yb = x_ # learner.yb: non-masked 4d tensor
        self.mask = mask_   # mask: [bs x num_patch x n_vars]09

    def multi_patch_masking(self):
        """

        """
        bs, L, n_var = self.xb.shape
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)

        x_mask_ = xb_patch.unsqueeze(0) #  1 x bs x num_patch x n_var x patch_len
        
        for _ in range(self.mask_nums):
            xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)
            xb_mask = xb_mask.unsqueeze(0)
            x_mask_ = torch.cat((x_mask_, xb_mask),dim=0)  #  (M+1) x bs x num_patch x n_var x patch_len


        x_mask_ = x_mask_[1:,:,:,:,:]
        # 对齐维度
        x_mask_ = x_mask_.view(bs*(self.mask_nums), num_patch, n_var, -1)

        self.learner.xb = x_mask_       # learner.xb: masked 4D tensor    
        self.learner.yb = xb_patch      # learner.yb: non-masked 4d tensor
        self.mask = torch.ones_like(self.mask).bool().to(self.device)   # mask: [bs x num_patch x n_vars]


    def freq_masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb_patch: [bs x num_patch x n_vars x patch_len]
        xb_mask= freq_random_masking(self.xb, self.mask_ratio,self.learner.p)                         # xb_mask:  [bs x seq_len x n_vars]
        xb_mask_patch, num_patch= create_patch(xb_mask, self.patch_len, self.stride)
        # xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)   # xb_mask_patch: [bs x num_patch x n_vars x patch_len]
        self.mask = torch.ones([xb_mask_patch.shape[0],xb_mask_patch.shape[1],xb_mask_patch.shape[2]], device=self.xb.device)
        self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
        self.learner.xb = xb_mask_patch       # learner.xb: masked 4D tensor   
        self.learner.yb = xb_patch      # learner.yb: non-masked 4d tensor
    
    def freq_multi_masking(self):
        bs, T, C = self.xb.shape
        xb_ifft = freq_multi_mask(self.xb, self.threshold_ratio_list, self.learner.p, self.mask_nums, self.patch_len, self.stride)
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)
        self.learner.xb = xb_ifft       # learner.xb: masked 4D tensor    
        self.learner.yb = xb_patch    # learner.yb: non-masked 4d tensor
        self.mask = torch.ones((bs, num_patch, C)).bool().to(self.device)
 

    def _loss(self, preds, target):        
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        """
        if preds.shape != target.shape:
            preds,_ = create_patch(preds,self.patch_len,self.stride)
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.mask).sum() / self.mask.sum()
        return loss


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch


class Patch(nn.Module):
    def __init__(self,seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        tgt_len = patch_len  + stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)                 # xb: [bs x num_patch x n_vars x patch_len]
        return x


def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, nvars,device=xb.device)  # noise in [0, 1], bs x L x nvars
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove 得到原始序列概率从小到大的索引
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # restore the index of the sample  # ids_restore: [bs x L x nvars] 对索引进行排列，从而到原始序列的

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]     # ids_keep: [bs x len_keep x nvars]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)                 # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)                                  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


def freq_random_masking(xb, mask_ratio,p):
    xb_fft = torch.fft.rfft(xb,dim=1)
    seq_rrf = xb_fft.shape[1]
    truncation = int(seq_rrf * mask_ratio)
    for i in range(xb.shape[2]):
        if random.random()>p:
            xb_fft[:,:truncation,i] = 0
        else:
            xb_fft[:,truncation:,i] = 0
    xb_ifft = torch.fft.irfft(xb_fft,dim=1)
    return xb_ifft


def freq_multi_mask(xb, threshold_ratio_list, p, mask_nums, patch_len, stride):
    bs, T, C = xb.shape
    xb_patch, num_patch = create_patch(xb, patch_len, stride) # xb: [bs x num_patch x n_vars x patch_len]

    x_fft = xb_patch.unsqueeze(-1) # 1 x bs x num_patch x n_vars x patch_len
    for i in range(mask_nums):
        x_fft_ = freq_random_mask(xb, threshold_ratio_list[i], p)
        x_fft_patch, _ = create_patch(x_fft_, patch_len, stride)
        x_fft_patch = x_fft_patch.unsqueeze(-1)
        x_fft = torch.concat([x_fft, x_fft_patch],dim=-1) # mask_nums+1 x bs x num_patch x n_vars x patch_len
    x_fft = x_fft[:,:,:,:,1:] # mask_nums x bs x num_patch x n_vars x patch_len
    # x_fft = x_fft.reshape(bs*mask_nums, num_patch, C, -1)
    return x_fft

def freq_random_mask(xb, threshold_ratio, p):
    xb_fft = torch.fft.rfft(xb, dim=1)
    seq_rrf = xb_fft.shape[1]
    
    truncation = int(seq_rrf * threshold_ratio)
    if random.random() > p: # mask low 
        xb_fft[:,:truncation,:] = 0
    else: # mask high
        xb_fft[:,truncation:,:] = 0
    xb_ifft = torch.fft.irfft(xb_fft, dim=1)
    return xb_ifft


def multi_masking(xb, mask_ratio):
    
    pass


def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]                                                 # ids_keep: [bs x len_keep]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))        # x_kept: [bs x len_keep x dim]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, D, device=xb.device)                        # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D))    # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)                                          # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore


if __name__ == "__main__":
    x = torch.randn((64, 512, 7))


