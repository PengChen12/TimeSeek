import os
# os.environ['CUDA_VISIBLE_DEVICES']='7'
import numpy as np
import pandas as pd

import torch
from torch import nn

##这里替换成我们的模型MOE_Time
from src.models.GenTS_fc import GenTS
#from src.models.patchTST_decoder_predict import PatchTST

from src.learner import Learner, transfer_weights
from src.callback.core import *
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *
import time
import argparse


seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
# Pretraining and Finetuning
parser.add_argument('--is_finetune', type=int, default=0, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')
parser.add_argument('--is_transfer', type=int, default=1, help='do finetuning or not')
# Dataset and dataloader
parser.add_argument('--dset_finetune', type=str, default='ettm1', help='dataset name')
parser.add_argument('--context_points', type=int, default=2304, help='sequence length')
parser.add_argument('--target_points', type=int, default=720, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--img', type=int, default=0, help='if using the image loader')
# Patch
parser.add_argument('--patch_len', type=int, default=192, help= 'patch length')
parser.add_argument('--stride', type=int, default=192, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--e_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--d_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=8, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=256, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--attn_dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
parser.add_argument('--alpha', type=float, default=1, help='head dropout')
parser.add_argument('--channel_key', type=int, default=1)
parser.add_argument('--num_slots', type=int, default=10)
# Optimization args
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--n_epochs_freeze', type=int, default=4, help='number of finetuning epochs')
parser.add_argument('--patience', type=int, default=10, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
# Pretrained model name
parser.add_argument('--pretrained_model', type=str, default='./checkpoints/GenTS_image_verison_2/patchtst_pretrained_cw576_patch48_stride48_epochs-pretrain30_mask0.25_model1/checkpoints_6.pth')
# parser.add_argument('--pretrained_model', type=str, default='/home/Decoder_version_2/checkpoints/monash_decoder_wo_img_256_rope/patchtst_pretrained_cw528_patch48_stride48_epochs-pretrain50_mask0.3_model1/checkpoints_30.pth', help='pretrained model name')
# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
parser.add_argument('--is_half', type=float, default=1, help='half of the train_set')
parser.add_argument('--is_all', type=float, default=0, help='half of the train_set')
# visualize 
parser.add_argument('--visual_stride', type=int, default=5, help='the visual frequency')
parser.add_argument('--visual_save_path', type=str, default='/home/Decoder_version_2/visualization/', help='the visual frequency')


args = parser.parse_args()
print('args:', args)
args.save_path = 'saved_models/' + args.dset_finetune + '/masked_patchtst/' + args.model_type + '/'        
if not os.path.exists(args.save_path): os.makedirs(args.save_path)
if not os.path.exists(args.visual_save_path + args.dset_finetune + '_' + str(args.context_points) + '_' + str(args.patch_len)): os.makedirs(args.visual_save_path + args.dset_finetune + '_' + str(args.context_points) + '_' + str(args.patch_len))
# args.save_finetuned_model = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.finetuned_model_id)
suffix_name = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_model' + str(args.finetuned_model_id)
if args.is_finetune: args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name
elif args.is_linear_probe: args.save_finetuned_model = args.dset_finetune+'_patchtst_linear-probe'+suffix_name
else: args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name

# get available GPU devide
set_device()

def get_model(c_in, args, head_type, weight_path=None):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    
    # get model
    model = GenTS(c_in=c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                e_layers=args.e_layers,
                d_layers=args.d_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                attn_dropout=args.attn_dropout,
                head_dropout=args.head_dropout,
                act='relu',
                channel_key=args.channel_key,
                num_slots=args.num_slots,
                head_type=head_type,
                res_attention=False,
                learn_pe=False,
                is_finetune=args.is_finetune
                )    
    # if weight_path: model = transfer_weights(weight_path, model)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


class loss_freq_tmp(nn.Module):
    def __init__(self, alpha):
        super(loss_freq_tmp, self).__init__()
        self.alpha = alpha
        self.L1loss = torch.nn.L1Loss()

    def forward(self, outputs, batch_y):
        #loss_freq = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()
        loss_tmp = self.L1loss(outputs, batch_y)
        return loss_tmp
        #return loss_tmp * self.alpha + loss_freq * (1 - self.alpha)


def find_lr(head_type):
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args, head_type)
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    # model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = loss_freq_tmp(alpha=args.alpha)
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    #cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
        
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=args.lr, 
                        cbs=cbs,
                        )                        
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def save_recorders(learn):
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_finetuned_model + '_losses.csv', float_format='%.6f', index=False)


def finetune_func(lr=args.lr):
    print('end-to-end finetuning')
    
    # get dataloader
    dls = get_dls(args)
    # get model 
    model = get_model(dls.vars, args, head_type='prediction')
    # transfer weight
    weight_path = args.pretrained_model + '.pth'
    if args.is_transfer:
        model = transfer_weights(args.pretrained_model, model, exclude_head=False)
    # get loss
    loss_func = loss_freq_tmp(alpha=args.alpha) 
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
        # PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path),
         EarlyStoppingCB(monitor='valid_loss', patient=args.patience)
        ]
    # define learner

    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse]
                        )       
    # fit the data to the model
    #learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=args.n_epochs_freeze)
    save_recorders(learn)

def linear_probe_func(lr=args.lr):
    print('linear probing')
    # get dataloader
    dls = get_dls(args)
    # get model 
    model = get_model(dls.vars, args, head_type='prediction')
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')    
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse]
                        )                            
    # fit the data to the model
    learn.linear_probe(n_epochs=args.n_epochs_finetune, base_lr=lr)
    save_recorders(learn)


def test_func(weight_path):
    # args.dset_finetune = 'etth1'
    # args.dset = 'etth1'
    # args.target_points = 1080
    # args.patch_len = 192
    # args.stride = 192
    # args.context_points = 2112
    if not os.path.exists(args.visual_save_path + args.dset_finetune + '_' + str(args.context_points) + '_' + str(args.patch_len)): os.makedirs(args.visual_save_path + args.dset_finetune + '_' + str(args.context_points) + '_' + str(args.patch_len))
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type='prediction').to('cuda')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    #cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs)
    if args.is_transfer and args.is_finetune == 0:
        model = transfer_weights(args.pretrained_model, model, exclude_head=False)
        learn = Learner(dls, model,cbs=cbs)
        # learn.test_cka(dls.test) 
        out  = learn.test(dls.test, scores=[mse,mae,mape])         # out: a list of [pred, targ, score]
        learn.get_visual(dls.test, args.visual_stride, args.visual_save_path + args.dset_finetune + '_' + str(args.context_points) + '_' + str(args.patch_len))
    else:
        model = transfer_weights(weight_path +'.pth', model, exclude_head=False)
        learn = Learner(dls, model,cbs=cbs)
        out  = learn.test(dls.test ,scores=[mse,mae,mape])         # out: a list of [pred, targ, score]
        learn.get_visual(dls.test, args.visual_stride, save_path= args.visual_save_path + args.dset_finetune + '_' + str(args.context_points) + '_' + str(args.patch_len))
    print('score:', out[2])
    # save results
    pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['mse','mae', 'mape']).to_csv(args.save_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
    return out



if __name__ == '__main__':
        
    if args.is_finetune:
        args.dset = args.dset_finetune
        # Finetune
        # suggested_lr = find_lr(head_type='prediction')
        suggested_lr = args.lr
        finetune_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func(args.save_path+args.save_finetuned_model)         
        print('----------- Complete! -----------')

    elif args.is_linear_probe:
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = find_lr(head_type='prediction')        
        linear_probe_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func(args.save_path+args.save_finetuned_model)        
        print('----------- Complete! -----------')

    else:
        args.dset = args.dset_finetune
        weight_path = args.save_path+args.dset_finetune+'_patchtst_finetuned'+suffix_name
        # Test
        #####
        start_time = time.time()
        out = test_func(weight_path)
        end_time = time.time()
        print('----------- Complete! -----------')
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")