import os
# os.environ['CUDA_VISIBLE_DEVICES']='6'

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

#from src.models.patchTST_decoder_ad import PatchTST
from src.models.TimeSeek_ad import TimeSeek
from src.learner_ad import Learner, transfer_weights
from src.callback.core import *
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *
from src.ad_data_provider.datamodule import *

import argparse

seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
# Pretraining and Finetuning
parser.add_argument('--is_finetune', type=int, default=1, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')
parser.add_argument('--is_transfer', type=int, default=0, help='do finetuning or not')
# Dataset and dataloader
parser.add_argument('--dset_finetune', type=str, default='PSM', help='dataset name')
parser.add_argument('--dset_path', type=str, default='./data/ad_datasets/', help='dataset name')
parser.add_argument('--win_size', type=int, default=96, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='sequence length')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--percentage', type=float, default=1, help='dataset path')
# Patch
parser.add_argument('--patch_len', type=int, default=48, help='patch length')
parser.add_argument('--stride', type=int, default=48, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--e_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--d_layers', type=int, default=6, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=8, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=256, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
parser.add_argument('--channel_key', type=int, default=0)
# Optimization args
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--n_epochs_freeze', type=int, default=10, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# Pretrained model name
parser.add_argument('--pretrained_model', type=str, default='/home/Decoder_version_2/checkpoints/monash_decoder_wo_img_256/patchtst_pretrained_cw528_patch48_stride48_epochs-pretrain50_mask0.4_model1/checkpoints_14.pth', help='pretrained model name')
# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int, default=3, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


args = parser.parse_args()
print('args:', args)
args.save_path = 'saved_models/' + args.dset_finetune + '/masked_patchtst/' + args.model_type + '/'        
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

# args.save_finetuned_model = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.finetuned_model_id)
suffix_name = '_cw'+str(args.win_size)+'_tw' + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_model' + str(args.finetuned_model_id)
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
    num_patch = (max(args.win_size, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    
    # get model
    model = TimeSeek(c_in=c_in,
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
                head_dropout=args.head_dropout,
                act='relu',
                head_type=head_type,
                channel_key=args.channel_key,
                res_attention=False,
                is_finetune=args.is_finetune
                )    
    # if weight_path: model = transfer_weights(weight_path, model)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model



def find_lr(head_type):
    # get dataloader
    dls = get_dls_ad(args)    
    model = get_model(1, args, head_type)
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    # model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(1, denorm=False, pretrain=False)] if args.revin else []
    # cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
        
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
    dls = get_dls_ad(args)
    # get model 
    model = get_model(1, args, head_type='anomalydetection')
    if args.is_transfer:
        model = transfer_weights(args.pretrained_model, model, exclude_head=False)
    model = nn.DataParallel(model, device_ids=[0])
    # transfer weight
    weight_path = args.pretrained_model + '.pth'
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')   
    # get callbacks
    cbs = [RevInCB(1, denorm=False, pretrain=False)] if args.revin else []
    cbs += [
         # PatchCB(patch_len=args.patch_len, stride=args.stride),
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
    #learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=args.n_epochs_freeze)
    save_recorders(learn)


def linear_probe_func(lr=args.lr):
    print('linear probing')
    # get dataloader
    dls = get_dls(args)
    # get model 
    model = get_model(dls.vars, args, head_type='anomalydetection')
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.L1Loss(reduction='mean')    
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         # PatchCB(patch_len=args.patch_len, stride=args.stride),
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
    # get dataloader

    qs_dict = {
        "CICIDS": [0.01,0.015,0.02,0.025,0.027,0.028,0.029,0.03,0.031],
        "NIPS_TS_Creditcard": [0.015,0.02,0.025,0.03], # best 0.02
        "NIPS_TS_GECCO": [0.018,0.019,0.020,0.021,0.022], # best 0.02
        "NIPS_TS_SWAN": [0.01,0.015,0.02,0.025,0.027,0.028,0.029,0.03,0.031],
        "MSL": [0.02,0.03,0.04,0.041, 0.042,0.045,0.046,0.05,0.055,0.06,0.07,0.08,0.09,0.10], # best 0.05
        "PSM": [0.019,0.02,0.021,0.022,0.023,0.0235,0.024,0.0245,0.025,0.026,0.0265,0.027,0.03,0.035,0.04,0.045,0.05], # best 0.019
        "SMAP": [0.018,0.019,0.02,0.025,0.03,0.035,0.036,0.037,0.038,0.039,0.04,0.043,0.044,0.045,0.046,0.047,0.05,0.055,0.06,0.09], # best 0.03
        "SMD": [0.018,0.0182,0.0184,0.0186,0.0188,0.019,0.0192,0.0194,0.0196,0.0198,0.02,0.0202, 0.0204, 0.0206, 0.0208, 0.021,0.022],
        "SWAT": [ 0.025,  0.03, 0.031,  0.035, 0.0375, 0.0376, 0.0378, 0.0379, 0.0380, 0.0382, 0.0384, 0.0386, 0.0388, 0.0390, 0.0392, 0.0394, 0.0396, 0.0398, 0.04,0.0405, 0.0408, 0.0410, 0.0415, 0.0418, 0.0420, 0.0422, 0.0425, 0.045, 0.0475, 0.05, 0.0525, 0.055, 0.0575, 0.060],
        "PUMP": [0.006,0.008,0.01,0.12,0.014],
        "daphnet_S01R02": [0.005,0.01,0.012,0.015,0.02,0.022,0.025],
        "daphnet_S02R01": [0.005,0.01,0.012,0.015,0.02,0.022,0.025],
        "daphnet_S03R01": [0.005,0.01,0.012,0.015,0.02,0.022,0.025],
        "daphnet_S03R02": [0.005,0.01,0.012,0.015,0.02,0.022,0.025],
        "daphnet_S07R01": [0.005,0.01,0.012,0.015,0.02,0.022,0.025],
        "daphnet_S07R02": [0.005,0.01,0.012,0.015,0.02,0.022,0.025],
        "daphnet_S08R01": [0.005,0.01,0.012,0.015,0.02,0.022,0.025],
        "Genesis": [0.01,0.015,0.02,0.025,0.03,0.035],
    }


    dls = get_dls_ad(args)
    model = get_model(1, args, head_type='anomalydetection').to('cuda')
    # get callbacks
    # cbs = [RevInCB(1, denorm=True)] if args.revin else []
    cbs = [RevInCB(1, denorm=True)]
    # cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs)


    if args.is_transfer and args.is_finetune == 0:
        folder_path='./ad_results/' + args.dset_finetune +'zero_shot'+'_model_' + str(args.finetuned_model_id)
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        model = transfer_weights(args.pretrained_model, model, exclude_head=False)
        learn = Learner(dls, model,cbs=cbs)
        learn.test_ad_spot(dls,metrics=['auc_roc' ,'affiliation'], folder_path=folder_path, qs=qs_dict[args.dset_finetune])         # out: a list of [pred, targ, score]
    else:
        folder_path='./ad_results/' + args.dset_finetune +'_model_' + str(args.finetuned_model_id)
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        model = transfer_weights(weight_path + '.pth', model, exclude_head=False)
        # learn.test_ad_spot(dls, weight_path=weight_path +'.pth' ,metrics=['auc_roc' ,'affiliation'], folder_path=folder_path, qs=qs_dict[args.dset_finetune])         # out: a list of [pred, targ, score]
        learn = Learner(dls, model, cbs=cbs)
        learn.test_ad_spot(dls, metrics=['auc_roc', 'affiliation'], folder_path=folder_path,
                           qs=qs_dict[args.dset_finetune])  # out: a list of [pred, targ, score]

    # print('score:', out[2])
    # # save results
    # pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['mse','mae']).to_csv(args.save_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)



if __name__ == '__main__':
        
    if args.is_finetune:
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = 0.0001
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
        out = test_func(weight_path)        
        print('----------- Complete! -----------')