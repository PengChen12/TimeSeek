import os
# os.environ['CUDA_VISIBLE_DEVICES']="7"

import numpy as np
import pandas as pd
import torch
from torch import nn

# from src.models.patchTST_decoder_cls import PatchTST
from src.models.GenTS_cls import GenTS
from src.learner_cls import Learner, transfer_weights
from src.callback.core import *
from src.callback.tracking_cls import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *
from src.cls_data_provider.datamodule import *

import argparse

seed=2023
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
parser.add_argument('--dset_finetune', type=str, default='EthanolConcentration', help='dataset name')
parser.add_argument('--loader', type=str, default='UEA', help='UCR or UEA')
parser.add_argument('--dset_path', type=str, default='/home/Decoder_version_2/data/cls_datasets/', help='dataset name')
parser.add_argument('--target_points', type=int, default=96, help='sequence length')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--percentage', type=float, default=1, help='dataset path')
# Patch
parser.add_argument('--patch_len', type=int, default=192, help='patch length')
parser.add_argument('--stride', type=int, default=192, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=0, help='reversible instance normalization')
# Model args
parser.add_argument('--e_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--d_layers', type=int, default=0, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=8, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=256, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
parser.add_argument('--channel_key', type=int, default=1)
# Optimization args
parser.add_argument('--n_epochs_finetune', type=int, default=100, help='number of finetuning epochs')
parser.add_argument('--n_epochs_freeze', type=int, default=0, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
# Pretrained model name
parser.add_argument('--pretrained_model', type=str, default='./checkpoints/GenTS_image_verison_6/patchtst_pretrained_cw576_patch48_stride48_epochs-pretrain25_mask0.25_model1/TimeSeek_checkpoints_10.pth', help='pretrained model name')
# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int, default=3, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


args = parser.parse_args()
print('args:', args)
args.save_path = 'saved_models/' + args.dset_finetune + '/masked_patchtst/' + args.model_type + '/'        
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

# args.save_finetuned_model = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.finetuned_model_id)
suffix_name =  '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_model' + str(args.finetuned_model_id)
if args.is_finetune: args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name
elif args.is_linear_probe: args.save_finetuned_model = args.dset_finetune+'_patchtst_linear-probe'+suffix_name
else: args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name

# get available GPU devide
# set_device()

def get_model(c_in, args, head_type, weight_path=None):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.sample_len, args.patch_len)-args.patch_len) // args.stride + 1    
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
                head_dropout=args.head_dropout,
                channel_key=args.channel_key,
                act='relu',
                head_type=head_type,
                res_attention=False
                )    
    # if weight_path: model = transfer_weights(weight_path, model,exclude_head=False)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model



def find_lr():
    # get dataloader
    dls = get_dls_cls(args)
    args.target_points =  dls.label_num  
    args.sample_len = dls.sample_len
    model = get_model(dls.n_vars, args, head_type='classification')
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    # model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = nn.CrossEntropyLoss()
    # get callbacks
    cbs = [RevInCB(1, denorm=False, pretrain=False)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
        
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
    dls = get_dls_cls(args)
    args.target_points =  dls.label_num  
    args.sample_len = dls.sample_len
    # get model 
    model = get_model(dls.n_vars, args, head_type='classification')
    if args.is_transfer:
        model = transfer_weights(args.pretrained_model, model, exclude_head=False)
    model = nn.DataParallel(model, device_ids=[0])
    # transfer weight
    weight_path = args.pretrained_model + '.pth'
    # get loss
    loss_func = nn.CrossEntropyLoss()   
    # get callbacks
    cbs = [RevInCB(dls.n_vars, denorm=False, pretrain=False)] if args.revin else []
    cbs += [
         # PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_accuracy', fname=args.save_finetuned_model, path=args.save_path)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        )                            
    # fit the data to the model
    #learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=args.n_epochs_freeze)
    # save_recorders(learn)


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
    loss_func = torch.nn.L1Loss(reduction='mean')    
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


def test_func(weight_path, result_path=None, dataset=None):
    # get dataloade
    dls = get_dls_cls(args)
    args.target_points =  dls.label_num  
    args.sample_len = dls.sample_len
    model = get_model(dls.n_vars, args, head_type='classification').to('cuda')
    # get callbacks
    cbs = [RevInCB(1, denorm=False)] if args.revin else []
    # cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    if args.is_transfer and args.is_finetune == 0:
        model = transfer_weights(args.pretrained_model, model, exclude_head=False)
        learn = Learner(dls, model,cbs=cbs)
        learn.test_cls(dls.test, result_path=result_path, dataset=dataset)         # out: a list of [pred, targ, score]
    else:
        learn = Learner(dls, model,cbs=cbs)
        learn.test_cls(dls.test, weight_path=weight_path+'.pth', result_path=result_path, dataset=dataset)         # out: a list of [pred, targ, score]

if __name__ == '__main__':
    result_path = './cls_result/w_img/' + str(args.is_finetune) 
    if not os.path.exists(result_path): os.makedirs(result_path)
    result_path = result_path + '/' + str(args.model_type)
    if not os.path.exists(result_path): os.makedirs(result_path)
    result_path = result_path + '/result.csv' 
    
    if args.is_finetune:
        args.dset = args.dset_finetune
        # Finetune
        # suggested_lr = find_lr() 
        suggested_lr = args.lr
        finetune_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func(args.save_path+args.save_finetuned_model, result_path=result_path, dataset=args.dset_finetune)         
        print('----------- Complete! -----------')

    elif args.is_linear_probe:
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = find_lr(head_type='prediction')        
        linear_probe_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func(args.save_path+args.save_finetuned_model, result_path=result_path, dataset=args.dset_finetune)        
        print('----------- Complete! -----------')

    else:
        args.dset = args.dset_finetune
        weight_path = args.save_path+args.dset_finetune+'_patchtst_finetuned'+suffix_name
        # Test
        out = test_func(weight_path, result_path=result_path, dataset=args.dset_finetune)        
        print('----------- Complete! -----------')