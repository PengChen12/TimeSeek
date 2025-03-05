export CUDA_VISIBLE_DEVICES=5
is_finetune=0
is_linear_probe=0
is_transfer=1


win_size=96
batch_size=64
percentage=1

patch_len=2
stride=2

revin=1

model_type='TimeSeek_anomaly_detection_zero_shot'

n_epochs_finetune=0
n_epochs_freeze=0


for dset_finetune in 'PSM' 'MSL'
do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/AnomalyDetection" ]; then
        mkdir logs/AnomalyDetection
    fi
    if [ ! -d "logs/AnomalyDetection/$model_type" ]; then
        mkdir logs/AnomalyDetection/$model_type
    fi
    if [ ! -d "logs/LongForecasting/$model_type/$dset_finetune" ]; then
        mkdir logs/AnomalyDetection/$model_type/$dset_finetune
    fi

    python -u First_version_decoder_finetune_ad.py \
    --is_finetune $is_finetune \
    --is_linear_probe $is_linear_probe \
    --is_transfer $is_transfer \
    --dset_finetune $dset_finetune \
    --percentage $percentage \
    --win_size $win_size \
    --batch_size $batch_size \
    --patch_len 8\
    --stride 8\
    --revin 1 \
    --e_layers 3\
    --d_layers 2\
    --n_heads 8 \
    --d_model 256 \
    --d_ff 512\
    --dropout 0.2\
    --head_drop 0.2 \
    --n_epochs_finetune $n_epochs_finetune\
    --n_epochs_freeze $n_epochs_freeze\
    --lr 1e-4 \
    --finetuned_model_id 1\
    --pretrained_model  ./checkpoints/pretrained_cw576_patch48_stride48_epochs-pretrain30_mask0.25_model/checkpoints.pth\
    --model_type $model_type\  >logs/AnomalyDetection/$model_type/$dset_finetune/'percentage'$percentage'_finetune'$is_finetune.log 
done


for dset_finetune in 'SWAT' 'SMD' 'SMAP'
do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/AnomalyDetection" ]; then
        mkdir logs/AnomalyDetection
    fi
    if [ ! -d "logs/AnomalyDetection/$model_type" ]; then
        mkdir logs/AnomalyDetection/$model_type
    fi
    if [ ! -d "logs/LongForecasting/$model_type/$dset_finetune" ]; then
        mkdir logs/AnomalyDetection/$model_type/$dset_finetune
    fi

    python -u First_version_decoder_finetune_ad.py \
    --is_finetune $is_finetune \
    --is_linear_probe $is_linear_probe \
    --is_transfer $is_transfer \
    --dset_finetune $dset_finetune \
    --percentage $percentage \
    --win_size $win_size \
    --batch_size $batch_size \
    --patch_len 2\
    --stride 2\
    --revin 1 \
    --e_layers 3\
    --d_layers 2\
    --n_heads 8 \
    --d_model 256 \
    --d_ff 512\
    --dropout 0.2\
    --head_drop 0.2 \
    --n_epochs_finetune $n_epochs_finetune\
    --n_epochs_freeze $n_epochs_freeze\
    --lr 1e-4 \
    --finetuned_model_id 1\
    --pretrained_model  ./checkpoints/pretrained_cw576_patch48_stride48_epochs-pretrain30_mask0.25_model/checkpoints.pth\
    --model_type $model_type\  >logs/AnomalyDetection/$model_type/$dset_finetune/'percentage'$percentage'_finetune'$is_finetune.log 
done
