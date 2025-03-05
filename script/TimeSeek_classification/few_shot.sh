export CUDA_VISIBLE_DEVICES=5
is_finetune=1
is_linear_probe=0
is_transfer=1


batch_size=16

patch_len=16
stride=16

revin=1

model_type='TimeSeek_Classification'

n_epochs_finetune=200
n_epochs_freeze=0
percentage=0.1

channel_key=0





# # 

    UEAlist=(
            # 'ArticularyWordRecognition'
            # 'AtrialFibrillation'
            # 'BasicMotions'
            # 'CharacterTrajectories'
            # 'Cricket'
            # 'DuckDuckGeese'
            # 'EigenWorms'
            # 'Epilepsy'
            'EthanolConcentration'
            # 'ERing'
            # 'FaceDetection'
            # 'FingerMovements'
            # 'HandMovementDirection'
            'Handwriting'
            'Heartbeat'
            # 'InsectWingbeat'
            'JapaneseVowels'
            # 'Libras'
            # 'LSST'
            # 'MotorImagery'
            # 'NATOPS'
            # 'PenDigits'
            # 'PEMS-SF'
            # 'PhonemeSpectra'
            # 'RacketSports'
            #'SelfRegulationSCP1'
            'SelfRegulationSCP2'
            # 'SpokenArabicDigits'
            # 'StandWalkJump'
            # 'UWaveGestureLibrary'
    )


for dset_finetune in ${UEAlist[@]}
do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/Classification" ]; then
        mkdir logs/Classification
    fi
    if [ ! -d "logs/Classification/$model_type" ]; then
        mkdir logs/Classification/$model_type
    fi
    if [ ! -d "logs/Classification/$model_type/UEA" ]; then
        mkdir logs/Classification/$model_type/UEA
    fi
    if [ ! -d "logs/LongForecasting/$model_type/UEA/$dset_finetune" ]; then
        mkdir logs/Classification/$model_type/UEA/$dset_finetune
    fi

    python -u First_version_decoder_finetune_cls_learnable_token.py \
    --is_finetune $is_finetune \
    --is_linear_probe $is_linear_probe \
    --is_transfer $is_transfer \
    --dset_finetune $dset_finetune \
    --batch_size $batch_size \
    --patch_len $patch_len\
    --loader 'UEA' \
    --stride $stride\
    --revin 0 \
    --e_layers 3\
    --d_layers 2\
    --n_heads 8 \
    --d_model 256 \
    --channel_key 1 \
    --d_ff 512\
    --dropout 0.2\
    --head_drop 0.2 \
    --n_epochs_finetune $n_epochs_finetune\
    --n_epochs_freeze $n_epochs_freeze\
    --lr 2e-4 \
    --finetuned_model_id 1\
    --pretrained_model ./checkpoints/GenTS_image_verison_6/patchtst_pretrained_cw576_patch48_stride48_epochs-pretrain25_mask0.25_model1/TimeSeek_checkpoints_10.pth\
    --model_type $model_type\  >logs/Classification/$model_type/UEA/$dset_finetune/'percentage'$percentage'_finetune'$is_finetune.log 
done






for dset_finetune in 'SpokenArabicDigits' 
do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/Classification" ]; then
        mkdir logs/Classification
    fi
    if [ ! -d "logs/Classification/$model_type" ]; then
        mkdir logs/Classification/$model_type
    fi
    if [ ! -d "logs/Classification/$model_type/UEA" ]; then
        mkdir logs/Classification/$model_type/UEA
    fi
    if [ ! -d "logs/LongForecasting/$model_type/UEA/$dset_finetune" ]; then
        mkdir logs/Classification/$model_type/UEA/$dset_finetune
    fi

    python -u First_version_decoder_finetune_cls.py \
    --is_finetune $is_finetune \
    --is_linear_probe $is_linear_probe \
    --is_transfer $is_transfer \
    --dset_finetune $dset_finetune \
    --batch_size 8 \
    --patch_len $patch_len\
    --percentage $percentage \
    --channel_key 0 \
    --loader 'UEA' \
    --stride $stride\
    --revin 0 \
    --e_layers 3\
    --d_layers 2\
    --n_heads 8 \
    --d_model 256 \
    --d_ff 512\
    --dropout 0.2\
    --head_drop 0.2 \
    --n_epochs_finetune $n_epochs_finetune\
    --n_epochs_freeze $n_epochs_freeze\
    --lr 2e-3 \
    --finetuned_model_id 1\
    --pretrained_model ./checkpoints/pretrained_cw576_patch48_stride48_epochs-pretrain30_mask0.25_model/checkpoints_zeroshot.pth\
    --model_type $model_type\  >logs/Classification/$model_type/UEA/$dset_finetune/'percentage'$percentage'_finetune'$is_finetune.log 
done





for dset_finetune in 'FaceDetection' 
do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/Classification" ]; then
        mkdir logs/Classification
    fi
    if [ ! -d "logs/Classification/$model_type" ]; then
        mkdir logs/Classification/$model_type
    fi
    if [ ! -d "logs/Classification/$model_type/UEA" ]; then
        mkdir logs/Classification/$model_type/UEA
    fi
    if [ ! -d "logs/LongForecasting/$model_type/UEA/$dset_finetune" ]; then
        mkdir logs/Classification/$model_type/UEA/$dset_finetune
    fi

    python -u First_version_decoder_finetune_cls.py \
    --is_finetune $is_finetune \
    --is_linear_probe $is_linear_probe \
    --is_transfer $is_transfer \
    --dset_finetune $dset_finetune \
    --batch_size 8 \
    --patch_len $patch_len\
    --percentage $percentage \
    --channel_key 0 \
    --loader 'UEA' \
    --stride $stride\
    --revin 0 \
    --e_layers 3\
    --d_layers 2\
    --n_heads 8 \
    --d_model 256 \
    --d_ff 512\
    --dropout 0.2\
    --head_drop 0.2 \
    --n_epochs_finetune $n_epochs_finetune\
    --n_epochs_freeze $n_epochs_freeze\
    --lr 5e-4 \
    --finetuned_model_id 1\
    --pretrained_model ./checkpoints/pretrained_cw576_patch48_stride48_epochs-pretrain30_mask0.25_model/checkpoints_zeroshot.pth\
    --model_type $model_type\  >logs/Classification/$model_type/UEA/$dset_finetune/'percentage'$percentage'_finetune'$is_finetune.log 
done






for dset_finetune in 'UWaveGestureLibrary' 
do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/Classification" ]; then
        mkdir logs/Classification
    fi
    if [ ! -d "logs/Classification/$model_type" ]; then
        mkdir logs/Classification/$model_type
    fi
    if [ ! -d "logs/Classification/$model_type/UEA" ]; then
        mkdir logs/Classification/$model_type/UEA
    fi
    if [ ! -d "logs/LongForecasting/$model_type/UEA/$dset_finetune" ]; then
        mkdir logs/Classification/$model_type/UEA/$dset_finetune
    fi

    python -u First_version_decoder_finetune_cls.py \
    --is_finetune $is_finetune \
    --is_linear_probe $is_linear_probe \
    --is_transfer $is_transfer \
    --dset_finetune $dset_finetune \
    --batch_size 8 \
    --patch_len $patch_len\
    --percentage $percentage \
    --channel_key 1 \
    --loader 'UEA' \
    --stride $stride\
    --revin 0 \
    --e_layers 3\
    --d_layers 2\
    --n_heads 8 \
    --d_model 256 \
    --d_ff 512\
    --dropout 0.2\
    --head_drop 0.2 \
    --n_epochs_finetune $n_epochs_finetune\
    --n_epochs_freeze $n_epochs_freeze\
    --lr 1e-3 \
    --finetuned_model_id 1\
    --pretrained_model ./checkpoints/pretrained_cw576_patch48_stride48_epochs-pretrain30_mask0.25_model/checkpoints_zeroshot.pth\
    --model_type $model_type\  >logs/Classification/$model_type/UEA/$dset_finetune/'percentage'$percentage'_finetune'$is_finetune.log 
done





for dset_finetune in 'SelfRegulationSCP1' 
do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/Classification" ]; then
        mkdir logs/Classification
    fi
    if [ ! -d "logs/Classification/$model_type" ]; then
        mkdir logs/Classification/$model_type
    fi
    if [ ! -d "logs/Classification/$model_type/UEA" ]; then
        mkdir logs/Classification/$model_type/UEA
    fi
    if [ ! -d "logs/LongForecasting/$model_type/UEA/$dset_finetune" ]; then
        mkdir logs/Classification/$model_type/UEA/$dset_finetune
    fi

    python -u First_version_decoder_finetune_cls.py \
    --is_finetune $is_finetune \
    --is_linear_probe $is_linear_probe \
    --is_transfer $is_transfer \
    --dset_finetune $dset_finetune \
    --batch_size 4 \
    --patch_len $patch_len\
    --percentage $percentage \
    --channel_key 1 \
    --loader 'UEA' \
    --stride $stride\
    --revin 0 \
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
    --pretrained_model ./checkpoints/pretrained_cw576_patch48_stride48_epochs-pretrain30_mask0.25_model/checkpoints_zeroshot.pth\
    --model_type $model_type\  >logs/Classification/$model_type/UEA/$dset_finetune/'percentage'$percentage'_finetune'$is_finetune.log 
done
