python3 main.py \
--epochs 17 \
--lr 3e-4 --weight_decay 1e-5 \
--batch_size 64 \
--dropout 0.2 \
--optim adamw --timm_optim 1 \
--scheduler 1 --one_cycle 1 \
--model_name model10_synth_new_full \
--val_check_interval 1.0 \
--feature_extractor resnet \
--prediction attention \
--height 64 --width 256 \
--grayscale 0 --keep_ratio_with_pad 0 --stn_on 1 \
--focal_loss 1 --synth 1 --num_synth 100000 --train 0