python3 main.py \
--epochs 23 \
--lr 3e-4 \
--batch_size 64 \
--dropout 0.2 \
--optim adamw --timm_optim 1 \
--scheduler 1 --one_cycle 1 \
--model_name model7_synth_full \
--val_check_interval 1.0 \
--feature_extractor resnet \
--prediction attention \
--height 64 \
--width 256 \
--grayscale 0 --keep_ratio_with_pad 0 \
--stn_on 1 --count_case 1 --count_mark 1 --synth 2 --train 0