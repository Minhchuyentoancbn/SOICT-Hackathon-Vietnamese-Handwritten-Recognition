python3 main.py \
--epochs 35 \
--lr 1e-4 \
--weight_decay 1e-4 \
--batch_size 128 \
--dropout 0.2 \
--optim adamw --timm_optim 1 \
--scheduler 1 --one_cycle 1 \
--model_name model6_new_full \
--val_check_interval 1.0 \
--feature_extractor resnet \
--prediction srn \
--height 64 --width 256 \
--grayscale 0 --keep_ratio_with_pad 1 --stn_on 0 --train 0