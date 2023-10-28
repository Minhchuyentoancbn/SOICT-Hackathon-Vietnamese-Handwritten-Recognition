python3 main.py \
--epochs 28 \
--lr 3e-4 \
--momentum 0.9 \
--weight_decay 1e-4 \
--batch_size 128 \
--dropout 0.2 \
--optim adamw --timm_optim 1 \
--scheduler 1 --one_cycle 1 \
--model_name model6_synth_full \
--val_check_interval 1.0 \
--feature_extractor resnet \
--prediction srn \
--height 64 --width 256 \
--grayscale 0 --keep_ratio_with_pad 1 \
--stn_on 0 --synth 2 --train 0