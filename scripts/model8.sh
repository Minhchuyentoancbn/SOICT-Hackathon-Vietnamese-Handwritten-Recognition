python3 main.py \
--epochs 31 \
--lr 1e-4 \
--momentum 0.9 \
--weight_decay 0 \
--batch_size 64 \
--dropout 0.2 \
--optim adamw \
--scheduler 1 \
--decay_epochs 18 30 \
--model_name model8 \
--val_check_interval 1.0 \
--feature_extractor resnet \
--prediction attention \
--height 64 \
--width 256 \
--grayscale 0 --keep_ratio_with_pad 0 \
--stn_on 1 --count_case 1