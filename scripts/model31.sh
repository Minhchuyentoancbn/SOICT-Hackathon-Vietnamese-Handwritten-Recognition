python3 main.py \
--epochs 65 \
--lr 3e-4 \
--weight_decay 5e-4 \
--batch_size 128 \
--optim adamw --timm_optim 1 \
--scheduler 1 --one_cycle 1 \
--model_name model31 \
--val_check_interval 1.0 \
--prediction cppd \
--feature_extractor resnet \
--height 32 --width 128 \
--grayscale 0 --keep_ratio_with_pad 0 --stn_on 1 \
--label_smoothing 0.1 --count_mark 1