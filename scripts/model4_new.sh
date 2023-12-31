python3 main.py \
--epochs 30 \
--lr 3e-4 \
--weight_decay 1e-5 \
--batch_size 64 \
--dropout 0.2 \
--optim adamw --timm_optim 1 --clip_grad_val 20 \
--scheduler 1 --one_cycle 1 \
--model_name model4_new \
--val_check_interval 1.0 \
--feature_extractor resnet \
--prediction attention \
--height 64 \
--width 256 \
--grayscale 0 --keep_ratio_with_pad 0 \
--stn_on 1 --count_mark 1