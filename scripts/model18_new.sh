python3 main.py \
--epochs 30 \
--lr 3e-4 \
--weight_decay 1e-4 \
--batch_size 128 \
--optim adamw --timm_optim 1 \
--scheduler 1 --one_cycle 1 \
--model_name model18_new \
--val_check_interval 1.0 \
--transformer 1 \
--transformer_model vitstr_small_patch16_224 \
--height 224 --width 224 \
--grayscale 1 --keep_ratio_with_pad 1 \
--stn_on 0 --label_smoothing 0.1