python3 main.py \
--epochs 42 \
--lr 3e-4 \
--weight_decay 5e-4 \
--batch_size 128 \
--optim adamw --timm_optim 1 \
--scheduler 1 --one_cycle 1 \
--model_name model26_synth \
--val_check_interval 1.0 \
--prediction ctc \
--feature_extractor svtr \
--height 32 --width 64 \
--grayscale 0 --keep_ratio_with_pad 0 --stn_on 1 --focal_loss 1 --synth 1 --num_synth 100000