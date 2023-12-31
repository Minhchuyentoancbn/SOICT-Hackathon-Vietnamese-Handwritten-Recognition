python3 main.py \
--epochs 55 \
--lr 5e-4 \
--weight_decay 1e-3 \
--batch_size 128 \
--optim adamw --timm_optim 1 --clip_grad_val 20 \
--scheduler 1 --one_cycle 1 \
--model_name model14 \
--val_check_interval 1.0 \
--prediction parseq --parseq_use_transformer 0 \
--height 32 \
--width 256 \
--grayscale 1 --keep_ratio_with_pad 1 --stn_on 1 --synth 2