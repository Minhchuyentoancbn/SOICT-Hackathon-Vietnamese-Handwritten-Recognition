python3 main.py \
--epochs 60 \
--lr 5e-4 \
--weight_decay 1e-3 \
--batch_size 128 \
--optim adamw --timm_optim 1 --clip_grad_val 20 \
--scheduler 1 --one_cycle 1 \
--model_name model29_full \
--val_check_interval 1.0 \
--prediction abinet \
--height 32 --width 128 --label_smoothing 0.1 --train 0