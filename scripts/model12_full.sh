python3 main.py \
--epochs 50 \
--lr 5e-4 \
--weight_decay 1e-5 \
--batch_size 128 \
--optim adamw --timm_optim 1 --clip_grad_val 20 \
--scheduler 1 --one_cycle 1 \
--model_name model12_full \
--val_check_interval 1.0 \
--prediction parseq --parseq_use_transformer 1 \
--height 32 \
--width 128 \
--keep_ratio_with_pad 1 --stn_on 1 --synth 2 --label_smoothing 0.1 --train 0