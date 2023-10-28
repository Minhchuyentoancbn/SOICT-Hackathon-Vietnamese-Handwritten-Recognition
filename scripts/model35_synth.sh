python3 main.py \
--epochs 7 \
--lr 2e-4 \
--weight_decay 1e-5 \
--batch_size 32 \
--optim adam --timm_optim 1 \
--scheduler 1 --one_cycle 1 \
--model_name model35_synth \
--val_check_interval 1.0 \
--prediction corner \
--height 32 --width 128 \
--grayscale 0 --keep_ratio_with_pad 1 \
--label_smoothing 0.1 --synth 1 --num_synth 100000