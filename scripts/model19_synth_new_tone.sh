python3 main.py \
--epochs 9 \
--lr 5e-4 \
--weight_decay 1e-5 \
--batch_size 64 \
--optim adamw --timm_optim 1 --clip_grad_val 20 \
--scheduler 1 --one_cycle 1 \
--model_name model19_synth_new_tone \
--val_check_interval 1.0 \
--prediction parseq --parseq_pretrained 1 --parseq_use_transformer 1 --parseq_model base \
--height 224 \
--width 224 \
--grayscale 1 --keep_ratio_with_pad 1 \
--stn_on 0 --label_smoothing 0.1 --tone 1 --synth 1 --num_synth 100000