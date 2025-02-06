#/bin/bash
accelerate launch \
    train.py \
    --train_data_dir /disk1/ryotaro/data/pix2gestalt_occlusions_release/ \
    --max_val_samples 8 \
    --output_dir pix2gestalt-finetuned \
    --resolution 256 \
    --train_batch_size 16 \
    --val_batch_size 8 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 6 \
    --learning_rate 1e-4 \
    --use_8bit_adam \
    --allow_tf32 \
    --use_ema \
    --offload_ema \
    --foreach_ema \
    --dataloader_num_workers 32 \
    --mixed_precision no \
    --checkpoints_total_limit 4 \
    --enable_xformers_memory_efficient_attention \
    --validation_steps 500 \
    --checkpointing_steps 500 \
    --resume_from_checkpoint latest
