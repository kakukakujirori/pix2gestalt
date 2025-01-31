#/bin/bash
accelerate launch \
    train.py \
    --train_data_dir /disk1/ryotaro/data/pix2gestalt_occlusions_release/ \
    --max_val_samples 8 \
    --output_dir pix2gestalt-finetuned \
    --resolution 256 \
    --train_batch_size 64 \
    --num_train_epochs 100 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --use_8bit_adam \
    --allow_tf32 \
    --use_ema \
    --offload_ema \
    --foreach_ema \
    --dataloader_num_workers 32 \
    --mixed_precision bf16 \
    --checkpoints_total_limit 4 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps 5000 \
    # --resume_from_checkpoint latest
