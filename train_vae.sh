#/bin/bash
accelerate launch \
    train_vae.py \
    --train_data_dir /disk1/ryotaro/data/pix2gestalt_occlusions_release/occlusion \
    --max_val_samples 8 \
    --output_dir vae-finetuned \
    --resolution 256 \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 6 \
    --learning_rate 1e-7 \
    --mixed_precision no \
    --use_8bit_adam \
    --allow_tf32 \
    --use_ema \
    --offload_ema \
    --foreach_ema \
    --dataloader_num_workers 32 \
    --checkpoints_total_limit 4 \
    --enable_xformers_memory_efficient_attention \
    --validation_steps 500 \
    --checkpointing_steps 500 \
    --freeze_encoder \
    --resume_from_checkpoint latest
