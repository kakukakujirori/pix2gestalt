#/bin/bash
# torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 \
accelerate launch \
    train.py \
    --train_data_dir /disk1/ryotaro/data/pix2gestalt_occlusions_release/ \
    --max_val_samples 8 \
    --output_dir pix2gestalt-finetuned \
    --resolution 256 \
    --random_flip \
    --train_batch_size 4 \
    --num_train_epochs 100 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --allow_tf32 \
    --use_ema \
    --offload_ema \
    --foreach_ema \
    --dataloader_num_workers 16 \
    --mixed_precision bf16 \
    --checkpoints_total_limit 4 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps 5000 \
    --resume_from_checkpoint latest
