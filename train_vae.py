"""
Taken from https://github.com/huggingface/diffusers/issues/3726#issuecomment-2008567922
"""

import argparse
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

from src.arguments_vae import parse_args
from src.image_dataset import ImageDataset

logger = get_logger(__name__, log_level="INFO")


@torch.inference_mode()
def log_validation(vae, args, repo_id, accelerator, test_dataloader, weight_dtype, epoch):
    logger.info("Running validation... ")

    vae_model = accelerator.unwrap_model(vae)

    if args.enable_xformers_memory_efficient_attention:
        vae_model.enable_xformers_memory_efficient_attention()

    images = []
    for batch in test_dataloader:
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            dtype = next(vae.parameters()).dtype
            autocast_ctx = torch.autocast(accelerator.device.type, dtype=dtype)  # DTYPE AUTOCAST NEEDED

        with autocast_ctx:
            reconst = vae_model(batch["pixel_values"]).sample

        x = (batch["pixel_values"] / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().float().numpy()
        reconst = (reconst / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().float().numpy()
        ret = np.concatenate([x, reconst], axis=1)
        images.extend([x for x in ret])

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack(images, axis=0)
            tracker.writer.add_images(
                "Original (top), Reconstruction (bottom)", np_images, epoch, dataformats="NHWC"
            )
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "Original (top), Reconstruction (bottom)": [
                        wandb.Image(torchvision.utils.make_grid(image))
                        for _, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.gen_images}")

    del vae_model
    torch.cuda.empty_cache()


# def make_image_grid(imgs, rows, cols):

#     w, h = imgs[0].size
#     grid = Image.new("RGB", size=(cols * w, rows * h))

#     for i, img in enumerate(imgs):
#         grid.paste(img, box=(i % cols * w, i // cols * h))
#     return grid


# def save_model_card(
#     args,
#     repo_id: str,
#     images=None,
#     repo_folder=None,
# ):
#     # img_str = ""
#     # if len(images) > 0:
#     #     image_grid = make_image_grid(images, 1, "example")
#     #     image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
#     #     img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

#     yaml = f"""
# ---
# license: creativeml-openrail-m
# base_model: {args.pretrained_model_name_or_path}
# datasets:
# - {args.dataset_name}
# tags:
# - stable-diffusion
# - stable-diffusion-diffusers
# - text-to-image
# - diffusers
# inference: true
# ---
#     """
#     model_card = f"""
# # Text-to-image finetuning - {repo_id}

# This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: Nothing: \n

# ## Training info

# These are the key hyperparameters used during training:

# * Epochs: {args.num_train_epochs}
# * Learning rate: {args.learning_rate}
# * Batch size: {args.train_batch_size}
# * Gradient accumulation steps: {args.gradient_accumulation_steps}
# * Image resolution: {args.resolution}
# * Mixed-precision: {args.mixed_precision}

# """
#     wandb_info = ""
#     if is_wandb_available():
#         wandb_run_url = None
#         if wandb.run is not None:
#             wandb_run_url = wandb.run.url

#     if wandb_run_url is not None:
#         wandb_info = f"""
# More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
# """

#     model_card += wandb_info

#     with open(os.path.join(repo_folder, "README.md"), "w") as f:
#         f.write(yaml + model_card)


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
        else:
            repo_id = None

    # Load vae
    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, weight_dtype=torch.float32
        )
    except:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.revision, weight_dtype=torch.float32
        )

    # Create EMA for the vae.
    if args.use_ema:
        try:
            ema_vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        except:
            ema_vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, revision=args.revision, weight_dtype=torch.float32)
        ema_vae = EMAModel(
            ema_vae.parameters(),
            model_cls=AutoencoderKL,
            model_config=ema_vae.config,
            foreach=args.foreach_ema,
        )

    vae.requires_grad_(True)
    vae.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            vae.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_vae.save_pretrained(os.path.join(output_dir, "vae_ema"))

            for model in models:
                if isinstance(unwrap_model(model), AutoencoderKL):
                    model = unwrap_model(model)
                    model.save_pretrained(os.path.join(output_dir, "vae"), safe_serialization=True, max_shard_size="5GB")
                else:
                    raise NotImplementedError(f"[save_model_hook] Invalid model: {type(model)}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "vae_ema"), AutoencoderKL)
                ema_vae.load_state_dict(load_model.state_dict())
                if args.offload_ema:
                    ema_vae.pin_memory()
                else:
                    ema_vae.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(unwrap_model(model), AutoencoderKL):
                    load_model = AutoencoderKL.from_pretrained(input_dir, subfolder="vae")
                else:
                    raise NotImplementedError(f"[load_model_hook] Unknown model: {type(model)}")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes` or `pip install bitsandbytes-windows` for Windows"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        vae.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets
    with accelerator.main_process_first():
        train_dataset = ImageDataset(
            args.train_data_dir,
            resolution=args.resolution,
            max_train_samples=args.max_train_samples,
            is_train=True,
        )
        test_dataset = ImageDataset(
            args.train_data_dir,
            resolution=args.resolution,
            max_val_samples=args.max_val_samples,
            is_train=False,
        )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.val_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    vae, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    if args.use_ema:
        if args.offload_ema:
            ema_vae.pin_memory()
        else:
            ema_vae.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    ################################################################

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num test samples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device, dtype=weight_dtype)
    lpips_loss_fn.requires_grad_(False)

    for epoch in range(first_epoch, args.num_train_epochs):
        vae.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(vae):
                target = batch["pixel_values"].to(weight_dtype)

                # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py
                if accelerator.num_processes > 1:
                    posterior = vae.module.encode(target).latent_dist
                else:
                    posterior = vae.encode(target).latent_dist

                z = posterior.sample()
                if accelerator.num_processes > 1:
                    pred = vae.module.decode(z).sample
                else:
                    pred = vae.decode(z).sample

                kl_loss = posterior.kl().mean()
                mse_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                lpips_loss = lpips_loss_fn(pred.to(dtype=weight_dtype), target).mean()
                if not torch.isfinite(lpips_loss):
                    lpips_loss = torch.tensor(0)

                loss = (
                    mse_loss + args.lpips_scale * lpips_loss + args.kl_scale * kl_loss
                )

                if not torch.isfinite(loss):
                    logger.info("\nWARNING: non-finite loss, ending training ")
                    accelerator.end_training()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.offload_ema:
                    ema_vae.to(device="cuda", non_blocking=True)
                    ema_vae.step(vae.parameters())
                    if args.offload_ema:
                        ema_vae.to(device="cpu", non_blocking=True)
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "mse": mse_loss.detach().item(),
                "lpips": lpips_loss.detach().item(),
                "kl": kl_loss.detach().item(),
            }
            progress_bar.set_postfix(**logs)
            if accelerator.sync_gradients:
                accelerator.log(logs, step=global_step)

            # validation
            if accelerator.sync_gradients and accelerator.is_main_process:
                if global_step % args.validation_steps == 0:
                    if args.use_ema:
                        # Store the VAE parameters temporarily and load the EMA parameters to perform inference.
                        ema_vae.store(vae.parameters())
                        ema_vae.copy_to(vae.parameters())
                    log_validation(vae, args, repo_id, accelerator, test_dataloader, weight_dtype, global_step)
                    if args.use_ema:
                        # Switch back to the original VAE parameters.
                        ema_vae.restore(vae.parameters())

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = unwrap_model(vae)
        if args.use_ema:
            ema_vae.copy_to(vae.parameters())
        vae.save_pretrained(args.output_dir)

        # if args.push_to_hub:
        #     try:
        #         save_model_card(args, repo_id, images, repo_folder=args.output_dir)
        #         upload_folder(
        #             repo_id=repo_id,
        #             folder_path=args.output_dir,
        #             commit_message="End of training",
        #             ignore_patterns=["step_*", "epoch_*"],
        #         )
        #     except:
        #         logger.info(f"UserWarning: Your huggingface's memory is limited. The weights will be saved only local path : {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
