r"""Launch training of forecasting diffusion prior."""

import argparse
import dawgz
import torch
import torch.distributed as dist
import wandb

from azula.denoise import KarrasDenoiser
from azula.noise import RectifiedSchedule
from omegaconf import OmegaConf
from shaggy.optimizer import SOAP, safe_gradient_step
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from neptune.config import PATH_MODELS
from neptune.data.dataloader import get_dataloaders
from neptune.data.dataset import get_forecast_latent_datasets
from neptune.diffusion import LatentViT, load, save
from neptune.distributed import reduce_mean, setup_distributed
from neptune.schedulers import warmup_cosine_decay
from neptune.tools import (
    generate_run_name_diff_forecast,
    get_wandb_hyperparameters,
    load_configuration,
)


# fmt: off
#
def training(
    config_state: dict,
    config_training: dict,
    config_arch: dict,
    config_schedule: dict,
    config_wandb: dict,
    config_cluster: dict,
) -> None:
    r"""Launch the training of a forecasting diffusion prior."""

    # Initialize distributed setup
    rank, local_rank, world_size, device, is_distributed = setup_distributed()

    (
        saving,
        ae_checkpoint_name,
        diff_checkpoint_name,
        alpha_min,
        sigma_min,
        steps_update,
        steps_logging,
        steps_saving,
        batch_size_per_step,
        batch_size_per_gpu,
        num_workers,
        prefetch_factor,
        lr_start,
        lr_peak,
        lr_end,
        warmup_steps,
    ) = (
        config_state["saving"],
        config_state["checkpoint_name_ae"],
        config_state["checkpoint_name_diff_fc"],
        config_schedule["alpha_min"],
        config_schedule["sigma_min"],
        config_training["steps_update"],
        config_training["steps_logging"],
        config_training["steps_saving"],
        config_training["batch_size_per_step"],
        config_training["batch_size_per_gpu"],
        config_training["num_workers"],
        config_training["prefetch_factor"],
        config_training["learning_rate_start"],
        config_training["learning_rate_peak"],
        config_training["learning_rate_end"],
        config_training["warmup_steps"],
    )

    input_states  = config_arch["input_states"]
    output_states = config_arch["output_states"]

    # Weights & Biases
    run_name = generate_run_name_diff_forecast(
        ae_checkpoint_name = ae_checkpoint_name,
        hid_channels       = config_arch["hid_channels"],
        hid_blocks         = config_arch["hid_blocks"],
        patch_size         = config_arch["patch_size"],
        input_states       = input_states,
        output_states      = output_states,
        previous_run_name  = diff_checkpoint_name,
    )

    if rank == 0:
        wandb.init(
            **config_wandb,
            name=run_name,
            config={
                "State"           : config_state,
                "Training"        : config_training,
                "Architecture"    : config_arch,
                "Schedule"        : config_schedule,
                "Cluster"         : config_cluster,
                "Hyperparameters" : get_wandb_hyperparameters([config_training, config_arch, config_schedule]),
            },
        )
    else:
        wandb.init(mode="disabled")

    # Gradient accumulation
    batch_size_per_process      = batch_size_per_gpu * world_size
    steps_gradient_accumulation = max(1, (batch_size_per_step + batch_size_per_process - 1) // batch_size_per_process)
    batches                     = [steps_update * steps_gradient_accumulation, None, None]

    dataloader_training, _, _ = get_dataloaders(
        batch_size      = batch_size_per_gpu,
        num_workers     = num_workers,
        prefetch_factor = prefetch_factor,
        get_datasets_fn = get_forecast_latent_datasets,
        batches         = batches,
        shuffle         = [True, False, False],
        infinite        = [True, False, False],
        rank            = rank,
        world_size      = world_size,
        is_distributed  = is_distributed,
        checkpoint_name = ae_checkpoint_name,
        input_size      = input_states,
        output_size     = output_states,
    )

    # Infering latent shape from a sample
    _ds_tmp                     = get_forecast_latent_datasets(ae_checkpoint_name, input_size=input_states, output_size=output_states)[0]
    _, C_LAT, H_LAT, W_LAT     = _ds_tmp[0][0].shape
    del _ds_tmp

    # Model | Loading checkpoint or new
    if diff_checkpoint_name is not None:
        ckpt_path = PATH_MODELS / diff_checkpoint_name
        backbone  = load(ckpt_path, device=str(device)).train()
    else:
        backbone = LatentViT(
            lat_channels  = output_states * C_LAT,
            cond_channels = 1 + input_states * C_LAT,
            **{k: v for k, v in config_arch.items() if k not in ("input_states", "output_states")},
        )

    # Preparing diffusion setup
    backbone = backbone.to(device)
    schedule = RectifiedSchedule(alpha_min=alpha_min, sigma_min=sigma_min)
    denoiser = KarrasDenoiser(backbone, schedule).to(device)

    # Model | Defining if DDP or DataParallel
    if is_distributed:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        backbone   = DDP(backbone, **ddp_kwargs)
    elif torch.cuda.device_count() > 1:
        backbone = torch.nn.DataParallel(backbone, device_ids=list(range(torch.cuda.device_count()))).to(device)

    # Log trainable parameters
    if rank == 0:
        wandb.log({"Informations/Trainable Parameters [M]": sum(p.numel() for p in backbone.parameters() if p.requires_grad) / 1e6})

    # SOAP optimizer and warmup-cosine scheduler
    optimizer = SOAP(backbone.parameters(), lr=lr_peak, max_precond_size=128)
    scheduler = warmup_cosine_decay(
        optimizer    = optimizer,
        lr_start     = lr_start,
        lr_peak      = lr_peak,
        lr_end       = lr_end,
        warmup_steps = warmup_steps,
        total_steps  = steps_update,
    )

    scaler                   = GradScaler(enabled=False)
    loss_accumulator         = 0.0
    loss_logging_accumulator = 0.0
    loss_mean                = float("inf")
    loss_best                = float("inf")
    gradient_norm            = float("inf")
    optimizer_step           = 0

    # Wait for all processes to be ready
    if is_distributed:
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)

    for step, (z_in, z_out, dates_in, _) in enumerate(dataloader_training):

        # Pushing to device
        z_in  = z_in.to(device)   # (B, input_states,  C_LAT, H_LAT, W_LAT)
        z_out = z_out.to(device)  # (B, output_states, C_LAT, H_LAT, W_LAT)

        # Computing conditioning: year_progress of current day + past states
        dates_current = list(dates_in[-1])
        year_cond     = LatentViT.day_of_year_to_conditioning(dates_current, H_LAT, W_LAT, device)
        cond          = torch.cat([year_cond, z_in.flatten(1, 2)], dim=1)

        # Flattening future states as denoising target: (B, output_states * C_LAT, H_LAT, W_LAT)
        z_out = z_out.flatten(1, 2)
        t     = torch.rand(z_out.shape[0], device=device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):

            # Forward pass and loss computation
            loss = denoiser.loss(z_out, t, cond=cond).mean()

        # Gradient accumulation
        loss              = loss / steps_gradient_accumulation
        loss_accumulator += loss.item()

        # Console logging when WandB is disabled
        if config_wandb["mode"] == "disabled":
            print(f"Step {optimizer_step:6d} | Loss: {loss_accumulator:.4f} | γ: {scheduler.get_last_lr()[0]:.6f} | ∇: {gradient_norm:.4f}")

        # Only sync gradients on last accumulation step
        is_last_accumulation_step = ((step + 1) % steps_gradient_accumulation == 0)

        if is_distributed and not is_last_accumulation_step:
            with backbone.no_sync():
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()

        # Cleaning up memory
        del z_in, z_out

        # Optimization step
        if is_last_accumulation_step:
            gradient_norm             = safe_gradient_step(optimizer=optimizer, scaler=scaler, grad_clip=1.0)
            loss_logging_accumulator += loss_accumulator
            loss_accumulator          = 0.0
            optimizer_step           += 1
            scheduler.step()
            del loss

        # Logging results
        if optimizer_step % steps_logging == 0 and is_last_accumulation_step:

            loss_mean                = loss_logging_accumulator / steps_logging
            loss_logging_accumulator = 0.0

            if is_distributed:
                loss_mean = reduce_mean(loss_mean, device)

            if rank == 0:
                wandb.log({
                    "Training/Loss"              : loss_mean,
                    "Informations/Steps Update"  : optimizer_step,
                    "Informations/Samples Seen"  : (step + 1) * batch_size_per_gpu * world_size,
                    "Informations/Gradient Norm" : gradient_norm,
                    "Informations/Learning Rate" : scheduler.get_last_lr()[0],
                })

        # Saving checkpoint
        if saving and optimizer_step % steps_saving == 0 and is_last_accumulation_step and optimizer_step > 0 and rank == 0:
            if loss_mean < loss_best:

                # Creating checkpoint
                save_dir     = PATH_MODELS / wandb.run.name
                raw_backbone = backbone.module if hasattr(backbone, "module") else backbone
                config_save  = OmegaConf.create({
                    "lat_channels"  : output_states * C_LAT,
                    "cond_channels" : 1 + input_states * C_LAT,
                    "input_states"  : input_states,
                    "output_states" : output_states,
                    "h_lat"         : H_LAT,
                    "w_lat"         : W_LAT,
                    **config_arch,
                    **config_schedule,
                })

                # Saving checkpoint
                save(raw_backbone, config_save, save_dir)

                # Updating best loss
                loss_best = loss_mean

    # Closing run
    wandb.finish()
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch a forecasting diffusion prior training.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the training .yml configuration file.",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend, 'slurm' for cluster-based scheduling and 'async' for local execution.",
    )

    args           = parser.parse_args()
    configs        = load_configuration(args.config)
    config_wandb   = configs[0]["WandB"]
    config_cluster = configs[0]["Cluster"]

    nodes         = config_cluster["nodes"]
    gpus_per_node = config_cluster["gpus-per-node"]
    cpus_per_node = config_cluster["cpus-per-node"]
    ram_per_node  = config_cluster["ram-per-node"]

    # Local
    if args.backend == "async":
        diff = configs[0]["Diffusion"]
        training(
            config_state    = diff["state"],
            config_training = diff["training"],
            config_arch     = diff["architecture"],
            config_schedule = diff["schedule"],
            config_wandb    = config_wandb,
            config_cluster  = config_cluster,
        )

    # Cluster
    else:
        if nodes > 1:
            interpreter = (
                f"torchrun --nnodes {nodes} --nproc-per-node {gpus_per_node} "
                f"--rdzv_backend=c10d --rdzv_endpoint=$SLURMD_NODENAME:$((20000 + SLURM_JOB_ID % 10000)) "
                f"--rdzv_id=$SLURM_JOB_ID"
            )
        else:
            interpreter = f"torchrun --nnodes 1 --nproc-per-node {gpus_per_node} --standalone"

        @dawgz.job(
            array=len(configs),
            nodes=nodes,
            gpus=gpus_per_node,
            cpus=cpus_per_node,
            ram=ram_per_node,
            time=config_cluster["time"],
            account=config_cluster["account"],
            partition=config_cluster["partition"],
        )
        def train(i: int) -> None:
            diff = configs[i]["Diffusion"]
            training(
                config_state    = diff["state"],
                config_training = diff["training"],
                config_arch     = diff["architecture"],
                config_schedule = diff["schedule"],
                config_wandb    = config_wandb,
                config_cluster  = config_cluster,
            )

        dawgz.schedule(
            train,
            name="DIFF-TRAIN-FORECASTING",
            backend="slurm",
            interpreter=interpreter,
            export="ALL",
        )
