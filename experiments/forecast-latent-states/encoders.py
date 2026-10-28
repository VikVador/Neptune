r"""Launch the training of the deterministic encoder of a group of variables, preserving the geometry of its states."""

import argparse
import dask
import math
import torch
import torch.distributed as dist
import wandb

from omegaconf import OmegaConf
from shaggy.loss import loss_geometry_embedding
from shaggy.optimizers.gradients import safe_gradient_step
from shaggy.optimizers.soap import SOAP
from shaggy.tools import load, load_config, save
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

import neptune.data
import neptune.data.dataloader
import neptune.data.dataset
import neptune.data.weights
import neptune.model.encoders

from neptune.config import PATH_MODELS
from neptune.data.dataloader import get_dataloaders
from neptune.data.weights import get_weights_conditioning
from neptune.distributed import reduce_mean, setup_distributed
from neptune.model.encoders import LatentEncoder, OceanEncoder, SurfaceEncoder
from neptune.schedulers import warmup_cosine_decay
from neptune.tools import (
    generate_run_name,
    get_wandb_hyperparameters,
    load_configuration,
    schedule_training,
)

# fmt: off
#
ENCODERS = {"surface": SurfaceEncoder, "ocean": OceanEncoder}


def _build_model(
    group           : str,
    config_model    : dict,
    checkpoint_name : str | None,
    device          : torch.device,
) -> tuple[LatentEncoder, dict]:
    r"""Build an encoder from scratch, or resume it from a checkpoint.

    Arguments:
        group           : Group of variables, 'surface' or 'ocean'.
        config_model    : Constructor kwargs of the encoder, used when training from scratch.
        checkpoint_name : Name of the checkpoint to resume from (if applicable).
        device          : Target device.

    Returns:
        model  : Encoder in training mode, on the target device.
        config : Constructor kwargs.
    """

    if checkpoint_name is not None:
        ckpt_path = PATH_MODELS / checkpoint_name
        config    = OmegaConf.to_container(load_config(ckpt_path))
        model     = load(ckpt_path, ENCODERS[group], device=str(device)).train()
    else:
        config = config_model
        model  = ENCODERS[group](**config).to(device)

    return model, config


def forward_loss(
    model  : torch.nn.Module,
    batch  : tuple,
    group  : str,
    device : torch.device,
) -> Tensor:
    r"""Encode the states of a batch, with their conditioning, and compute the GME loss between the states and their latents.

    Arguments:
        model  : Encoder, possibly wrapped by DDP.
        batch  : Single state with its conditioning and date, from NeptuneDataset.
        group  : Group of variables, 'surface' or 'ocean'.
        device : Target device.

    Returns:
        loss : GME loss of the encoder.
    """

    x_inp_s, x_inp_o, _, _, c_inp, dates = batch

    # The states are encoded with their conditioning: the date and ERA5 at the surface, the date in the ocean (the collate transposes the dates)
    cond_s, cond_o = get_weights_conditioning(dates[0], c_inp[:, 0], device=device)
    states, cond   = (x_inp_s[:, 0], cond_s) if group == "surface" else (x_inp_o[:, 0], cond_o)
    states         = states.to(device)

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        z = model(torch.cat([states, cond], dim=1))

    # The loss only compares the geometry of the states, the conditioning being an input of the encoder
    return loss_geometry_embedding(states, z.float())


def training(
    config_state    : dict,
    config_training : dict,
    config_model    : dict,
    config_wandb    : dict,
    config_cluster  : dict,
) -> None:
    r"""Launch the training of the encoder of a group of variables.

    Arguments:
        config_state    : Saving option and checkpoint to resume from.
        config_training : Batch sizes, optimizer steps and learning rate schedule.
        config_model    : Group of variables, latent channels and architecture of the encoder.
        config_wandb    : Weights & Biases entity, project and mode.
        config_cluster  : Slurm resources, logged alongside the run for traceability.
    """

    # Initialize distributed setup
    rank, local_rank, world_size, device, is_distributed = setup_distributed()

    # Prevent xarray/dask deadlocks inside DataLoader workers
    dask.config.set(scheduler="synchronous")

    group = config_model["group"]

    (
        saving,
        checkpoint_name,
        steps_update,
        steps_logging,
        steps_validation,
        batches_validation,
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
        config_state["checkpoint_name"],
        config_training["steps_update"],
        config_training["steps_logging"],
        config_training["steps_validation"],
        config_training["batches_validation"],
        config_training["batch_size_per_step"],
        config_training["batch_size_per_gpu"],
        config_training["num_workers"],
        config_training["prefetch_factor"],
        config_training["learning_rate_start"],
        config_training["learning_rate_peak"],
        config_training["learning_rate_end"],
        config_training["warmup_steps"],
    )

    # Security
    assert batch_size_per_gpu > 1, f"ERROR - The GME loss pairs the samples of a GPU, batch_size_per_gpu must be > 1, got {batch_size_per_gpu}."

    # The model is loaded from a checkpoint or built from scratch, the group selecting the class of the encoder
    config        = {key: value for key, value in config_model.items() if key != "group"}
    model, config = _build_model(group, config, checkpoint_name, device)

    # The Weights & Biases run is named after the architecture
    run_name = generate_run_name(f"ENC_{group}_lc{model.lat_channels}_tk{model.tokens}", checkpoint_name)

    if rank == 0:
        wandb.init(
            **config_wandb,
            name=run_name,
            config={
                "State"           : config_state,
                "Training"        : config_training,
                "Model"           : config,
                "Cluster"         : config_cluster,
                "Hyperparameters" : get_wandb_hyperparameters({**config_training, **config}),
            },
        )
        wandb.log({
            "Informations/Trainable Parameters [M]" : sum(p.numel() for p in model.parameters()) / 1e6,
            "Informations/Tokens"                   : model.tokens,
        })
    else:
        wandb.init(mode="disabled")

    # Number of steps to accumulate gradients before updating model parameters
    steps_gradient_accumulation = max(1, math.ceil(batch_size_per_step / (batch_size_per_gpu * world_size)))
    batches = [
        steps_update * steps_gradient_accumulation,
        steps_update // steps_validation * batches_validation,
        None,
    ]

    # Each sample is a single state of the group, with the conditioning of its date
    dataloader_training, dataloader_validation, _ = get_dataloaders(
        batch_size      = batch_size_per_gpu,
        num_workers     = num_workers,
        prefetch_factor = prefetch_factor,
        batches         = batches,
        shuffle         = [True, True, False],
        infinite        = [True, True, False],
        rank            = rank,
        world_size      = world_size,
        is_distributed  = is_distributed,
        input_states    = 1,
        output_states   = 0,
        group           = group,
    )

    # Masks and meshes are identical on every process, so they are not broadcast
    if is_distributed:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        model      = DDP(model, broadcast_buffers=False, gradient_as_bucket_view=True, **ddp_kwargs)

    # Setting up training tools
    optimizer = SOAP(
        model.parameters(),
        lr=lr_peak,
        max_precond_size=1024,
    )

    scheduler = warmup_cosine_decay(
        optimizer    = optimizer,
        lr_start     = lr_start,
        lr_peak      = lr_peak,
        lr_end       = lr_end,
        warmup_steps = warmup_steps,
        total_steps  = steps_update,
    )

    loss_accumulator         = 0.0
    loss_logging_accumulator = 0.0
    loss_validation_best     = float("inf")
    gradient_norm            = float("inf")
    optimizer_step           = 0

    # Waiting for processes to be ready
    if is_distributed:
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)

    for step, batch in enumerate(dataloader_training):

        # Forward pass and loss
        loss = forward_loss(model, batch, group, device)

        # Gradient accumulation
        loss              = loss / steps_gradient_accumulation
        loss_accumulator += loss.item()

        # Only sync gradients on last accumulation step
        is_last_accumulation_step = ((step + 1) % steps_gradient_accumulation == 0)

        if is_distributed and not is_last_accumulation_step:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

        # Cleaning up memory
        del batch, loss

        if not is_last_accumulation_step:
            continue

        # Optimization step
        gradient_norm             = safe_gradient_step(optimizer=optimizer, grad_clip=1.0)
        loss_step                 = loss_accumulator
        loss_logging_accumulator += loss_accumulator
        loss_accumulator          = 0.0
        optimizer_step           += 1
        scheduler.step()

        # Logging to console if not using WandB
        if config_wandb["mode"] == "disabled":
            print(f"Step {optimizer_step:6d} | Loss: {loss_step:.4f} | γ: {scheduler.get_last_lr()[0]:.6f} | ∇: {gradient_norm:.4f}")

        # Logging results, averaged over the logging window and the processes
        if optimizer_step % steps_logging == 0:
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

        # Validation, and saving the best model
        if optimizer_step % steps_validation == 0:
            model.eval()
            loss_validation = 0.0
            with torch.no_grad():
                for _ in range(batches_validation):
                    loss_validation += forward_loss(model, next(dataloader_validation), group, device).item() / batches_validation
            model.train()

            if is_distributed:
                loss_validation = reduce_mean(loss_validation, device)

            if rank == 0:
                wandb.log({
                    "Validation/Loss"           : loss_validation,
                    "Informations/Steps Update" : optimizer_step,
                })

                if saving and loss_validation < loss_validation_best:
                    raw_model            = model.module if hasattr(model, "module") else model
                    loss_validation_best = loss_validation
                    save(raw_model, config, PATH_MODELS / run_name)

    # Closing run
    wandb.finish()
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch the training of the encoder of a group of variables.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the .yml configuration of the encoder, e.g. configs/encoder/encoder-ocean.yml.",
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
    config         = load_configuration(args.config)[0]
    config_cluster = config["Cluster"]

    def train() -> None:
        r"""Train the encoder of the configuration."""

        training(
            config_state    = config["State"],
            config_training = config["Training"],
            config_model    = config["Model"],
            config_wandb    = config["WandB"],
            config_cluster  = config_cluster,
        )

    # Local
    if args.backend == "async":
        train()

    # Cluster
    else:
        schedule_training(
            train,
            name           = f"NEPT-ENC-{config['Model']['group'].upper()}",
            config_cluster = config_cluster,
            modules        = [
                neptune.data,
                neptune.data.dataset,
                neptune.data.weights,
                neptune.data.dataloader,
                neptune.model.encoders,
            ],
        )
