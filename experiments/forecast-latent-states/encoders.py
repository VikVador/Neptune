r"""Launch the trainings of the deterministic surface and ocean encoders, preserving the geometry of the states."""

import argparse
import cloudpickle
import dask
import math
import secrets
import torch
import torch.distributed as dist
import wandb

from dawgz import job, schedule
from omegaconf import OmegaConf
from shaggy.loss import loss_geometry_embedding
from shaggy.optimizers.gradients import safe_gradient_step
from shaggy.optimizers.soap import SOAP
from shaggy.tools import load, load_config, save
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from neptune.config import PATH_MODELS
from neptune.data.dataloader import get_dataloaders
from neptune.data.weights import get_weights_conditioning
from neptune.distributed import reduce_mean, setup_distributed
from neptune.model.encoders import LatentEncoder, OceanEncoder, SurfaceEncoder
from neptune.schedulers import warmup_cosine_decay
from neptune.tools import extract_model_hash, get_wandb_hyperparameters, load_configuration

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

    # States | Encoded with their conditioning, the date and ERA5 at the surface, the date in the ocean (the collate transposes the dates)
    cond_s, cond_o = get_weights_conditioning(dates[0], c_inp[:, 0], device=device)
    states, cond   = (x_inp_s[:, 0], cond_s) if group == "surface" else (x_inp_o[:, 0], cond_o)
    states         = states.to(device)

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        z = model(torch.cat([states, cond], dim=1))

    # Loss | Geometry of the states only, the conditioning being an input of the encoder
    return loss_geometry_embedding(states, z.float())


def training(
    group           : str,
    config_state    : dict,
    config_training : dict,
    config_model    : dict,
    config_wandb    : dict,
    config_cluster  : dict,
) -> None:
    r"""Launch the training of the encoder of a group of variables.

    Arguments:
        group           : Group of variables, 'surface' or 'ocean'.
        config_state    : Saving option and checkpoint of each encoder to resume from.
        config_training : Batch sizes of each encoder, optimizer steps and learning rate schedule.
        config_model    : Shared latent channels and architecture of each encoder.
        config_wandb    : Weights & Biases entity, project and mode.
        config_cluster  : Slurm resources, logged alongside the run for traceability.
    """

    # Initialize distributed setup
    rank, local_rank, world_size, device, is_distributed = setup_distributed()

    # Prevent xarray/dask deadlocks inside DataLoader workers
    dask.config.set(scheduler="synchronous")

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
        config_state[f"checkpoint_name_{group}"],
        config_training["steps_update"],
        config_training["steps_logging"],
        config_training["steps_validation"],
        config_training["batches_validation"],
        config_training[f"batch_size_per_step_{group}"],
        config_training[f"batch_size_per_gpu_{group}"],
        config_training["num_workers"],
        config_training["prefetch_factor"],
        config_training["learning_rate_start"],
        config_training["learning_rate_peak"],
        config_training["learning_rate_end"],
        config_training["warmup_steps"],
    )

    # Security
    assert batch_size_per_gpu > 1, f"ERROR - The GME loss pairs the samples of a GPU, batch_size_per_gpu must be > 1, got {batch_size_per_gpu}."

    # Model | Loading a checkpoint or building from scratch
    config        = {"lat_channels": config_model["lat_channels"], **config_model[f"config_{group}"]}
    model, config = _build_model(group, config, checkpoint_name, device)

    # Weights & Biases | Run named after the architecture
    run_name = f"ENC_{group}_lc{model.lat_channels}_tk{model.tokens}__{secrets.token_hex(2).upper()}"
    if checkpoint_name is not None:
        run_name = f"{run_name}_{extract_model_hash(checkpoint_name)}"

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

    # Data | Single states of the group only, with the conditioning of their date
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

    # Model | Masks and meshes are identical on every process, no need to broadcast them
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

    parser = argparse.ArgumentParser(description="Launch the trainings of the latent encoders.")
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
    config         = load_configuration(args.config)[0]
    config_wandb   = config["WandB"]
    config_cluster = config["Cluster"]
    groups         = list(ENCODERS)

    nodes         = config_cluster["nodes"]
    gpus_per_node = config_cluster["gpus-per-node"]
    cpus_per_node = config_cluster["cpus-per-node"]
    ram_per_node  = config_cluster["ram-per-node"]

    # Local
    if args.backend == "async":
        for group in groups:
            training(
                group           = group,
                config_state    = config["State"],
                config_training = config["Training"],
                config_model    = config["Model"],
                config_wandb    = config_wandb,
                config_cluster  = config_cluster,
            )

    # Cluster | One job per encoder, with the same resources
    else:

        # Freeze modules to avoid pickling issues
        import neptune.data
        import neptune.data.dataloader
        import neptune.data.dataset
        import neptune.data.weights
        import neptune.model.encoders
        for _mod in [
            neptune.data,
            neptune.data.dataset,
            neptune.data.weights,
            neptune.data.dataloader,
            neptune.model.encoders,
        ]:
            cloudpickle.register_pickle_by_value(_mod)

        if nodes > 1:
            interpreter = (
                f"torchrun --nnodes {nodes} --nproc-per-node {gpus_per_node} "
                f"--rdzv_backend=c10d --rdzv_endpoint=$SLURMD_NODENAME:$((20000 + SLURM_JOB_ID % 10000)) "
                f"--rdzv_id=$SLURM_JOB_ID"
            )
        else:
            interpreter = f"torchrun --nnodes 1 --nproc-per-node {gpus_per_node} --standalone"

        @job(
            array     = len(groups),
            nodes     = nodes,
            gpus      = gpus_per_node,
            cpus      = cpus_per_node,
            ram       = ram_per_node,
            time      = config_cluster["time"],
            account   = config_cluster["account"],
            partition = config_cluster["partition"],
        )
        def train(i: int) -> None:
            r"""Train the encoder of the i-th group of variables."""

            training(
                group           = groups[i],
                config_state    = config["State"],
                config_training = config["Training"],
                config_model    = config["Model"],
                config_wandb    = config_wandb,
                config_cluster  = config_cluster,
            )

        schedule(
            train,
            name="NEPT-ENC",
            backend="slurm",
            interpreter=interpreter,
            export="ALL",
        )
