r"""Launch the training of a Functional Generative Network (FGN) forecasting the next day."""

import argparse
import cloudpickle
import dask
import math
import torch
import torch.distributed as dist
import wandb

from collections.abc import Sequence
from dawgz import job, schedule
from omegaconf import OmegaConf
from shaggy.optimizers.gradients import safe_gradient_step
from shaggy.optimizers.soap import SOAP
from shaggy.tools import load, load_config, save
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from neptune.config import PATH_MODELS
from neptune.data.dataloader import get_dataloaders
from neptune.data.weights import get_weights_date, get_weights_loss
from neptune.distributed import reduce_mean, setup_distributed
from neptune.loss import loss_crps
from neptune.model import FGN
from neptune.schedulers import warmup_cosine_decay
from neptune.tools import generate_run_name_fgn, get_wandb_hyperparameters, load_configuration


# fmt: off
#
def _build_model(
    config_model    : dict,
    checkpoint_name : str | None,
    device          : torch.device,
) -> tuple[FGN, dict]:
    r"""Build a FGN from scratch, or resume one from a checkpoint.

    Arguments:
        config_model    : Constructor kwargs of the FGN, used when training from scratch.
        checkpoint_name : Name of the checkpoint to resume from (if applicable).
        device          : Target device.

    Returns:
        model  : FGN in training mode, on the target device.
        config : Constructor kwargs.
    """

    if checkpoint_name is not None:
        ckpt_path = PATH_MODELS / checkpoint_name
        config    = OmegaConf.to_container(load_config(ckpt_path))
        model     = load(ckpt_path, FGN, device=str(device)).train()
    else:
        config = config_model
        model  = FGN(**config).to(device)

    return model, config


def _conditioning(dates: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
    r"""Encode the dates of a batch as the conditioning of the FGN.

    Arguments:
        dates  : Date strings 'YYYY-MM-DD' of the forecasted states (B,).
        device : Target device.

    Returns:
        cond_s : Surface conditioning (B, 2, Y, X).
        cond_o : Ocean conditioning (B, 2, Z, Y, X).
    """

    encodings = [get_weights_date(date, dim=2, device=device) for date in dates]

    return torch.cat([cond_s for cond_s, _ in encodings]), torch.cat([cond_o for _, cond_o in encodings])


def forward_loss(
    model   : torch.nn.Module,
    batch   : tuple,
    members : int,
    weights : tuple[Tensor, Tensor],
    device  : torch.device,
) -> tuple[Tensor, Tensor]:
    r"""Forecast the next states of a batch with an ensemble and compute its CRPS.

    Arguments:
        model   : FGN, possibly wrapped by DDP.
        batch   : Window of previous and future states, with their dates, from NeptuneDataset.
        members : Number of ensemble members E.
        weights : Loss weights of the surface and ocean variables, from get_weights_loss.
        device  : Target device.

    Returns:
        loss_surface : Weighted CRPS of the surface variables.
        loss_ocean   : Weighted CRPS of the ocean variables.
    """

    x_inp_s, x_inp_o, x_out_s, x_out_o, dates = batch
    fgn = model.module if hasattr(model, "module") else model

    # The collate transposes the dates, dates[N] holds the date of the next state of each sample
    cond_s, cond_o   = _conditioning(dates[x_inp_s.shape[1]], device)
    x_inp_s, x_inp_o = x_inp_s.to(device), x_inp_o.to(device)
    x_s, x_o         = x_out_s[:, 0].to(device), x_out_o[:, 0].to(device)

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        x_pred_s, x_pred_o = model(x_inp_s, x_inp_o, cond_s, cond_o, members=members)

    return loss_crps(
        x_pred_s.float(), x_pred_o.float(), x_s, x_o, fgn.mask_surface, fgn.mask_ocean, *weights
    )


def training(
    config_state    : dict,
    config_training : dict,
    config_model    : dict,
    config_wandb    : dict,
    config_cluster  : dict,
) -> None:
    r"""Launch the training of a FGN forecasting the next day.

    Arguments:
        config_state    : Checkpointing and saving options.
        config_training : Ensemble, batch sizes, optimizer steps and learning rate schedule.
        config_model    : Architecture of the FGN.
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
        members,
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
        config_training["members"],
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

    # Model | Loading a checkpoint or building from scratch
    model, config = _build_model(config_model, checkpoint_name, device)

    # Weights & Biases | Run named after the architecture
    run_name = generate_run_name_fgn(
        input_states      = model.input_states,
        lat_channels      = model.lat_channels,
        compression       = model.compression()[2],
        tokens            = sum(model.tokens),
        hid_channels      = model.processor.in_proj.out_features,
        hid_blocks        = len(model.processor.blocks),
        previous_run_name = checkpoint_name,
    )

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
            "Informations/Tokens"                   : sum(model.tokens),
            "Informations/Compression"              : model.compression()[2],
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
        input_states    = model.input_states,
        output_states   = 1,
    )

    # Loss | CRPS of the normalized increments, averaged over the non-constant variable-levels
    weights = get_weights_loss(device=device)

    # Model | Masks, meshes and statistics are identical on every process, no need to broadcast them
    if is_distributed:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        model      = DDP(model, broadcast_buffers=False, **ddp_kwargs)

    # Setting up training tools
    optimizer = SOAP(
        model.parameters(),
        lr=lr_peak,
    )

    scheduler = warmup_cosine_decay(
        optimizer    = optimizer,
        lr_start     = lr_start,
        lr_peak      = lr_peak,
        lr_end       = lr_end,
        warmup_steps = warmup_steps,
        total_steps  = steps_update,
    )

    loss_accumulator         = torch.zeros(2)
    loss_logging_accumulator = torch.zeros(2)
    loss_validation_best     = float("inf")
    gradient_norm            = float("inf")
    optimizer_step           = 0

    # Waiting for processes to be ready
    if is_distributed:
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)

    for step, batch in enumerate(dataloader_training):

        # Forward pass and loss (surface, ocean)
        loss_surface, loss_ocean = forward_loss(model, batch, members, weights, device)

        # Gradient accumulation
        loss              = (loss_surface + loss_ocean) / steps_gradient_accumulation
        loss_accumulator += torch.tensor([loss_surface.item(), loss_ocean.item()]) / steps_gradient_accumulation

        # Only sync gradients on last accumulation step
        is_last_accumulation_step = ((step + 1) % steps_gradient_accumulation == 0)

        if is_distributed and not is_last_accumulation_step:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

        # Cleaning up memory
        del batch, loss, loss_surface, loss_ocean

        if not is_last_accumulation_step:
            continue

        # Optimization step
        gradient_norm             = safe_gradient_step(optimizer=optimizer, grad_clip=1.0)
        loss_step                 = loss_accumulator.sum().item()
        loss_logging_accumulator += loss_accumulator
        loss_accumulator          = torch.zeros(2)
        optimizer_step           += 1
        scheduler.step()

        # Logging to console if not using WandB
        if config_wandb["mode"] == "disabled":
            print(f"Step {optimizer_step:6d} | Loss: {loss_step:.4f} | γ: {scheduler.get_last_lr()[0]:.6f} | ∇: {gradient_norm:.4f}")

        # Logging results, averaged over the logging window and the processes
        if optimizer_step % steps_logging == 0:
            loss_mean_surface, loss_mean_ocean = (loss_logging_accumulator / steps_logging).tolist()
            loss_logging_accumulator           = torch.zeros(2)

            if is_distributed:
                loss_mean_surface = reduce_mean(loss_mean_surface, device)
                loss_mean_ocean   = reduce_mean(loss_mean_ocean, device)

            if rank == 0:
                wandb.log({
                    "Training/Loss"              : loss_mean_surface + loss_mean_ocean,
                    "Training/Loss (Surface)"    : loss_mean_surface,
                    "Training/Loss (Ocean)"      : loss_mean_ocean,
                    "Informations/Steps Update"  : optimizer_step,
                    "Informations/Samples Seen"  : (step + 1) * batch_size_per_gpu * world_size,
                    "Informations/Gradient Norm" : gradient_norm,
                    "Informations/Learning Rate" : scheduler.get_last_lr()[0],
                })

        # Validation, and saving the best model
        if optimizer_step % steps_validation == 0:
            model.eval()
            loss_validation = torch.zeros(2)
            with torch.no_grad():
                for _ in range(batches_validation):
                    loss_surface, loss_ocean = forward_loss(model, next(dataloader_validation), members, weights, device)
                    loss_validation         += torch.tensor([loss_surface.item(), loss_ocean.item()]) / batches_validation
            model.train()

            loss_validation_surface, loss_validation_ocean = loss_validation.tolist()
            if is_distributed:
                loss_validation_surface = reduce_mean(loss_validation_surface, device)
                loss_validation_ocean   = reduce_mean(loss_validation_ocean, device)

            if rank == 0:
                wandb.log({
                    "Validation/Loss"           : loss_validation_surface + loss_validation_ocean,
                    "Validation/Loss (Surface)" : loss_validation_surface,
                    "Validation/Loss (Ocean)"   : loss_validation_ocean,
                    "Informations/Steps Update" : optimizer_step,
                })

                if saving and loss_validation_surface + loss_validation_ocean < loss_validation_best:
                    raw_model            = model.module if hasattr(model, "module") else model
                    loss_validation_best = loss_validation_surface + loss_validation_ocean
                    save(raw_model, config, PATH_MODELS / run_name)

    # Closing run
    wandb.finish()
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch a FGN training pipeline.")
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
        for config in configs:
            training(
                config_state    = config["State"],
                config_training = config["Training"],
                config_model    = config["Model"],
                config_wandb    = config_wandb,
                config_cluster  = config_cluster,
            )

    # Cluster
    else:

        # Freeze modules to avoid pickling issues
        import neptune.data
        import neptune.data.dataloader
        import neptune.data.dataset
        import neptune.data.weights
        import neptune.loss
        import neptune.model.fgn
        for _mod in [
            neptune.data,
            neptune.data.dataset,
            neptune.data.weights,
            neptune.data.dataloader,
            neptune.loss,
            neptune.model.fgn,
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
            array     = len(configs),
            nodes     = nodes,
            gpus      = gpus_per_node,
            cpus      = cpus_per_node,
            ram       = ram_per_node,
            time      = config_cluster["time"],
            account   = config_cluster["account"],
            partition = config_cluster["partition"],
        )
        def train(i: int) -> None:
            r"""Train the FGN of the i-th configuration."""

            training(
                config_state    = configs[i]["State"],
                config_training = configs[i]["Training"],
                config_model    = configs[i]["Model"],
                config_wandb    = config_wandb,
                config_cluster  = config_cluster,
            )

        schedule(
            train,
            name="NEPT-TRAIN",
            backend="slurm",
            interpreter=interpreter,
            export="ALL",
        )
