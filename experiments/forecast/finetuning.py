r"""Launch the fine-tuning of a pre-trained Functional Generative Network (FGN) on its own rollouts."""

import argparse
import cloudpickle
import dask
import math
import torch
import torch.distributed as dist

from dawgz import job, schedule
from omegaconf import OmegaConf
from shaggy.optimizers.gradients import safe_gradient_step
from shaggy.optimizers.soap import SOAP
from shaggy.tools import load, load_config, save
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from neptune.config import PATH_MODELS
from neptune.data.dataloader import get_dataloaders
from neptune.data.weights import get_weights_conditioning, get_weights_loss
from neptune.distributed import reduce_mean, setup_distributed
from neptune.loss import loss_crps_rollout
from neptune.model import FGN
from neptune.schedulers import warmup_cosine_decay
from neptune.tools import generate_run_name_fgn, get_wandb_hyperparameters, load_configuration


# fmt: off
#
def _load_model(
    checkpoint_name : str,
    device          : torch.device,
) -> tuple[FGN, dict]:
    r"""Load a pre-trained FGN, the fine-tuning never starting from scratch.

    Arguments:
        checkpoint_name : Name of the checkpoint to fine-tune.
        device          : Target device.

    Returns:
        model  : FGN in training mode, on the target device.
        config : Constructor kwargs.
    """

    # Security
    assert checkpoint_name is not None, "ERROR - The fine-tuning requires a pre-trained checkpoint_name."

    ckpt_path = PATH_MODELS / checkpoint_name
    config    = OmegaConf.to_container(load_config(ckpt_path))
    model     = load(ckpt_path, FGN, device=str(device)).train()

    return model, config


def forward_loss(
    model   : torch.nn.Module,
    batch   : tuple,
    members : int,
    weights : tuple[Tensor, Tensor, Tensor],
    noise   : list[float],
    device  : torch.device,
) -> tuple[Tensor, Tensor]:
    r"""Forecast autoregressively the next K states of a batch with an ensemble and compute its weighted CRPS.

    Arguments:
        model   : FGN, possibly wrapped by DDP.
        batch   : Window of previous and future states, with the conditioning and dates.
        members : Number of ensemble members E.
        weights : Loss weights of the days of the rollout.
        noise   : Standard deviation of the noise added to the ERA5 conditioning of each step (K,).
        device  : Target device.

    Returns:
        loss_surface : CRPS of the surface variables, weighted over the days of the rollout.
        loss_ocean   : CRPS of the ocean variables, weighted over the days of the rollout.
    """

    x_inp_s, x_inp_o, x_out_s, x_out_o, c_inp, dates = batch
    fgn  = model.module if hasattr(model, "module") else model
    n, k = x_inp_s.shape[1], x_out_s.shape[1]

    # Each member is rolled out as a sample of its own, (B, ...) → (B * E, ...)
    x_inp_s, x_inp_o = (t.to(device).repeat_interleave(members, dim=0) for t in (x_inp_s, x_inp_o))

    x_pred_s, x_pred_o = [], []
    for step in range(k):

        # Conditioning | Date and (noisy) ERA5 of the last input state, the same for every member
        cond_s, cond_o = get_weights_conditioning(dates[n - 1 + step], c_inp[:, step], noise=noise[step], device=device)
        cond_s, cond_o = cond_s.repeat_interleave(members, dim=0), cond_o.repeat_interleave(members, dim=0)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            x_s, x_o = model(x_inp_s, x_inp_o, cond_s, cond_o)

        x_pred_s.append(x_s[:, 0])
        x_pred_o.append(x_o[:, 0])

        # Sliding window | Oldest state out, forecast in
        x_inp_s = torch.cat([x_inp_s[:, 1:], x_s], dim=1)
        x_inp_o = torch.cat([x_inp_o[:, 1:], x_o], dim=1)

    # Members gathered along the ensemble dimension, (B * E, K, ...) → (B, E, K, ...)
    x_pred_s = torch.stack(x_pred_s, dim=1).unflatten(0, (-1, members)).float()
    x_pred_o = torch.stack(x_pred_o, dim=1).unflatten(0, (-1, members)).float()

    return loss_crps_rollout(
        x_pred_s, x_pred_o, x_out_s.to(device), x_out_o.to(device), fgn.mask_surface, fgn.mask_ocean, *weights
    )


def training(
    config_state    : dict,
    config_training : dict,
    config_wandb    : dict,
    config_cluster  : dict,
) -> None:
    r"""Launch the fine-tuning of a pre-trained FGN on its own rollouts.

    Arguments:
        config_state    : Checkpoint to fine-tune and saving options.
        config_training : Ensemble, rollout, noise of the forcing, batch sizes, optimizer steps and learning rate schedule.
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
        rollout_days,
        rollout_weights,
        noise_forcing,
        noise_forcing_levels,
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
        config_training["rollout_days"],
        config_training["rollout_weights"],
        config_training["noise_forcing"],
        config_training["noise_forcing_levels"],
    )

    # Security
    assert len(rollout_weights) == rollout_days, f"ERROR - rollout_weights must have {rollout_days} elements, got {len(rollout_weights)}."
    assert all(w >= 0 for w in rollout_weights) and sum(rollout_weights) > 0, f"ERROR - rollout_weights must be non-negative, with a positive sum, got {rollout_weights}."
    assert len(noise_forcing_levels) == rollout_days, f"ERROR - noise_forcing_levels must have {rollout_days} elements, got {len(noise_forcing_levels)}."

    # Model | Loading the pre-trained checkpoint
    model, config = _load_model(checkpoint_name, device)

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
        output_states   = rollout_days,
    )

    # Loss | CRPS of the normalized increments, averaged over the non-constant variable-levels and weighted over the days
    weights = (torch.tensor(rollout_weights, device=device), *get_weights_loss(device=device))

    # Noise | Added to the ERA5 conditioning of each step, during the training and the validation
    noise = noise_forcing_levels if noise_forcing else [0.0] * rollout_days

    # Model | Masks, meshes and statistics are identical on every process, no need to broadcast them, and the
    #         static graph allows the parameters to be used once per step of the rollout before the backward pass
    if is_distributed:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        model      = DDP(model, broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True, **ddp_kwargs)

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
        loss_surface, loss_ocean = forward_loss(model, batch, members, weights, noise, device)

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
                    loss_surface, loss_ocean = forward_loss(model, next(dataloader_validation), members, weights, noise, device)
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

    parser = argparse.ArgumentParser(description="Launch the fine-tuning of a pre-trained FGN on its own rollouts.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the fine-tuning .yml configuration file.",
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
        def finetune(i: int) -> None:
            r"""Fine-tune the FGN of the i-th configuration."""

            training(
                config_state    = configs[i]["State"],
                config_training = configs[i]["Training"],
                config_wandb    = config_wandb,
                config_cluster  = config_cluster,
            )

        schedule(
            finetune,
            name="NEPT-FINETUNE",
            backend="slurm",
            interpreter=interpreter,
            export="ALL",
        )
