r"""Launch the training of the deterministic encoder of a group of variables, preserving the geometry of its states."""

import argparse
import dask
import math
import torch
import torch.distributed as dist
import wandb

from omegaconf import OmegaConf
from pathlib import Path
from shaggy.distributed import reduce_mean, setup_distributed
from shaggy.loss import loss_geometry_embedding
from shaggy.optimizers.gradients import safe_gradient_step
from shaggy.optimizers.soap import SOAP
from shaggy.schedulers import warmup_cosine_decay
from shaggy.tools import load, load_config, load_weights, save
from shaggy.training import has_stagnated, load_state, save_state
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

import neptune.data
import neptune.data.dataloader
import neptune.data.dataset
import neptune.data.weights
import neptune.model.encoders
import neptune.training

from neptune.config import PATH_MODELS
from neptune.data.dataloader import get_dataloaders
from neptune.data.weights import get_weights_distance, get_weights_distance_era5
from neptune.model.encoders import Encoder
from neptune.tools import (
    generate_run_name,
    get_wandb_hyperparameters,
    load_configuration,
    schedule_training,
)
from neptune.training import get_run_id, submit_resume


# fmt: off
#
def _build_model(
    config_model    : dict,
    checkpoint_name : str | None,
    device          : torch.device,
) -> tuple[Encoder, dict]:
    r"""Build an encoder from scratch, or resume it from a checkpoint.

    Arguments:
        config_model    : Constructor kwargs of the encoder (group, ...), used when training from scratch.
        checkpoint_name : Name of the checkpoint to resume from (if applicable).
        device          : Target device.

    Returns:
        model  : Encoder in training mode, on the target device.
        config : Constructor kwargs.
    """

    if checkpoint_name is not None:
        ckpt_path = PATH_MODELS / checkpoint_name
        config    = OmegaConf.to_container(load_config(ckpt_path))
        model     = load(ckpt_path, Encoder, device=str(device)).train()
    else:
        config = config_model
        model  = Encoder(**config).to(device)

    return model, config


def forward_loss(
    model         : torch.nn.Module,
    batch         : tuple,
    group         : str,
    weights       : Tensor,
    device        : torch.device,
    memory_format : torch.memory_format,
) -> Tensor:
    r"""Encode the states of a batch and compute the GME loss between the states and their latents.

    Arguments:
        model         : Encoder, possibly wrapped by DDP.
        batch         : Single state, from NeptuneDataset.
        group         : Group of variables, 'surface', 'ocean' or 'era5' (the forcing of the same day).
        weights       : Weight σ/τ of each channel in the distances, (C, 1, 1) or (C, Z, 1, 1).
        device        : Target device.
        memory_format : Layout of the states in memory, the channels last during the training.

    Returns:
        loss : GME loss of the encoder.
    """

    x_inp_s, x_inp_o, _, _, c_inp, _ = batch
    states = {"surface": x_inp_s, "ocean": x_inp_o, "era5": c_inp}[group][:, 0].to(device, memory_format=memory_format)

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        z = model(states)

    # The distances between states are counted in typical days of change, whatever the variable and the level
    return loss_geometry_embedding(states * weights, z.float())


def training(
    config_state    : dict,
    config_training : dict,
    config_model    : dict,
    config_wandb    : dict,
    config_cluster  : dict,
) -> None:
    r"""Launch the training of the encoder of a group of variables.

    Arguments:
        config_state    : Saving option, checkpoint to resume from and whether its optimizer state is resumed too.
        config_training : Batch sizes, optimizer steps, learning rate schedule, validation and stopping.
        config_model    : Group of variables, latent channels and architecture of the encoder.
        config_wandb    : Weights & Biases entity, project and mode.
        config_cluster  : Slurm resources and number of jobs that may resume the training after the first one.
    """

    # Initialize distributed setup
    rank, local_rank, world_size, device, is_distributed = setup_distributed()

    # A job stopped by its time limit leaves the next one waiting in the queue, which resumes the same run
    if rank == 0:
        submit_resume(config_cluster["restarts"])

    # Prevent xarray/dask deadlocks inside DataLoader workers
    dask.config.set(scheduler="synchronous")

    group = config_model["group"]

    (
        saving,
        checkpoint_name,
        resume_optimizer,
        steps_update,
        steps_logging,
        steps_validation,
        steps_checkpoint,
        samples_validation,
        stopping_window,
        stopping_threshold,
        scale_distances,
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
        config_state["resume_optimizer"],
        config_training["steps_update"],
        config_training["steps_logging"],
        config_training["steps_validation"],
        config_training["steps_checkpoint"],
        config_training["samples_validation"],
        config_training["stopping_window"],
        config_training["stopping_threshold"],
        config_training["scale_distances"],
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
    # Each GPU reads its share of the fixed days, whole batches only: any other number would drop days without a word
    samples_per_step = batch_size_per_gpu * world_size
    assert samples_validation % samples_per_step == 0, f"ERROR - samples_validation counts the days of all the GPUs, it must be a multiple of {samples_per_step}, got {samples_validation}."

    # The model is loaded from a checkpoint or built from scratch
    model, config = _build_model(config_model, checkpoint_name, device)

    # Weight of each channel in the distances between states, computed once
    if group == "era5":
        weights = get_weights_distance_era5(device=device)
    else:
        weights_surface, weights_ocean = get_weights_distance(device=device)
        weights = weights_surface if group == "surface" else weights_ocean

    # Without the scaling, every channel weighs 1, except the constant levels that stay out of the distances
    if not scale_distances:
        weights = (weights > 0).float()

    # The Weights & Biases run is named after the architecture, the jobs of a same training sharing its name
    previous_name = None if checkpoint_name is None else Path(checkpoint_name).parts[0]
    run_name      = generate_run_name(f"ENC_{group}_lc{model.lat_channels}_tk{model.tokens}", previous_name, get_run_id())
    run_path      = PATH_MODELS / run_name

    # A job restarted after a crash or a time limit always resumes exactly where its run stopped
    state_path = run_path / "last" / "state.pt"
    state      = None
    if state_path.exists():
        state = load_state(state_path, device)
        model = load_weights(model, run_path / "last", device=str(device)).train()
        if rank == 0:
            print(f"Resuming {run_name} at update {state['optimizer_step']}", flush=True)
    elif checkpoint_name is not None and resume_optimizer:
        state_path_resumed = PATH_MODELS / checkpoint_name / "state.pt"
        assert state_path_resumed.exists(), f"ERROR - No optimizer state in {checkpoint_name}, use its last/ folder or resume_optimizer: false."
        state = load_state(state_path_resumed, device)

    if rank == 0:
        wandb.init(
            **config_wandb,
            name=run_name,
            id=run_name,
            resume="allow",
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
        wandb.define_metric("*", step_metric="Informations/Steps Update")
    else:
        wandb.init(mode="disabled")

    # The updates redone after a resume are not logged again, so that the curves continue from their last point
    update_logged = wandb.run.summary.get("Informations/Steps Update", 0) if rank == 0 else 0

    # Number of steps to accumulate gradients before updating model parameters
    steps_gradient_accumulation = max(1, math.ceil(batch_size_per_step / (batch_size_per_gpu * world_size)))
    optimizer_step              = 0 if state is None else state["optimizer_step"]
    batches                     = [(steps_update - optimizer_step) * steps_gradient_accumulation, None, None]

    # Each sample is a single state of the group
    dataloader_training, dataloader_validation, _ = get_dataloaders(
        batch_size         = batch_size_per_gpu,
        num_workers        = num_workers,
        prefetch_factor    = prefetch_factor,
        batches            = batches,
        shuffle            = [True, False, False],
        infinite           = [True, False, False],
        rank               = rank,
        world_size         = world_size,
        is_distributed     = is_distributed,
        samples_validation = samples_validation,
        seed               = optimizer_step,
        input_states       = 1,
        output_states      = 0,
        group              = group,
    )

    # The convolutions run faster in bfloat16 with the channels last, a layout of the training only: the shapes do
    # not change, and a saved encoder loads back with the usual layout
    spatial       = len(model.shape)
    memory_format = torch.channels_last if spatial == 2 else torch.channels_last_3d
    for parameter in model.parameters():
        if parameter.dim() == spatial + 2:
            parameter.data = parameter.data.contiguous(memory_format=memory_format)

    # The network is compiled at its first batch, its normalizations and activations fused around the convolutions
    model.compile()

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

    # The learning rate schedule and the moments of the optimizer come back with a state, not with the weights alone
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    loss_accumulator         = 0.0
    loss_logging_accumulator = 0.0
    loss_training_window     = 0.0
    updates_training_window  = 0
    gradient_norm            = float("inf")
    samples_seen             = 0   if state is None else state["samples_seen"]
    losses_validation        = []  if state is None else state["losses_validation"]
    loss_validation_best     = float("inf") if state is None else state["loss_validation_best"]
    loss_training_best       = float("inf") if state is None else state["loss_training_best"]
    raw_model                = model.module if hasattr(model, "module") else model

    # Waiting for processes to be ready
    if is_distributed:
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)

    for step, batch in enumerate(dataloader_training):

        # Forward pass and loss
        loss = forward_loss(model, batch, group, weights, device, memory_format)
        samples_seen += batch_size_per_gpu * world_size

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
        loss_training_window     += loss_accumulator
        updates_training_window  += 1
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

            if rank == 0 and optimizer_step > update_logged:
                wandb.log({
                    "Training/Loss"              : loss_mean,
                    "Informations/Steps Update"  : optimizer_step,
                    "Informations/Samples Seen"  : samples_seen,
                    "Informations/Gradient Norm" : gradient_norm,
                    "Informations/Learning Rate" : scheduler.get_last_lr()[0],
                })

        # Validation, on the same days each time, and saving the best models
        stop = False
        if optimizer_step % steps_validation == 0:
            model.eval()
            loss_validation = 0.0
            with torch.no_grad():
                for batch in dataloader_validation:
                    loss_validation += forward_loss(model, batch, group, weights, device, memory_format).item() / len(dataloader_validation)
            model.train()

            # The training loss of one batch is noisy, the mean since the last validation is what gets compared
            loss_training           = loss_training_window / updates_training_window
            loss_training_window    = 0.0
            updates_training_window = 0

            if is_distributed:
                loss_validation = reduce_mean(loss_validation, device)
                loss_training   = reduce_mean(loss_training, device)

            losses_validation.append(loss_validation)

            if rank == 0:
                # A job can stop during a validation, after logging the training loss of this same update
                if optimizer_step >= update_logged:
                    wandb.log({
                        "Validation/Loss"           : loss_validation,
                        "Informations/Steps Update" : optimizer_step,
                    })

                if saving and loss_validation < loss_validation_best:
                    save(raw_model, config, run_path / "best-validation")
                if saving and loss_training < loss_training_best:
                    save(raw_model, config, run_path / "best-training")

            loss_validation_best = min(loss_validation_best, loss_validation)
            loss_training_best   = min(loss_training_best, loss_training)

            # Every process reads the same reduced losses, so that they all stop at the same update
            stop = stopping_window is not None and has_stagnated(losses_validation, stopping_window, stopping_threshold)

        # Everything needed to resume from this exact update, also written at the very end to continue a finished run
        is_last = stop or optimizer_step == steps_update
        if saving and rank == 0 and (optimizer_step % steps_checkpoint == 0 or is_last):
            save(raw_model, config, run_path / "last")
            save_state(state_path, {
                "optimizer_step"       : optimizer_step,
                "samples_seen"         : samples_seen,
                "optimizer"            : optimizer.state_dict(),
                "scheduler"            : scheduler.state_dict(),
                "losses_validation"    : losses_validation,
                "loss_validation_best" : loss_validation_best,
                "loss_training_best"   : loss_training_best,
            })

        if stop:
            if rank == 0:
                wandb.log({"Informations/Stopped Early": 1, "Informations/Steps Update": optimizer_step})
            break

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
                neptune.training,
            ],
        )
