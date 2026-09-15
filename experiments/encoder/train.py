r"""Launch the training of a single (surface or ocean) encoder."""

import argparse
import cloudpickle
import dask
import dawgz
import secrets
import torch
import torch.distributed as dist
import wandb

from omegaconf import OmegaConf
from shaggy.loss import loss_geometry_embedding
from shaggy.models.cae import ConvEncoder
from shaggy.optimizers.gradients import safe_gradient_step
from shaggy.optimizers.soap import SOAP
from shaggy.tools import load_config, load_weights
from shaggy.tools import save as s_save
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from neptune.config import PATH_MODELS
from neptune.data import DATASET_VARIABLES_OCEAN, DATASET_VARIABLES_SURFACE
from neptune.data.dataloader import get_dataloaders
from neptune.data.weights import get_weights_mask
from neptune.distributed import reduce_mean, setup_distributed
from neptune.schedulers import warmup_cosine_decay
from neptune.tools import generate_run_name_ae, get_wandb_hyperparameters, load_configuration


# fmt: off
#
def _build_encoder(
    in_channels     : int,
    spatial         : int,
    arch            : dict,
    checkpoint_name : str | None,
    device          : torch.device,
) -> tuple[torch.nn.Module, dict]:
    r"""Build a ConvEncoder from scratch, or resume one from a checkpoint.

    Arguments:
        in_channels     : Number of input channels (including the mask channel), if training from scratch.
        spatial         : Number of spatial dimensions (2 for surface, 3 for ocean).
        arch            : Architecture config (hid_channels, hid_blocks, lat_channels, checkpointing, ...).
        checkpoint_name : Name of the checkpoint to resume from, or None to start fresh.
        device          : Target device.

    Returns:
        encoder : The (possibly resumed) ConvEncoder, in train mode.
        config  : The ConvEncoder constructor kwargs, reused as-is when saving a checkpoint.
    """
    if checkpoint_name is not None:
        ckpt_path = PATH_MODELS / checkpoint_name
        config    = OmegaConf.to_container(load_config(ckpt_path))
        encoder   = load_weights(ConvEncoder(**config), ckpt_path, device=str(device)).train()
    else:
        config = {
            "in_channels"  : in_channels,
            "out_channels" : arch["lat_channels"],
            "spatial"      : spatial,
            **{k: v for k, v in arch.items() if k != "lat_channels"},
        }
        encoder = ConvEncoder(**config).to(device)

    return encoder, config


def training(
    role: str,
    joint_hash: str,
    config_state: dict,
    config_training: dict,
    config_encoder: dict,
    config_wandb: dict,
    config_cluster: dict,
) -> None:
    r"""Launch the training of a single (surface or ocean) encoder."""

    n_channels, spatial = {
        "surface" : (len(DATASET_VARIABLES_SURFACE), 2),
        "ocean"   : (len(DATASET_VARIABLES_OCEAN), 3),
    }[role]

    # Initialize distributed setup
    rank, local_rank, world_size, device, is_distributed = setup_distributed()

    # Prevent xarray/dask deadlocks inside DataLoader workers
    dask.config.set(scheduler="synchronous")

    # Weights & Biases | One run per encoder, named after its architecture and joint_hash
    checkpoint_name = config_state[f"checkpoint_name_{role}"]
    run_name = generate_run_name_ae(
        joint_hash        = joint_hash,
        in_channels       = n_channels,
        lat_channels      = config_encoder["lat_channels"],
        hid_channels      = config_encoder["hid_channels"],
        hid_blocks        = config_encoder["hid_blocks"],
        stride            = config_encoder["stride"],
        spatial           = spatial,
        previous_run_name = checkpoint_name,
    )

    if rank == 0:
        wandb.init(
            **config_wandb,
            name=run_name,
            config={
                "State"           : config_state,
                "Training"        : config_training,
                "Architecture"    : config_encoder,
                "Cluster"         : config_cluster,
                "Hyperparameters" : get_wandb_hyperparameters([config_training, config_encoder]),
            },
        )
    else:
        wandb.init(mode="disabled")

    (
        saving,
        checkpointing,
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
        config_state["checkpointing"],
        config_training["steps_update"],
        config_training["steps_logging"],
        config_training["steps_saving"],
        config_training[f"batch_size_per_step_{role}"],
        config_training[f"batch_size_per_gpu_{role}"],
        config_training["num_workers"],
        config_training["prefetch_factor"],
        config_training["learning_rate_start"],
        config_training["learning_rate_peak"],
        config_training["learning_rate_end"],
        config_training["warmup_steps"],
    )

    # Number of steps to accumulate gradients before updating model parameters
    batch_size_per_process      = batch_size_per_gpu * world_size
    steps_gradient_accumulation = max(1, (batch_size_per_step + batch_size_per_process - 1) // batch_size_per_process)
    batches                     = [steps_update * steps_gradient_accumulation, None, None]

    dataloader_training, _, _ = get_dataloaders(
        batch_size      = batch_size_per_gpu,
        num_workers     = num_workers,
        prefetch_factor = prefetch_factor,
        batches         = batches,
        shuffle         = [True, False, False],
        infinite        = [True, False, False],
        rank            = rank,
        world_size      = world_size,
        is_distributed  = is_distributed,
    )

    # Land/sea mask | One extra input channel, appended along the channel axis
    mask_full = get_weights_mask(dim=1, device=device)                  # (Z, Y, X)
    w_mask    = mask_full[0][None, None] if role == "surface" else mask_full[None, None]

    # Model | Loading a checkpoint or building from scratch
    encoder, ckpt_config = _build_encoder(
        in_channels     = n_channels + 1,
        spatial         = spatial,
        arch            = {**config_encoder, "checkpointing": checkpointing},
        checkpoint_name = checkpoint_name,
        device          = device,
    )

    # Model | Defining if DDP or DataParallel
    if is_distributed:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        encoder    = DDP(encoder, **ddp_kwargs)
    elif torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder, device_ids=list(range(torch.cuda.device_count()))).to(device)

    # Logging number of trainable parameters
    if rank == 0:
        n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        wandb.log({"Informations/Trainable Parameters [M]": n_params / 1e6})

    # Setting up training tools
    optimizer = SOAP(encoder.parameters(), lr=lr_peak, max_precond_size=128)
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

    # Waiting for processes to be ready
    if is_distributed:
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)

    for step, sample in enumerate(dataloader_training):
        x = sample[0] if role == "surface" else sample[1]

        # Pushing to device and concatenating the land/sea mask
        x = x.to(device)
        x_in = torch.cat([x, w_mask.expand(x.shape[0], *([-1] * (w_mask.dim() - 1)))], dim=1)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):

            # Forward pass
            z = encoder(x_in)

            # Computing loss
            loss = loss_geometry_embedding(x, z)

        # Gradient accumulation
        loss              = loss / steps_gradient_accumulation
        loss_accumulator += loss.item()

        # Logging to console if not using WandB
        if config_wandb["mode"] == "disabled":
            print(f"Step {optimizer_step:6d} | Loss: {loss_accumulator:.4f} | γ: {scheduler.get_last_lr()[0]:.6f} | ∇: {gradient_norm:.4f}")

        # Only sync gradients on last accumulation step
        is_last_accumulation_step = ((step + 1) % steps_gradient_accumulation == 0)

        if is_distributed and not is_last_accumulation_step:
            with encoder.no_sync():
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()

        # Cleaning up memory
        del x, x_in, z

        # Optimization step
        if is_last_accumulation_step:
            gradient_norm             = safe_gradient_step(optimizer=optimizer, scaler=scaler, grad_clip=1.0)
            loss_to_log               = loss_accumulator
            loss_logging_accumulator += loss_to_log
            loss_accumulator          = 0.0
            optimizer_step           += 1
            scheduler.step()
            del loss

        # Logging results
        if optimizer_step % steps_logging == 0 and is_last_accumulation_step:

            # Average loss over logging window
            loss_mean                = loss_logging_accumulator / steps_logging
            loss_logging_accumulator = 0.0

            # Average across distributed processes
            if is_distributed:
                loss_mean = reduce_mean(loss_mean, device)

            # Logging
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

                # Extracting raw model and saving (overwrites the previous checkpoint for this run)
                raw_encoder = encoder.module if hasattr(encoder, "module") else encoder
                s_save(raw_encoder, ckpt_config, PATH_MODELS / run_name)

                # Updating best loss
                loss_best = loss_mean

    # Closing run
    wandb.finish()
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Submit the surface and ocean encoder trainings.")
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
    joint_hash     = secrets.token_hex(2).upper()

    nodes         = config_cluster["nodes"]
    gpus_per_node = config_cluster["gpus-per-node"]
    cpus_per_node = config_cluster["cpus-per-node"]
    ram_per_node  = config_cluster["ram-per-node"]

    # Local
    if args.backend == "async":
        for role in ("surface", "ocean"):
            training(
                role=role,
                joint_hash=joint_hash,
                config_state=configs[0]["State"],
                config_training=configs[0]["Training"],
                config_encoder=configs[0]["Encoders"][role.capitalize()],
                config_wandb=config_wandb,
                config_cluster=config_cluster,
            )

    # Cluster
    else:

        # Freeze modules to avoid pickling issues
        import neptune.data
        import neptune.data.dataloader
        import neptune.data.dataset
        import neptune.data.weights
        for _mod in [neptune.data, neptune.data.dataset, neptune.data.weights, neptune.data.dataloader]:
            cloudpickle.register_pickle_by_value(_mod)

        if nodes > 1:
            interpreter = (
                f"torchrun --nnodes {nodes} --nproc-per-node {gpus_per_node} "
                f"--rdzv_backend=c10d --rdzv_endpoint=$SLURMD_NODENAME:$((20000 + SLURM_JOB_ID % 10000)) "
                f"--rdzv_id=$SLURM_JOB_ID"
            )
        else:
            interpreter = f"torchrun --nnodes 1 --nproc-per-node {gpus_per_node} --standalone"

        job_kwargs = dict(
            array=len(configs),
            nodes=nodes,
            gpus=gpus_per_node,
            cpus=cpus_per_node,
            ram=ram_per_node,
            time=config_cluster["time"],
            account=config_cluster["account"],
            partition=config_cluster["partition"],
        )

        @dawgz.job(**job_kwargs)
        def train_surface(i: int) -> None:
            training(
                role="surface",
                joint_hash=joint_hash,
                config_state=configs[i]["State"],
                config_training=configs[i]["Training"],
                config_encoder=configs[i]["Encoders"]["Surface"],
                config_wandb=config_wandb,
                config_cluster=config_cluster,
            )

        @dawgz.job(**job_kwargs)
        def train_ocean(i: int) -> None:
            training(
                role="ocean",
                joint_hash=joint_hash,
                config_state=configs[i]["State"],
                config_training=configs[i]["Training"],
                config_encoder=configs[i]["Encoders"]["Ocean"],
                config_wandb=config_wandb,
                config_cluster=config_cluster,
            )

        dawgz.schedule(
            train_surface,
            train_ocean,
            name="E3D-TRAIN",
            backend="slurm",
            interpreter=interpreter,
            export="ALL"
        )
