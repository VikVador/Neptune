r"""Training Autoencoder."""

import argparse
import dask
import dawgz
import torch
import torch.distributed as dist
import wandb

from omegaconf import OmegaConf
from shaggy.loss import AELoss
from shaggy.models.cae import create_ConvAE
from shaggy.optimizer import SOAP, safe_gradient_step
from shaggy.tools import load as s_load
from shaggy.tools import save as s_save
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from neptune.config import PATH_MODELS
from neptune.data import DATASET_REGION, DATASET_VARIABLES_OCEAN, DATASET_VARIABLES_SURFACE
from neptune.data.dataloader import get_dataloaders
from neptune.data.weights import get_weights_loss, get_weights_mask
from neptune.distributed import reduce_mean, setup_distributed
from neptune.tools import load_configuration


# fmt: off
#
def training(
    config_state: dict,
    config_training: dict,
    config_arch: dict,
    config_wandb: dict,
    config_cluster: dict,
) -> None:
    r"""Launch the training of an autoencoder."""

    # Initialize distributed setup
    rank, local_rank, world_size, device, is_distributed = setup_distributed()

    # Prevent xarray/dask deadlocks inside DataLoader workers
    dask.config.set(scheduler="synchronous")

    # Weights & Biases
    if rank == 0:
        wandb.init(
            **config_wandb,
            config={
                "State"        : config_state,
                "Training"     : config_training,
                "Architecture" : config_arch,
                "Cluster"      : config_cluster,
            },
        )
    else:
        wandb.init(mode="disabled")

    (
        steps_update,
        steps_logging,
        steps_validation,
        steps_saving,
        samples_validation,
        batch_size_per_step,
        batch_size_per_gpu,
        learning_rate,
        num_workers,
        prefetch_factor,
    ) = (
        config_training["steps_update"],
        config_training["steps_logging"],
        config_training["steps_validation"],
        config_training["steps_saving"],
        config_training["samples_validation"],
        config_training["batch_size_per_step"],
        config_training["batch_size_per_gpu"],
        config_training["learning_rate"],
        config_training["num_workers"],
        config_training["prefetch_factor"],
    )

    # Number of steps to accumulate gradients before updating model parameters
    batch_size_per_process = batch_size_per_gpu * world_size
    steps_gradient_accumulation = max(
        1, (batch_size_per_step + batch_size_per_process - 1) // batch_size_per_process
    )

    # Number of batches to yield per dataloader
    n_batches_validation = max(1, (samples_validation + batch_size_per_gpu - 1) // batch_size_per_gpu)
    batches = [
        steps_update * steps_gradient_accumulation,
        steps_update * n_batches_validation,
        None,
    ]

    dataloader_training, dataloader_validation, _ = get_dataloaders(
        batch_size      = batch_size_per_gpu,
        num_workers     = num_workers,
        prefetch_factor = prefetch_factor,
        batches         = batches,
        shuffle         = [True,  True, False],
        infinite        = [True, True, False],
        rank            = rank,
        world_size      = world_size,
        is_distributed  = is_distributed,
    )

    # Waiting for processes to be ready (1)
    if is_distributed:
        dist.barrier()

    # Constants
    Z     = DATASET_REGION["deptht"].stop - DATASET_REGION["deptht"].start     # Depth levels
    C     = len(DATASET_VARIABLES_SURFACE) + len(DATASET_VARIABLES_OCEAN) * Z  # Aggregated levels
    IN_C  = C + Z                                                              # State = Variables + Mask
    OUT_C = C                                                                  # State = Variables

    # Initializing weighting tensors
    w_mask, w_loss = (
        get_weights_mask(dim=2,              device=device), # (1,     Z, Y, X)
        get_weights_loss(dim=2, scale=100.0, device=device), # (1, OUT_C, 1, 1)
    )

    # Model | 1 | Loading checkpoint
    if config_state["checkpoint_name"] is not None:
        ckpt_path = PATH_MODELS / config_state["checkpoint_name"]
        model     = s_load(ckpt_path, device=str(device)).train()

    # Model | 2 | New
    else:
        model = create_ConvAE(
            in_channels  = IN_C,
            out_channels = OUT_C,
            **config_arch
        ).to(device)

    # Model | 3 | Defining if DDP or DataParallel
    if is_distributed:
        ddp_kwargs = {
            "device_ids": [local_rank],
            "output_device": local_rank
        } if device.type == "cuda" else {}

        model = DDP(model, **ddp_kwargs)

    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=list(range(torch.cuda.device_count()))
        ).to(device)

    # Logging number of trainable parameters
    if rank == 0:
        wandb.log({
            "Informations/Trainable Parameters [M]": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ) / 1e6,
        })

    # Setting up training tools
    optimizer                = SOAP(model.parameters(), lr=learning_rate)
    scheduler                = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler                   = GradScaler(enabled=True)
    loss_function            = AELoss(weights=w_loss)
    loss_accumulator         = 0.0
    loss_logging_accumulator = 0.0
    optimizer_step           = 0

    # Waiting for processes to be ready (2)
    if is_distributed:
        dist.barrier()

    for step, (x, _) in enumerate(dataloader_training):

        # Pushing to device and concatenating mask
        x = x.to(device)
        x = torch.cat([x, w_mask.expand(x.shape[0], -1, -1, -1)], dim=1) # (B, IN_C, Y, X)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):

            # Forward pass
            _, x_hat = model(x)

            # Computing loss
            loss = loss_function(x_hat, x[:, :C])

        # Gradient accumulation
        loss              = loss / steps_gradient_accumulation
        loss_accumulator += loss.item()

        # Only sync gradients on last accumulation step
        is_last_accumulation_step = ((step + 1) % steps_gradient_accumulation == 0)

        if is_distributed and not is_last_accumulation_step:
            with model.no_sync():
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()

        # Cleaning up memory
        del x, x_hat

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
                    "Training & Validation/Loss (Training)" : loss_mean,
                    "Informations/Steps Update [-]"         : optimizer_step,
                    "Informations/Samples Seen [-]"         : (step + 1) * batch_size_per_gpu * world_size,
                    "Informations/Gradient Norm [-]"        : gradient_norm,
                })

        # Computing validation
        if optimizer_step % steps_validation == 0 and is_last_accumulation_step and optimizer_step > 0:

            # Switching to evaluation mode (dropout)
            model.eval()
            val_loss, n_val_batches = 0.0, 0

            with torch.no_grad():
                for x_val, _ in dataloader_validation:

                    # Pushing to device and concatenating mask
                    x_val = x_val.to(device)
                    x_val = torch.cat([x_val, w_mask.expand(x_val.shape[0], -1, -1, -1)], dim=1)

                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):

                        # Forward pass
                        _, x_hat_val = model(x_val)

                        # Computing loss
                        val_loss += loss_function(x_hat_val, x_val[:, :C]).item()

                    # Cleaning
                    del x_val, x_hat_val

                    # Dataloader is infinite, we need to manually break the loop after enough batches seen
                    n_val_batches += 1
                    if n_val_batches >= n_batches_validation:
                        break

            # Computing mean validation loss accross batches and processes
            val_loss_mean = val_loss / max(1, n_val_batches)
            if is_distributed:
                val_loss_mean = reduce_mean(val_loss_mean, device)

            # Logging results
            if rank == 0:
                wandb.log({"Training & Validation/Loss (Validation)": val_loss_mean})

            # Switching back to training mode
            model.train()

        # Saving checkpoint
        if optimizer_step % steps_saving == 0 and is_last_accumulation_step and optimizer_step > 0 and rank == 0:

            # Extracting raw model from DDP/DataParallel wrapper if needed
            raw_model = model.module if hasattr(model, "module") else model

            # Creating checkpoint configuration
            ckpt_config = OmegaConf.create({
                "in_channels": IN_C,
                "out_channels": OUT_C,
                **config_arch
            })

            # Saving model (overwrites previous checkpoint for this run)
            s_save(raw_model, ckpt_config, PATH_MODELS / wandb.run.name)

    # Closing run
    wandb.finish()
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch an autoencoder training pipeline.")
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

    nodes         = config_cluster.get("nodes", 1)
    gpus_per_node = config_cluster.get("gpus-per-node", 1)
    cpus_per_node = config_cluster.get("cpus-per-node", 1)
    ram_per_node  = config_cluster.get("ram-per-node", "8GB")

    # Local
    if args.backend == "async":
        ae = configs[0]["Autoencoder"]
        training(
            config_state=ae["state"],
            config_training=ae["training"],
            config_arch=ae["architecture"],
            config_wandb=config_wandb,
            config_cluster=config_cluster,
        )

    # Cluster
    else:
        if nodes > 1:
            interpreter = (
                f"torchrun --nnodes {nodes} --nproc-per-node {gpus_per_node} "
                f"--rdzv_backend=c10d --rdzv_endpoint=$SLURMD_NODENAME:12345 "
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
            time=config_cluster.get("time", "00:10:00"),
            account=config_cluster.get("account"),
            partition=config_cluster.get("partition"),
        )
        def train(i: int) -> None:
            ae = configs[i]["Autoencoder"]
            training(
                config_state=ae["state"],
                config_training=ae["training"],
                config_arch=ae["architecture"],
                config_wandb=config_wandb,
                config_cluster=config_cluster,
            )

        dawgz.schedule(
            train,
            name="AE-TRAIN",
            backend="slurm",
            interpreter=interpreter,
            export="ALL"
        )
