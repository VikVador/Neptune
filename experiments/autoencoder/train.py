r"""Launch training of autoencoder."""

import argparse
import cloudpickle
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
from neptune.data import C_IN, C_OUT, C, Z
from neptune.data.dataloader import get_dataloaders
from neptune.data.weights import get_weights_loss, get_weights_mask
from neptune.distributed import reduce_mean, setup_distributed
from neptune.schedulers import warmup_cosine_decay
from neptune.tools import generate_run_name_ae, get_wandb_hyperparameters, load_configuration


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
    run_name = generate_run_name_ae(
        in_channels       = C,
        lat_channels      = config_arch["lat_channels"],
        hid_channels      = config_arch["hid_channels"],
        hid_blocks        = config_arch["hid_blocks"],
        stride            = config_arch["stride"],
        previous_run_name = config_state["checkpoint_name"],
    )

    if rank == 0:
        wandb.init(
            **config_wandb,
            name=run_name,
            config={
                "State"           : config_state,
                "Training"        : config_training,
                "Architecture"    : config_arch,
                "Cluster"         : config_cluster,
                "Hyperparameters" : get_wandb_hyperparameters([config_training, config_arch]),
            },
        )
    else:
        wandb.init(mode="disabled")

    (
        saving,
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

    # Initializing weighting tensors
    w_mask, w_loss = (
        get_weights_mask(dim=2,                                          device=device),
        get_weights_loss(dim=2, depths=(min(47, Z - 1), min(37, Z - 1)), device=device),
    )

    # Model | Loading checkpoint or new
    if config_state["checkpoint_name"] is not None:
        ckpt_path = PATH_MODELS / config_state["checkpoint_name"]
        model     = s_load(ckpt_path, device=str(device)).train()
    else:
        model = create_ConvAE(in_channels  = C_IN, out_channels = C_OUT, **config_arch).to(device)

    # Model | Defining if DDP or DataParallel
    if is_distributed:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        model      = DDP(model, **ddp_kwargs)
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)

    # Logging number of trainable parameters
    if rank == 0:
        wandb.log({"Informations/Trainable Parameters [M]": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,})

    # Setting up training tools
    optimizer = SOAP(model.parameters(), lr=lr_peak, max_precond_size=128)
    scheduler = warmup_cosine_decay(
        optimizer    = optimizer,
        lr_start     = lr_start,
        lr_peak      = lr_peak,
        lr_end       = lr_end,
        warmup_steps = warmup_steps,
        total_steps  = steps_update,
    )

    scaler                   = GradScaler(enabled=False)
    loss_function            = AELoss(weights=w_loss)
    loss_accumulator         = 0.0
    loss_logging_accumulator = 0.0
    loss_mean                = float("inf")
    loss_best                = float("inf")
    gradient_norm            = float("inf")
    optimizer_step           = 0

    # Waiting for processes to be ready
    if is_distributed:
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)

    for step, (x, _) in enumerate(dataloader_training):

        # Pushing to device and concatenating mask
        x = x.to(device)
        x = torch.cat([x, w_mask.expand(x.shape[0], -1, -1, -1)], dim=1)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):

            # Forward pass
            _, x_hat = model(x)

            # Computing loss
            loss = loss_function(x_hat, x[:, :C_OUT])

        # Gradient accumulation
        loss              = loss / steps_gradient_accumulation
        loss_accumulator += loss.item()

        # Logging to console if not using WandB
        if config_wandb["mode"] == "disabled":
            print(f"Step {optimizer_step:6d} | Loss: {loss_accumulator:.4f} | γ: {scheduler.get_last_lr()[0]:.6f} | ∇: {gradient_norm:.4f}")

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
                    "Training/Loss"              : loss_mean,
                    "Informations/Steps Update"  : optimizer_step,
                    "Informations/Samples Seen"  : (step + 1) * batch_size_per_gpu * world_size,
                    "Informations/Gradient Norm" : gradient_norm,
                    "Informations/Learning Rate" : scheduler.get_last_lr()[0],
                })

        # Saving checkpoint
        if saving and optimizer_step % steps_saving == 0 and is_last_accumulation_step and optimizer_step > 0 and rank == 0:
            if loss_mean < loss_best:

                # Extracting raw model and creating checkpoint configuration
                raw_model   = model.module if hasattr(model, "module") else model
                ckpt_config = OmegaConf.create({"in_channels": C_IN, "out_channels": C_OUT, **config_arch})

                # Saving model (overwrites previous checkpoint for this run)
                s_save(raw_model, ckpt_config, PATH_MODELS / wandb.run.name)

                # Updating best loss
                loss_best = loss_mean

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

    nodes         = config_cluster["nodes"]
    gpus_per_node = config_cluster["gpus-per-node"]
    cpus_per_node = config_cluster["cpus-per-node"]
    ram_per_node  = config_cluster["ram-per-node"]

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
