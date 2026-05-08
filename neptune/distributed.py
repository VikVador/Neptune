r"""Distributed training utilities."""

import os
import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int, torch.device, bool]:
    r"""Initialize distributed training and return the process configuration.

    Returns:
        rank           : Global rank of the current process.
        local_rank     : Local rank, i.e. GPU index on this node.
        world_size     : Total number of processes across all nodes.
        device         : Torch device assigned to this process.
        is_distributed : True when running under torchrun, False otherwise.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Running under torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Initialize process group (torchrun sets up env variables)
        dist.init_process_group(backend="nccl", init_method="env://")

        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Set matmul precision for performance on modern GPUs
        torch.set_float32_matmul_precision("high")
        return rank, local_rank, world_size, device, True

    else:
        # Single-GPU or CPU fallback
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return rank, local_rank, world_size, device, False
