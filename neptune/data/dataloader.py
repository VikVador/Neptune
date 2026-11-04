r"""Dataloaders."""

__all__ = [
    "get_dataloaders",
    "get_fixed_indices",
]

import ctypes
import numpy as np

from collections.abc import Iterator, Sequence
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from typing import Any

from neptune.data.dataset import get_datasets


def _init_worker(worker_id: int) -> None:
    r"""Serve the large arrays of a worker with mmap, such that they are returned to the system once freed.

    Arguments:
        worker_id : Index of the worker.
    """
    ctypes.CDLL("libc.so.6").mallopt(-3, 2**16)


def infinite_dataloader(
    dataloader: DataLoader,
    batches: int,
    is_distributed: bool = False,
) -> Iterator:
    r"""Make a basic PyTorch dataloader 'infinite' by cycling over it.

    Arguments:
        dataloader     : Dataloader to iterate over.
        batches        : Maximum number of batches to yield before stopping.
        is_distributed : Whether running in distributed mode (updates the sampler epoch each cycle).

    Returns:
        batch : Successive batches, cycling past the end of the dataloader as many times as needed.
    """

    epoch = 0
    batches_remaining = batches

    while batches_remaining > 0:
        if is_distributed and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        for batch in dataloader:
            yield batch
            batches_remaining -= 1
            if batches_remaining <= 0:
                return

        epoch += 1


def get_fixed_indices(length: int, samples: int, seed: int = 0) -> list[int]:
    r"""Choose samples spread evenly through time, in an order mixing the seasons within each batch.

    Arguments:
        length  : Number of samples of the dataset.
        samples : Number of samples to keep.
        seed    : Seed of the order of the samples.

    Returns:
        indices : Indices of the samples, always the same for a given length, number of samples and seed.
    """

    # Security
    assert 0 < samples <= length, f"ERROR - samples must be in [1, {length}], got {samples}."

    indices = np.linspace(0, length - 1, samples).round().astype(int)
    return np.random.default_rng(seed).permutation(indices).tolist()


def get_dataloaders(
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: tuple[bool, bool, bool] = (True, False, False),
    infinite: Sequence[bool] | None = None,
    batches: Sequence[int] | None = None,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
    samples_validation: int | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> tuple[DataLoader | Iterator, DataLoader | Iterator, DataLoader | Iterator]:
    r"""Build and return train, validation, and test dataloaders.

    Arguments:
        batch_size      : Number of samples per batch (per GPU in distributed mode).
        num_workers     : Number of worker processes for data loading.
        prefetch_factor : Number of batches prefetched per worker.
        shuffle         : Whether to shuffle each split (train, val, test).
        infinite        : Whether to cycle each dataloader indefinitely.
        batches         : Maximum number of batches to yield before stopping.
        rank            : Global rank of the current process (distributed mode).
        world_size      : Total number of processes (distributed mode).
        is_distributed  : Whether to use DistributedSampler for each dataloader.
        samples_validation : Number of samples of a fixed validation over all the GPUs (every sample if None).
        seed            : Seed of the distributed shuffling, changed when a training resumes to draw other samples.
        kwargs          : Forwarded to get_datasets().

    Returns:
        train_loader : Training dataloader, or an infinite iterator.
        val_loader   : Validation dataloader, or an infinite iterator.
        test_loader  : Test dataloader, or an infinite iterator.
    """

    if infinite is None:
        infinite = [False, False, False]
    if batches is None:
        batches = [None, None, None]

    # Security
    assert len(infinite) == 3, f"ERROR - infinite must have 3 elements, got {len(infinite)}."
    assert len(batches) == 3, f"ERROR - batches must have 3 elements, got {len(batches)}."
    for inf, bst in zip(infinite, batches, strict=True):
        assert not (inf and bst is None), "ERROR - batches[i] must be set when infinite[i]=True."

    datasets = list(get_datasets(**kwargs))

    # Comparing two models on the same pairs of days halves the noise of the comparison
    if samples_validation is not None:
        samples = min(samples_validation, len(datasets[1]))
        datasets[1] = Subset(datasets[1], get_fixed_indices(len(datasets[1]), samples))

    dataloaders = []
    for i, dataset in enumerate(datasets):
        if is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle[i],
                seed=seed,
                drop_last=True,
            )
            dataloader_shuffle = False
        else:
            sampler = None
            dataloader_shuffle = shuffle[i]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=dataloader_shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=_init_worker,
            drop_last=True,
        )
        dataloaders.append(dataloader)

    dataloaders = [
        infinite_dataloader(dl, bst, is_distributed) if inf else dl
        for inf, bst, dl in zip(infinite, batches, dataloaders, strict=True)
    ]

    return tuple(dataloaders)
