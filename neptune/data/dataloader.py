r"""Dataloaders."""

__all__ = [
    "get_dataloaders",
]

from collections.abc import Callable, Iterator, Sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Any

from neptune.data.dataset import get_datasets


def infinite_dataloader(
    dataloader: DataLoader,
    batches: int,
    is_distributed: bool = False,
) -> Any:
    r"""Makes a basic PyTorch dataloader 'infinite'.

    Arguments:
        dataloader     : A PyTorch dataloader to iterate over.
        batches        : Maximum number of batches to yield before stopping.
        is_distributed : Whether running in distributed mode (updates sampler epoch for proper shuffling).
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


def get_dataloaders(
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    get_datasets_fn: Callable = get_datasets,
    shuffle: tuple[bool, bool, bool] = (True, False, False),
    infinite: Sequence[bool] | None = None,
    batches: Sequence[int] | None = None,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
    **kwargs: Any,
) -> tuple[DataLoader | Iterator, DataLoader | Iterator, DataLoader | Iterator]:
    r"""Build and return train, validation, and test dataloaders.

    Arguments:
        batch_size      : Number of samples per batch (per GPU in distributed mode).
        num_workers     : Number of worker processes for data loading.
        prefetch_factor : Number of batches prefetched per worker.
        get_datasets_fn : Function returning (train, val, test) datasets.
        shuffle         : Whether to shuffle each split (train, val, test).
        infinite        : Whether to cycle each dataloader indefinitely.
        batches         : Maximum number of batches to yield before stopping.
        rank            : Global rank of the current process (distributed mode).
        world_size      : Total number of processes (distributed mode).
        is_distributed  : Whether to use DistributedSampler for each dataloader.
        kwargs          : Forwarded to get_datasets_fn().

    Returns:
        train_loader : Training dataloader, or an infinite iterator.
        val_loader   : Validation dataloader, or an infinite iterator.
        test_loader  : Test dataloader, or an infinite iterator.
    """
    if infinite is None:
        infinite = [False, False, False]
    if batches is None:
        batches = [None, None, None]

    if len(infinite) != 3:
        raise ValueError(f"ERROR - infinite must have exactly 3 elements, got {len(infinite)}.")
    if len(batches) != 3:
        raise ValueError(f"ERROR - batches must have exactly 3 elements, got {len(batches)}.")

    for inf, bst in zip(infinite, batches, strict=True):
        assert not (inf and bst is None), "ERROR - batches[i] must be set when infinite[i]=True."

    datasets = get_datasets_fn(**kwargs)

    dataloaders = []
    for i, dataset in enumerate(datasets):
        if is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle[i],
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
        )
        dataloaders.append(dataloader)

    dataloaders = [
        infinite_dataloader(dl, bst, is_distributed) if inf else dl
        for inf, bst, dl in zip(infinite, batches, dataloaders, strict=True)
    ]

    return tuple(dataloaders)
