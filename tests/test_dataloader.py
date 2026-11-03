r"""Tests for neptune.data.dataloader."""

import pytest
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from neptune.data.dataloader import get_dataloaders


def _fake_get_datasets(**kwargs) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    r"""Fake training, validation and test datasets of 8 samples each."""
    return tuple(TensorDataset(torch.randn(8, 4, 8, 8)) for _ in range(3))


def test_get_dataloaders_typical(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if three dataloaders are built with the batch size and the dataset kwargs."""

    received = {}

    def _capturing_get_datasets(**kwargs) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
        r"""Fake datasets, recording the kwargs they are built with."""

        received.update(kwargs)

        return _fake_get_datasets()

    monkeypatch.setattr("neptune.data.dataloader.get_datasets", _capturing_get_datasets)
    loaders = get_dataloaders(batch_size=3, num_workers=0, prefetch_factor=1, input_states=2)

    assert len(loaders) == 3
    assert all(isinstance(loader, DataLoader) and loader.batch_size == 3 for loader in loaders)
    assert received == {"input_states": 2}


def test_get_dataloaders_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if distributed dataloaders use a DistributedSampler."""

    monkeypatch.setattr("neptune.data.dataloader.get_datasets", _fake_get_datasets)
    loaders = get_dataloaders(
        batch_size=2,
        num_workers=0,
        prefetch_factor=1,
        rank=0,
        world_size=2,
        is_distributed=True,
    )

    assert all(isinstance(loader.sampler, DistributedSampler) for loader in loaders)


def test_get_dataloaders_fixed_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if the validation keeps the same samples, in the same order, whatever the seed of the training."""

    datasets = tuple(TensorDataset(torch.arange(n).float()) for n in (8, 8, 8))
    monkeypatch.setattr("neptune.data.dataloader.get_datasets", lambda **kwargs: datasets)

    first = get_dataloaders(
        batch_size=2, num_workers=0, prefetch_factor=1, samples_validation=4, seed=0
    )[1]
    second = get_dataloaders(
        batch_size=2, num_workers=0, prefetch_factor=1, samples_validation=4, seed=5
    )[1]

    assert len(first.dataset) == 4
    assert [b[0].tolist() for b in first] == [b[0].tolist() for b in second]
