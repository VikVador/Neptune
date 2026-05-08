r"""Tests for neptune.data.dataloader."""

import pytest
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from neptune.data.dataloader import get_dataloaders, infinite_dataloader


def _make_tiny_dataset(n: int = 8) -> TensorDataset:
    return TensorDataset(torch.randn(n, 4, 8, 8), torch.zeros(n))


def _fake_get_datasets(**kwargs) -> tuple:
    return _make_tiny_dataset(), _make_tiny_dataset(), _make_tiny_dataset()


def test_get_dataloaders_returns_three_dataloaders(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if get_dataloaders returns a tuple of three DataLoaders."""
    monkeypatch.setattr("neptune.data.dataloader.get_datasets", _fake_get_datasets)
    result = get_dataloaders(batch_size=2, num_workers=0, prefetch_factor=1)
    assert isinstance(result, tuple)
    assert len(result) == 3
    for loader in result:
        assert isinstance(loader, DataLoader)


def test_get_dataloaders_batch_size_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if the batch_size argument is propagated to each DataLoader."""
    monkeypatch.setattr("neptune.data.dataloader.get_datasets", _fake_get_datasets)
    train, val, test = get_dataloaders(batch_size=3, num_workers=0, prefetch_factor=1)
    assert train.batch_size == 3
    assert val.batch_size == 3
    assert test.batch_size == 3


def test_get_dataloaders_infinite_requires_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if infinite=True without a batch count raises an AssertionError."""
    monkeypatch.setattr("neptune.data.dataloader.get_datasets", _fake_get_datasets)
    with pytest.raises(AssertionError):
        get_dataloaders(
            batch_size=2,
            num_workers=0,
            prefetch_factor=1,
            infinite=[True, False, False],
            batches=[None, None, None],
        )


def test_get_dataloaders_distributed_uses_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if is_distributed=True attaches a DistributedSampler to each DataLoader."""
    monkeypatch.setattr("neptune.data.dataloader.get_datasets", _fake_get_datasets)
    train, val, test = get_dataloaders(
        batch_size=2,
        num_workers=0,
        prefetch_factor=1,
        rank=0,
        world_size=2,
        is_distributed=True,
    )
    for loader in (train, val, test):
        assert isinstance(loader.sampler, DistributedSampler)


def test_get_dataloaders_kwargs_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if extra kwargs are forwarded to get_datasets."""
    received: dict = {}

    def _capturing_get_datasets(**kwargs) -> tuple:
        received.update(kwargs)
        return _make_tiny_dataset(), _make_tiny_dataset(), _make_tiny_dataset()

    monkeypatch.setattr("neptune.data.dataloader.get_datasets", _capturing_get_datasets)
    get_dataloaders(batch_size=2, num_workers=0, prefetch_factor=1, standardized=False)
    assert received.get("standardized") is False


def test_infinite_dataloader_yields_correct_count() -> None:
    r"""Determines if infinite_dataloader yields exactly the requested number of batches."""
    loader = DataLoader(TensorDataset(torch.arange(16).float()), batch_size=4)
    assert len(list(infinite_dataloader(loader, batches=10))) == 10


def test_infinite_dataloader_cycles_across_epochs() -> None:
    r"""Determines if infinite_dataloader cycles past the end of the dataset."""
    loader = DataLoader(TensorDataset(torch.arange(4).float()), batch_size=2)
    assert len(list(infinite_dataloader(loader, batches=7))) == 7
