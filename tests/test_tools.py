r"""Tests for neptune.tools."""

import pytest

import neptune.tools

from neptune.tools import extract_model_hash, generate_run_name, schedule_training

# Slurm resources of a training on one GPU node
CLUSTER = {
    "nodes": 1,
    "gpus-per-node": 4,
    "cpus-per-node": 32,
    "ram-per-node": "240GB",
    "time": "48:00:00",
    "account": "bsmfc",
    "partition": "gpu",
}


def test_generate_run_name_typical() -> None:
    r"""Determines if a run name holds its own hash, and the hash of the run it resumes."""

    first = generate_run_name("ENC_surface_lc128_tk576")
    second = generate_run_name("ENC_surface_lc128_tk576", previous_run_name=first)

    assert first.startswith("ENC_surface_lc128_tk576__")
    assert len(extract_model_hash(first)) == 4
    assert second.endswith(f"_{extract_model_hash(first)}")


def test_schedule_training_typical(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if an array of trainings is scheduled on a single node, with one process per GPU."""

    scheduled = {}
    monkeypatch.setattr(neptune.tools, "freeze_modules", lambda modules: None)
    monkeypatch.setattr(
        neptune.tools, "schedule", lambda job, **kwargs: scheduled.update(job=job, **kwargs)
    )

    def train(i: int) -> None:
        r"""Train the i-th configuration."""

    schedule_training(train, "NEPT-TRAIN", CLUSTER, modules=[], array=3)

    assert scheduled["name"] == "NEPT-TRAIN"
    assert scheduled["backend"] == "slurm"
    assert scheduled["interpreter"] == "torchrun --nnodes 1 --nproc-per-node 4 --standalone"
    assert scheduled["job"].name == "train"
    assert sorted(scheduled["job"].array) == [0, 1, 2]
    assert scheduled["job"].settings["account"] == "bsmfc"
    assert scheduled["job"].settings["partition"] == "gpu"


def test_schedule_training_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if a single training on several nodes meets at a rendezvous, without job array."""

    scheduled = {}
    monkeypatch.setattr(neptune.tools, "freeze_modules", lambda modules: None)
    monkeypatch.setattr(
        neptune.tools, "schedule", lambda job, **kwargs: scheduled.update(job=job, **kwargs)
    )

    def train() -> None:
        r"""Train the configuration."""

    schedule_training(train, "NEPT-ENC-OCEAN", {**CLUSTER, "nodes": 2}, modules=[])

    assert scheduled["interpreter"].startswith("torchrun --nnodes 2 --nproc-per-node 4 ")
    assert "--rdzv_backend=c10d" in scheduled["interpreter"]
    assert scheduled["job"].array is None
