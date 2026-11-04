r"""Tests for neptune.training and the fixed validation of neptune.data.dataloader."""

import pytest
import torch

from pathlib import Path
from shaggy.optimizers.soap import SOAP

import neptune.training

from neptune.data.dataloader import get_fixed_indices
from neptune.schedulers import warmup_cosine_decay
from neptune.tools import generate_run_name, load_configuration
from neptune.training import has_stagnated, load_state, save_state, submit_resume


def test_has_stagnated_flat() -> None:
    r"""Determines if it stops on a flat validation."""

    history = [1.0, 0.5, 0.25] + [0.1, 0.1001, 0.0999, 0.1] * 3

    assert has_stagnated(history, window=10, threshold=0.01)


def test_has_stagnated_falling() -> None:
    r"""Determines if it continues while the validation falls."""

    history = [0.9**i for i in range(30)]

    assert not has_stagnated(history, window=10, threshold=0.01)


def test_has_stagnated_rising() -> None:
    r"""Determines if it stops on a rising validation, whose spread stays large."""

    history = [0.1 * 1.1**i for i in range(20)]

    assert has_stagnated(history, window=10, threshold=0.01)


def test_has_stagnated_too_short() -> None:
    r"""Determines if it waits for a full window, without a previous best to compare with."""

    assert not has_stagnated([1.0, 2.0, 3.0], window=10, threshold=0.01)
    assert not has_stagnated([1.0 + i for i in range(10)], window=10, threshold=0.01)


def test_get_fixed_indices_typical() -> None:
    r"""Determines if the days are always the same, spread through time and mixed."""

    indices = get_fixed_indices(1096, 512)

    assert indices == get_fixed_indices(1096, 512)
    assert len(set(indices)) == 512
    assert min(indices) == 0 and max(indices) == 1095
    assert indices != sorted(indices)
    assert max(b - a for a, b in zip(sorted(indices), sorted(indices)[1:], strict=False)) <= 3


def test_get_fixed_indices_all() -> None:
    r"""Determines if asking for every day keeps each of them once."""

    assert sorted(get_fixed_indices(16, 16)) == list(range(16))


def test_generate_run_name_same_id() -> None:
    r"""Determines if the jobs of a same training share their name, unlike two trainings."""

    assert generate_run_name("ENC", run_id="1") == generate_run_name("ENC", run_id="1")
    assert generate_run_name("ENC", run_id="1") != generate_run_name("ENC", run_id="2")


def test_state_resumes_exactly(tmp_path: Path) -> None:
    r"""Determines if a training resumed from a saved state follows the same path as if never stopped."""

    def build() -> tuple:
        torch.manual_seed(0)
        model = torch.nn.Linear(8, 4)
        optimizer = SOAP(model.parameters(), lr=1e-2, max_precond_size=1024)
        scheduler = warmup_cosine_decay(optimizer, 1e-3, 1e-2, 1e-4, 2, 40)
        return model, optimizer, scheduler

    def step(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        i: int,
    ) -> None:
        torch.manual_seed(i)
        model(torch.randn(16, 8)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    model, optimizer, scheduler = build()
    for i in range(20):
        step(model, optimizer, scheduler, i)

    path = tmp_path / "last" / "state.pt"
    save_state(path, {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()})
    weights = {k: v.clone() for k, v in model.state_dict().items()}

    for i in range(20, 30):
        step(model, optimizer, scheduler, i)

    resumed, optimizer_resumed, scheduler_resumed = build()
    state = load_state(path, torch.device("cpu"))
    resumed.load_state_dict(weights)
    optimizer_resumed.load_state_dict(state["optimizer"])
    scheduler_resumed.load_state_dict(state["scheduler"])

    for i in range(20, 30):
        step(resumed, optimizer_resumed, scheduler_resumed, i)

    assert path.exists() and not path.with_suffix(".tmp").exists()
    assert scheduler_resumed.get_last_lr() == scheduler.get_last_lr()
    for key, value in model.state_dict().items():
        assert torch.allclose(value, resumed.state_dict()[key], atol=1e-5)


def test_submit_resume_typical(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if a job submits its own script after itself, without the variables of the job and of torchrun."""

    calls = []

    def run(command: list[str], **kwargs) -> object:
        calls.append((command, kwargs.get("env")))
        return type("Result", (), {"stdout": "JobId=42 Command=/a/0000_train.sh WorkDir=/a"})()

    for key, value in {
        "SLURM_JOB_ID": "42",
        "RANK": "0",
        "MASTER_PORT": "1",
        "PATH": "/bin",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("NEPTUNE_RUN_ID", raising=False)
    monkeypatch.delenv("NEPTUNE_RESTART", raising=False)
    monkeypatch.setattr(neptune.training.subprocess, "run", run)

    submit_resume(restarts=3)
    command, env = calls[-1]

    assert command[0] == "sbatch" and command[-1] == "/a/0000_train.sh"
    assert "--dependency=afternotok:42" in command
    assert (
        env["NEPTUNE_RUN_ID"] == "42" and env["NEPTUNE_RESTART"] == "1" and env["PATH"] == "/bin"
    )
    assert (
        not any(k.startswith("SLURM_") for k in env)
        and "RANK" not in env
        and "MASTER_PORT" not in env
    )


def test_submit_resume_last(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if the last allowed job and a run outside Slurm submit nothing."""

    calls = []
    monkeypatch.setattr(
        neptune.training.subprocess, "run", lambda *args, **kwargs: calls.append(args)
    )
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("NEPTUNE_RESTART", "3")

    submit_resume(restarts=3)
    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.setenv("NEPTUNE_RESTART", "0")
    submit_resume(restarts=3)

    assert calls == []


def test_load_configuration_encoders() -> None:
    r"""Determines if both encoders have a complete configuration, with whole validation batches."""

    folder = (
        Path(__file__).parents[1]
        / "experiments"
        / "forecast-latent-states"
        / "configs"
        / "encoder"
    )

    for group in ("ocean", "surface"):
        config = load_configuration(folder / f"encoder-{group}.yml")[0]
        training, cluster = config["Training"], config["Cluster"]
        samples_per_step = (
            training["batch_size_per_gpu"] * cluster["nodes"] * cluster["gpus-per-node"]
        )

        assert "batches_validation" not in training
        assert config["State"] == {
            "saving": True,
            "checkpoint_name": None,
            "resume_optimizer": True,
        }
        assert training["samples_validation"] % samples_per_step == 0
        assert cluster["restarts"] >= 0
