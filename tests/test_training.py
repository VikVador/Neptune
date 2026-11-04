r"""Tests for the resubmission of neptune.training, the run names and the configurations of the encoders."""

import pytest

from pathlib import Path

import neptune.training

from neptune.tools import generate_run_name, load_configuration
from neptune.training import submit_resume


def test_generate_run_name_same_id() -> None:
    r"""Determines if the jobs of a same training share their name, unlike two trainings."""

    assert generate_run_name("ENC", run_id="1") == generate_run_name("ENC", run_id="1")
    assert generate_run_name("ENC", run_id="1") != generate_run_name("ENC", run_id="2")


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
