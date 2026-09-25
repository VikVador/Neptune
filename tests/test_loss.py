r"""Tests for neptune.loss."""

import torch

from neptune.loss import loss_crps


def test_loss_crps_typical() -> None:
    r"""Determines if the loss is the fair CRPS of two members, averaged over sea points only."""

    x_s, x_o = torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 1, 2, 2)
    x_pred_s = torch.zeros(1, 2, 1, 2, 2)
    x_pred_s[:, 0] = 1.0  # Members 1 and 0 → CRPS = (1 + 0) / 2 - 1 / 2 = 0
    x_pred_s[:, :, :, 0, 0] = 4.0  # Both members 4 → CRPS = 4, but this point is land
    x_pred_o = torch.full((1, 2, 1, 1, 2, 2), 2.0)  # Both members 2 → CRPS = 2 everywhere

    mask_surface = torch.ones(1, 2, 2)
    mask_surface[:, 0, 0] = 0
    mask_ocean = torch.ones(1, 1, 2, 2)

    loss_s, loss_o = loss_crps(x_pred_s, x_pred_o, x_s, x_o, mask_surface, mask_ocean)
    assert torch.isclose(loss_s, torch.tensor(0.0))
    assert torch.isclose(loss_o, torch.tensor(2.0))

    _, loss_o = loss_crps(
        x_pred_s,
        x_pred_o,
        x_s,
        x_o,
        mask_surface,
        mask_ocean,
        weights_surface=torch.ones(1),
        weights_ocean=torch.full((1, 1), 0.5),
    )
    assert torch.isclose(loss_o, torch.tensor(1.0))


def test_loss_crps_gradient() -> None:
    r"""Determines if gradients are finite and zero on land, despite the NaN-masked land points."""

    x_pred_s = torch.randn(2, 2, 3, 4, 4, requires_grad=True)
    x_pred_o = torch.randn(2, 2, 3, 2, 4, 4, requires_grad=True)
    mask_surface = (torch.rand(1, 4, 4) > 0.3).float()
    mask_ocean = (torch.rand(1, 2, 4, 4) > 0.3).float()

    loss_s, loss_o = loss_crps(
        x_pred_s,
        x_pred_o,
        torch.randn(2, 3, 4, 4),
        torch.randn(2, 3, 2, 4, 4),
        mask_surface,
        mask_ocean,
    )
    (loss_s + loss_o).backward()

    assert torch.isfinite(x_pred_s.grad).all()
    assert torch.isfinite(x_pred_o.grad).all()
    assert (x_pred_s.grad[:, :, :, mask_surface[0] == 0] == 0).all()
