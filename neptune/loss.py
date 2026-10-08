r"""Training losses."""

__all__ = [
    "loss_crps",
    "loss_crps_rollout",
]

from albus.metrics import continuous_ranked_probability_score
from torch import Tensor


def loss_crps(
    x_pred_s: Tensor,
    x_pred_o: Tensor,
    x_s: Tensor,
    x_o: Tensor,
    mask_surface: Tensor,
    mask_ocean: Tensor,
    weights_surface: Tensor | None = None,
    weights_ocean: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Compute the (optionally weighted) fair CRPS of surface and ocean ensemble forecasts.

    References:
        | Fair scores for ensemble forecasts (Ferro, 2014)
        | https://doi.org/10.1002/qj.2270

    Arguments:
        x_pred_s        : Forecasted surface states (B, E, C_s, Y, X).
        x_pred_o        : Forecasted ocean states (B, E, C_o, Z, Y, X).
        x_s             : Target surface states (B, C_s, Y, X).
        x_o             : Target ocean states (B, C_o, Z, Y, X).
        mask_surface    : Land/sea mask of the surface (1, Y, X).
        mask_ocean      : Land/sea mask of the ocean (1, Z, Y, X).
        weights_surface : Weight of each surface variable (C_s,), or None to average them.
        weights_ocean   : Weight of each ocean variable-level (C_o, Z), or None to average them.

    Returns:
        loss_surface : CRPS of the surface variables.
        loss_ocean   : CRPS of the ocean variables.
    """

    crps_s = continuous_ranked_probability_score(
        x_s,
        x_pred_s,
        dims="B E C Y X",
        ensemble="E",
        reduce="Y X",
        mask=mask_surface,
    )

    crps_o = continuous_ranked_probability_score(
        x_o,
        x_pred_o,
        dims="B E C Z Y X",
        ensemble="E",
        reduce="Y X",
        mask=mask_ocean,
    )

    if weights_surface is None:
        loss_surface = crps_s.mean()
    else:
        loss_surface = (crps_s * weights_surface).sum(dim=-1).mean()

    if weights_ocean is None:
        loss_ocean = crps_o.mean()
    else:
        loss_ocean = (crps_o * weights_ocean).sum(dim=(-2, -1)).mean()

    return loss_surface, loss_ocean


def loss_crps_rollout(
    x_pred_s: Tensor,
    x_pred_o: Tensor,
    x_s: Tensor,
    x_o: Tensor,
    mask_surface: Tensor,
    mask_ocean: Tensor,
    weights_rollout: Tensor,
    weights_surface: Tensor | None = None,
    weights_ocean: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Compute the fair CRPS of surface and ocean ensemble rollouts, weighted over their days.

    Arguments:
        x_pred_s        : Forecasted surface states (B, E, K, C_s, Y, X).
        x_pred_o        : Forecasted ocean states (B, E, K, C_o, Z, Y, X).
        x_s             : Target surface states (B, K, C_s, Y, X).
        x_o             : Target ocean states (B, K, C_o, Z, Y, X).
        mask_surface    : Land/sea mask of the surface (1, Y, X).
        mask_ocean      : Land/sea mask of the ocean (1, Z, Y, X).
        weights_rollout : Weight of each day of the rollout (K,), normalized to sum to 1.
        weights_surface : Weight of each surface variable (C_s,), or None to average them.
        weights_ocean   : Weight of each ocean variable-level (C_o, Z), or None to average them.

    Returns:
        loss_surface : CRPS of the surface variables, weighted over the days of the rollout.
        loss_ocean   : CRPS of the ocean variables, weighted over the days of the rollout.
    """

    weights_rollout = weights_rollout / weights_rollout.sum()
    loss_surface, loss_ocean = 0.0, 0.0

    for k, weight in enumerate(weights_rollout):
        crps_s, crps_o = loss_crps(
            x_pred_s[:, :, k],
            x_pred_o[:, :, k],
            x_s[:, k],
            x_o[:, k],
            mask_surface,
            mask_ocean,
            weights_surface,
            weights_ocean,
        )

        loss_surface += weight * crps_s
        loss_ocean += weight * crps_o

    return loss_surface, loss_ocean
