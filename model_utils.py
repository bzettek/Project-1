"""Shared helpers for model training/inference."""

import numpy as np

def augment_features(X):
    """Return base features plus engineered terms for tree models.

    Parameters
    ----------
    X : array-like of shape (n_samples, 5)
        Columns ordered as [capacity, fill, wins, losses, prcp].
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError("augment_features expects shape (n_samples, 5)")

    cap = np.clip(arr[:, 0], 0.0, None)
    fill = np.clip(arr[:, 1], 0.0, 1.0)
    wins = np.clip(arr[:, 2], 0.0, None)
    losses = np.clip(arr[:, 3], 0.0, None)
    prcp = np.clip(arr[:, 4], 0.0, None)

    total_games = wins + losses
    with np.errstate(divide="ignore", invalid="ignore"):
        win_pct = np.divide(wins, total_games, out=np.zeros_like(wins), where=total_games > 0)

    expected = cap * fill
    net_record = wins - losses
    slack = cap - expected
    rain_pressure = prcp * cap
    log_cap = np.log1p(cap)

    return np.column_stack([
        cap,
        fill,
        wins,
        losses,
        prcp,
        win_pct,
        expected,
        net_record,
        slack,
        rain_pressure,
        log_cap,
    ])
