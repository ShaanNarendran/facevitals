"""Evaluation metrics required by the SP Cup brief."""
from __future__ import annotations

import numpy as np


def hr_metrics(est: np.ndarray, ref: np.ndarray) -> dict:
    m = np.isfinite(est) & np.isfinite(ref)
    e, r = np.asarray(est)[m], np.asarray(ref)[m]
    if len(e) < 3:
        return dict(n=int(len(e)))
    d = e - r
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return dict(
        n=int(len(e)),
        mae=float(np.abs(d).mean()),
        rmse=float(np.sqrt((d ** 2).mean())),
        r=float(np.corrcoef(e, r)[0, 1]) if e.std() > 0 and r.std() > 0 else float("nan"),
        ba_bias=bias,
        ba_loa_low=bias - 1.96 * sd,
        ba_loa_high=bias + 1.96 * sd,
        within5=float((np.abs(d) <= 5).mean()),
    )
