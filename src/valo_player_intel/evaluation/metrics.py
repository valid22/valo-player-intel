from __future__ import annotations

import numpy as np
import pandas as pd


def expected_calibration_error(actual: pd.Series, probability: pd.Series, bins: int = 10) -> float:
    frame = pd.DataFrame({"actual": actual.astype(float), "probability": probability.astype(float)})
    frame["bin"] = pd.cut(frame["probability"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    grouped = frame.groupby("bin", observed=False)
    ece = 0.0
    n = len(frame)
    for _, group in grouped:
        if group.empty:
            continue
        ece += abs(group["actual"].mean() - group["probability"].mean()) * (len(group) / n)
    return float(ece)


def reliability_curve(actual: pd.Series, probability: pd.Series, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": actual.astype(float), "probability": probability.astype(float)})
    frame["bin_index"] = pd.cut(frame["probability"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True, labels=False)
    curve = (
        frame.groupby("bin_index", observed=False)
        .agg(mean_predicted=("probability", "mean"), empirical_win_rate=("actual", "mean"), count=("actual", "size"))
        .reset_index()
    )
    return curve
