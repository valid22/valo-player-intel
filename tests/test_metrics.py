import pandas as pd

from valo_player_intel.evaluation.metrics import expected_calibration_error, reliability_curve


def test_reliability_metrics_are_bounded():
    actual = pd.Series([0, 0, 1, 1])
    probability = pd.Series([0.1, 0.2, 0.8, 0.9])

    curve = reliability_curve(actual, probability, bins=2)
    ece = expected_calibration_error(actual, probability, bins=2)

    assert not curve.empty
    assert 0.0 <= ece <= 1.0
