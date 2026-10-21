import numpy as np
import pytest

from training.visibility.calibrate import auc, fit_logreg


def test_auc_basic():
    y = np.array([0, 0, 1, 1])
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5


def test_logreg_recovers_direction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(2 * X[:, 0] - 1)))).astype(float)
    w, b = fit_logreg(X, y, l2=0.0)
    assert w[0] == pytest.approx(2, abs=0.3) and abs(w[1]) < 0.3 and b == pytest.approx(-1, abs=0.3)
