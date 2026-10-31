import numpy as np
import pytest

from labeltriage.eval import metrics as M


def test_accuracy_and_nll():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    labels = np.array([0, 1, 1])
    assert M.accuracy(probs, labels) == pytest.approx(2 / 3)
    assert M.nll(probs, labels) == pytest.approx(-np.log([0.9, 0.8, 0.4]).mean())


def test_ece_hand_computed():
    # Two bins used: confidence 0.9 (1 of 2 correct) and 0.6 (both correct).
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1, 0, 1])
    expected = 0.5 * abs(0.5 - 0.9) + 0.5 * abs(1.0 - 0.6)
    assert M.ece(probs, labels) == pytest.approx(expected)


def test_ece_zero_when_perfectly_calibrated():
    probs = np.tile([1.0, 0.0], (10, 1))
    assert M.ece(probs, np.zeros(10, dtype=int)) == pytest.approx(0.0)


def test_divergences_vanish_on_identical_distributions():
    p = np.array([[0.5, 0.3, 0.2], [0.0, 1.0, 0.0]])
    for fn in (M.kl, M.js, M.manhattan, M.wasserstein):
        assert fn(p, p) == pytest.approx(0.0, abs=1e-9)


def test_divergence_values():
    h = np.array([[1.0, 0.0, 0.0]])
    q = np.array([[0.0, 0.0, 1.0]])
    assert M.manhattan(h, q) == pytest.approx(2.0)
    assert M.wasserstein(h, q) == pytest.approx(2.0)
    assert M.js(h, q) == pytest.approx(np.log(2))
    assert M.kl(np.array([[0.5, 0.5]]), np.array([[0.25, 0.75]])) == pytest.approx(
        0.5 * np.log(2) + 0.5 * np.log(0.5 / 0.75)
    )


def test_auroc_and_auprc():
    y = np.array([0, 0, 1, 1])
    assert M.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert M.auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert np.isnan(M.auroc(np.ones(3), np.zeros(3)))
    assert M.auprc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_report_keys():
    probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    rep = M.classification_report(probs, np.array([0, 1]), probs, prefix="test/")
    assert set(rep) == {
        f"test/{k}" for k in ("acc", "ece", "nll", "kl", "js", "manhattan", "wasserstein")
    }
