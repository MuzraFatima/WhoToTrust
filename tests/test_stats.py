import numpy as np
import pytest

from labeltriage.eval import stats


def test_cluster_bootstrap_covers_the_truth():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(200), 5)
    x = rng.normal(1.0, 1.0, 200)[clusters] + rng.normal(0, 0.1, 1000)
    point, lo, hi = stats.cluster_bootstrap(lambda idx: x[idx].mean(), clusters, n_boot=500)
    assert lo < 1.0 < hi
    assert lo < point < hi
    # Cluster resampling must give a wider interval than pretending votes are independent.
    naive = 1.96 * x.std() / np.sqrt(len(x))
    assert (hi - lo) / 2 > 1.5 * naive


def test_paired_wilcoxon():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert stats.paired_wilcoxon(a + 1, a) < 0.01
    assert stats.paired_wilcoxon(a, a) == 1.0


def test_holm_step_down():
    rej = stats.holm({"h1": 0.001, "h2": 0.015, "h3": 0.04, "h4": 0.5})
    assert rej == {"h1": True, "h2": True, "h3": False, "h4": False}


def test_mde_shrinks_with_more_seeds():
    assert stats.minimum_detectable_effect(1.0, 5) < stats.minimum_detectable_effect(1.0, 3)
    assert stats.minimum_detectable_effect(0.5, 3) == pytest.approx(
        2.80 * 0.5 * np.sqrt(2 / 3), rel=0.01
    )
