import numpy as np

from labeltriage import crossfit


def test_folds_are_stable_and_balanced():
    ids = [f"i{k}" for k in range(5000)]
    a = crossfit.fold_ids(ids, 5, seed=1)
    assert (a == crossfit.fold_ids(ids, 5, seed=1)).all()
    assert not (a == crossfit.fold_ids(ids, 5, seed=2)).all()
    counts = np.bincount(a, minlength=5)
    assert counts.min() > 900


def test_fold_of_an_item_does_not_depend_on_its_neighbours():
    ids = [f"i{k}" for k in range(100)]
    assert crossfit.fold_ids(ids[50:], 5)[0] == crossfit.fold_ids(ids, 5)[50]


def test_cross_predict_never_scores_an_item_with_itself():
    ids = np.array([f"i{k}" for k in range(200)])

    def fit_predict(fold, train, held):
        trained_on = set(ids[train])
        return np.array([item in trained_on for item in ids[held]], dtype=float)

    leaked = crossfit.cross_predict(ids, fit_predict)
    assert leaked.sum() == 0
    assert leaked.shape == (200,)
