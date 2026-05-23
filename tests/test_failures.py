import numpy as np

from viz.failures import top_k_confused_pairs


def test_top_k_confused_pairs_excludes_diagonal():
    cm = np.array([
        [10, 5, 1],
        [4, 12, 2],
        [3, 1, 15],
    ])
    pairs = top_k_confused_pairs(cm, k=2)
    assert pairs[0] == (0, 1, 5)   # (true, pred, count)
    assert pairs[1] == (1, 0, 4)


def test_top_k_confused_pairs_returns_at_most_k():
    cm = np.zeros((5, 5), dtype=int)
    pairs = top_k_confused_pairs(cm, k=3)
    assert len(pairs) == 3
    assert all(count == 0 for _, _, count in pairs)


def test_top_k_confused_pairs_descending_order():
    cm = np.array([
        [0, 1, 3],
        [2, 0, 4],
        [5, 6, 0],
    ])
    pairs = top_k_confused_pairs(cm, k=6)
    counts = [c for _, _, c in pairs]
    assert counts == sorted(counts, reverse=True)
