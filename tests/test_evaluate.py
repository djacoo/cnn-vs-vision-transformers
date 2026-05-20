from src.evaluate import compute_metrics


def test_compute_metrics_perfect_prediction():
    y_true = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y_true, y_true)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_compute_metrics_known_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]   # 3/4 correct
    m = compute_metrics(y_true, y_pred)
    assert abs(m["accuracy"] - 0.75) < 1e-9
    assert 0.0 <= m["macro_precision"] <= 1.0
    assert 0.0 <= m["macro_recall"] <= 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0
