from __future__ import annotations

from cascade_defect.eval.l2_threshold_sweep import _score


def test_error_decisions_are_counted_as_misclassifications() -> None:
    records = [
        {"track": "A", "true_polarity": "defective", "trace": []},
        {"track": "A", "true_polarity": "normal", "trace": []},
    ]

    row = _score(records, threshold=0.5, uncertain_mode="truth_aware")

    assert row["errors"] == 2
    assert row["fn"] == 1
    assert row["fp"] == 1
    assert row["f1"] == 0.0


def test_uncertain_error_mode_penalizes_metrics() -> None:
    records = [
        {
            "track": "A",
            "true_polarity": "defective",
            "trace": [{"layer": 3, "decision": "uncertain"}],
        },
        {
            "track": "A",
            "true_polarity": "normal",
            "trace": [{"layer": 3, "decision": "uncertain"}],
        },
    ]

    row = _score(records, threshold=0.5, uncertain_mode="error")

    assert row["uncertain"] == 2
    assert row["errors"] == 2
    assert row["fn"] == 1
    assert row["fp"] == 1
    assert row["f1"] == 0.0
