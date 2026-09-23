"""End-to-end test of the assignment workflow using synthetic data."""

import sqlite3

import pandas as pd

import main as assignment_main


def test_complete_workflow_creates_required_outputs(
    tmp_path, monkeypatch, tiny_train, tiny_ideal
):
    """Run the same orchestration as the submitted program without personal data."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    tiny_train.to_csv(raw / "train.csv", index=False)
    tiny_ideal.to_csv(raw / "ideal.csv", index=False)
    pd.DataFrame(
        [
            {"x": 0.0, "y": 0.1},
            {"x": 1.0, "y": 10.9},
            {"x": 2.0, "y": 999.0},
        ]
    ).to_csv(raw / "test.csv", index=False)

    monkeypatch.chdir(tmp_path)
    exit_code = assignment_main.main()

    assert exit_code == 0
    database_path = tmp_path / "output" / "results.db"
    visualization_path = tmp_path / "output" / "visualization.html"
    assert database_path.exists()
    assert visualization_path.exists()

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        mapped_count = connection.execute(
            "SELECT COUNT(*) FROM mapped_results"
        ).fetchone()[0]

    assert tables == {"training_data", "ideal_functions", "mapped_results"}
    assert mapped_count == 2
