"""Automated tests for database manager."""
import sqlite3

import pandas as pd

from src.database_manager import DatabaseManager


def test_database_contains_required_tables_and_columns(tmp_path, tiny_train, tiny_ideal):
    """Verify that database contains required tables and columns."""
    db_path = tmp_path / "result.db"
    manager = DatabaseManager(str(db_path))
    mapped = pd.DataFrame(
        [{"x": 0.0, "y": 0.1, "delta_y": 0.1, "ideal_function": "y13"}]
    )

    manager.save_input_tables(tiny_train, tiny_ideal)
    manager.save_mapped_results(mapped)
    manager.dispose()

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"training_data", "ideal_functions", "mapped_results"} <= tables
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(mapped_results)")
        ]
        assert columns == ["x", "y", "delta_y", "ideal_function"]


def test_database_preserves_required_input_schemas(tmp_path, tiny_train, tiny_ideal):
    """Verify that database preserves required input schemas."""
    db_path = tmp_path / "schemas.db"
    manager = DatabaseManager(str(db_path))
    manager.save_input_tables(tiny_train, tiny_ideal)
    manager.dispose()

    with sqlite3.connect(db_path) as connection:
        training_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(training_data)")
        ]
        ideal_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(ideal_functions)")
        ]

    assert training_columns == ["x", "y1", "y2", "y3", "y4"]
    assert ideal_columns == ["x"] + [f"y{i}" for i in range(1, 51)]


def test_missing_mapping_column_raises_project_database_error(tmp_path):
    """Verify that missing mapping column raises project database error."""
    import pytest

    from src.exceptions import DatabaseOperationError

    manager = DatabaseManager(str(tmp_path / "broken.db"))
    broken = pd.DataFrame([{"x": 0.0, "y": 1.0, "delta_y": 0.1}])
    try:
        with pytest.raises(DatabaseOperationError):
            manager.save_mapped_results(broken)
    finally:
        manager.dispose()
