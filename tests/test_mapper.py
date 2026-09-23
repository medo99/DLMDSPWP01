"""Automated tests for mapper."""
import pandas as pd

from src.mapper import Mapper


def test_accepted_and_rejected_points_are_distinguished():
    """Verify that accepted and rejected points are distinguished."""
    ideal = pd.DataFrame({"x": [0.0], "y13": [1.0], "y24": [5.0]})
    allowed = {
        "y13": {"training_function": "y1", "allowed_deviation": 0.5},
        "y24": {"training_function": "y2", "allowed_deviation": 0.5},
    }
    rows = [{"x": 0.0, "y": 1.2}, {"x": 0.0, "y": 3.0}]

    test_df, mapped_df = Mapper(ideal, allowed).map_test_rows(rows)

    assert len(test_df) == 2
    assert len(mapped_df) == 1
    assert mapped_df.iloc[0]["ideal_function"] == "y13"


def test_nearest_qualifying_function_wins_when_multiple_match():
    """Verify that nearest qualifying function wins when multiple match."""
    ideal = pd.DataFrame({"x": [-1.6], "y13": [-8.2], "y24": [-8.192]})
    allowed = {
        "y13": {"training_function": "y1", "allowed_deviation": 0.7},
        "y24": {"training_function": "y2", "allowed_deviation": 0.7},
    }

    _, mapped_df = Mapper(ideal, allowed).map_test_rows(
        [{"x": -1.6, "y": -8.079187}]
    )

    assert mapped_df.iloc[0]["ideal_function"] == "y24"


def test_threshold_boundary_is_accepted():
    """Verify that threshold boundary is accepted."""
    ideal = pd.DataFrame({"x": [1.0], "y13": [10.0]})
    allowed = {
        "y13": {"training_function": "y1", "allowed_deviation": 0.5},
    }

    _, mapped_df = Mapper(ideal, allowed).map_test_rows([{"x": 1.0, "y": 10.5}])
    assert len(mapped_df) == 1


def test_missing_x_cannot_be_mapped():
    """Verify that missing x cannot be mapped."""
    ideal = pd.DataFrame({"x": [1.0], "y13": [10.0]})
    allowed = {
        "y13": {"training_function": "y1", "allowed_deviation": 1.0},
    }

    test_df, mapped_df = Mapper(ideal, allowed).map_test_rows([{"x": 2.0, "y": 10.0}])
    assert len(test_df) == 1
    assert mapped_df.empty
    assert list(mapped_df.columns) == ["x", "y", "delta_y", "ideal_function"]
