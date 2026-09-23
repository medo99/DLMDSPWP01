"""Shared synthetic fixtures for unit tests."""

import pandas as pd
import pytest


@pytest.fixture
def tiny_train() -> pd.DataFrame:
    """Return a compact training DataFrame for isolated tests."""
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y1": [0.0, 1.0, 2.0],
            "y2": [10.0, 11.0, 12.0],
            "y3": [20.0, 21.0, 22.0],
            "y4": [30.0, 31.0, 32.0],
        }
    )


@pytest.fixture
def tiny_ideal() -> pd.DataFrame:
    """Return a compact 50-function ideal DataFrame for isolated tests."""
    data = {"x": [0.0, 1.0, 2.0]}
    for number in range(1, 51):
        data[f"y{number}"] = [100.0 + number] * 3
    data["y13"] = [0.1, 1.1, 2.1]
    data["y24"] = [9.9, 10.9, 11.9]
    data["y36"] = [20.2, 21.2, 22.2]
    data["y40"] = [30.3, 31.3, 32.3]
    return pd.DataFrame(data)
