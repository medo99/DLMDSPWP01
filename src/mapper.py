"""Map streamed test observations to qualifying selected ideal functions."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


class Mapper:
    """Map each test point to the qualifying ideal function with minimum deviation."""

    MAPPED_COLUMNS = ["x", "y", "delta_y", "ideal_function"]

    def __init__(
        self,
        ideal_df: pd.DataFrame,
        allowed_deviations: dict[str, dict[str, float | str]],
    ) -> None:
        """Store ideal data and per-function acceptance thresholds."""
        self.ideal_df = ideal_df
        self.allowed_deviations = allowed_deviations

    def map_test_rows(
        self,
        test_rows: Iterable[dict[str, float]],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate streamed test rows and return all observations plus accepted mappings.

        If a point satisfies more than one threshold, the candidate with the smallest
        absolute y-deviation is selected.  This makes the assignment deterministic and
        avoids dependence on dictionary iteration order.
        """
        all_rows: list[dict[str, float]] = []
        mapped_results: list[dict[str, float | str]] = []

        for test_row in test_rows:
            x_value = float(test_row["x"])
            y_value = float(test_row["y"])
            all_rows.append({"x": x_value, "y": y_value})

            matches = self.ideal_df[np.isclose(self.ideal_df["x"], x_value)]
            if matches.empty:
                continue

            ideal_row = matches.iloc[0]
            eligible: list[tuple[str, float]] = []
            for ideal_column, info in self.allowed_deviations.items():
                delta = abs(y_value - float(ideal_row[ideal_column]))
                allowed = float(info["allowed_deviation"])
                if delta <= allowed:
                    eligible.append((ideal_column, delta))

            if eligible:
                ideal_column, delta = min(eligible, key=lambda item: (item[1], item[0]))
                mapped_results.append(
                    {
                        "x": x_value,
                        "y": y_value,
                        "delta_y": float(delta),
                        "ideal_function": ideal_column,
                    }
                )

        test_df = pd.DataFrame(all_rows, columns=["x", "y"])
        mapped_df = pd.DataFrame(mapped_results, columns=self.MAPPED_COLUMNS)
        return test_df, mapped_df
