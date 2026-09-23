"""Calculate mapping thresholds from training-to-ideal deviations."""

import numpy as np
import pandas as pd


class DeviationCalculator:
    """Apply the assignment's sqrt(2) maximum-deviation threshold rule."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        selected_functions: dict[str, dict[str, float | str]],
    ) -> None:
        """Store validated data and selected ideal-function metadata."""
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.selected_functions = selected_functions

    def calculate_allowed_deviations(self) -> dict[str, dict[str, float | str]]:
        """Calculate maximum residual and allowed sqrt(2)-scaled deviation."""
        ideal_indexed = self.ideal_df.set_index("x")
        results: dict[str, dict[str, float | str]] = {}

        for train_column, info in self.selected_functions.items():
            ideal_column = str(info["ideal_function"])
            training = self.train_df[["x", train_column]].set_index("x")[train_column]
            deviation = (
                training - ideal_indexed.loc[training.index, ideal_column]
            ).abs()
            max_deviation = float(deviation.max())
            allowed_deviation = float(max_deviation * np.sqrt(2))

            results[ideal_column] = {
                "training_function": train_column,
                "max_deviation": max_deviation,
                "allowed_deviation": allowed_deviation,
            }

        return results
