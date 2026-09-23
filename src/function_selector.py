"""Select the least-squares ideal function for each training function."""

import pandas as pd


class FunctionSelector:
    """Choose four ideal functions by minimizing sum of squared errors (SSE)."""

    TRAIN_COLUMNS = ["y1", "y2", "y3", "y4"]

    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame) -> None:
        """Store validated training and ideal datasets."""
        self.train_df = train_df
        self.ideal_df = ideal_df

    def find_best_functions(self) -> dict[str, dict[str, float | str]]:
        """Return the minimum-SSE ideal function for every training function.

        Training and ideal rows are explicitly aligned by ``x`` before residuals are
        calculated, avoiding dependence on input row order.
        """
        ideal_indexed = self.ideal_df.set_index("x")
        selected: dict[str, dict[str, float | str]] = {}

        for train_column in self.TRAIN_COLUMNS:
            training = self.train_df[["x", train_column]].set_index("x")[train_column]
            best_ideal_column: str | None = None
            lowest_sse = float("inf")

            for ideal_column in ideal_indexed.columns:
                deviation = training - ideal_indexed.loc[training.index, ideal_column]
                sse = float((deviation ** 2).sum())
                if sse < lowest_sse:
                    lowest_sse = sse
                    best_ideal_column = ideal_column

            selected[train_column] = {
                "ideal_function": str(best_ideal_column),
                "sse": lowest_sse,
            }

        return selected
