"""Validate assignment datasets before numerical processing."""

import numpy as np
import pandas as pd

from src.base_processor import BaseProcessor
from src.exceptions import DataValidationError


class DataValidator(BaseProcessor):
    """Validate training and ideal data using reusable base-class checks."""

    TRAIN_COLUMNS = ["x", "y1", "y2", "y3", "y4"]
    IDEAL_COLUMNS = ["x"] + [f"y{i}" for i in range(1, 51)]

    def validate_training_and_ideal(
        self,
        train_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
    ) -> None:
        """Validate schemas, values, x uniqueness and x-domain compatibility.

        Raises:
            DataValidationError: If either dataset is unsuitable for the algorithm.
        """
        self._require_non_empty(train_df, "training dataset")
        self._require_non_empty(ideal_df, "ideal dataset")
        self._require_columns(train_df, self.TRAIN_COLUMNS, "training dataset")
        self._require_columns(ideal_df, self.IDEAL_COLUMNS, "ideal dataset")
        self._require_complete_numeric(train_df, "training dataset")
        self._require_complete_numeric(ideal_df, "ideal dataset")

        if train_df["x"].duplicated().any():
            raise DataValidationError("training dataset contains duplicate x-values.")
        if ideal_df["x"].duplicated().any():
            raise DataValidationError("ideal dataset contains duplicate x-values.")

        train_x = np.sort(train_df["x"].to_numpy(dtype=float))
        ideal_x = np.sort(ideal_df["x"].to_numpy(dtype=float))
        if len(train_x) != len(ideal_x) or not np.allclose(
            train_x, ideal_x, rtol=0.0, atol=1e-12
        ):
            raise DataValidationError(
                "training and ideal datasets must contain the same x-values."
            )
