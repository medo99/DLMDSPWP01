"""Reusable validation helpers for processing components."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.exceptions import DataValidationError


class BaseProcessor:
    """Provide common DataFrame validation behaviour to derived processors.

    The assignment requires inheritance.  This base class contains validation behaviour
    that is genuinely reused by :class:`DataValidator`, rather than existing only as an
    empty parent class.
    """

    @staticmethod
    def _require_columns(
        dataframe: pd.DataFrame,
        expected_columns: Iterable[str],
        dataset_name: str,
    ) -> None:
        """Require an exact ordered column schema for a dataset.

        Args:
            dataframe: DataFrame whose schema is checked.
            expected_columns: Column names expected by the assignment.
            dataset_name: Human-readable name used in error messages.

        Raises:
            DataValidationError: If the columns differ from the expected schema.
        """
        expected = list(expected_columns)
        actual = list(dataframe.columns)
        if actual != expected:
            raise DataValidationError(
                f"{dataset_name} columns must be {expected}, but were {actual}."
            )

    @staticmethod
    def _require_non_empty(dataframe: pd.DataFrame, dataset_name: str) -> None:
        """Reject an empty DataFrame."""
        if dataframe.empty:
            raise DataValidationError(f"{dataset_name} must not be empty.")

    @staticmethod
    def _require_complete_numeric(dataframe: pd.DataFrame, dataset_name: str) -> None:
        """Require non-null, numeric and finite values in every dataset column."""
        if dataframe.isna().any().any():
            raise DataValidationError(f"{dataset_name} contains missing values.")

        non_numeric = [
            column
            for column in dataframe.columns
            if not pd.api.types.is_numeric_dtype(dataframe[column])
        ]
        if non_numeric:
            raise DataValidationError(
                f"{dataset_name} contains non-numeric columns: {non_numeric}."
            )

        values = dataframe.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataValidationError(
                f"{dataset_name} contains infinite or non-finite values."
            )
