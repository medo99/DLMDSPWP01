"""Load the assignment CSV datasets."""

import csv
import math
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.exceptions import (
    DataFileNotFoundError,
    DataLoadingError,
    DataValidationError,
)


class DataLoader:
    """Load training/ideal datasets and stream the test dataset line by line."""

    def __init__(
        self,
        train_path: str = "data/raw/train.csv",
        ideal_path: str = "data/raw/ideal.csv",
        test_path: str = "data/raw/test.csv",
    ) -> None:
        """Initialize the three input paths used by the assignment."""
        self.train_path = Path(train_path)
        self.ideal_path = Path(ideal_path)
        self.test_path = Path(test_path)

    @staticmethod
    def _read_dataframe(file_path: Path) -> pd.DataFrame:
        """Read a CSV into a DataFrame with explicit exception handling.

        Standard/library exceptions are caught and translated into project-specific
        exceptions so callers receive a clear assignment-level error.
        """
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError as exc:
            raise DataFileNotFoundError(
                f"Required dataset was not found: {file_path}"
            ) from exc
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise DataLoadingError(f"Could not parse dataset: {file_path}") from exc
        except OSError as exc:
            raise DataLoadingError(f"Could not read dataset: {file_path}") from exc

    def load_training_and_ideal(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the two datasets that must be persisted before test processing."""
        return (
            self._read_dataframe(self.train_path),
            self._read_dataframe(self.ideal_path),
        )

    def iter_test_rows(self) -> Iterator[dict[str, float]]:
        """Yield test observations one CSV line at a time.

        The written-assignment task explicitly requires the test data to be loaded
        line-by-line.  ``csv.DictReader`` therefore streams one record at a time instead
        of loading the entire test file into a DataFrame.

        Yields:
            Dictionaries containing numeric ``x`` and ``y`` values.

        Raises:
            DataFileNotFoundError: If the test file does not exist.
            DataLoadingError: If the file cannot be read.
            DataValidationError: If its header or a row is malformed/non-numeric.
        """
        try:
            with self.test_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != ["x", "y"]:
                    raise DataValidationError(
                        "test dataset columns must be ['x', 'y'], "
                        f"but were {reader.fieldnames}."
                    )

                row_count = 0
                for line_number, row in enumerate(reader, start=2):
                    row_count += 1
                    try:
                        x_value = float(row["x"])
                        y_value = float(row["y"])
                    except (TypeError, ValueError, KeyError) as exc:
                        raise DataValidationError(
                            f"Invalid numeric test row at CSV line {line_number}: {row}."
                        ) from exc

                    if not math.isfinite(x_value) or not math.isfinite(y_value):
                        raise DataValidationError(
                            f"Non-finite test value at CSV line {line_number}."
                        )
                    yield {"x": x_value, "y": y_value}

                if row_count == 0:
                    raise DataValidationError("test dataset must not be empty.")
        except FileNotFoundError as exc:
            raise DataFileNotFoundError(
                f"Required dataset was not found: {self.test_path}"
            ) from exc
        except DataValidationError:
            raise
        except (UnicodeDecodeError, csv.Error, OSError) as exc:
            raise DataLoadingError(
                f"Could not read test dataset: {self.test_path}"
            ) from exc
