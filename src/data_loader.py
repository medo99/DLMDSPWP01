from pathlib import Path

import pandas as pd

from src.base_processor import BaseProcessor
from src.exceptions import DataFileNotFoundError


class DataLoader(BaseProcessor):
    """Load training, ideal, and test datasets."""

    def __init__(
        self,
        train_path="data/raw/train.csv",
        ideal_path="data/raw/ideal.csv",
        test_path="data/raw/test.csv",
    ):
        """Initialize dataset file paths."""
        super().__init__()
        self.train_path = train_path
        self.ideal_path = ideal_path
        self.test_path = test_path

    def _read_csv(self, file_path):
        """Read a CSV file and raise a custom error if it is missing."""
        if not Path(file_path).exists():
            raise DataFileNotFoundError(f"Missing required file: {file_path}")

        return pd.read_csv(file_path)

    def load_data(self):
        """Load all project datasets."""
        train_df = self._read_csv(self.train_path)
        ideal_df = self._read_csv(self.ideal_path)
        test_df = self._read_csv(self.test_path)

        return train_df, ideal_df, test_df