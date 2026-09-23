"""Persist assignment data to a SQLite database through SQLAlchemy."""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from src.exceptions import DatabaseOperationError


class DatabaseManager:
    """Manage the three SQLite tables required by the written assignment."""

    def __init__(self, database_path: str = "output/results.db") -> None:
        """Create a SQLAlchemy engine for a file-based SQLite database."""
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.database_path}")

    def save_input_tables(
        self,
        train_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
    ) -> None:
        """Persist training and ideal tables before test-data processing."""
        try:
            with self.engine.begin() as connection:
                train_df.to_sql(
                    "training_data", connection, if_exists="replace", index=False
                )
                ideal_df.to_sql(
                    "ideal_functions", connection, if_exists="replace", index=False
                )
        except SQLAlchemyError as exc:
            raise DatabaseOperationError(
                "Could not persist training and ideal datasets."
            ) from exc

    def save_mapped_results(self, mapped_df: pd.DataFrame) -> None:
        """Persist the required four-column test mapping table."""
        try:
            mapped_df[["x", "y", "delta_y", "ideal_function"]].to_sql(
                "mapped_results", self.engine, if_exists="replace", index=False
            )
        except (SQLAlchemyError, KeyError) as exc:
            raise DatabaseOperationError("Could not persist mapped results.") from exc

    def dispose(self) -> None:
        """Release SQLAlchemy connection resources."""
        self.engine.dispose()
