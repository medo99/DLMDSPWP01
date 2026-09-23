"""Automated tests for data loader."""
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.exceptions import DataFileNotFoundError, DataValidationError


def test_load_training_and_ideal(tmp_path, tiny_train, tiny_ideal):
    """Verify that load training and ideal."""
    train_path = tmp_path / "train.csv"
    ideal_path = tmp_path / "ideal.csv"
    tiny_train.to_csv(train_path, index=False)
    tiny_ideal.to_csv(ideal_path, index=False)

    loader = DataLoader(train_path=str(train_path), ideal_path=str(ideal_path))
    loaded_train, loaded_ideal = loader.load_training_and_ideal()

    pd.testing.assert_frame_equal(loaded_train, tiny_train)
    pd.testing.assert_frame_equal(loaded_ideal, tiny_ideal)


def test_missing_file_raises_custom_exception(tmp_path):
    """Verify that missing file raises custom exception."""
    loader = DataLoader(
        train_path=str(tmp_path / "missing.csv"),
        ideal_path=str(tmp_path / "also_missing.csv"),
    )
    with pytest.raises(DataFileNotFoundError):
        loader.load_training_and_ideal()


def test_test_rows_are_streamed_line_by_line(tmp_path):
    """Verify that test rows are streamed line by line."""
    test_path = tmp_path / "test.csv"
    test_path.write_text("x,y\n0,1.5\n1,2.5\n", encoding="utf-8")
    loader = DataLoader(test_path=str(test_path))

    rows = list(loader.iter_test_rows())

    assert rows == [{"x": 0.0, "y": 1.5}, {"x": 1.0, "y": 2.5}]


def test_invalid_test_header_is_rejected(tmp_path):
    """Verify that invalid test header is rejected."""
    test_path = tmp_path / "test.csv"
    test_path.write_text("x,value\n0,1.5\n", encoding="utf-8")
    loader = DataLoader(test_path=str(test_path))

    with pytest.raises(DataValidationError):
        list(loader.iter_test_rows())


def test_empty_training_csv_raises_data_loading_error(tmp_path):
    """Verify that empty training csv raises data loading error."""
    from src.exceptions import DataLoadingError

    train_path = tmp_path / "train.csv"
    ideal_path = tmp_path / "ideal.csv"
    train_path.write_text("", encoding="utf-8")
    ideal_path.write_text("x,y1\n0,1\n", encoding="utf-8")

    loader = DataLoader(train_path=str(train_path), ideal_path=str(ideal_path))
    with pytest.raises(DataLoadingError):
        loader.load_training_and_ideal()


def test_empty_test_dataset_is_rejected(tmp_path):
    """Verify that empty test dataset is rejected."""
    test_path = tmp_path / "test.csv"
    test_path.write_text("x,y\n", encoding="utf-8")
    loader = DataLoader(test_path=str(test_path))

    with pytest.raises(DataValidationError):
        list(loader.iter_test_rows())
