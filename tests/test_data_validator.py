"""Automated tests for data validator."""
import numpy as np
import pytest

from src.data_validator import DataValidator
from src.exceptions import DataValidationError


def test_valid_training_and_ideal_pass(tiny_train, tiny_ideal):
    """Verify that valid training and ideal pass."""
    DataValidator().validate_training_and_ideal(tiny_train, tiny_ideal)


def test_missing_training_column_is_rejected(tiny_train, tiny_ideal):
    """Verify that missing training column is rejected."""
    with pytest.raises(DataValidationError):
        DataValidator().validate_training_and_ideal(
            tiny_train.drop(columns=["y4"]), tiny_ideal
        )


def test_missing_values_are_rejected(tiny_train, tiny_ideal):
    """Verify that missing values are rejected."""
    broken = tiny_train.copy()
    broken.loc[0, "y1"] = np.nan
    with pytest.raises(DataValidationError):
        DataValidator().validate_training_and_ideal(broken, tiny_ideal)


def test_duplicate_x_is_rejected(tiny_train, tiny_ideal):
    """Verify that duplicate x is rejected."""
    broken = tiny_train.copy()
    broken.loc[1, "x"] = broken.loc[0, "x"]
    with pytest.raises(DataValidationError):
        DataValidator().validate_training_and_ideal(broken, tiny_ideal)


def test_mismatched_x_domains_are_rejected(tiny_train, tiny_ideal):
    """Verify that mismatched x domains are rejected."""
    broken = tiny_ideal.copy()
    broken.loc[2, "x"] = 99.0
    with pytest.raises(DataValidationError):
        DataValidator().validate_training_and_ideal(tiny_train, broken)


def test_validator_uses_required_inheritance():
    """Verify that validator uses required inheritance."""
    from src.base_processor import BaseProcessor

    assert issubclass(DataValidator, BaseProcessor)
