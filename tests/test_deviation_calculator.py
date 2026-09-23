"""Automated tests for deviation calculator."""
import numpy as np

from src.deviation_calculator import DeviationCalculator
from src.function_selector import FunctionSelector


def test_allowed_deviation_is_sqrt2_times_max_residual(tiny_train, tiny_ideal):
    """Verify that allowed deviation is sqrt2 times max residual."""
    selected = FunctionSelector(tiny_train, tiny_ideal).find_best_functions()
    result = DeviationCalculator(tiny_train, tiny_ideal, selected).calculate_allowed_deviations()

    assert np.isclose(result["y13"]["max_deviation"], 0.1)
    assert np.isclose(result["y13"]["allowed_deviation"], 0.1 * np.sqrt(2))
