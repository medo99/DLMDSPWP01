"""Automated tests for function selector."""
from src.function_selector import FunctionSelector


def test_find_best_functions_uses_minimum_sse(tiny_train, tiny_ideal):
    """Verify that find best functions uses minimum sse."""
    result = FunctionSelector(tiny_train, tiny_ideal).find_best_functions()

    assert result["y1"]["ideal_function"] == "y13"
    assert result["y2"]["ideal_function"] == "y24"
    assert result["y3"]["ideal_function"] == "y36"
    assert result["y4"]["ideal_function"] == "y40"


def test_selection_is_independent_of_ideal_row_order(tiny_train, tiny_ideal):
    """Verify that selection is independent of ideal row order."""
    reversed_ideal = tiny_ideal.iloc[::-1].reset_index(drop=True)
    result = FunctionSelector(tiny_train, reversed_ideal).find_best_functions()
    assert result["y1"]["ideal_function"] == "y13"
