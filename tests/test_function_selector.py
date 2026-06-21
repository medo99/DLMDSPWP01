import pandas as pd

from src.function_selector import FunctionSelector


def test_find_best_functions():
    train_df = pd.read_csv("data/raw/train.csv")
    ideal_df = pd.read_csv("data/raw/ideal.csv")

    selector = FunctionSelector(train_df, ideal_df)
    result = selector.find_best_functions()

    assert result["y1"]["ideal_function"] == "y13"
    assert result["y2"]["ideal_function"] == "y24"
    assert result["y3"]["ideal_function"] == "y36"
    assert result["y4"]["ideal_function"] == "y40"