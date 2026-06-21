from src.data_loader import DataLoader
from src.function_selector import FunctionSelector
from src.deviation_calculator import DeviationCalculator
from src.test_mapper import TestMapper


def test_mapped_count():
    loader = DataLoader()

    train_df, ideal_df, test_df = loader.load_data()

    selector = FunctionSelector(train_df, ideal_df)
    selected = selector.find_best_functions()

    calculator = DeviationCalculator(
        train_df,
        ideal_df,
        selected
    )

    allowed = calculator.calculate_allowed_deviations()

    mapper = TestMapper(
        ideal_df,
        test_df,
        allowed
    )

    mapped_df = mapper.map_test_points()

    assert len(mapped_df) == 34