from src.data_loader import DataLoader
from src.function_selector import FunctionSelector
from src.deviation_calculator import DeviationCalculator
from src.test_mapper import TestMapper
from src.database_manager import DatabaseManager

def main():
    loader = DataLoader()
    train_df, ideal_df, test_df = loader.load_data()

    selector = FunctionSelector(train_df, ideal_df)
    selected = selector.find_best_functions()

    calculator = DeviationCalculator(train_df, ideal_df, selected)
    allowed_deviations = calculator.calculate_allowed_deviations()

    mapper = TestMapper(ideal_df, test_df, allowed_deviations)
    mapped_df = mapper.map_test_points()

    print(selected)
    print(allowed_deviations)
    print(mapped_df.head())
    print("Mapped count:", len(mapped_df))
    db_manager = DatabaseManager()

    db_manager.save_dataframe(train_df, "training_data")
    db_manager.save_dataframe(ideal_df, "ideal_functions")
    db_manager.save_dataframe(mapped_df, "mapped_results")

    print("Data saved to SQLite database.")

if __name__ == "__main__":
    main()