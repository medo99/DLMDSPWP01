from src.data_loader import DataLoader
from src.function_selector import FunctionSelector
from src.deviation_calculator import DeviationCalculator
from src.mapper import Mapper
from src.database_manager import DatabaseManager
from src.plot_generator import PlotGenerator
def main():
    loader = DataLoader()
    train_df, ideal_df, test_df = loader.load_data()

    selector = FunctionSelector(train_df, ideal_df)
    selected = selector.find_best_functions()

    calculator = DeviationCalculator(train_df, ideal_df, selected)
    allowed_deviations = calculator.calculate_allowed_deviations()

    mapper = Mapper(ideal_df, test_df, allowed_deviations)
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
    plot_generator = PlotGenerator()
    plot_path = plot_generator.create_plot(
        train_df,
        ideal_df,
        mapped_df,
        selected,
    )

    print(f"Visualization saved to {plot_path}")
if __name__ == "__main__":
    main()