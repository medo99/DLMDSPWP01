"""Run the complete DLMDSPWP01 assignment workflow."""

from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.database_manager import DatabaseManager
from src.deviation_calculator import DeviationCalculator
from src.exceptions import ProjectError
from src.function_selector import FunctionSelector
from src.mapper import Mapper
from src.plot_generator import PlotGenerator


def main() -> int:
    """Execute validation, persistence, fitting, mapping, visualization and reporting."""
    database_manager = DatabaseManager()
    try:
        loader = DataLoader()
        train_df, ideal_df = loader.load_training_and_ideal()

        validator = DataValidator()
        validator.validate_training_and_ideal(train_df, ideal_df)

        # The assignment specifies that training and ideal data are loaded into SQLite
        # before the test dataset is processed line-by-line.
        database_manager.save_input_tables(train_df, ideal_df)

        selector = FunctionSelector(train_df, ideal_df)
        selected = selector.find_best_functions()

        calculator = DeviationCalculator(train_df, ideal_df, selected)
        allowed_deviations = calculator.calculate_allowed_deviations()

        mapper = Mapper(ideal_df, allowed_deviations)
        test_df, mapped_df = mapper.map_test_rows(loader.iter_test_rows())
        database_manager.save_mapped_results(mapped_df)

        plot_path = PlotGenerator().create_plot(
            train_df=train_df,
            ideal_df=ideal_df,
            test_df=test_df,
            mapped_df=mapped_df,
            selected_functions=selected,
            allowed_deviations=allowed_deviations,
        )

        print("Selected ideal functions:")
        for train_column, info in selected.items():
            print(
                f"  {train_column} -> {info['ideal_function']} "
                f"(SSE={float(info['sse']):.12f})"
            )
        print(f"Mapped test observations: {len(mapped_df)} of {len(test_df)}")
        print(f"SQLite database: {database_manager.database_path}")
        print(f"Bokeh visualization: {plot_path}")
        return 0
    except ProjectError as exc:
        print(f"Project error: {exc}")
        return 1
    finally:
        database_manager.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
