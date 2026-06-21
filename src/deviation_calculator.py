import numpy as np


class DeviationCalculator:
    def __init__(self, train_df, ideal_df, selected_functions):
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.selected_functions = selected_functions

    def calculate_allowed_deviations(self):
        results = {}

        for train_col, info in self.selected_functions.items():
            ideal_col = info["ideal_function"]

            deviation = abs(self.train_df[train_col] - self.ideal_df[ideal_col])
            max_deviation = deviation.max()
            allowed_deviation = max_deviation * np.sqrt(2)

            results[ideal_col] = {
                "training_function": train_col,
                "max_deviation": float(max_deviation),
                "allowed_deviation": float(allowed_deviation),
            }

        return results