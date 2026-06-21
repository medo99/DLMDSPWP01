import numpy as np
import pandas as pd


class Mapper:
    __test__ = False
    def __init__(self, ideal_df, test_df, allowed_deviations):
        self.ideal_df = ideal_df
        self.test_df = test_df
        self.allowed_deviations = allowed_deviations

    def map_test_points(self):
        mapped_results = []

        for _, test_row in self.test_df.iterrows():
            x_value = test_row["x"]
            y_value = test_row["y"]

            ideal_row = self.ideal_df[np.isclose(self.ideal_df["x"], x_value)]

            if ideal_row.empty:
                continue

            for ideal_col, info in self.allowed_deviations.items():
                allowed = info["allowed_deviation"]
                ideal_y = ideal_row.iloc[0][ideal_col]
                delta = abs(y_value - ideal_y)

                if delta <= allowed:
                    mapped_results.append({
                        "x": float(x_value),
                        "y": float(y_value),
                        "delta_y": float(delta),
                        "ideal_function": ideal_col,
                    })
                    break

        return pd.DataFrame(mapped_results)