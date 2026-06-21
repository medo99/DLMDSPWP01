class FunctionSelector:
    def __init__(self, train_df, ideal_df):
        self.train_df = train_df
        self.ideal_df = ideal_df

    def find_best_functions(self):
        selected = {}

        for train_column in ["y1", "y2", "y3", "y4"]:
            best_ideal_column = None
            lowest_sse = float("inf")

            for ideal_column in self.ideal_df.columns:
                if ideal_column == "x":
                    continue

                deviation = self.train_df[train_column] - self.ideal_df[ideal_column]
                sse = (deviation ** 2).sum()

                if sse < lowest_sse:
                    lowest_sse = sse
                    best_ideal_column = ideal_column

            selected[train_column] = {
    "ideal_function": best_ideal_column,
    "sse": float(lowest_sse),
}

        return selected