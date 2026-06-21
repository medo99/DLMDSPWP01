from bokeh.plotting import figure, output_file, save


class PlotGenerator:
    """Generate Bokeh visualizations for training, ideal, and mapped test data."""

    def __init__(self, output_path="output/visualization.html"):
        """Initialize the visualization output path."""
        self.output_path = output_path

    def create_plot(self, train_df, ideal_df, mapped_df, selected_functions):
        """Create and save an HTML visualization using Bokeh."""
        output_file(self.output_path)

        plot = figure(
            title="Training Data, Selected Ideal Functions, and Mapped Test Points",
            x_axis_label="X",
            y_axis_label="Y",
            width=1000,
            height=600,
        )

        for train_col, info in selected_functions.items():
            ideal_col = info["ideal_function"]

            plot.line(
                train_df["x"],
                train_df[train_col],
                legend_label=f"Training {train_col}",
                line_width=2,
            )

            plot.line(
                ideal_df["x"],
                ideal_df[ideal_col],
                legend_label=f"Ideal {ideal_col}",
                line_width=2,
                line_dash="dashed",
            )

        if not mapped_df.empty:
            plot.scatter(
                mapped_df["x"],
                mapped_df["y"],
                legend_label="Mapped Test Points",
                size=7,
                marker="circle",
            )

        plot.legend.click_policy = "hide"

        save(plot)

        return self.output_path