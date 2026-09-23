"""Create logical Bokeh visualizations for every assignment dataset."""

from pathlib import Path

import pandas as pd
from bokeh.layouts import column
from bokeh.models import Span
from bokeh.plotting import figure, output_file, save


class PlotGenerator:
    """Generate an HTML report containing overview, pair and deviation plots."""

    def __init__(self, output_path: str = "output/visualization.html") -> None:
        """Initialize the visualization output path and ensure its directory exists."""
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def create_plot(
        self,
        train_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        test_df: pd.DataFrame,
        mapped_df: pd.DataFrame,
        selected_functions: dict[str, dict[str, float | str]],
        allowed_deviations: dict[str, dict[str, float | str]],
    ) -> str:
        """Visualize all test data, each training/ideal pair and mapping deviations."""
        output_file(str(self.output_path), title="DLMDSPWP01 Assignment Results")

        overview = figure(
            title="All Test Observations",
            x_axis_label="x",
            y_axis_label="y",
            width=1000,
            height=350,
        )
        if not test_df.empty:
            overview.scatter(
                test_df["x"], test_df["y"], legend_label="All test observations", size=6
            )
        if not mapped_df.empty:
            overview.scatter(
                mapped_df["x"],
                mapped_df["y"],
                legend_label="Mapped observations",
                marker="diamond",
                size=8,
            )
        overview.legend.click_policy = "hide"

        pair_plots = []
        for train_column, info in selected_functions.items():
            ideal_column = str(info["ideal_function"])
            pair_plot = figure(
                title=f"{train_column} and selected ideal function {ideal_column}",
                x_axis_label="x",
                y_axis_label="y",
                width=1000,
                height=320,
            )
            pair_plot.line(
                train_df["x"],
                train_df[train_column],
                legend_label=f"Training {train_column}",
                line_width=2,
            )
            pair_plot.line(
                ideal_df["x"],
                ideal_df[ideal_column],
                legend_label=f"Ideal {ideal_column}",
                line_dash="dashed",
                line_width=2,
            )
            assigned = mapped_df[mapped_df["ideal_function"] == ideal_column]
            if not assigned.empty:
                pair_plot.scatter(
                    assigned["x"],
                    assigned["y"],
                    legend_label=f"Mapped to {ideal_column}",
                    marker="circle",
                    size=7,
                )
            pair_plot.legend.click_policy = "hide"
            pair_plots.append(pair_plot)

        deviation_plot = figure(
            title="Mapped-Point Deviation Relative to Its Allowed Threshold",
            x_axis_label="Mapped observation index",
            y_axis_label="delta_y / allowed_deviation",
            width=1000,
            height=320,
        )
        if not mapped_df.empty:
            ratios = [
                float(row.delta_y)
                / float(allowed_deviations[str(row.ideal_function)]["allowed_deviation"])
                for row in mapped_df.itertuples()
            ]
            deviation_plot.scatter(
                list(range(1, len(ratios) + 1)),
                ratios,
                legend_label="Normalized mapped deviation",
                size=7,
            )
        deviation_plot.add_layout(Span(location=1.0, dimension="width", line_dash="dashed"))

        layout = column(overview, *pair_plots, deviation_plot)
        save(layout)
        return str(self.output_path)
