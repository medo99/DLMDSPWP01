"""Automated tests for plot generator."""
import pandas as pd

from src.plot_generator import PlotGenerator


def test_plot_generator_creates_html(tmp_path, tiny_train, tiny_ideal):
    """Verify that plot generator creates html."""
    mapped = pd.DataFrame(
        [{"x": 0.0, "y": 0.1, "delta_y": 0.0, "ideal_function": "y13"}]
    )
    test_df = pd.DataFrame([{"x": 0.0, "y": 0.1}, {"x": 1.0, "y": 999.0}])
    selected = {
        "y1": {"ideal_function": "y13", "sse": 0.03},
        "y2": {"ideal_function": "y24", "sse": 0.03},
        "y3": {"ideal_function": "y36", "sse": 0.12},
        "y4": {"ideal_function": "y40", "sse": 0.27},
    }
    allowed = {
        "y13": {"training_function": "y1", "allowed_deviation": 0.2},
        "y24": {"training_function": "y2", "allowed_deviation": 0.2},
        "y36": {"training_function": "y3", "allowed_deviation": 0.3},
        "y40": {"training_function": "y4", "allowed_deviation": 0.4},
    }
    output = tmp_path / "visualization.html"

    result = PlotGenerator(str(output)).create_plot(
        tiny_train, tiny_ideal, test_df, mapped, selected, allowed
    )

    assert result == str(output)
    assert output.exists()
    assert output.stat().st_size > 0
