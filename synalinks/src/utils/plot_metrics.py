# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import os

import matplotlib.pyplot as plt
import numpy as np

from synalinks.src.api_export import synalinks_export
from synalinks.src.utils.plot_utils import generate_distinct_colors


@synalinks_export("synalinks.utils.plot_metrics")
def plot_metrics(
    metrics,
    to_file="evaluation_metrics.png",
    to_folder=None,
    xlabel="Metrics",
    ylabel="Scores",
    title="Evaluation metrics",
    grid=True,
    metrics_filter=None,
    **kwargs,
):
    """Plots the evaluation metrics of a program and saves it to a file.

    Code Example:

    ```python
    program.compile(...)
    metrics = await program.evaluate(...)

    synalinks.utils.plot_metrics(metrics)
    ```

    Example:

    ![evaluation_metrics.png](../../assets/evaluation_metrics.png)

    Args:
        metrics (dict): The metrics from a program evaluation.
        to_file (str): The file path where the plot will be saved.
            Default to "evaluation_metrics.png".
        to_folder (str, optional): The folder where the plot will be saved.
            If provided, will be combined with to_file.
        xlabel (str): Optional. The label for the x-axis. Default to "Metrics".
        ylabel (str): Optional. The label for the y-axis. Default to "Scores".
        title (str): Optional. The title of the plot. Default to "Evaluation metrics".
        grid (bool): Whether to display the grid on the plot. Default to True.
        metrics_filter (list, optional): List of specific metrics to plot.
            If None, all metrics will be plotted.
        **kwargs (keyword arguments): Additional keyword arguments
            forwarded to `plt.bar()`

    Raises:
        ValueError: If there are unrecognized keyword arguments.

    Returns:
        (IPython.display.Image | marimo.Image | str):
            If running in a Jupyter notebook, returns an IPython Image object
            for inline display. If running in a Marimo notebook returns a marimo image.
            Otherwise returns the filepath where the image has been saved.
    """
    all_metrics = list(metrics.keys())

    if metrics_filter is not None:
        if not all(metric in all_metrics for metric in metrics_filter):
            raise ValueError(f"Requested metrics {metrics_filter} not found in metrics")
        filtered_metrics = {k: v for k, v in metrics.items() if k in metrics_filter}
    else:
        filtered_metrics = metrics

    metric_names = list(filtered_metrics.keys())
    metric_values = list(filtered_metrics.values())

    colors = generate_distinct_colors(len(metric_names))

    plt.bar(metric_names, metric_values, color=colors, **kwargs)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # Set y-axis limits: minimum 0.0, maximum 1.0 but allow exceeding if needed
    max_value = max(metric_values) if metric_values else 1.0
    plt.ylim(0.0, max(1.0, max_value * 1.05))  # 5% padding above max value
    plt.grid(grid)

    # Rotate x-axis labels if there are many metrics
    if len(metric_names) > 5:
        plt.xticks(rotation=45, ha="right")

    if to_folder:
        to_file = os.path.join(to_folder, to_file)

    plt.savefig(to_file, dpi=300, bbox_inches="tight")
    plt.close()

    try:
        import marimo as mo

        if mo.running_in_notebook():
            return mo.image(src=to_file).center()
    except ImportError:
        pass

    try:
        from IPython import display

        return display.Image(filename=to_file)
    except ImportError:
        pass

    return to_file


@synalinks_export("synalinks.utils.plot_metrics_comparison")
def plot_metrics_comparison(
    metrics_dict,
    to_file="evaluation_metrics_comparison.png",
    to_folder=None,
    xlabel="Metrics",
    ylabel="Scores",
    title="Metrics Comparison",
    grid=True,
    metrics_filter=None,
    bar_width=0.35,
    **kwargs,
):
    """Plots comparison of evaluation metrics across different runs/models/conditions.

    Code Example:

    ```python
    # Compare metrics from different models
    metrics_comparison = {
        "Program A": metrics_a,
        "Program B": metrics_b,
        "Program C": metrics_c,
    }

    synalinks.utils.plot_metrics_comparison(metrics_comparison)
    ```

    ![evaluation_metrics_comparison.png](../../assets/evaluation_metrics_comparison.png)

    Args:
        metrics_dict (dict): Dictionary where keys are condition names (e.g., model names)
            and values are metrics dictionaries. Format:
            {"condition1": {"metric1": value1, "metric2": value2}, ...}
        to_file (str): The file path where the plot will be saved.
            Default to "evaluation_metrics_comparison.png".
        to_folder (str, optional): The folder where the plot will be saved.
            If provided, will be combined with to_file.
        xlabel (str): Optional. The label for the x-axis. Default to "Metrics".
        ylabel (str): Optional. The label for the y-axis. Default to "Scores".
        title (str): Optional. The title of the plot. Default to "Metrics Comparison".
        grid (bool): Whether to display the grid on the plot. Default to True.
        metrics_filter (list, optional): List of specific metrics to plot.
            If None, all metrics will be plotted.
        bar_width (float): Width of the bars. Default to 0.35.
        **kwargs (keyword arguments): Additional keyword arguments
            forwarded to `plt.bar()`

    Raises:
        ValueError: If metrics_dict is empty, has inconsistent metric names,
            or if there are unrecognized keyword arguments.

    Returns:
        (IPython.display.Image | marimo.Image | str):
            If running in a Jupyter notebook, returns an IPython Image object
            for inline display. If running in a Marimo notebook returns a marimo image.
            Otherwise returns the filepath where the image has been saved.
    """
    if not metrics_dict:
        raise ValueError("metrics_dict cannot be empty")

    # Get all metric names and validate consistency
    condition_names = list(metrics_dict.keys())
    all_metric_names = list(metrics_dict[condition_names[0]].keys())

    # Validate that all conditions have the same metrics
    for condition in condition_names:
        if set(metrics_dict[condition].keys()) != set(all_metric_names):
            raise ValueError(
                f"Condition '{condition}' has inconsistent metric names. "
                f"Expected: {all_metric_names}, "
                f"Got: {list(metrics_dict[condition].keys())}"
            )

    if metrics_filter is not None:
        if not all(metric in all_metric_names for metric in metrics_filter):
            raise ValueError(f"Requested metrics {metrics_filter} not found in metrics")
        metric_names = metrics_filter
    else:
        metric_names = all_metric_names

    # Set up the plot
    x = np.arange(len(metric_names))
    num_conditions = len(condition_names)
    colors = generate_distinct_colors(num_conditions)

    # Calculate bar positions
    bar_positions = []
    for i in range(num_conditions):
        pos = x + (i - num_conditions / 2 + 0.5) * bar_width
        bar_positions.append(pos)

    # Plot bars for each condition
    for i, condition in enumerate(condition_names):
        values = [metrics_dict[condition][metric] for metric in metric_names]
        plt.bar(
            bar_positions[i],
            values,
            bar_width,
            label=condition,
            color=colors[i],
            **kwargs,
        )

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # Set y-axis limits: minimum 0.0, maximum 1.0 but allow exceeding if needed
    all_values = [
        metrics_dict[cond][metric] for cond in condition_names for metric in metric_names
    ]
    max_value = max(all_values) if all_values else 1.0
    plt.ylim(0.0, max(1.0, max_value * 1.05))  # 5% padding above max value

    plt.grid(grid)
    plt.xticks(x, metric_names)
    plt.legend()

    # Rotate x-axis labels if there are many metrics
    if len(metric_names) > 5:
        plt.xticks(x, metric_names, rotation=45, ha="right")

    if to_folder:
        to_file = os.path.join(to_folder, to_file)

    plt.savefig(to_file, dpi=300, bbox_inches="tight")
    plt.close()

    try:
        import marimo as mo

        if mo.running_in_notebook():
            return mo.image(src=to_file).center()
    except ImportError:
        pass

    try:
        from IPython import display

        return display.Image(filename=to_file)
    except ImportError:
        pass

    return to_file


@synalinks_export("synalinks.utils.plot_metrics_comparison_with_mean_and_std")
def plot_metrics_comparison_with_mean_and_std(
    metrics_comparison_dict,
    to_file="evaluation_metrics_comparison_with_mean_and_std.png",
    to_folder=None,
    xlabel="Metrics",
    ylabel="Scores",
    title="Metrics Comparison with Mean and Std",
    grid=True,
    show_values=False,
    capsize=5,
    metrics_filter=None,
    bar_width=0.35,
    **kwargs,
):
    """Plots comparison of evaluation metrics with mean and standard deviation
        across conditions.

    Calculates mean and standard deviation for each condition across multiple runs and
    displays them as grouped bar plots with error bars for comparison.

    Code Example:
    ```python
    # Compare metrics from different models with multiple runs each
    metrics_comparison = {
        "Program A": metrics_list_a,
        "Program B": metrics_list_b
    }

    synalinks.utils.plot_metrics_comparison_with_mean_and_std(
        metrics_comparison,
        show_values=True,
    )
    ```

    ![evaluation_comparaison_with_mean_and_std.png](../../assets/evaluation_comparaison_with_mean_and_std.png)

    Args:
        metrics_comparison_dict (dict): Dictionary where keys are condition names and
            values are lists of metrics dictionaries. Format:
            {"condition1": [{"metric1": val, "metric2": val}, ...], ...}
        to_file (str): The file path where the plot will be saved.
            Default to "evaluation_metrics_comparison_with_mean_and_std.png".
        to_folder (str, optional): The folder where the plot will be saved.
            If provided, will be combined with to_file.
        xlabel (str): Optional. The label for the x-axis. Default to "Metrics".
        ylabel (str): Optional. The label for the y-axis. Default to "Scores".
        title (str): Optional. The title of the plot.
            Default to "Metrics Comparison with Mean and Std".
        grid (bool): Whether to display the grid on the plot. Default to True.
        show_values (bool): Whether to display mean values on top of bars
            (Default to False).
        capsize (float): Size of the error bar caps. Default to 5.
        metrics_filter (list, optional): List of specific metrics to plot.
            If None, all metrics will be plotted.
        bar_width (float): Width of the bars. Default to 0.35.
        **kwargs (keyword arguments): Additional keyword arguments
            forwarded to `plt.bar()`

    Raises:
        ValueError: If metrics_comparison_dict is empty, has inconsistent structures,
            or if there are unrecognized keyword arguments.

    Returns:
        (IPython.display.Image | marimo.Image | str):
            If running in a Jupyter notebook, returns an IPython Image object
            for inline display. If running in a Marimo notebook returns a marimo image.
            Otherwise returns the filepath where the image has been saved.
    """
    if not metrics_comparison_dict:
        raise ValueError("metrics_comparison_dict cannot be empty")

    condition_names = list(metrics_comparison_dict.keys())

    # Validate structure and get metric names
    for condition in condition_names:
        if not isinstance(metrics_comparison_dict[condition], list):
            raise ValueError(
                f"Values for condition '{condition}' must be a list of"
                " metric dictionaries"
            )
        if not metrics_comparison_dict[condition]:
            raise ValueError(f"Metrics list for condition '{condition}' cannot be empty")

    # Get metric names from first condition's first run
    all_metric_names = list(metrics_comparison_dict[condition_names[0]][0].keys())

    # Validate consistency across all conditions and runs
    for condition in condition_names:
        for i, metrics_dict in enumerate(metrics_comparison_dict[condition]):
            if not isinstance(metrics_dict, dict):
                raise ValueError(
                    f"Entry {i} for condition '{condition}' is not a dictionary"
                )
            if set(metrics_dict.keys()) != set(all_metric_names):
                raise ValueError(
                    f"Entry {i} for condition '{condition}' has inconsistent"
                    " metric names. "
                    f"Expected: {all_metric_names}, Got: {list(metrics_dict.keys())}"
                )

    if metrics_filter is not None:
        if not all(metric in all_metric_names for metric in metrics_filter):
            raise ValueError(f"Requested metrics {metrics_filter} not found in metrics")
        metric_names = metrics_filter
    else:
        metric_names = all_metric_names

    # Calculate means and stds for each condition
    condition_stats = {}
    for condition in condition_names:
        means = []
        stds = []
        for metric_name in metric_names:
            values = [
                metrics_dict[metric_name]
                for metrics_dict in metrics_comparison_dict[condition]
            ]
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
        condition_stats[condition] = {"means": means, "stds": stds}

    # Set up the plot
    x = np.arange(len(metric_names))
    num_conditions = len(condition_names)
    colors = generate_distinct_colors(num_conditions)

    # Calculate bar positions
    bar_positions = []
    for i in range(num_conditions):
        pos = x + (i - num_conditions / 2 + 0.5) * bar_width
        bar_positions.append(pos)

    # Plot bars for each condition
    bars_list = []
    for i, condition in enumerate(condition_names):
        means = condition_stats[condition]["means"]
        stds = condition_stats[condition]["stds"]

        bars = plt.bar(
            bar_positions[i],
            means,
            bar_width,
            yerr=stds,
            label=condition,
            color=colors[i],
            capsize=capsize,
            **kwargs,
        )
        bars_list.append(bars)

        # Add value labels on top of bars if requested
        if show_values:
            for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + std + 0.01,
                    f"{mean:.3f}±{std:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90 if num_conditions > 2 else 0,
                )

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # Set y-axis limits: minimum 0.0, maximum 1.0 but allow exceeding if needed
    all_means = [
        mean
        for condition in condition_names
        for mean in condition_stats[condition]["means"]
    ]
    all_stds = [
        std for condition in condition_names for std in condition_stats[condition]["stds"]
    ]
    max_val = max(np.array(all_means) + np.array(all_stds)) if all_means else 1.0
    y_padding = 0.15 if show_values else 0.05
    plt.ylim(0.0, max(1.0, max_val + y_padding))

    plt.grid(grid)
    plt.xticks(x, metric_names)
    plt.legend()

    # Rotate x-axis labels if there are many metrics
    if len(metric_names) > 5:
        plt.xticks(x, metric_names, rotation=45, ha="right")

    if to_folder:
        to_file = os.path.join(to_folder, to_file)

    plt.savefig(to_file, dpi=300, bbox_inches="tight")
    plt.close()

    try:
        import marimo as mo

        if mo.running_in_notebook():
            return mo.image(src=to_file).center()
    except ImportError:
        pass

    try:
        from IPython import display

        return display.Image(filename=to_file)
    except ImportError:
        pass

    return to_file


@synalinks_export("synalinks.utils.plot_metrics_with_mean_and_std")
def plot_metrics_with_mean_and_std(
    metrics_list,
    to_file="evaluation_metrics_with_mean_and_std.png",
    to_folder=None,
    xlabel="Metrics",
    ylabel="Scores",
    title="Evaluation metrics with mean and std",
    grid=True,
    show_values=False,
    capsize=5,
    metrics=None,
    **kwargs,
):
    """Plots the evaluation metrics with mean and standard deviation error bars.

    Calculates mean and standard deviation across multiple evaluation runs and
    displays them as bar plots with error bars.

    Code Example:
    ```python
    program.compile(...)
    metrics_list = []
    for i in range(5):  # Multiple evaluation runs
        metrics = await program.evaluate(...)
        metrics_list.append(metrics)

    synalinks.utils.plot_metrics_with_mean_and_std(metrics_list)
    ```

    Example:

    ![evaluation_metrics_with_mean_and_std.png](../../assets/evaluation_metrics_with_mean_and_std.png)

    Args:
        metrics_list (list[dict]): List of metrics dictionaries from multiple
            program evaluations. Each dict should have format:
            {'metric_name': float_value, ...}
        to_file (str): The file path where the plot will be saved.
            Default to "evaluation_metrics_with_mean_and_std.png".
        to_folder (str, optional): The folder where the plot will be saved.
            If provided, will be combined with to_file.
        xlabel (str): Optional. The label for the x-axis. Default to "Metrics".
        ylabel (str): Optional. The label for the y-axis. Default to "Scores".
        title (str): Optional. The title of the plot.
            Default to "Evaluation metrics with mean and std".
        grid (bool): Whether to display the grid on the plot. Default to True.
        show_values (bool): Whether to display mean values on top of bars
            (Default to True).
        capsize (float): Size of the error bar caps. Default to 5.
        metrics (list, optional): List of specific metrics to plot.
            If None, all metrics will be plotted.
        **kwargs (keyword arguments): Additional keyword arguments
            forwarded to `plt.bar()`

    Raises:
        ValueError: If metrics_list is empty, not a list, contains inconsistent
            metric names, or if there are unrecognized keyword arguments.

    Returns:
        (IPython.display.Image | marimo.Image | str):
            If running in a Jupyter notebook, returns an IPython Image object
            for inline display. If running in a Marimo notebook returns a marimo image.
            Otherwise returns the filepath where the image has been saved.
    """
    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    if not isinstance(metrics_list, list):
        raise ValueError("metrics_list must be a list of metric dictionaries")

    # Get metric names from first run and validate consistency
    all_metric_names = list(metrics_list[0].keys())

    if metrics is not None:
        if not all(metric in all_metric_names for metric in metrics):
            raise ValueError(f"Requested metrics {metrics} not found in metrics_list")
        metric_names = metrics
    else:
        metric_names = all_metric_names

    for i, metrics_dict in enumerate(metrics_list):
        if not isinstance(metrics_dict, dict):
            raise ValueError(f"Entry {i} in metrics_list is not a dictionary")
        if set(metrics_dict.keys()) != set(all_metric_names):
            raise ValueError(
                f"Entry {i} has inconsistent metric names. "
                f"Expected: {all_metric_names}, Got: {list(metrics_dict.keys())}"
            )

    # Calculate mean and std for each metric
    means = []
    stds = []

    for metric_name in metric_names:
        values = [metrics_dict[metric_name] for metrics_dict in metrics_list]
        means.append(np.mean(values))
        stds.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)

    colors = generate_distinct_colors(len(metric_names))

    # Create bar plot with error bars
    bars = plt.bar(
        metric_names, means, yerr=stds, color=colors, capsize=capsize, **kwargs
    )

    # Add value labels on top of bars if requested
    if show_values:
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.01,
                f"{mean:.3f}±{std:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # Set y-axis limits: minimum 0.0, maximum 1.0 but allow exceeding if needed
    max_val = max(np.array(means) + np.array(stds)) if means else 1.0
    y_padding = 0.1 if show_values else 0.05
    plt.ylim(0.0, max(1.0, max_val + y_padding))

    plt.grid(grid)

    # Rotate x-axis labels if there are many metrics
    if len(metric_names) > 5:
        plt.xticks(rotation=45, ha="right")

    if to_folder:
        to_file = os.path.join(to_folder, to_file)

    plt.savefig(to_file, dpi=300, bbox_inches="tight")
    plt.close()

    try:
        import marimo as mo

        if mo.running_in_notebook():
            return mo.image(src=to_file).center()
    except ImportError:
        pass

    try:
        from IPython import display

        return display.Image(filename=to_file)
    except ImportError:
        pass

    return to_file
