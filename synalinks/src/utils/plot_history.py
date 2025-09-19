# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from synalinks.src.api_export import synalinks_export
from synalinks.src.utils.plot_utils import generate_distinct_colors


@synalinks_export("synalinks.utils.plot_history")
def plot_history(
    history,
    to_file="training_history.png",
    to_folder=None,
    xlabel="Epochs",
    ylabel="Scores",
    title="Training history",
    grid=True,
    metrics_filter=None,
    **kwargs,
):
    """Plots the training history of a program and saves it to a file.

    Code Example:

    ```python
    program.compile(...)
    history = await program.fit(...)

    synalinks.utils.plot_history(history)
    ```

    Example:

    ![training_history.png](../../assets/training_history.png)

    Args:
        history (History): The training history.
        to_file (str): The file path where the plot will be saved.
            Default to "training_history.png".
        to_folder (str, optional): The folder where the plot will be saved.
            If provided, will be combined with to_file.
        xlabel (str): Optional. The label for the x-axis. Default to "Epochs".
        ylabel (str): Optional. The label for the y-axis. Default to "Scores".
        title (str): Optional. The title of the plot. Default to "Training history".
        grid (bool): Whether to display the grid on the plot. Default to True.
        metrics_filter (list, optional): List of specific metrics to plot.
            If None, all metrics will be plotted.
        **kwargs (keyword arguments): Additional keyword arguments
            forwarded to `plt.plot()`

    Raises:
        ValueError: If there are unrecognized keyword arguments.

    Returns:
        (IPython.display.Image | marimo.Image | str):
            If running in a Jupyter notebook, returns an IPython Image object
            for inline display. If running in a Marimo notebook returns a marimo image.
            Otherwise returns the filepath where the image has been saved.
    """
    all_metrics = list(history.history.keys())

    if metrics_filter is not None:
        if not all(metric in all_metrics for metric in metrics_filter):
            raise ValueError(f"Requested metrics {metrics_filter} not found in history")
        all_metrics = metrics_filter

    colors = generate_distinct_colors(len(all_metrics))

    for i, metric in enumerate(all_metrics):
        plt.plot(history.history[metric], label=metric, color=colors[i], **kwargs)

    if xlabel:
        plt.xlabel(xlabel)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # Set y-axis limits: minimum 0.0, maximum 1.0 but allow exceeding if needed
    all_values = [val for metric in all_metrics for val in history.history[metric]]
    max_value = max(all_values) if all_values else 1.0
    plt.ylim(0.0, max(1.0, max_value * 1.05))  # 5% padding above max value

    plt.legend()
    plt.grid(grid)

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


@synalinks_export("synalinks.utils.plot_history_with_mean_and_std")
def plot_history_with_mean_and_std(
    history_list,
    to_file="training_history_with_mean_and_std.png",
    to_folder=None,
    xlabel="Epochs",
    ylabel="Scores",
    title="Training history with mean and std",
    grid=True,
    alpha=0.2,
    metrics_filter=None,
    **kwargs,
):
    """Plots the mean and standard deviation of multiple training history list.

    This function takes a list of history objects from multiple runs of the same model
    and plots the mean and standard deviation for each metric.

    Code Example:

    ```python
    program.compile(...)
    history_list = []
    for i in range(5):  # run 5 times
        history = await program.fit(...)
        history_list.append(history)

    synalinks.utils.plot_history_with_mean_and_std(history_list)
    ```

    Example:

    ![training_history_with_mean_and_std.png](../../assets/training_history_with_mean_and_std.png)

    Args:
        history_list (list): A list of History objects from multiple runs.
        to_file (str): The file path where the plot will be saved.
            Default to "training_history_with_mean_and_std.png".
        to_folder (str, optional): The folder where the plot will be saved.
            If provided, will be combined with to_file.
        xlabel (str): Optional. The label for the x-axis. Default to "Epochs".
        ylabel (str): Optional. The label for the y-axis. Default to "Scores".
        title (str): Optional. The title of the plot.
            Default to "Training history with mean and std".
        grid (bool): Whether to display the grid on the plot. Default to True.
        alpha (float): The transparency of the standard deviation area. Default to 0.2.
        metrics_filter (list, optional): List of specific metrics to plot.
            If None, all metrics will be plotted.
        **kwargs (keyword arguments): Additional keyword arguments
            forwarded to `plt.plot()` for the mean lines.

    Raises:
        ValueError: If `history_list` is empty, not a list, or if metrics_filter
            don't match across `history_list`.

    Returns:
        (IPython.display.Image | marimo.Image | str):
            If running in a Jupyter notebook, returns an IPython Image object
            for inline display. If running in a Marimo notebook returns a marimo image.
            Otherwise returns the filepath where the image has been saved.
    """
    if not history_list:
        raise ValueError("history_list cannot be empty")

    if not isinstance(history_list, list):
        raise ValueError("history_list must be a list of History objects")

    all_metrics = list(history_list[0].history.keys())

    if metrics_filter is not None:
        if not all(metric in all_metrics for metric in metrics_filter):
            raise ValueError(
                f"Requested metrics {metrics_filter} not found in history_list"
            )
        all_metrics = metrics_filter

    for i, history in enumerate(history_list):
        if not all(metric in history.history for metric in all_metrics):
            raise ValueError(
                f"Entry {i}: All history objects must contain the same metrics"
            )

    min_epochs = min(len(history.history[all_metrics[0]]) for history in history_list)

    all_values = {metric: [] for metric in all_metrics}

    for history in history_list:
        for metric in all_metrics:
            all_values[metric].append(history.history[metric][:min_epochs])

    mean_values = {}
    std_values = {}
    for metric in all_metrics:
        values_array = np.array(all_values[metric])
        mean_values[metric] = np.mean(values_array, axis=0)
        std_values[metric] = (
            np.std(values_array, axis=0, ddof=1)
            if len(all_values[metric]) > 1
            else np.zeros_like(np.mean(values_array, axis=0))
        )

    colors = generate_distinct_colors(len(all_metrics))

    plt.figure(figsize=(10, 6))

    for i, metric in enumerate(all_metrics):
        color = colors[i]
        x = range(min_epochs)
        mean = mean_values[metric]
        std = std_values[metric]

        plt.plot(x, mean, label=f"{metric} (mean)", color=color, **kwargs)

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=alpha,
            label=f"{metric} (±std)",
        )

    if xlabel:
        plt.xlabel(xlabel)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # Set y-axis limits: minimum 0.0, maximum 1.0 but allow exceeding if needed
    all_vals = []
    for metric in all_metrics:
        all_vals.extend(mean_values[metric] + std_values[metric])
    max_value = max(all_vals) if all_vals else 1.0
    plt.ylim(0.0, max(1.0, max_value * 1.05))  # 5% padding above max value

    plt.legend()
    plt.grid(grid)

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


@synalinks_export("synalinks.utils.plot_history_comparison")
def plot_history_comparison(
    history_dict,
    to_file="training_history_comparison.png",
    to_folder=None,
    xlabel="Epochs",
    ylabel="Scores",
    title="Training History Comparison",
    grid=True,
    metrics_filter=None,
    linestyle_cycle=None,
    **kwargs,
):
    """Plots comparison of training histories across different conditions/models.

    Code Example:

    ```python
    import synalinks
    import asyncio

    NB_RUN = 5

    async def main():

        # ... program definition

        program.compile(...)

        history_list = []
        for i in range(NB_RUN):
            history = await program.fit(...)
            history_list.append(history)

        # ... program_1 definition

        program_1.compile(...)

        history_list_1 = []
        for i in range(NB_RUN):
            history = await program.fit(...)
            history_list_1.append(history)

        history_comparaison = {
            "program_a": history_list
            "program_b: history_list_1
        }

        synalinks.utils.plot_history_comparison(history_comparison)
    ```

    Args:
        history_dict (dict): Dictionary where keys are condition names (e.g., model names)
            and values are History objects. Format:
            {"condition1": history1, "condition2": history2, ...}
        to_file (str): The file path where the plot will be saved.
            Default to "training_history_comparison.png".
        to_folder (str, optional): The folder where the plot will be saved.
            If provided, will be combined with to_file.
        xlabel (str): Optional. The label for the x-axis. Default to "Epochs".
        ylabel (str): Optional. The label for the y-axis. Default to "Scores".
        title (str): Optional. The title of the plot
            (Default to "Training History Comparison").
        grid (bool): Whether to display the grid on the plot. Default to True.
        metrics_filter (list, optional): List of specific metrics to plot.
            If None, all metrics will be plotted.
        linestyle_cycle (list, optional): List of line styles to cycle through
            for conditions (Default to ['-', '--', '-.', ':']).
        **kwargs (keyword arguments): Additional keyword arguments
            forwarded to `plt.plot()`

    Raises:
        ValueError: If history_dict is empty, has inconsistent metric names,
            or if there are unrecognized keyword arguments.

    Returns:
        (IPython.display.Image | marimo.Image | str):
            If running in a Jupyter notebook, returns an IPython Image object
            for inline display. If running in a Marimo notebook returns a marimo image.
            Otherwise returns the filepath where the image has been saved.
    """
    if not history_dict:
        raise ValueError("history_dict cannot be empty")

    # Get all metric names and validate consistency
    condition_names = list(history_dict.keys())
    all_metric_names = list(history_dict[condition_names[0]].history.keys())

    # Validate that all conditions have the same metrics
    for condition in condition_names:
        if set(history_dict[condition].history.keys()) != set(all_metric_names):
            raise ValueError(
                f"Condition '{condition}' has inconsistent metric names. "
                f"Expected: {all_metric_names}, "
                f"Got: {list(history_dict[condition].history.keys())}"
            )

    if metrics_filter is not None:
        if not all(metric in all_metric_names for metric in metrics_filter):
            raise ValueError(f"Requested metrics {metrics_filter} not found in history")
        metric_names = metrics_filter
    else:
        metric_names = all_metric_names

    if linestyle_cycle is None:
        linestyle_cycle = ["-", "--", "-.", ":"]

    # Get colors for metrics and line styles for conditions
    colors = generate_distinct_colors(len(metric_names))

    plt.figure(figsize=(12, 8))

    # Plot each metric for each condition
    for metric_idx, metric in enumerate(metric_names):
        for cond_idx, condition in enumerate(condition_names):
            history = history_dict[condition]
            linestyle = linestyle_cycle[cond_idx % len(linestyle_cycle)]

            plt.plot(
                history.history[metric],
                label=f"{condition} - {metric}",
                color=colors[metric_idx],
                linestyle=linestyle,
                **kwargs,
            )

    if xlabel:
        plt.xlabel(xlabel)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # Set y-axis limits: minimum 0.0, maximum 1.0 but allow exceeding if needed
    all_values = []
    for condition in condition_names:
        for metric in metric_names:
            all_values.extend(history_dict[condition].history[metric])
    max_value = max(all_values) if all_values else 1.0
    plt.ylim(0.0, max(1.0, max_value * 1.05))  # 5% padding above max value

    plt.legend()
    plt.grid(grid)

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


@synalinks_export("synalinks.utils.plot_history_comparison_with_mean_and_std")
def plot_history_comparison_with_mean_and_std(
    history_comparison_dict,
    to_file="training_history_comparison_with_mean_and_std.png",
    to_folder=None,
    xlabel="Epochs",
    ylabel="Scores",
    title="Training History Comparison with Mean and Std",
    grid=True,
    alpha=0.2,
    metrics_filter=None,
    linestyle_cycle=None,
    **kwargs,
):
    """Plots comparison of training histories with mean and standard deviation
        across conditions.

    Calculates mean and standard deviation for each condition across multiple runs and
    displays them as line plots with error bands for comparison.

    Code Example:
    ```python
    # Compare training histories from different models with multiple runs each
    history_comparison = {
        "Model A": [history_a1, history_a2, history_a3],
        "Model B": [history_b1, history_b2, history_b3]
    }

    synalinks.utils.plot_history_comparison_with_mean_and_std(history_comparison)
    ```

    Args:
        history_comparison_dict (dict): Dictionary where keys are condition names and
            values are lists of History objects. Format:
            {"condition1": [history1, history2, ...], ...}
        to_file (str): The file path where the plot will be saved.
            Default to "training_history_comparison_with_mean_and_std.png".
        to_folder (str, optional): The folder where the plot will be saved.
            If provided, will be combined with to_file.
        xlabel (str): Optional. The label for the x-axis. Default to "Epochs".
        ylabel (str): Optional. The label for the y-axis. Default to "Scores".
        title (str): Optional. The title of the plot.
            Default to "Training History Comparison with Mean and Std".
        grid (bool): Whether to display the grid on the plot. Default to True.
        alpha (float): The transparency of the standard deviation area. Default to 0.2.
        metrics_filter (list, optional): List of specific metrics to plot.
            If None, all metrics will be plotted.
        linestyle_cycle (list, optional): List of line styles to cycle through
            for conditions (Default to ['-', '--', '-.', ':']).
        **kwargs (keyword arguments): Additional keyword arguments
            forwarded to `plt.plot()` for the mean lines.

    Raises:
        ValueError: If history_comparison_dict is empty, has inconsistent structures,
            or if there are unrecognized keyword arguments.

    Returns:
        (IPython.display.Image | marimo.Image | str):
            If running in a Jupyter notebook, returns an IPython Image object
            for inline display. If running in a Marimo notebook returns a marimo image.
            Otherwise returns the filepath where the image has been saved.
    """
    if not history_comparison_dict:
        raise ValueError("history_comparison_dict cannot be empty")

    condition_names = list(history_comparison_dict.keys())

    # Validate structure and get metric names
    for condition in condition_names:
        if not isinstance(history_comparison_dict[condition], list):
            raise ValueError(
                f"Values for condition '{condition}' must be a list of History objects"
            )
        if not history_comparison_dict[condition]:
            raise ValueError(f"History list for condition '{condition}' cannot be empty")

    # Get metric names from first condition's first history
    all_metric_names = list(history_comparison_dict[condition_names[0]][0].history.keys())

    # Validate consistency across all conditions and histories
    for condition in condition_names:
        for i, history in enumerate(history_comparison_dict[condition]):
            if set(history.history.keys()) != set(all_metric_names):
                raise ValueError(
                    f"History {i} for condition '{condition}' has inconsistent "
                    "metric names. "
                    f"Expected: {all_metric_names}, Got: {list(history.history.keys())}"
                )

    if metrics_filter is not None:
        if not all(metric in all_metric_names for metric in metrics_filter):
            raise ValueError(f"Requested metrics {metrics_filter} not found in history")
        metric_names = metrics_filter
    else:
        metric_names = all_metric_names

    if linestyle_cycle is None:
        linestyle_cycle = ["-", "--", "-.", ":"]

    # Calculate statistics for each condition and metric
    condition_stats = {}
    min_epochs = float("inf")

    for condition in condition_names:
        # Find minimum epochs across all histories for this condition
        cond_min_epochs = min(
            len(history.history[metric_names[0]])
            for history in history_comparison_dict[condition]
        )
        min_epochs = min(min_epochs, cond_min_epochs)

        condition_stats[condition] = {}
        for metric in metric_names:
            # Collect all values for this metric across runs
            all_values = []
            for history in history_comparison_dict[condition]:
                all_values.append(history.history[metric][:cond_min_epochs])

            # Calculate mean and std across runs
            values_array = np.array(all_values)
            mean_vals = np.mean(values_array, axis=0)[:min_epochs]
            std_vals = (
                np.std(values_array, axis=0, ddof=1)[:min_epochs]
                if len(all_values) > 1
                else np.zeros_like(mean_vals)
            )

            condition_stats[condition][metric] = {"mean": mean_vals, "std": std_vals}

    # Get colors for metrics
    colors = generate_distinct_colors(len(metric_names))

    plt.figure(figsize=(12, 8))

    # Plot each metric for each condition
    x = range(min_epochs)
    for metric_idx, metric in enumerate(metric_names):
        for cond_idx, condition in enumerate(condition_names):
            mean_vals = condition_stats[condition][metric]["mean"]
            std_vals = condition_stats[condition][metric]["std"]

            linestyle = linestyle_cycle[cond_idx % len(linestyle_cycle)]
            color = colors[metric_idx]

            # Plot mean line
            plt.plot(
                x,
                mean_vals,
                label=f"{condition} - {metric}",
                color=color,
                linestyle=linestyle,
                **kwargs,
            )

            # Plot std band
            plt.fill_between(
                x, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=alpha
            )

    if xlabel:
        plt.xlabel(xlabel)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # Set y-axis limits: minimum 0.0, maximum 1.0 but allow exceeding if needed
    all_values = []
    for condition in condition_names:
        for metric in metric_names:
            mean_vals = condition_stats[condition][metric]["mean"]
            std_vals = condition_stats[condition][metric]["std"]
            all_values.extend(mean_vals + std_vals)
    max_value = max(all_values) if all_values else 1.0
    plt.ylim(0.0, max(1.0, max_value * 1.05))  # 5% padding above max value

    plt.legend()
    plt.grid(grid)

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
