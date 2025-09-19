# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import itertools
import json
import os
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.gridspec import GridSpec

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.arcagi1_tasks import EVALUATION_TASK_NAMES
from synalinks.src.datasets.arcagi1_tasks import TRAINING_TASK_NAMES
from synalinks.src.utils import file_utils

ARCAGI1_BASE_URL = (
    "https://raw.githubusercontent.com/fchollet/ARC-AGI/refs/heads/master/data/"
)
ARCAGI2_BASE_URL = (
    "https://raw.githubusercontent.com/arcprize/ARC-AGI-2/refs/heads/main/data/"
)


class Color(int, Enum):
    BLACK: int = 0
    BLUE: int = 1
    RED: int = 2
    GREEN: int = 3
    YELLOW: int = 4
    GRAY: int = 5
    MAGENTA: int = 6
    ORANGE: int = 7
    LIGHT_BLUE: int = 8
    DARK_RED: int = 9


class ARCAGITask(DataModel):
    input_grid: List[List[Color]] = Field(
        description="The input grid (list of integer list)",
    )
    output_grid: List[List[Color]] = Field(
        description="The output grid (list of integer list)",
    )


class ARCAGIInput(DataModel):
    examples: List[ARCAGITask] = Field(
        description="A set of transformation examples",
    )
    input_grid: List[List[Color]] = Field(
        description="The input grid (list of integer list)",
    )


class ARCAGIOutput(DataModel):
    output_grid: List[List[Color]] = Field(
        description="The output grid (list of integer list)",
    )


@synalinks_export("synalinks.datasets.arcagi.default_instructions")
def default_instructions():
    return """
Find the common rules that maps an input grid to an output grid, given the examples below.
""".strip()


@synalinks_export("synalinks.datasets.arcagi.get_arcagi1_training_task_names")
def get_arcagi1_training_task_names():
    return TRAINING_TASK_NAMES


@synalinks_export("synalinks.datasets.arcagi.get_arcagi1_evaluation_task_names")
def get_arcagi1_evaluation_task_names():
    return EVALUATION_TASK_NAMES


@synalinks_export("synalinks.datasets.arcagi.get_arcagi2_training_task_names")
def get_arcagi2_training_task_names():
    url = f"{ARCAGI2_BASE_URL}/training.txt"
    file_path = file_utils.get_file(origin=url, progbar=False)
    with open(file_path, "r") as f:
        tasks = f.read()
    return tasks.splitlines()


@synalinks_export("synalinks.datasets.arcagi.get_arcagi2_evaluation_task_names")
def get_arcagi2_evaluation_task_names():
    url = f"{ARCAGI2_BASE_URL}/evaluation.txt"
    file_path = file_utils.get_file(origin=url, progbar=False)
    with open(file_path, "r") as f:
        tasks = f.read()
    return tasks.splitlines()


@synalinks_export("synalinks.datasets.arcagi.get_input_data_model")
def get_input_data_model():
    """
    Returns ARC-AGI input data model for pipeline configurations.

    Returns:
        (DataModel): The ARC-AGI input data model
    """
    return ARCAGIInput


@synalinks_export("synalinks.datasets.arcagi.get_output_data_model")
def get_output_data_model():
    """
    Returns ARC-AGI output data model for pipeline configurations.

    Returns:
        (DataModel): The ARC-AGI output data model
    """
    return ARCAGIOutput


@synalinks_export("synalinks.datasets.arcagi.load_data")
def load_data(
    task_name,
    filepath=None,
    arc_version=1,
    one_leave_out=True,
    permutation=False,
    repeat=1,
    curriculum_learning=True,
):
    """
    Load task data by name.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.arcagi1.load_data(
        task_name="62c24649",
        arc_version=1,
    )
    ```

    Args:
        task_name (str): The name of the task
        filepath (str): The task filepath
        arc_version (int): ARC-AGI version between 1 or 2
        one_leave_out (bool): If True create a traning set using the one-leave-out technique.
        permutation (bool): If True augment the training data using permutation of examples.
        repeat (int): The number of times to repeat the training data.
        curriculum_learning (bool): Wether or not to sort the training set by difficulty.
            In this case, the difficulty refer to the grid-size of the expected output.
            (Default to True)
    """
    if filepath:
        try:
            with open(filepath, "r") as f:
                json_data = json.loads(f.read())
        except Exception:
            raise ValueError(
                f"Could not find task data at '{filepath}', "
                "make sure the path is correct.",
            )
    else:
        if arc_version == 1:
            if task_name in get_arcagi1_training_task_names():
                url = f"{ARCAGI1_BASE_URL}/training/{task_name}.json"
            elif task_name in get_arcagi1_evaluation_task_names():
                url = f"{ARCAGI1_BASE_URL}/evaluation/{task_name}.json"
            else:
                raise ValueError(
                    f"Task '{task_name}' not recognized, make sure that"
                    " the task name is valid."
                )
        elif arc_version == 2:
            if task_name in get_arcagi2_training_task_names():
                url = f"{ARCAGI2_BASE_URL}/training/{task_name}.json"
            elif task_name in get_arcagi2_evaluation_task_names():
                url = f"{ARCAGI2_BASE_URL}/evaluation/{task_name}.json"
            else:
                raise ValueError(
                    f"Task '{task_name}' not recognized, make sure that"
                    " the task name is valid."
                )
        else:
            raise ValueError("Invalid `arc_version` provided, should be 1 or 2")
        file_path = file_utils.get_file(origin=url, progbar=False)
        with open(file_path, "r") as f:
            json_data = json.loads(f.read())

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    trainset = json_data.get("train")
    testset = json_data.get("test")

    for i in range(len(trainset)):
        if one_leave_out:
            other_examples = [j for j in range(len(trainset)) if j != i]
            if permutation:
                permutations = list(itertools.permutations(other_examples))
                for k, perm in enumerate(permutations):
                    examples = []
                    for j in perm:
                        input_grid_example = trainset[j].get("input")
                        output_grid_example = trainset[j].get("output")
                        task = ARCAGITask(
                            input_grid=input_grid_example,
                            output_grid=output_grid_example,
                        )
                        examples.append(task)
                    input_grid = trainset[i].get("input")
                    output_grid = trainset[i].get("output")
                    inputs = ARCAGIInput(
                        examples=examples,
                        input_grid=input_grid,
                    )
                    outputs = ARCAGIOutput(
                        output_grid=output_grid,
                    )
                    for i in range(repeat):
                        x_train.append(inputs)
                        y_train.append(outputs)
            else:
                examples = []
                for j in other_examples:
                    input_grid_example = trainset[j].get("input")
                    output_grid_example = trainset[j].get("output")
                    task = ARCAGITask(
                        input_grid=input_grid_example,
                        output_grid=output_grid_example,
                    )
                    examples.append(task)
                input_grid = trainset[i].get("input")
                output_grid = trainset[i].get("output")
                inputs = ARCAGIInput(
                    examples=examples,
                    input_grid=input_grid,
                )
                outputs = ARCAGIOutput(
                    output_grid=output_grid,
                )
                for i in range(repeat):
                    x_train.append(inputs)
                    y_train.append(outputs)

    for i in range(len(testset)):
        examples = []
        for j in range(len(trainset)):
            input_grid_example = trainset[j].get("input")
            output_grid_example = trainset[j].get("output")

            task = ARCAGITask(
                input_grid=input_grid_example,
                output_grid=output_grid_example,
            )
            examples.append(task)
        input_grid = testset[i].get("input")
        output_grid = testset[i].get("output")
        inputs = ARCAGIInput(
            examples=examples,
            input_grid=input_grid,
        )
        outputs = ARCAGIOutput(
            output_grid=output_grid,
        )
        x_test.append(inputs)
        y_test.append(outputs)

    if curriculum_learning:

        def get_output_grid_size(y_example):
            output_grid = y_example.output_grid
            if output_grid:
                return (
                    len(output_grid) * len(output_grid[0]) if len(output_grid) > 0 else 0
                )
            return 0

        training_pairs = list(zip(x_train, y_train))
        training_pairs.sort(key=lambda pair: get_output_grid_size(pair[1]))
        x_train, y_train = zip(*training_pairs)
        x_train = list(x_train)
        y_train = list(y_train)

    x_train = np.array(x_train, dtype="object")
    y_train = np.array(y_train, dtype="object")
    x_test = np.array(x_test, dtype="object")
    y_test = np.array(y_test, dtype="object")
    return (x_train, y_train), (x_test, y_test)


@synalinks_export("synalinks.datasets.arcagi.plot_task")
def plot_task(
    x=None,
    y_true=None,
    y_pred=None,
    task_name=None,
    to_file=None,
    to_folder=None,
    figsize=(12, 6),
    cmap=None,
    grid_color="white",
    grid_linewidth=0.8,
    title_fontsize=11,
    label_fontsize=9,
):
    # Data conversion (if needed)
    if isinstance(x, DataModel):
        x = x.to_json_data_model(name="x")
    if isinstance(y_true, DataModel):
        y_true = y_true.to_json_data_model(name="y_true")
    if isinstance(y_pred, DataModel):
        y_pred = y_pred.to_json_data_model(name="y_pred")

    # Set default filename
    if not to_file and task_name:
        to_file = f"{task_name}.png"
    if not to_file:
        to_file = "arc_agi_task.png"
    filepath = os.path.join(to_folder, to_file) if to_folder else to_file

    # Define colormap
    if cmap is None:
        cmap = colors.ListedColormap(
            [
                "#000000",  # 0: Black
                "#0074D9",  # 1: Blue
                "#FF4136",  # 2: Red
                "#2ECC40",  # 3: Green
                "#FFDC00",  # 4: Yellow
                "#AAAAAA",  # 5: Gray
                "#F012BE",  # 6: Magenta
                "#FF851B",  # 7: Orange
                "#7FDBFF",  # 8: Light Blue
                "#870C25",  # 9: Dark Red
            ]
        )
    norm = colors.Normalize(vmin=0, vmax=9)

    # Determine layout
    nb_examples = len(x.get("examples", []))
    n_cols = 3  # Input, True Output, Predicted Output
    n_rows = nb_examples + 1  # +1 for test input/output

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)

    def plot_grid(ax, grid_data, title):
        """Helper function to plot a grid with consistent styling."""
        im = ax.imshow(grid_data, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=title_fontsize, pad=8, fontweight="semibold")

        # Enhanced grid
        ax.set_xticks(np.arange(-0.5, grid_data.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_data.shape[0], 1), minor=True)
        ax.grid(which="minor", color=grid_color, linewidth=grid_linewidth, alpha=0.8)
        ax.tick_params(which="minor", size=0)
        ax.tick_params(which="major", size=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Add subtle border
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#333333")

        # Add dimensions as text
        height, width = grid_data.shape
        ax.text(
            0.02,
            0.98,
            f"{height}×{width}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )
        return im

    # Plot examples in rows
    for i, example in enumerate(x.get("examples", [])):
        inputs = np.array(example.get("input_grid"))
        outputs = np.array(example.get("output_grid"))

        # Input example (row i, column 0)
        ax_in = fig.add_subplot(gs[i, 0])
        plot_grid(ax_in, inputs, f"Input Example {i + 1}")

        # Output example (row i, column 1)
        ax_out = fig.add_subplot(gs[i, 1])
        plot_grid(ax_out, outputs, f"Output Example {i + 1}")

    # Plot test input (last row, column 0)
    test_inputs = np.array(x.get("input_grid"))
    ax_test_in = fig.add_subplot(gs[nb_examples, 0])
    plot_grid(ax_test_in, test_inputs, "Test Input")

    # Plot outputs (last row, columns 1 and 2)
    if y_true and y_pred:
        y_true_data = np.array(y_true.get("output_grid"))
        y_pred_data = np.array(y_pred.get("output_grid"))

        ax_true = fig.add_subplot(gs[nb_examples, 1])
        plot_grid(ax_true, y_true_data, "True Output")

        ax_pred = fig.add_subplot(gs[nb_examples, 2])
        plot_grid(ax_pred, y_pred_data, "Predicted Output")

    elif y_true:
        y_true_data = np.array(y_true.get("output_grid"))
        ax_true = fig.add_subplot(gs[nb_examples, 1])
        plot_grid(ax_true, y_true_data, "True Output")

    elif y_pred:
        y_pred_data = np.array(y_pred.get("output_grid"))
        ax_pred = fig.add_subplot(gs[nb_examples, 1])
        plot_grid(ax_pred, y_pred_data, "Predicted Output")

    else:
        raise ValueError("Either y_true or y_pred must be provided.")

    # Manually adjust subplot spacing
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.9,
        bottom=0.1,
        hspace=0.4,
        wspace=0.3,
    )

    # Save and display
    plt.savefig(filepath, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()

    try:
        from IPython import display

        return display.Image(filename=filepath)
    except ImportError:
        pass
