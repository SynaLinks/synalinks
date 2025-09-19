# Modified from: keras/src/utils/summary_utils.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import copy
import json
import re
import shutil

import rich
import rich.table

from synalinks.src import tree
from synalinks.src.utils import io_utils


def count_params(variables):
    return int(sum(len(v.get_json()) for v in variables))


def highlight_number(x):
    """Themes numbers in a summary using rich markup.

    We use a separate color for `None`s, e.g. in a module schema.
    """
    if x is None:
        return f"[color(45)]{x}[/]"
    else:
        return f"[color(34)]{x}[/]"


def highlight_symbol(x):
    """Themes keras symbols in a summary using rich markup."""
    return f"[color(33)]{x}[/]"


def bold_text(x, color=None):
    """Bolds text using rich markup."""
    if color:
        return f"[bold][color({color})]{x}[/][/]"
    return f"[bold]{x}[/]"


def format_module_schema(module):
    if not module._inbound_nodes and not module._build_schemas_dict:
        return "?"

    def format_schema(schema):
        schema = copy.deepcopy(schema)
        schema.pop("$defs")
        return json.dumps(schema, indent=2)

    # There are 2 approaches to get output schemas:
    # 1. Using `module._inbound_nodes`, which is possible if the program is a
    # Sequential or Functional.
    # 2. Using `module._build_schemas_dict`, which is possible if users manually
    # build the module.
    if len(module._inbound_nodes) > 0:
        for i in range(len(module._inbound_nodes)):
            outputs = module._inbound_nodes[i].output_data_models
            output_schemas = tree.map_structure(
                lambda x: format_schema(x.get_schema()), outputs
            )
    else:
        try:
            if hasattr(module, "output_schema"):
                output_schemas = format_schema(module.output_schema)
            else:
                outputs = module.compute_output_schema(**module._build_schemas_dict)
                output_schemas = tree.map_schema_structure(
                    lambda x: format_schema(x), outputs
                )
        except NotImplementedError:
            return "?"
    if len(output_schemas) == 1:
        return output_schemas[0]
    out = "\n---\n".join(output_schemas)
    return out


def print_summary(
    program,
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    show_trainable=False,
    module_range=None,
):
    """Prints a summary of a program.

    Args:
        program: Synalinks program instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[0.3, 0.6, 0.70, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
            It defaults to `print` (prints to stdout).
        return_string: If True, return the summary string instead of printing.
        expand_nested: Whether to expand the nested programs.
            If not provided, defaults to `False`.
        show_trainable: Whether to show if a module is trainable.
            If not provided, defaults to `False`.
        module_range: List or tuple containing two strings,
            the starting module name and ending module name (both inclusive),
            indicating the range of modules to be printed in the summary. The
            strings could also be regexes instead of an exact name. In this
             case, the starting module will be the first module that matches
            `module_range[0]` and the ending module will be the last element that
            matches `module_range[1]`. By default (`None`) all
            modules in the program are included in the summary.
    """
    from synalinks.src.programs import Functional
    from synalinks.src.programs import Sequential

    if not print_fn and not io_utils.is_interactive_logging_enabled():
        print_fn = io_utils.print_msg

    if isinstance(program, Sequential):
        sequential_like = True
        modules = program.modules
    elif not isinstance(program, Functional):
        # We treat subclassed programs as a simple sequence of modules, for logging
        # purposes.
        sequential_like = True
        modules = program.modules
    else:
        modules = program._operations
        sequential_like = True
        nodes_by_depth = program._nodes_by_depth.values()
        nodes = []
        for v in nodes_by_depth:
            if (len(v) > 1) or (
                len(v) == 1 and len(tree.flatten(v[0].input_data_models)) > 1
            ):
                # if the program has multiple nodes
                # or if the nodes have multiple inbound_modules
                # the program is no longer sequential
                sequential_like = False
                break
            nodes += v
        if sequential_like:
            # search for shared modules
            for module in program.modules:
                flag = False
                for node in module._inbound_nodes:
                    if node in nodes:
                        if flag:
                            sequential_like = False
                            break
                        else:
                            flag = True
                if not sequential_like:
                    break

    if sequential_like:
        default_line_length = 88
        positions = positions or [0.45, 0.80, 1.0]
        # header names for the different log elements
        header = ["Module (type)", "Output Schema", "Variable #"]
        alignment = ["left", "left", "right"]
    else:
        default_line_length = 108
        positions = positions or [0.3, 0.56, 0.74, 1.0]
        # header names for the different log elements
        header = [
            "Module (type)",
            "Output Schema",
            "Variable #",
            "Connected to",
        ]
        alignment = ["left", "left", "right", "left"]
        relevant_nodes = []
        for v in program._nodes_by_depth.values():
            relevant_nodes += v

    if show_trainable:
        default_line_length += 12
        positions = [p * 0.90 for p in positions] + [1.0]
        header.append("Trainable")
        alignment.append("center")

    # Compute columns widths
    default_line_length = min(default_line_length, shutil.get_terminal_size().columns - 4)
    line_length = line_length or default_line_length
    column_widths = []
    current = 0
    for pos in positions:
        width = int(pos * line_length) - current
        if width < 4:
            raise ValueError("Insufficient console width to print summary.")
        column_widths.append(width)
        current += width

    # Render summary as a rich table.
    columns = []
    # Right align parameter counts.
    for i, name in enumerate(header):
        column = rich.table.Column(
            name,
            justify=alignment[i],
            width=column_widths[i],
        )
        columns.append(column)

    table = rich.table.Table(*columns, width=line_length, show_lines=True)

    def get_connections(module):
        connections = ""
        for node in module._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue
            for dm in node.input_data_models:
                synalinks_history = dm._synalinks_history
                inbound_module = synalinks_history.operation
                node_index = highlight_number(synalinks_history.node_index)
                data_model_index = highlight_number(synalinks_history.data_model_index)
                if connections:
                    connections += ", "
                connections += f"{inbound_module.name}[{node_index}][{data_model_index}]"
        if not connections:
            connections = "-"
        return connections

    def get_module_fields(module, prefix=""):
        output_schema = format_module_schema(module)
        name = prefix + module.name
        cls_name = module.__class__.__name__
        name = rich.markup.escape(name)
        name += f" ({highlight_symbol(rich.markup.escape(cls_name))})"

        if not hasattr(module, "built"):
            params = highlight_number(0)
        elif not module.built:
            params = highlight_number(0) + " (unbuilt)"
        else:
            params = highlight_number(f"{count_params(module.variables):,}")

        fields = [name, output_schema, params]
        if not sequential_like:
            fields.append(get_connections(module))
        if show_trainable:
            if hasattr(module, "variables") and len(module.variables) > 0:
                fields.append(
                    bold_text("Y", color=34)
                    if module.trainable
                    else bold_text("N", color=9)
                )
            else:
                fields.append(bold_text("-"))
        return fields

    def print_module(module, nested_level=0):
        if nested_level:
            prefix = "   " * nested_level + "└" + " "
        else:
            prefix = ""

        fields = get_module_fields(module, prefix=prefix)

        rows = [fields]
        if expand_nested and hasattr(module, "modules") and module.modules:
            nested_modules = module.modules
            nested_level += 1
            for i in range(len(nested_modules)):
                rows.extend(print_module(nested_modules[i], nested_level=nested_level))
        return rows

    # Render all modules to the rich table.
    module_range = get_module_index_bound_by_module_name(modules, module_range)
    for module in modules[module_range[0] : module_range[1]]:
        for row in print_module(module):
            table.add_row(*row)

    # Create a rich console for printing. Capture for non-interactive logging.
    if print_fn:
        console = rich.console.Console(
            highlight=False, force_terminal=False, color_system=None
        )
        console.begin_capture()
    else:
        console = rich.console.Console(highlight=False)

    # Print the to the console.
    console.print(f"Program: {rich.markup.escape(program.name)}")
    console.print(f"description: '{rich.markup.escape(program.description)}'")
    console.print(table)


def get_module_index_bound_by_module_name(modules, module_range=None):
    """Get the module indexes from the model based on module names.

    The module indexes can be used to slice the model into sub models for
    display.

    Args:
        model: `Model` instance.
        module_names: a list or tuple of 2 strings, the starting module name and
            ending module name (both inclusive) for the result. All modules will
            be included when `None` is provided.

    Returns:
        The index value of module based on its unique name (module_names).
        Output will be [first_module_index, last_module_index + 1].
    """
    if module_range is not None:
        if len(module_range) != 2:
            raise ValueError(
                "module_range must be a list or tuple of length 2. Received: "
                f"module_range = {module_range} of length {len(module_range)}"
            )
        if not isinstance(module_range[0], str) or not isinstance(module_range[1], str):
            raise ValueError(
                f"module_range should contain string type only. Received: {module_range}"
            )
    else:
        return [0, len(modules)]

    lower_index = [
        idx
        for idx, module in enumerate(modules)
        if re.match(module_range[0], module.name)
    ]
    upper_index = [
        idx
        for idx, module in enumerate(modules)
        if re.match(module_range[1], module.name)
    ]

    if not lower_index or not upper_index:
        raise ValueError(
            "Passed module_names do not match the module names in the model. "
            f"Received: {module_range}"
        )

    if min(lower_index) > max(upper_index):
        return [min(upper_index), max(lower_index) + 1]
    return [min(lower_index), max(upper_index) + 1]
