# Modified from: keras/src/utils/model_visualization.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import copy
import os
import sys

import orjson

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.utils import io_utils

try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # pydotplus is an improved version of pydot
    try:
        import pydotplus as pydot
    except ImportError:
        # Fall back on pydot if necessary.
        try:
            import pydot
        except ImportError:
            pydot = None


def check_pydot():
    # Returns True if PyDot is available.
    return pydot is not None


def check_graphviz():
    # Returns True if both PyDot and Graphviz are available.
    if not check_pydot():
        return False
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
        return True
    except (OSError, Exception):
        return False


def add_edge(dot, src, dst):
    if not dot.get_edge(src, dst):
        edge = pydot.Edge(src, dst)
        edge.set("penwidth", "2")
        dot.add_edge(edge)


def get_module_activation_name(module):
    if hasattr(module.activation, "name"):
        activation_name = module.activation.name
    elif hasattr(module.activation, "__name__"):
        activation_name = module.activation.__name__
    else:
        activation_name = str(module.activation)
    return activation_name


def make_module_label(module, **kwargs):
    class_name = module.__class__.__name__

    show_module_names = kwargs.pop("show_module_names")
    show_schemas = kwargs.pop("show_schemas")
    show_defs = kwargs.pop("show_defs")
    show_trainable = kwargs.pop("show_trainable")
    if kwargs:
        raise ValueError(f"Invalid kwargs: {kwargs}")

    table = '<<table border="0" cellborder="1" bgcolor="black">'

    colspan_max = sum(int(x) for x in (False, show_trainable))
    if show_schemas:
        colspan_max += 2
    colspan = max(1, colspan_max)

    if show_module_names:
        table += (
            f'<tr><td colspan="{colspan}" bgcolor="black" cellpadding="10">'
            '<font point-size="16" color="white">'
            f"<b>{module.name}</b> ({class_name})"
            "</font></td></tr>"
        )
    else:
        table += (
            f'<tr><td colspan="{colspan}" bgcolor="black" cellpadding="10">'
            '<font point-size="16" color="white">'
            f"<b>{class_name}</b>"
            "</font></td></tr>"
        )

    cols = []
    if show_schemas:
        input_schema = None
        output_schema = None
        try:
            input_schema = tree.map_structure(
                lambda x: copy.deepcopy(x.get_schema()), module.input
            )
            output_schema = tree.map_structure(
                lambda x: copy.deepcopy(x.get_schema()), module.output
            )
        except (ValueError, AttributeError):
            pass

        def format_schema(schema, inputs=False, defs=False):
            if not defs:
                try:
                    schema.pop("$defs")
                except Exception:
                    pass
            prefix = "Input" if inputs else "Output"
            if schema is None:
                return "?"
            if not isinstance(schema, (list, tuple)):
                schema_list = [schema]
            else:
                schema_list = schema
            nested_table = '<table border="0" cellborder="1" cellpadding="10">'
            for i in range(len(schema_list)):
                schema_str = f"\n\n{orjson.dumps(schema_list[i], option=orjson.OPT_INDENT_2).decode()}\n"
                schema_str = schema_str.replace("\n", '<br align="left"/>')
                nested_table += (
                    f'<tr><td align="left">'
                    f"<b>{prefix} Schema {i + 1}:</b>"
                    f"{schema_str}</td></tr>"
                )
            return nested_table + "</table>"

        if class_name != "InputModule":
            formatted_schema = format_schema(input_schema, inputs=True, defs=show_defs)
            cols.append((f'<td bgcolor="white">{formatted_schema}</td>'))
        formatted_schema = format_schema(output_schema, inputs=False, defs=show_defs)
        cols.append((f'<td bgcolor="white">{formatted_schema}</td>'))
    if show_trainable and hasattr(module, "trainable") and module.variables:
        if module.trainable:
            cols.append(
                (
                    '<td bgcolor="forestgreen">'
                    '<font point-size="14" color="white">'
                    "<b>Trainable</b></font></td>"
                )
            )
        else:
            cols.append(
                (
                    '<td bgcolor="firebrick">'
                    '<font point-size="14" color="white">'
                    "<b>Non-trainable</b></font></td>"
                )
            )
    if cols:
        colspan = len(cols)
    else:
        colspan = 1

    if cols:
        table += "<tr>" + "".join(cols) + "</tr>"
    table += "</table>>"
    return table


def make_node(module, **kwargs):
    node = pydot.Node(str(id(module)), label=make_module_label(module, **kwargs))
    node.set("fontname", "Helvetica")
    node.set("border", "0")
    node.set("margin", "0")
    node.set("shape", "box")
    return node


def remove_unused_edges(dot):
    nodes = [v.get_name() for v in dot.get_nodes()]
    for edge in dot.get_edges():
        if edge.get_destination() not in nodes:
            dot.del_edge(edge.get_source(), edge.get_destination())
    return dot


@synalinks_export("synalinks.utils.program_to_dot")
def program_to_dot(
    program,
    show_schemas=False,
    show_defs=False,
    show_module_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    subgraph=False,
    show_trainable=False,
    **kwargs,
):
    """Convert a Synalinks program to dot format.

    Args:
        program (Program): A Synalinks program instance.
        show_schemas (bool): whether to display schema information.
        show_defs (bool): whether to display schema defs information.
        show_module_names (bool): whether to display module names.
        rankdir (str): `rankdir` argument passed to PyDot,
            a string specifying the format of the plot: `"TB"`
            creates a vertical plot; `"LR"` creates a horizontal plot.
        expand_nested (bool): whether to expand nested Functional programs
            into clusters.
        dpi (int): Image resolution in dots per inch.
        subgraph (bool): whether to return a `pydot.Cluster` instance.
        show_trainable (bool): whether to display if a module is trainable.

    Returns:
        (pydot.Dot | pydot.Cluster): A `pydot.Dot` instance representing the
            program or a `pydot.Cluster` instance representing
            nested program if `subgraph=True`.
    """
    from synalinks.src.ops.function import make_node_key

    if not program.built:
        raise ValueError(
            "This program has not yet been built. "
            "Build the program first by calling `build()` or by calling "
            "the program on a batch of data."
        )

    from synalinks.src.programs import functional
    from synalinks.src.programs import sequential

    if not check_pydot():
        raise ImportError(
            "You must install pydot (`pip install pydot`) for program_to_dot to work."
        )

    if subgraph:
        dot = pydot.Cluster(style="dashed", graph_name=program.name)
        dot.set("label", program.name)
        dot.set("labeljust", "l")
    else:
        dot = pydot.Dot()
        dot.set("rankdir", rankdir)
        dot.set("concentrate", True)
        dot.set("dpi", dpi)
        dot.set("splines", "ortho")
        dot.set_node_defaults(schema="record")

    if kwargs.pop("module_range", None) is not None:
        raise ValueError("Argument `module_range` is no longer supported.")
    if kwargs:
        raise ValueError(f"Unrecognized keyword arguments: {kwargs}")

    kwargs = {
        "show_module_names": show_module_names,
        "show_schemas": show_schemas,
        "show_defs": show_defs,
        "show_trainable": show_trainable,
    }

    if isinstance(program, sequential.Sequential):
        modules = program.modules
    elif not isinstance(program, functional.Functional):
        # We treat subclassed programs as a single node.
        node = make_node(program, **kwargs)
        dot.add_node(node)
        return dot
    else:
        modules = program._operations

    # Create graph nodes.
    sub_n_first_node = {}
    sub_n_last_node = {}
    for i, module in enumerate(modules):
        # Process nested functional programs.
        if expand_nested and isinstance(module, functional.Functional):
            subprogram = program_to_dot(
                module,
                show_schemas,
                show_module_names,
                rankdir,
                expand_nested,
                subgraph=True,
                show_trainable=show_trainable,
            )
            # sub_n : subprogram
            sub_n_nodes = subprogram.get_nodes()
            sub_n_first_node[module.name] = sub_n_nodes[0]
            sub_n_last_node[module.name] = sub_n_nodes[-1]
            dot.add_subgraph(subprogram)

        else:
            node = make_node(module, **kwargs)
            dot.add_node(node)

    # Connect nodes with edges.
    # Sequential case.
    if isinstance(program, sequential.Sequential):
        for i in range(len(modules) - 1):
            inbound_module_id = str(id(modules[i]))
            module_id = str(id(modules[i + 1]))
            add_edge(dot, inbound_module_id, module_id)
        return dot

    # Functional case.
    for i, module in enumerate(modules):
        module_id = str(id(module))
        for i, node in enumerate(module._inbound_nodes):
            node_key = make_node_key(module, i)
            if node_key in program._nodes:
                for parent_node in node.parent_nodes:
                    inbound_module = parent_node.operation
                    inbound_module_id = str(id(inbound_module))
                    if not expand_nested:
                        assert dot.get_node(inbound_module_id)
                        assert dot.get_node(module_id)
                        add_edge(dot, inbound_module_id, module_id)
                    else:
                        # if inbound_module is not Functional
                        if not isinstance(inbound_module, functional.Functional):
                            # if current module is not Functional
                            if not isinstance(module, functional.Functional):
                                assert dot.get_node(inbound_module_id)
                                assert dot.get_node(module_id)
                                add_edge(dot, inbound_module_id, module_id)
                            # if current module is Functional
                            elif isinstance(module, functional.Functional):
                                add_edge(
                                    dot,
                                    inbound_module_id,
                                    sub_n_first_node[module.name].get_name(),
                                )
                        # if inbound_module is Functional
                        elif isinstance(inbound_module, functional.Functional):
                            name = sub_n_last_node[inbound_module.name].get_name()
                            if isinstance(module, functional.Functional):
                                output_name = sub_n_first_node[module.name].get_name()
                                add_edge(dot, name, output_name)
                            else:
                                add_edge(dot, name, module_id)
    return dot


@synalinks_export("synalinks.utils.plot_program")
def plot_program(
    program,
    to_file=None,
    to_folder=None,
    show_schemas=False,
    show_module_names=False,
    show_defs=False,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    show_trainable=False,
    **kwargs,
):
    """Converts a Synalinks program to dot format and save to a file.

    Code example:

    ```python
    inputs = ...
    outputs = ...
    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
    )

    synalinks.utils.plot_program(
        program,
        to_file="program_1.png",
        to_folder="/tmp",
        show_schemas=True,
        show_defs=True,
        show_trainable=True,
    )
    ```

    Example:

    ![chain_of_thought.png](../../assets/chain_of_thought.png)

    Args:
        program (Program): A Synalinks program instance
        to_file (str | None): Optional. File name of the plot image.
        show_schemas (bool): whether to display schema information.
        show_defs (bool): whether to display defs schema information.
        show_module_names (bool): whether to display module names.
        rankdir (str): `rankdir` argument passed to PyDot,
            a string specifying the format of the plot: `"TB"`
            creates a vertical plot; `"LR"` creates a horizontal plot.
        expand_nested (bool): whether to expand nested Functional programs
            into clusters.
        dpi (int): Image resolution in dots per inch.
        show_trainable (bool): whether to display if a module is trainable.

    Returns:
        (IPython.display.Image | marimo.Image | str):
            If running in a Jupyter notebook, returns an IPython Image object
            for inline display. If running in a Marimo notebook returns a marimo image.
            Otherwise returns the filepath where the image have been saved.
    """

    if not to_file:
        to_file = f"{program.name}.png"

    if not program.built:
        raise ValueError(
            "This program has not yet been built. "
            "Build the program first by calling `build()` or by calling "
            "the program on a batch of data."
        )
    if not check_pydot():
        message = (
            "You must install pydot (`pip install pydot`) for `plot_program` to work."
        )
        if "IPython.core.magics.namespace" in sys.modules:
            # We don't raise an exception here in order to avoid crashing
            # notebook tests where graphviz is not available.
            io_utils.print_msg(message)
            return
        else:
            raise ImportError(message)
    if not check_graphviz():
        message = (
            "You must install graphviz "
            "(see instructions at https://graphviz.gitlab.io/download/) "
            "for `plot_program` to work."
        )
        if "IPython.core.magics.namespace" in sys.modules:
            # We don't raise an exception here in order to avoid crashing
            # notebook tests where graphviz is not available.
            io_utils.print_msg(message)
            return
        else:
            raise ImportError(message)

    if kwargs:
        raise ValueError(f"Unrecognized keyword arguments: {kwargs}")

    dot = program_to_dot(
        program,
        show_schemas=show_schemas,
        show_defs=show_defs,
        show_module_names=show_module_names,
        rankdir=rankdir,
        expand_nested=expand_nested,
        dpi=dpi,
        show_trainable=show_trainable,
    )
    to_file = str(to_file)
    if dot is None:
        return
    dot = remove_unused_edges(dot)
    _, extension = os.path.splitext(to_file)
    if to_folder:
        to_file = os.path.join(to_folder, to_file)
    if not extension:
        extension = "png"
    else:
        extension = extension[1:]
    # Save image to disk.
    dot.write(to_file, format=extension)
    # Return the image as a Jupyter Image object or Marimo Image object, to be
    # displayed in-line. Note that we cannot easily detect whether the code is
    # running in a Jupyter notebook, and thus we always return the Image if
    # Jupyter is available.
    if extension != "pdf":
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
    else:
        try:
            import marimo as mo

            if mo.running_in_notebook():
                return mo.pdf(src=to_file)
        except ImportError:
            pass
    return to_file
