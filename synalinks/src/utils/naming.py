# Modified from: keras/src/utils/naming.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import collections
import re

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common import global_state


def auto_name(prefix):
    prefix = to_snake_case(prefix)
    return uniquify(prefix)


def uniquify(name):
    object_name_uids = global_state.get_global_attribute(
        "object_name_uids",
        default=collections.defaultdict(int),
        set_to_default=True,
    )
    if name in object_name_uids:
        unique_name = f"{name}_{object_name_uids[name]}"
    else:
        unique_name = name
    object_name_uids[name] += 1
    return unique_name


def to_snake_case(text: str) -> str:
    """Convert a string to snake_case.

    The mirror image of `to_pascal_case`: splits on non-
    alphanumeric separators *and* on case boundaries, then lowercases
    each remaining word and joins them with underscores.

    Examples:

    - ``"FirstName"`` → ``"first_name"``
    - ``"firstName"`` → ``"first_name"``
    - ``"First Name"`` / ``"first-name"`` / ``"first.name"``
      → ``"first_name"``
    - ``"FIRST_NAME"`` → ``"first_name"``
    - ``"HTTPResponse"`` → ``"http_response"``
    - ``"id"`` → ``"id"``

    Used to canonicalize column / property names so that database
    schemas always use snake_case regardless of the casing convention
    in the source file.

    Args:
        text (str): The text to convert.

    Returns:
        (str): The snake_case form, or ``""`` when no alphanumeric content remains.
    """
    if not text:
        return ""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    parts = re.split(r"[^a-zA-Z0-9]+", name)
    return "_".join(p.lower() for p in parts if p)


def to_pascal_case(text: str) -> str:
    """Convert a string to PascalCase.

    Splits on non-alphanumeric separators (``-``, ``_``, spaces, dots,
    etc.) and on case boundaries (``myDocs`` and ``XMLParser`` both
    have boundaries) and then capitalizes each remaining word.

    Examples:

    - ``"my-docs"`` → ``"MyDocs"``
    - ``"my_docs"`` → ``"MyDocs"``
    - ``"my docs"`` → ``"MyDocs"``
    - ``"myDocs"`` → ``"MyDocs"``
    - ``"XMLParser"`` → ``"XmlParser"``
    - ``"docs"`` → ``"Docs"``

    Used to coerce free-form names (filenames, user-supplied table
    titles) into a shape acceptable to SQL-identifier validators —
    note that this does *not* validate identifier rules itself (a
    leading digit, for example, survives this step and must be
    rejected downstream).

    Args:
        text (str): The text to convert.

    Returns:
        (str): The PascalCase form, or ``""`` when no alphanumeric content remains.
    """
    if not text:
        return ""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    parts = re.split(r"[^a-zA-Z0-9]+", name)
    return "".join(p[:1].upper() + p[1:].lower() for p in parts if p)


def to_pkg_name(name):
    name = name.replace("-", "_")
    name = name.replace(" ", "_")
    name = to_snake_case(name)
    return name


@synalinks_export("synalinks.backend.get_uid")
def get_uid(prefix=""):
    """Associates a string prefix with an integer counter.

    Args:
        prefix: String prefix to index.

    Returns:
        Unique integer ID.

    Example:

    >>> get_uid('action')
    1
    >>> get_uid('action')
    2
    """
    object_name_uids = global_state.get_global_attribute(
        "object_name_uids",
        default=collections.defaultdict(int),
        set_to_default=True,
    )
    object_name_uids[prefix] += 1
    return object_name_uids[prefix]


def reset_uids():
    global_state.set_global_attribute("object_name_uids", collections.defaultdict(int))


def get_object_name(obj):
    if hasattr(obj, "name"):  # Most synalinks objects.
        return obj.name
    elif hasattr(obj, "__name__"):  # Function.
        return to_snake_case(obj.__name__)
    elif hasattr(obj, "__class__"):  # Class instance.
        return to_snake_case(obj.__class__.__name__)
    return to_snake_case(str(obj))
