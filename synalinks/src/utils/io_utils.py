# Modified from: keras/src/utils/io_utils.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import sys

from absl import logging

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common import global_state


@synalinks_export(
    [
        "synalinks.config.enable_interactive_logging",
        "synalinks.utils.enable_interactive_logging",
    ]
)
def enable_interactive_logging():
    """Turn on interactive logging.

    When interactive logging is enabled, Keras displays logs via stdout.
    This provides the best experience when using Keras in an interactive
    environment such as a shell or a notebook.
    """
    global_state.set_global_attribute("interactive_logging", True)


@synalinks_export(
    [
        "synalinks.config.disable_interactive_logging",
        "synalinks.utils.disable_interactive_logging",
    ]
)
def disable_interactive_logging():
    """Turn off interactive logging.

    When interactive logging is disabled, Keras sends logs to `absl.logging`.
    This is the best option when using Keras in a non-interactive
    way, such as running a training or inference job on a server.
    """
    global_state.set_global_attribute("interactive_logging", False)


@synalinks_export(
    [
        "synalinks.config.is_interactive_logging_enabled",
        "synalinks.utils.is_interactive_logging_enabled",
    ]
)
def is_interactive_logging_enabled():
    """Check if interactive logging is enabled.

    To switch between writing logs to stdout and `absl.logging`, you may use
    `synalinks.config.enable_interactive_logging()` and
    `synalinks.config.disable_interactive_logging()`.

    Returns:
        Boolean, `True` if interactive logging is enabled,
        and `False` otherwise.
    """
    return global_state.get_global_attribute("interactive_logging", True)


def set_logging_verbosity(level):
    """Sets the verbosity level for logging.

    Supported log levels are as follows:

    - `"FATAL"` (least verbose)
    - `"ERROR"`
    - `"WARNING"`
    - `"INFO"`
    - `"DEBUG"` (most verbose)

    Args:
        level: A string corresponding to the level of verbosity for logging.
    """
    valid_levels = {
        "FATAL": logging.FATAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    verbosity = valid_levels.get(level)
    if verbosity is None:
        raise ValueError(
            "Please pass a valid level for logging verbosity. "
            f"Expected one of: {set(valid_levels.keys())}. "
            f"Received: {level}"
        )
    logging.set_verbosity(verbosity)


def print_msg(message, line_break=True):
    """Print the message to absl logging or stdout."""
    message = str(message)
    if is_interactive_logging_enabled():
        message = message + "\n" if line_break else message
        try:
            sys.stdout.write(message)
        except UnicodeEncodeError:
            # If the encoding differs from UTF-8, `sys.stdout.write` may fail.
            # To address this, replace special unicode characters in the
            # message, and then encode and decode using the target encoding.
            message = _replace_special_unicode_character(message)
            message_bytes = message.encode(sys.stdout.encoding, errors="ignore")
            message = message_bytes.decode(sys.stdout.encoding)
            sys.stdout.write(message)
        sys.stdout.flush()
    else:
        logging.info(message)


def ask_to_proceed_with_overwrite(filepath):
    """Produces a prompt asking about overwriting a file.

    Args:
        filepath: the path to the file to be overwritten.

    Returns:
        True if we can proceed with overwrite, False otherwise.
    """
    overwrite = (
        input(f"[WARNING] {filepath} already exists - overwrite? [y/n]").strip().lower()
    )
    while overwrite not in ("y", "n"):
        overwrite = input('Enter "y" (overwrite) or "n" (cancel).').strip().lower()
    if overwrite == "n":
        return False
    print_msg("[TIP] Next time specify overwrite=True!")
    return True


def _replace_special_unicode_character(message):
    message = str(message).replace("━", "=")  # Fall back to Keras2 behavior.
    return message
