# Modified from: keras/src/backend/config.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json
import logging
import os
import random

import numpy as np

from synalinks.src.api_export import synalinks_export

# The type of float to use throughout a session.
_FLOATX = "float32"

# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 1e-7

# The Synalinks API key.
_SYNALINKS_API_KEY = None

# Default backend: Pydantic.
_BACKEND = "pydantic"

# Default Synalinks api base
_SYNALINKS_API_BASE = "http://localhost:4248"

# Enable monitoring
_ENABLE_OBSERVABILITY = False

# Available backends
_AVAILABLE_BACKEND = ["pydantic"]

# Random seed for reproductability
_RANDOM_SEED = 123

np.random.seed(_RANDOM_SEED)
random.seed(_RANDOM_SEED)


@synalinks_export(["synalinks.config.floatx", "synalinks.backend.floatx"])
def floatx():
    """Return the default float type, as a string.

    E.g. `'bfloat16'`, `'float16'`, `'float32'`, `'float64'`.

    Returns:
        (str): The current default float type.

    Example:

    ```python
    >>> synalinks.config.floatx()
    'float32'
    ```

    """
    return _FLOATX


@synalinks_export(["synalinks.config.set_floatx", "synalinks.backend.set_floatx"])
def set_floatx(value):
    """Set the default float dtype.

    Note: It is not recommended to set this to `"float16"`,
    as this will likely cause numeric stability issues.
    Instead, use `float64` or `float32`.

    Args:
        value (str): The float type between `'bfloat16'`, `'float16'`, `'float32'`,
            or `'float64'`.

    Examples:

    ```python
    >>> synalinks.config.floatx()
    'float32'
    ```

    ```python
    >>> synalinks.config.set_floatx('float64')
    >>> synalinks.config.floatx()
    'float64'
    ```

    ```python
    >>> # Set it back to float32
    >>> synalinks.config.set_floatx('float32')
    ```

    Raises:
        ValueError: In case of invalid value.
    """
    global _FLOATX
    accepted_dtypes = {"bfloat16", "float16", "float32", "float64"}
    if value not in accepted_dtypes:
        raise ValueError(
            f"Unknown `floatx` value: {value}. Expected one of {accepted_dtypes}"
        )
    _FLOATX = str(value)


@synalinks_export(["synalinks.config.epsilon", "synalinks.backend.epsilon"])
def epsilon():
    """Return the value of the fuzz factor used in numeric expressions.

    Returns:
        (float): The epsilon value.

    Example:

    ```python
    >>> synalinks.config.epsilon()
    1e-07
    ```

    """
    return _EPSILON


@synalinks_export(["synalinks.config.set_epsilon", "synalinks.backend.set_epsilon"])
def set_epsilon(value):
    """Set the value of the fuzz factor used in numeric expressions.

    Args:
        value (float): The new value of epsilon.

    Examples:

    ```python
    >>> synalinks.config.epsilon()
    1e-07
    ```

    ```python
    >>> synalinks.config.set_epsilon(1e-5)
    >>> synalinks.config.epsilon()
    1e-05
    ```

    ```python
    >>> # Set it back to the default value.
    >>> synalinks.config.set_epsilon(1e-7)
    ```

    """
    global _EPSILON
    _EPSILON = value


@synalinks_export(
    [
        "synalinks.config.set_seed",
        "synalinks.backend.set_seed",
        "synalinks.set_seed",
    ]
)
def set_seed(value):
    """Set the value of the random seed for reproductability.

    Args:
        value (float): The new value of epsilon.
    """
    global _RANDOM_SEED
    _RANDOM_SEED = value
    np.random.seed(_RANDOM_SEED)
    random.seed(_RANDOM_SEED)


@synalinks_export(
    [
        "synalinks.config.get_seed",
        "synalinks.backend.get_seed",
        "synalinks.get_seed",
    ]
)
def get_seed():
    return _RANDOM_SEED


@synalinks_export(
    [
        "synalinks.config.enable_logging",
        "synalinks.backend.enable_logging",
        "synalinks.enable_logging",
    ]
)
def enable_logging(filename=None, debug=False):
    """
    Configures and enables logging for the application.

    This function sets up the logging configuration for the application, allowing logs to be output
    either to a specified file or to the console. The logging level can be set to DEBUG for more
    verbose logging or INFO for standard logging.

    Args:
        filename (str): The name of the file where logs should be written. If not provided,
            logs will be output to the console.
        debug (bool): If True, sets the logging level to DEBUG. If False, the logging level
            is set to INFO. Defaults to False.

    The log message format includes the timestamp, log level, and the log message itself.
    """
    level = logging.DEBUG if debug else logging.INFO
    if filename:
        logging.basicConfig(
            filename=filename,
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )


@synalinks_export(
    [
        "synalinks.config.is_observability_enabled",
        "synalinks.backend.is_observability_enabled",
        "synalinks.is_observability_enabled",
    ]
)
def is_observability_enabled():
    """Check if the observability is enabled

    Returns:
        (bool): True if the observability is enabled.
    """
    return _ENABLE_OBSERVABILITY


@synalinks_export(
    [
        "synalinks.config.api_base",
        "synalinks.backend.api_base",
        "synalinks.api_base",
    ]
)
def api_base():
    """Returns the Synalinks api base

    Returns:
        (str): The observability api base
    """
    return _API_BASE


@synalinks_export(
    [
        "synalinks.config.set_api_base",
        "synalinks.backend.set_api_base",
        "synalinks.set_api_base",
    ]
)
def set_api_base(api_base):
    """Set the observability api base

    Args:
        api_base (str): The observability api base
    """
    global _SYNALINKS_API_BASE
    _SYNALINKS_API_BASE = api_base


@synalinks_export(
    [
        "synalinks.config.enable_observability",
        "synalinks.backend.enable_observability",
        "synalinks.enable_observability",
    ]
)
def enable_observability(api_base=None):
    """
    Configures and enables observability for the application.

    This function sets up the observability configuration for the application,
    allowing traces and logs to be send to Synalinks Lab.

    Args:
        api_base (str): Optional. The api base to send the traces and logs to.
    """
    global _SYNALINKS_API_BASE
    global _ENABLE_OBSERVABILITY
    if api_base:
        _SYNALINKS_API_BASE = api_base
    _ENABLE_OBSERVABILITY = True


@synalinks_export(
    [
        "synalinks.config.api_key",
        "synalinks.backend.api_key",
    ]
)
def api_key():
    """Synalinks API key.

    Returns:
        (str): Synalinks API key.

    ```python
    >>> synalinks.config.api_key()
    'my-secret-api-key'
    ```

    """
    return _SYNALINKS_API_KEY


@synalinks_export(
    [
        "synalinks.config.set_api_key",
        "synalinks.backend.set_api_key",
    ]
)
def set_api_key(key):
    """Set Synalinks API key.

    Args:
        key (str): The API key value.

    The API key is retrieved from the env variables at start.

    ```python
    >>> os.environ["SYNALINKS_API_KEY"] = 'my-secret-api-key'
    ```

    Or you can setup it using the config

    ```python
    >>> synalinks.config.set_api_key('my-secret-api-key')
    >>> synalinks.config.api_key()
    'my-secret-api-key'
    ```

    Args:
        key (str): Synalinks API key.
    """
    global _SYNALINKS_API_KEY
    _SYNALINKS_API_KEY = key


# Set synalinks base dir path given synalinks_HOME env variable, if applicable.
# Otherwise either ~/.synalinks or /tmp.
if "SYNALINKS_HOME" in os.environ:
    _synalinks_DIR = os.environ.get("SYNALINKS_HOME")
else:
    _synalinks_base_dir = os.path.expanduser("~")
    if not os.access(_synalinks_base_dir, os.W_OK):
        _synalinks_base_dir = "/tmp"
    _synalinks_DIR = os.path.join(_synalinks_base_dir, ".synalinks")


@synalinks_export(["synalinks.config.synalinks_home", "synalinks.synalinks_home"])
def synalinks_home():
    # Private accessor for the synalinks home location.
    return _synalinks_DIR


@synalinks_export(["synalinks.config.backend", "synalinks.backend.backend"])
def backend():
    """Publicly accessible method for determining the current backend.

    Returns:
        (str): The name of the backend synalinks is currently using. like
            `"pydantic"`.

    Example:

    ```python
    >>> synalinks.config.backend()
    'pydantic'
    ```
    """
    return _BACKEND


@synalinks_export(["synalinks.config.set_backend", "synalinks.backend.set_backend"])
def set_backend(backend):
    if backend not in _AVAILABLE_BACKEND:
        raise ValueError(
            f"Cannot set backend to {backend}, "
            f"you should choose between: {_AVAILABLE_BACKEND}"
        )
    global _BACKEND
    _BACKEND = backend


# Attemp to get the API key from env variables
_api_key = os.getenv("SYNALINKS_API_KEY", None)
if _api_key:
    assert isinstance(_api_key, str)
    set_api_key(_api_key)

# Attempt to read synalinks config file.
_config_path = os.path.expanduser(os.path.join(_synalinks_DIR, "synalinks.json"))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _backend = _config.get("backend", _BACKEND)
    assert _backend in _AVAILABLE_BACKEND
    _floatx = _config.get("floatx", _FLOATX)
    assert _floatx in {"float16", "float32", "float64"}
    _epsilon = _config.get("epsilon", _EPSILON)
    assert isinstance(_epsilon, float)
    _seed = _config.get("seed", _RANDOM_SEED)
    assert isinstance(_seed, int)
    _api_base = _config.get("api_base", _SYNALINKS_API_BASE)
    assert isinstance(_api_base, str)

    set_backend(_backend)
    set_floatx(_floatx)
    set_epsilon(_epsilon)
    set_seed(_seed)
    set_api_base(_api_base)

# Save config file, if possible.
if not os.path.exists(_synalinks_DIR):
    try:
        os.makedirs(_synalinks_DIR)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

if not os.path.exists(_config_path):
    _config = {
        "backend": _BACKEND,
        "floatx": _FLOATX,
        "epsilon": _EPSILON,
        "seed": _RANDOM_SEED,
        "api_base": _SYNALINKS_API_BASE,
    }
    try:
        with open(_config_path, "w") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass
