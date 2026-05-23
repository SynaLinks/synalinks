# Modified from: keras/src/backend/config.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import logging
import os
import random
import re
import tempfile

import numpy as np
import orjson

from synalinks.src.api_export import synalinks_export

# The type of float to use throughout a session.
_FLOATX = "float32"

# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 1e-7

# The Synalinks API key.
_SYNALINKS_API_KEY = None

# Default backend: Pydantic.
_BACKEND = "pydantic"

# Default Synalinks Studio API base
_SYNALINKS_API_BASE = "http://localhost:8000/api"

# Enable observability
_ENABLE_OBSERVABILITY = False

# MLflow tracking URI for observability
_MLFLOW_TRACKING_URI = None

# MLflow experiment name for observability
_MLFLOW_EXPERIMENT_NAME = "synalinks_traces"

# Default language model (cached instance) and the identifier to persist.
# The instance is materialized lazily because language_models depends on
# modules which depends on backend.
_DEFAULT_LANGUAGE_MODEL = None
_DEFAULT_LANGUAGE_MODEL_IDENTIFIER = None

# Default embedding model (same shape as language model above).
_DEFAULT_EMBEDDING_MODEL = None
_DEFAULT_EMBEDDING_MODEL_IDENTIFIER = None

# Default knowledge base (same shape as language model above).
_DEFAULT_KNOWLEDGE_BASE = None
_DEFAULT_KNOWLEDGE_BASE_IDENTIFIER = None

# Available backends
_AVAILABLE_BACKEND = ["pydantic"]

# Random seed for reproductability
_RANDOM_SEED = 123

np.random.seed(_RANDOM_SEED)
random.seed(_RANDOM_SEED)

logger = logging.getLogger("synalinks")
logger.propagate = False
logger.setLevel(logging.CRITICAL)


class SynalinksLogFormatter(logging.Formatter):
    """Formatter for console logging with colors and prefix."""

    blue = "\x1b[34m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    prefix = "[Synalinks]"

    FORMATS = {
        logging.DEBUG: f"(DEBUG) {prefix}%(message)s",
        logging.INFO: f"{prefix}%(message)s",
        logging.WARNING: f"{prefix}%(message)s",
        logging.ERROR: f"{bold_red}%(message)s{reset}",
        logging.CRITICAL: f"{bold_red}{prefix}%(message)s{reset}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class SynalinksFileFormatter(logging.Formatter):
    """Formatter for file logging that removes ANSI escape codes."""

    ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")

    def format(self, record):
        record.msg = self.ANSI_ESCAPE_PATTERN.sub("", str(record.msg))
        return super().format(record)


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
def enable_logging(log_level="debug", log_to_file=False):
    """
    Configures and enables logging for the application.

    This function sets up the logging configuration for the application,
    allowing logs to be output either to a specified file or to the console.
    The logging level can be set to DEBUG for more verbose logging or INFO
    for standard logging.

    Args:
        log_level (str): The logging level.
        log_to_file (bool): If True save the logs into `synalinks.log`

    The log message format includes the timestamp, log level, and the log message itself.
    """
    global logger

    log_level = log_level.upper()
    if log_level and hasattr(logging, log_level):
        log_level = getattr(logging, log_level)

    logger.setLevel(log_level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(SynalinksLogFormatter())
    logger.addHandler(stream_handler)

    if log_to_file:
        file_handler = logging.FileHandler("synalinks.log", mode="w")
        file_handler.setLevel(log_level)
        formatter = SynalinksLogFormatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


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
    return _SYNALINKS_API_BASE


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
def enable_observability(tracking_uri=None, experiment_name=None):
    """
    Configures and enables observability for the application using MLflow.

    This function sets up the observability configuration for the application,
    enabling tracing of module calls via MLflow.

    Args:
        tracking_uri (str): Optional. The MLflow tracking server URI.
            If not provided, MLflow will use the default (local ./mlruns
            directory or MLFLOW_TRACKING_URI environment variable).
        experiment_name (str): Optional. The MLflow experiment name.
            Defaults to "synalinks_traces".

    Example:

    ```python
    import synalinks

    # Basic usage with defaults
    synalinks.enable_observability()

    # With custom MLflow tracking server
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="my_experiment"
    )
    ```
    """
    global _MLFLOW_TRACKING_URI
    global _MLFLOW_EXPERIMENT_NAME
    global _ENABLE_OBSERVABILITY

    if tracking_uri:
        _MLFLOW_TRACKING_URI = tracking_uri
    if experiment_name:
        _MLFLOW_EXPERIMENT_NAME = experiment_name
    _ENABLE_OBSERVABILITY = True


@synalinks_export(
    [
        "synalinks.config.mlflow_tracking_uri",
        "synalinks.backend.mlflow_tracking_uri",
    ]
)
def mlflow_tracking_uri():
    """Returns the MLflow tracking URI for observability.

    Returns:
        (str): The MLflow tracking URI, or None if not set.
    """
    return _MLFLOW_TRACKING_URI


@synalinks_export(
    [
        "synalinks.config.mlflow_experiment_name",
        "synalinks.backend.mlflow_experiment_name",
    ]
)
def mlflow_experiment_name():
    """Returns the MLflow experiment name for observability.

    Returns:
        (str): The MLflow experiment name.
    """
    return _MLFLOW_EXPERIMENT_NAME


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
# Otherwise either ~/.synalinks or, when home isn't writable, the system temp dir.
if "SYNALINKS_HOME" in os.environ:
    _synalinks_DIR = os.environ.get("SYNALINKS_HOME")
else:
    _synalinks_base_dir = os.path.expanduser("~")
    if not os.access(_synalinks_base_dir, os.W_OK):
        _synalinks_base_dir = tempfile.gettempdir()
    _synalinks_DIR = os.path.join(_synalinks_base_dir, ".synalinks")


@synalinks_export(
    [
        "synalinks.config.default_language_model",
        "synalinks.default_language_model",
    ]
)
def default_language_model():
    """Return the default `LanguageModel` instance, or `None` if unset.

    The instance is constructed lazily on first call when set from a
    persisted identifier (e.g. via `~/.synalinks/synalinks.json`).
    """
    global _DEFAULT_LANGUAGE_MODEL
    if _DEFAULT_LANGUAGE_MODEL is None and _DEFAULT_LANGUAGE_MODEL_IDENTIFIER is not None:
        from synalinks.src.modules.language_models import get as _get_lm

        _DEFAULT_LANGUAGE_MODEL = _get_lm(_DEFAULT_LANGUAGE_MODEL_IDENTIFIER)
    return _DEFAULT_LANGUAGE_MODEL


@synalinks_export(
    [
        "synalinks.config.set_default_language_model",
        "synalinks.set_default_language_model",
    ]
)
def set_default_language_model(identifier: "str | dict | object | None"):
    """Set the default `LanguageModel`.

    Args:
        identifier (str | dict | LanguageModel | None): A model string
            (e.g. `"openai/gpt-4o-mini"`), a config dict, an existing
            `LanguageModel` instance, or `None` to clear. Strings persist
            into the on-disk config; instances do not.
    """
    global _DEFAULT_LANGUAGE_MODEL, _DEFAULT_LANGUAGE_MODEL_IDENTIFIER
    if identifier is None:
        _DEFAULT_LANGUAGE_MODEL = None
        _DEFAULT_LANGUAGE_MODEL_IDENTIFIER = None
        _persist_config()
        return
    from synalinks.src.modules.language_models import get as _get_lm

    _DEFAULT_LANGUAGE_MODEL = _get_lm(identifier)
    _DEFAULT_LANGUAGE_MODEL_IDENTIFIER = (
        identifier if isinstance(identifier, str) else None
    )
    _persist_config()


@synalinks_export(
    [
        "synalinks.config.default_embedding_model",
        "synalinks.default_embedding_model",
    ]
)
def default_embedding_model():
    """Return the default `EmbeddingModel` instance, or `None` if unset."""
    global _DEFAULT_EMBEDDING_MODEL
    if (
        _DEFAULT_EMBEDDING_MODEL is None
        and _DEFAULT_EMBEDDING_MODEL_IDENTIFIER is not None
    ):
        from synalinks.src.modules.embedding_models import get as _get_em

        _DEFAULT_EMBEDDING_MODEL = _get_em(_DEFAULT_EMBEDDING_MODEL_IDENTIFIER)
    return _DEFAULT_EMBEDDING_MODEL


@synalinks_export(
    [
        "synalinks.config.set_default_embedding_model",
        "synalinks.set_default_embedding_model",
    ]
)
def set_default_embedding_model(identifier: "str | dict | object | None"):
    """Set the default `EmbeddingModel`.

    Args:
        identifier (str | dict | EmbeddingModel | None): A model string
            (e.g. `"openai/text-embedding-3-small"`), a config dict, an
            existing `EmbeddingModel` instance, or `None` to clear.
            Strings persist into the on-disk config; instances do not.
    """
    global _DEFAULT_EMBEDDING_MODEL, _DEFAULT_EMBEDDING_MODEL_IDENTIFIER
    if identifier is None:
        _DEFAULT_EMBEDDING_MODEL = None
        _DEFAULT_EMBEDDING_MODEL_IDENTIFIER = None
        _persist_config()
        return
    from synalinks.src.modules.embedding_models import get as _get_em

    _DEFAULT_EMBEDDING_MODEL = _get_em(identifier)
    _DEFAULT_EMBEDDING_MODEL_IDENTIFIER = (
        identifier if isinstance(identifier, str) else None
    )
    _persist_config()


@synalinks_export(
    [
        "synalinks.config.default_knowledge_base",
        "synalinks.default_knowledge_base",
    ]
)
def default_knowledge_base():
    """Return the default `KnowledgeBase` instance, or `None` if unset."""
    global _DEFAULT_KNOWLEDGE_BASE
    if _DEFAULT_KNOWLEDGE_BASE is None and _DEFAULT_KNOWLEDGE_BASE_IDENTIFIER is not None:
        from synalinks.src.knowledge_bases import get as _get_kb

        _DEFAULT_KNOWLEDGE_BASE = _get_kb(_DEFAULT_KNOWLEDGE_BASE_IDENTIFIER)
    return _DEFAULT_KNOWLEDGE_BASE


@synalinks_export(
    [
        "synalinks.config.set_default_knowledge_base",
        "synalinks.set_default_knowledge_base",
    ]
)
def set_default_knowledge_base(identifier: "str | dict | object | None"):
    """Set the default `KnowledgeBase`.

    Args:
        identifier (str | dict | KnowledgeBase | None): A URI string
            (e.g. `"duckdb://./my_database.db"`), a config dict, an
            existing `KnowledgeBase` instance, or `None` to clear.
            Strings persist into the on-disk config; instances do not.
    """
    global _DEFAULT_KNOWLEDGE_BASE, _DEFAULT_KNOWLEDGE_BASE_IDENTIFIER
    if identifier is None:
        _DEFAULT_KNOWLEDGE_BASE = None
        _DEFAULT_KNOWLEDGE_BASE_IDENTIFIER = None
        _persist_config()
        return
    from synalinks.src.knowledge_bases import get as _get_kb

    _DEFAULT_KNOWLEDGE_BASE = _get_kb(identifier)
    _DEFAULT_KNOWLEDGE_BASE_IDENTIFIER = (
        identifier if isinstance(identifier, str) else None
    )
    _persist_config()


def _persist_config():
    """Rewrite `~/.synalinks/synalinks.json` with the current config."""
    if not os.path.exists(_synalinks_DIR):
        try:
            os.makedirs(_synalinks_DIR)
        except OSError:
            return
    payload = {
        "backend": _BACKEND,
        "floatx": _FLOATX,
        "epsilon": _EPSILON,
        "seed": _RANDOM_SEED,
        "api_base": _SYNALINKS_API_BASE,
    }
    if _DEFAULT_LANGUAGE_MODEL_IDENTIFIER is not None:
        payload["language_model"] = _DEFAULT_LANGUAGE_MODEL_IDENTIFIER
    if _DEFAULT_EMBEDDING_MODEL_IDENTIFIER is not None:
        payload["embedding_model"] = _DEFAULT_EMBEDDING_MODEL_IDENTIFIER
    if _DEFAULT_KNOWLEDGE_BASE_IDENTIFIER is not None:
        payload["knowledge_base"] = _DEFAULT_KNOWLEDGE_BASE_IDENTIFIER
    try:
        with open(_config_path, "wb") as f:
            f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    except IOError:
        pass


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
        with open(_config_path, "rb") as f:
            _config = orjson.loads(f.read())
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
    _SYNALINKS_API_BASE = _config.get("api_base", _SYNALINKS_API_BASE)
    assert isinstance(_SYNALINKS_API_BASE, str)

    set_backend(_backend)
    set_floatx(_floatx)
    set_epsilon(_epsilon)
    set_seed(_seed)
    set_api_base(_SYNALINKS_API_BASE)

    # Default LM/Embedding identifiers are stored as strings; the cached
    # instances are materialized lazily on first `default_*_model()` call
    # because `language_models`/`embedding_models` aren't loaded yet.
    _lm_identifier = _config.get("language_model")
    if _lm_identifier is not None:
        assert isinstance(_lm_identifier, str)
        _DEFAULT_LANGUAGE_MODEL_IDENTIFIER = _lm_identifier
    _em_identifier = _config.get("embedding_model")
    if _em_identifier is not None:
        assert isinstance(_em_identifier, str)
        _DEFAULT_EMBEDDING_MODEL_IDENTIFIER = _em_identifier
    _kb_identifier = _config.get("knowledge_base")
    if _kb_identifier is not None:
        assert isinstance(_kb_identifier, str)
        _DEFAULT_KNOWLEDGE_BASE_IDENTIFIER = _kb_identifier

# Save config file with current values, creating the directory if needed.
if not os.path.exists(_config_path):
    _persist_config()
