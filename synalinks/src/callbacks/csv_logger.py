# Modified from: keras/src/callbacks/csv_logger.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import collections
import csv

import numpy as np

from synalinks.src.api_export import synalinks_export
from synalinks.src.callbacks.callback import Callback
from synalinks.src.utils import file_utils


@synalinks_export("synalinks.callbacks.CSVLogger")
class CSVLogger(Callback):
    """Callback that streams epoch results to a CSV file.

    Supports all values that can be represented as a string,
    including 1D iterables such as `np.ndarray`.

    Args:
        filepath (str | os.PathLike): Filepath of the CSV file, e.g. `'run/log.csv'`.
        separator (str): String used to separate elements in the CSV file.
        append (bool): True: append if file exists (useful for continuing
            training). False: overwrite existing file.

    Example:

    ```python
    csv_logger = CSVLogger(filepath='training.log')
    program.fit(x_train, y_train, callbacks=[csv_logger])
    ```
    """

    def __init__(self, filepath, separator=",", append=False):
        super().__init__()
        self.sep = separator
        self.filepath = file_utils.path_to_string(filepath)
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True

    def on_train_begin(self, logs=None):
        if self.append:
            if file_utils.exists(self.filepath):
                with file_utils.File(self.filepath, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = file_utils.File(self.filepath, mode)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
                return f'"[{", ".join(map(str, k))}]"'
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())
            # When validation_freq > 1, `val_` keys are not in first epoch logs
            # Add the `val_` keys so that its part of the fieldnames of writer.
            val_keys_found = False
            for key in self.keys:
                if key.startswith("val_"):
                    val_keys_found = True
                    break
            if not val_keys_found:
                self.keys.extend(["val_" + k for k in self.keys])

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update((key, handle_value(logs.get(key, "NA"))) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
