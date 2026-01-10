# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import utils
from synalinks.src.api_export import synalinks_export


@synalinks_export("synalinks.hooks.Hook")
class Hook:
    """Base hook class used to build new hooks.

    Hooks are callback-like objects that can be passed to the module's configuration, they
    intercept the `__call__()` method of the modules.

    They can be used to log the inputs/outputs of the modules as well as for
    streaming data or for observability.

    Attributes:
        params (dict): Hook parameters
            (eg. endpoint, log-level ...).
        module (Module): Instance of `Module`.
            Reference of the module being monitored.
    """

    def __init__(self):
        self.params = None
        self._module = None

    def set_params(self, params):
        self.params = params

    def set_module(self, module):
        self._module = module

    @property
    def module(self):
        return self._module

    @utils.default
    def on_call_begin(
        self,
        call_id,
        parent_call_id=None,
        inputs=None,
        kwargs=None,
    ):
        """Called at the beginning of the module execution.

        Args:
            call_id (str): The id of the module's call
            inputs (SymbolicDataModel | JsonDataModel | DataModel | list | dict | tuple):
                The module's inputs. The outputs can be data models or lists,
                dicts or tuples of data models.
            kwargs (dict): The keyword arguments passed to the module's call.
        """
        pass

    @utils.default
    def on_call_end(
        self,
        call_id,
        parent_call_id=None,
        outputs=None,
        exception=None,
    ):
        """Called at the end of the module execution.

        Args:
            call_id (str): The id of the module's call
            outputs (SymbolicDataModel | JsonDataModel | DataModel | list | dict | tuple):
                The module's outputs. The outputs can be data models or lists,
                dicts or tuples of data models.
            exception (str): Exception message if any.
        """
        pass
