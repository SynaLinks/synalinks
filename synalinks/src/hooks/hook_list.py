# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)


from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import is_observability_enabled
from synalinks.src.hooks.hook import Hook
from synalinks.src.hooks.logger import Logger
from synalinks.src.hooks.monitor import Monitor


@synalinks_export("synalinks.hooks.HookList")
class HookList(Hook):
    """Container abstracting a list of hooks.

    Container for `Hook` instances.

    This object wraps a list of `Hook` instances, making it possible
    to call them all at once via a single endpoint
    (e.g. `hook_list.on_module_begin(...)`).

    Args:
        hooks (list): List of `Hook` instances.
        add_logger (bool): Whether a `Logger` hook should be added, if one
            does not already exist in the `hooks` list.
        add_observability (bool): Whether a `Observability` hook should be added, if one
            does not already exist in the `hooks` list.
        module (Module): The `Module` these callbacks are used with.
        **params (keyword arguments): If provided, parameters will be passed to each
            `Hook` via `Hook.set_params`.
    """

    def __init__(
        self,
        hooks=None,
        add_logger=True,
        module=None,
        **params,
    ):
        self.hooks = tree.flatten(hooks) if hooks else []
        self._add_default_hooks(add_logger)
        self.set_module(module)
        self.set_params(params)

    def add_hook(self, hook):
        self.hooks.append(hook)

    def set_params(self, params):
        self.params = params
        if params:
            for hook in self.hooks:
                hook.set_params(params)

    def _add_default_hooks(self, add_logger):
        self._logger = None
        self._monitor = None

        for hook in self.hooks:
            if isinstance(hook, Logger):
                self._logger = hook

        if self._logger is None and add_logger:
            self._logger = Logger()
            self.hooks.append(self._logger)

        if self._monitor is None and is_observability_enabled():
            self._monitor = Monitor()
            self.hooks.append(self._monitor)

    def set_module(self, module):
        if not module:
            return
        super().set_module(module)
        for hook in self.hooks:
            hook.set_module(module)

    def on_call_begin(
        self,
        call_id,
        inputs=None,
    ):
        for hook in self.hooks:
            hook.on_call_begin(
                call_id,
                inputs=inputs,
            )

    def on_call_end(
        self,
        call_id,
        outputs=None,
        exception=None,
    ):
        for hook in self.hooks:
            hook.on_call_end(
                call_id,
                outputs=outputs,
                exception=exception,
            )
