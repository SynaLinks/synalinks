# Modified from: keras/src/backend/common/stateless_scope.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common import global_state


@synalinks_export("synalinks.StatelessScope")
class StatelessScope:
    """Scope to prevent any update to synalinks Variables.

    The values of variables to be used inside the scope
    should be passed via the `state_mapping` argument, a
    list of tuples `(k, v)` where `k` is a `Variable`
    and `v` is the intended value for this variable
    (a backend data model).

    Updated values can be collected on scope exit via
    `value = scope.get_current_value(variable)`. No updates
    will be applied in-place to any variables for the duration
    of the scope.

    Example:

    ```python
    state_mapping = [(k, ops.ones(k.shape, k.dtype)) for k in program.variables]
    with synalinks.StatelessScope(state_mapping) as scope:
        outputs = program.some_function(inputs)

    # All program variables remain unchanged. Their new values can be
    # collected via:
    for k in program.variables:
        new_value = scope.get_current_value(k)
        print(f"New value for {k}: {new_value})
    ```
    """

    def __init__(
        self,
        state_mapping=None,
        collect_rewards=False,
        initialize_variables=True,
    ):
        from synalinks.src import backend
        from synalinks.src.backend.common.variables import Variable

        self.collect_rewards = collect_rewards
        self.initialize_variables = initialize_variables
        self.rewards = []
        self.state_mapping = {}
        state_mapping = state_mapping or {}
        for k, v in state_mapping:
            if not isinstance(k, Variable):
                raise ValueError(
                    "Invalid reference variable in StatelessScope: "
                    "all keys in argument `mapping` must be Variable "
                    f"instances. Received instead: {k}"
                )
            v = backend.convert_to_json_data_model(v, dtype=k.dtype)
            if not backend.is_schema_equal(k.get_schema(), v.get_schema()):
                raise ValueError(
                    "Invalid variable value in StatelessScope: "
                    "all values in argument `mapping` must be data models with "
                    "a schema that matches the corresponding variable schema. "
                    f"For variable {k}, received invalid value {v} with schema "
                    f"{v.prettify_schema()}."
                )
            self.state_mapping[id(k)] = v

    def __enter__(self):
        self.original_scope = get_stateless_scope()
        global_state.set_global_attribute("stateless_scope", self)
        return self

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_update(self, update):
        variable, value = update
        self.state_mapping[id(variable)] = value

    def get_current_value(self, variable):
        return self.state_mapping.get(id(variable), None)

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute("stateless_scope", self.original_scope)
        if self.original_scope is None and self.initialize_variables:
            # We're back in eager scope;
            # if any variables were created within the stateless
            # scope, we initialize them here.
            from synalinks.src.backend.common.variables import initialize_all_variables

            initialize_all_variables()


def in_stateless_scope():
    return global_state.get_global_attribute("stateless_scope") is not None


def get_stateless_scope():
    return global_state.get_global_attribute("stateless_scope")
