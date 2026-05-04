# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import jsonschema
from jsonschema import ValidationError

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import Trainable
from synalinks.src.modules.module import Module
from synalinks.src.sandboxes.monty_sandbox import MontySandbox
from synalinks.src.saving import serialization_lib
from synalinks.src.saving.object_registration import get_registered_name
from synalinks.src.saving.object_registration import get_registered_object


class PythonScript(Trainable):
    """The python code to transform a JSON object into another JSON object.

    The script is executed inside the Monty
    (https://github.com/pydantic/monty) sandboxed Python interpreter, which
    implements only a subset of Python. Scripts must observe the following
    constraints:

    - The input JSON object is exposed as a dict named ``inputs``; the script
      must assign the output JSON object to a variable named ``result`` before
      it ends.
    - Only this subset of the standard library is importable: ``sys``,
      ``os``, ``typing``, ``asyncio``, ``re``, ``datetime``, ``json``,
      ``math``, ``pathlib``. Notably, ``time``, ``random``, ``itertools``,
      ``collections``, ``functools`` and the rest of the stdlib are **not**
      available.
    - No third-party libraries can be imported (e.g. ``numpy``, ``pandas``,
      ``pydantic``).
    - ``class`` definitions and ``match`` statements are not supported; use
      functions and ``if``/``elif`` chains instead.
    - The host filesystem, environment variables and network are not
      reachable from the script. ``os``, ``sys`` and ``pathlib`` import but
      their dangerous surface is pruned or gated: ``open()``, ``os.system``,
      ``os.listdir``, ``os.environ``, ``os.path``, ``sys.argv`` and
      ``Path.read_text`` are all unavailable.
    - ``asyncio`` is also a stub: only ``asyncio.run`` and ``asyncio.gather``
      are exposed. There is no ``asyncio.sleep``, ``wait_for``, ``Future``,
      ``create_task`` or ``TaskGroup``, and no time primitives of any kind
      (``time`` is not importable either).
    - Tools bound to the module are exposed as **global async callables**
      under their tool name. They must be awaited inside an ``async def``
      and driven with ``asyncio.run(...)``. Every tool call returns a
      **dict**: a tool wrapping ``async def f(x) -> int`` yields
      ``{"result": <value>}``, a tool that already returns a dict yields
      that dict directly. For example, with a bound tool ``web_search``:

      ```python
      import asyncio

      async def main():
          hits = await web_search(query=inputs.get("q"))
          # hits is a dict — index the field you need
          return {"answer": hits["results"][0]["title"]}

      result = asyncio.run(main())
      ```

      Independent tool calls can be fanned out with ``asyncio.gather``.
      Calling a tool without ``await`` returns a coroutine object, not the
      real value.
    - Execution is bounded by the module's ``timeout`` and by Monty's memory
      limits; long-running or allocation-heavy scripts will be aborted.
    """

    python_script: str = Field(
        description=(
            "A Python script that transforms a JSON input into a JSON "
            "output. The script reads the input from a dict named "
            "`inputs` and must assign the output dict to a variable "
            "named `result` before it ends. Exact language and stdlib "
            "constraints depend on the active sandbox."
        ),
    )


class PythonConsoleLog(DataModel):
    stdout: str = Field(description="The python console's stdout")
    stderr: str = Field(description="The python console's stderr")


def _adapt_tool_for_sandbox(tool):
    """Route a sandbox tool call through the `Tool` Module (preserving
    observability/retry) and return a plain dict the sandbox can marshal back.
    """

    async def adapter(**kwargs):
        result = await tool(**kwargs)
        if result is None:
            return None
        if hasattr(result, "get_json"):
            return result.get_json()
        return result

    return adapter


def _relabel_error(error: str) -> str:
    """Translate a MontyError class name into the module's legacy label.

    ``MontySandbox.run`` surfaces errors as ``"MontyXxxError: ..."``; the
    historical contract of this module is ``"Syntax Error: ..."`` /
    ``"Runtime Error: ..."`` (matching the rest of the stdout/stderr).
    """
    head, _, rest = error.partition(":")
    label = "Syntax Error" if head == "MontySyntaxError" else "Runtime Error"
    return f"{label}:{rest}"


async def _run_script(
    python_script,
    inputs_json,
    schema,
    timeout,
    tools,
    sandbox=None,
    sandbox_type=MontySandbox,
):
    """Execute the script inside a ``MontySandbox``.

    The script contract is unchanged: assign the output to a variable named
    ``result``. A trailing ``result`` expression is appended so the
    sandbox's last-expression return captures that value.

    ``tools`` is a ``{name: Tool}`` mapping; each ``Tool`` is exposed as a
    global async callable inside the sandbox and must be ``await``-ed.

    ``sandbox`` is optional. When ``None``, a fresh ``MontySandbox`` is
    built for just this call — the normal case, giving every input an
    independent namespace. When supplied, the caller owns the sandbox
    and state persists across calls (useful at training time to explore
    scripts that build on each other).
    """
    code = python_script + "\nresult\n"
    if sandbox is None:
        sandbox = sandbox_type(timeout=timeout)
    external_functions = (
        {name: _adapt_tool_for_sandbox(tool) for name, tool in tools.items()}
        if tools
        else None
    )

    execution = await sandbox.run(
        code,
        inputs={"inputs": inputs_json},
        external_functions=external_functions,
    )

    if execution.error:
        relabelled = _relabel_error(execution.error)
        # Syntax errors happen at compile time — no script output precedes them.
        if execution.error.startswith("MontySyntaxError"):
            return None, "", f"{relabelled}\n"
        return (
            None,
            execution.stdout,
            execution.stderr + f"{relabelled}\n",
        )

    result = execution.result
    if not result:
        return None, execution.stdout, execution.stderr

    try:
        jsonschema.validate(result, schema)
    except ValidationError as validation_error:
        return (
            None,
            execution.stdout,
            execution.stderr + f"Validation Error: {validation_error}\n",
        )

    return result, execution.stdout, execution.stderr


@synalinks_export(
    [
        "synalinks.modules.PythonSynthesis",
        "synalinks.PythonSynthesis",
    ]
)
class PythonSynthesis(Module):
    """A code Python code transformation on JSON data.

    The script runs inside the `Monty <https://github.com/pydantic/monty>`_
    sandboxed Python interpreter: the host filesystem, environment and network
    are unreachable from the script. Monty only supports a subset of Python
    (no third-party libraries, limited standard library, no class or match
    statements), so the generated script must stay within what Monty can
    execute.

    This module features a python code as trainable variable, allowing the optimizers
    to refine the code during the training loop based on iterative feedback and
    automatic selection of the best script.

    This module works **ONLY** with advanced optimizers (**NOT** the
    `RandomFewShot` optimizer).

    The module executes the entire Python script and expects the result to be stored
    in a variable named 'result' at the end of execution.
    
    Example:
    
    ```python
    import synalinks
    import asyncio
    
    default_python_script = \\
    \"\"\"
    def transform(inputs):
        # TODO implement the code to transform the input grid into the output grid
        return {"output_grid": inputs.get("input_grid")}
        
    result = transform(inputs)
    \"\"\"
    
    async def main():
        inputs = synalinks.Input(
            data_model=synalinks.datasets.arcagi.get_input_data_model(),
        )
        outputs = await synalinks.PythonSynthesis(
            data_model=synalinks.datasets.arcagi.get_output_data_model()
            python_script=default_python_script,
            default_return_value={"output_grid": [[]]},
        )(inputs)
        
        program = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="python_script_synthesis",
            description="A program to solve ARCAGI with python code",
        )
    ```
    
    If you want to explore the future of neuro-symbolic self-evolving systems, contact us.
    While these systems are not "hard" to code thanks to Synalinks, they requires 
    technical knowledge and a deep understanding of multiple AI paradigm.

    Args:
        schema (dict): The target JSON schema.
            If not provided use the `data_model` to infer it.
        data_model (DataModel | SymbolicDataModel | JsonDataModel): The target data
            model for structured output.
        python_script (str): The default Python script.
        seed_scripts (list): Optional. A list of Python scripts to use as seed
            for the evolution. If not provided, create a seed from the default
            configuration.
        default_return_value (dict): Default return value.
        return_python_script (bool): Wether or not to return the python script for
            evaluation. (Default to False).
        timeout (int): Maximum execution time in seconds. (Default 5 seconds).
        tools (list): Optional. A list of `Tool` (or MCP tools) exposed to the
            script as global async callables. Because `Tool`s are async,
            scripts must call them inside an `async def` and `await` them
            (see the ``PythonScript`` docs). Passing `None` or an empty list
            means no tools are bound.

            **Naming gotcha**: each tool is registered inside the sandbox
            under ``tool.name``, which is ``tool._func.__name__``. So
            ``Tool(_my_helper)`` registers as ``_my_helper`` (underscore
            preserved) and the script must call ``await _my_helper(...)``.
            Name your tool functions exactly as you want them to appear
            inside the generated script — rename the function, don't rely
            on an alias.
        sandbox_type (type): Optional. The ``Sandbox`` subclass used to
            build a fresh sandbox per call when no ``sandbox`` is
            injected. Defaults to ``MontySandbox``. Any ``Sandbox``
            subclass whose ``__init__`` accepts ``(timeout=..., name=...)``
            works; register custom subclasses with
            ``@register_synalinks_serializable`` so they round-trip
            through ``get_config`` / ``from_config``.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.

    ``call`` also accepts an optional ``sandbox`` kwarg. If given, the
    script executes inside the caller-supplied ``MontySandbox`` and any
    state (variables, imports, function defs) persists across successive
    calls — useful at training time when candidate scripts share
    cached state. When omitted, a fresh sandbox is built per call
    (independent execution, the normal runtime contract).
    """

    def __init__(
        self,
        *,
        schema=None,
        data_model=None,
        python_script=None,
        seed_scripts=None,
        default_return_value=None,
        return_python_script=False,
        timeout=5,
        tools=None,
        sandbox_type=None,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema

        if not python_script:
            raise ValueError("You should provide the `python_script` argument")
        self.python_script = python_script

        if not default_return_value:
            raise ValueError("You should provide the `default_return_value` argument")

        try:
            jsonschema.validate(default_return_value, self.schema)
        except ValidationError as e:
            raise ValueError(
                f"`default_return_value` parameter does not conform to schema: {e}"
            )

        self.default_return_value = default_return_value
        self.return_python_script = return_python_script
        self.timeout = timeout

        self.tools = {}
        if tools:
            for tool in tools:
                self.tools[tool.name] = tool

        # Which Sandbox subclass to instantiate when no caller-supplied
        # sandbox is passed to `call()`. Defaults to MontySandbox.
        self.sandbox_type = sandbox_type or MontySandbox

        if not seed_scripts:
            seed_scripts = []
        self.seed_scripts = seed_scripts

        seed_candidates = [
            {"python_script": seed_script} for seed_script in self.seed_scripts
        ]

        self.state = self.add_variable(
            initializer=PythonScript(
                python_script=self.python_script,
                seed_candidates=seed_candidates,
            ).get_json(),
            data_model=PythonScript,
            name="state_" + self.name,
        )

    async def execute(self, inputs, python_script, sandbox=None):
        """Execute the Python script in the sandbox with a timeout."""
        return await _run_script(
            python_script,
            inputs.get_json(),
            self.schema,
            self.timeout,
            self.tools,
            sandbox=sandbox,
            sandbox_type=self.sandbox_type,
        )

    async def call(self, inputs, training=False, sandbox=None):
        if not inputs:
            return None
        python_script = self.state.get("python_script")
        result, stdout, stderr = await self.execute(
            inputs, python_script, sandbox=sandbox
        )
        if training:
            predictions = self.state.get("current_predictions")
            if result:
                if self.return_python_script:
                    predictions.append(
                        {
                            "inputs": {
                                **inputs.get_json(),
                            },
                            "outputs": {
                                "python_script": python_script,
                                **result,
                                "stdout": stdout,
                                "stderr": stderr,
                            },
                            "reward": None,
                        }
                    )
                else:
                    predictions.append(
                        {
                            "inputs": {
                                **inputs.get_json(),
                            },
                            "outputs": {
                                **result,
                                "stdout": stdout,
                                "stderr": stderr,
                            },
                            "reward": None,
                        }
                    )
            else:
                if self.return_python_script:
                    predictions.append(
                        {
                            "inputs": {
                                **inputs.get_json(),
                            },
                            "outputs": {
                                "python_script": python_script,
                                "stdout": stdout,
                                "stderr": stderr,
                            },
                            "reward": None,
                        }
                    )
                else:
                    predictions.append(
                        {
                            "inputs": {
                                **inputs.get_json(),
                            },
                            "outputs": {
                                "stdout": stdout,
                                "stderr": stderr,
                            },
                            "reward": None,
                        }
                    )
        if result:
            if self.return_python_script:
                return JsonDataModel(
                    json={
                        "python_script": python_script,
                        **result,
                        "stdout": stdout,
                        "stderr": stderr,
                    },
                    schema=self.schema,
                    name=self.name,
                )
            else:
                return JsonDataModel(
                    json={
                        **result,
                        "stdout": stdout,
                        "stderr": stderr,
                    },
                    schema=self.schema,
                    name=self.name,
                )
        else:
            if self.return_python_script:
                return JsonDataModel(
                    json={
                        "python_script": python_script,
                        **self.default_return_value,
                        "stdout": stdout,
                        "stderr": stderr,
                    },
                    schema=self.schema,
                    name=self.name,
                )
            else:
                return JsonDataModel(
                    json={
                        **self.default_return_value,
                        "stdout": stdout,
                        "stderr": stderr,
                    },
                    schema=self.schema,
                    name=self.name,
                )

    async def compute_output_spec(self, inputs, training=False):
        if self.return_python_script:
            return await ops.concat(
                await ops.out_mask(
                    PythonScript.to_symbolic_data_model(),
                    mask=list(Trainable.keys()),
                    name="python_script_masked_" + self.name,
                ),
                await ops.concat(
                    SymbolicDataModel(schema=self.schema),
                    PythonConsoleLog,
                    name="python_logs_" + self.name,
                ),
                name=self.name,
            )
        else:
            return await ops.concat(
                SymbolicDataModel(schema=self.schema),
                PythonConsoleLog,
                name=self.name,
            )

    def get_config(self):
        config = {
            "schema": self.schema,
            "python_script": self.python_script,
            "seed_scripts": self.seed_scripts,
            "default_return_value": self.default_return_value,
            "return_python_script": self.return_python_script,
            "timeout": self.timeout,
            "sandbox_type": get_registered_name(self.sandbox_type),
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        tools_config = {
            "tools": [
                serialization_lib.serialize_synalinks_object(tool)
                for tool in self.tools.values()
            ]
        }
        return {**config, **tools_config}

    @classmethod
    def from_config(cls, config):
        tools = [
            serialization_lib.deserialize_synalinks_object(tool)
            for tool in config.pop("tools", [])
        ]
        sandbox_type_name = config.pop("sandbox_type", None)
        sandbox_type = (
            get_registered_object(sandbox_type_name) if sandbox_type_name else None
        )
        return cls(tools=tools or None, sandbox_type=sandbox_type, **config)
