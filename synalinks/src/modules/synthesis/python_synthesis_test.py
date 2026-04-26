from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.datasets.arcagi import get_arcagi1_evaluation_task_names
from synalinks.src.datasets.arcagi import get_input_data_model
from synalinks.src.datasets.arcagi import get_output_data_model
from synalinks.src.datasets.arcagi import load_data
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.synthesis.python_synthesis import PythonSynthesis
from synalinks.src.programs.program import Program
from synalinks.src.sandboxes.monty_sandbox import MontySandbox


class _IntIn(DataModel):
    value: int


class _IntOut(DataModel):
    doubled: int


async def triple(x: int) -> int:
    """Triple an integer.

    Args:
        x (int): the integer to triple.
    """
    return x * 3


async def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a (int): left operand.
        b (int): right operand.
    """
    return a + b


class PythonSynthesisTest(testing.TestCase):
    async def test_default_synthesis(self):
        task_names = get_arcagi1_evaluation_task_names()
        task_name = task_names[0]

        default_python_script = """
def transform(inputs):
    # TODO implement the code
    return {"output_grid": inputs.get("input_grid")}
    
result = transform(inputs)
"""
        inputs = Input(data_model=get_input_data_model())
        outputs = await PythonSynthesis(
            data_model=get_output_data_model(),
            python_script=default_python_script,
            default_return_value={"output_grid": [[]]},
        )(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
            name=f"arcagi_task_{task_name}",
            description=f"A python function to solve ARC-AGI task {task_name}",
        )

        (x_train, _), (_, _) = load_data(task_name=task_name)

        x = x_train[0]

        result = await program(x)
        self.assertEqual(result.get("output_grid"), x.get("input_grid"))

    async def test_program_synthesis_with_timeout(self):
        task_names = get_arcagi1_evaluation_task_names()
        task_name = task_names[0]

        default_python_script = """
def transform(inputs):
    while True:
        pass
    # TODO implement the code to transform an input grid into an output grid
    return {"output_grid": inputs.get("input_grid")}

result = transform(inputs)
"""
        inputs = Input(data_model=get_input_data_model())
        outputs = await PythonSynthesis(
            data_model=get_output_data_model(),
            python_script=default_python_script,
            default_return_value={"output_grid": [[]]},
        )(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
            name=f"arcagi_task_{task_name}",
            description=f"A python function to solve ARC-AGI task {task_name}",
        )

        (x_train, _), (_, _) = load_data(task_name=task_name)

        x = x_train[0]

        result = await program(x)
        self.assertEqual(result.get("output_grid"), [[]])

    async def _run_simple(self, python_script, **kwargs):
        """Build a tiny _IntIn -> _IntOut program and run it once with value=3."""
        inputs = Input(data_model=_IntIn)
        outputs = await PythonSynthesis(
            data_model=_IntOut,
            python_script=python_script,
            default_return_value={"doubled": -1},
            **kwargs,
        )(inputs)
        program = Program(inputs=inputs, outputs=outputs)
        return await program(_IntIn(value=3))

    async def test_simple_happy_path(self):
        script = """
result = {"doubled": inputs.get("value") * 2}
"""
        result = await self._run_simple(script)
        self.assertEqual(result.get("doubled"), 6)
        self.assertEqual(result.get("stderr"), "")

    async def test_syntax_error_falls_back_to_default(self):
        script = "def broken(: pass\nresult = {}\n"
        result = await self._run_simple(script)
        self.assertEqual(result.get("doubled"), -1)
        self.assertIn("Syntax Error", result.get("stderr"))

    async def test_runtime_error_falls_back_to_default(self):
        script = """
result = {"doubled": 1 / 0}
"""
        result = await self._run_simple(script)
        self.assertEqual(result.get("doubled"), -1)
        self.assertIn("Runtime Error", result.get("stderr"))
        self.assertIn("ZeroDivisionError", result.get("stderr"))

    async def test_missing_result_variable_falls_back_to_default(self):
        script = """
x = inputs.get("value") * 2
"""
        result = await self._run_simple(script)
        self.assertEqual(result.get("doubled"), -1)
        self.assertIn("Runtime Error", result.get("stderr"))
        self.assertIn("NameError", result.get("stderr"))

    async def test_schema_validation_failure_falls_back_to_default(self):
        script = """
result = {"doubled": "not an int"}
"""
        result = await self._run_simple(script)
        self.assertEqual(result.get("doubled"), -1)
        self.assertIn("Validation Error", result.get("stderr"))

    async def test_stdout_is_captured(self):
        script = """
print("hello from monty")
result = {"doubled": inputs.get("value") * 2}
"""
        result = await self._run_simple(script)
        self.assertEqual(result.get("doubled"), 6)
        self.assertIn("hello from monty", result.get("stdout"))

    async def test_return_python_script_true_includes_script(self):
        script = """
result = {"doubled": inputs.get("value") * 2}
"""
        result = await self._run_simple(script, return_python_script=True)
        self.assertEqual(result.get("doubled"), 6)
        self.assertEqual(result.get("python_script"), script)

    async def test_invalid_default_return_value_raises(self):
        with self.assertRaises(ValueError):
            PythonSynthesis(
                data_model=_IntOut,
                python_script="result = {}\n",
                default_return_value={"doubled": "not an int"},
            )

    async def test_missing_python_script_raises(self):
        with self.assertRaises(ValueError):
            PythonSynthesis(
                data_model=_IntOut,
                default_return_value={"doubled": -1},
            )

    async def test_missing_default_return_value_raises(self):
        with self.assertRaises(ValueError):
            PythonSynthesis(
                data_model=_IntOut,
                python_script="result = {}\n",
            )

    async def test_tool_call_happy_path(self):
        # Tool(triple) wraps the int return in `{"result": value}`, so the
        # script has to index that field before using it.
        script = """
import asyncio

async def main():
    tripled = (await triple(x=inputs.get("value")))["result"]
    return {"doubled": tripled}

result = asyncio.run(main())
"""
        inputs = Input(data_model=_IntIn)
        outputs = await PythonSynthesis(
            data_model=_IntOut,
            python_script=script,
            default_return_value={"doubled": -1},
            tools=[Tool(triple)],
        )(inputs)
        program = Program(inputs=inputs, outputs=outputs)

        result = await program(_IntIn(value=4))
        self.assertEqual(result.get("doubled"), 12)
        self.assertEqual(result.get("stderr"), "")

    async def test_multiple_tools_bound_by_name(self):
        script = """
import asyncio

async def main():
    s = (await add(a=inputs.get("value"), b=inputs.get("value")))["result"]
    t = (await triple(x=s))["result"]
    return {"doubled": t}

result = asyncio.run(main())
"""
        inputs = Input(data_model=_IntIn)
        outputs = await PythonSynthesis(
            data_model=_IntOut,
            python_script=script,
            default_return_value={"doubled": -1},
            tools=[Tool(triple), Tool(add)],
        )(inputs)
        program = Program(inputs=inputs, outputs=outputs)

        # (4 + 4) * 3 = 24
        result = await program(_IntIn(value=4))
        self.assertEqual(result.get("doubled"), 24)

    async def test_tool_called_without_await_is_a_runtime_error(self):
        # Forgetting to `await` an async tool yields a coroutine object, not
        # the value — monty rejects it when marshalling, falling back to the
        # default.
        script = """
result = {"doubled": triple(x=inputs.get("value"))}
"""
        inputs = Input(data_model=_IntIn)
        outputs = await PythonSynthesis(
            data_model=_IntOut,
            python_script=script,
            default_return_value={"doubled": -1},
            tools=[Tool(triple)],
        )(inputs)
        program = Program(inputs=inputs, outputs=outputs)

        result = await program(_IntIn(value=4))
        self.assertEqual(result.get("doubled"), -1)
        self.assertNotEqual(result.get("stderr"), "")

    async def test_injected_sandbox_preserves_state_across_calls(self):
        """When the caller hands in a ``MontySandbox``, variables defined
        by one script are visible to the next. Default (no sandbox)
        behaviour is fresh state per call, exercised elsewhere."""
        # First script seeds `memo`; second script references it. Without
        # a shared sandbox, the second run would error because `memo` is
        # not defined.
        seed_script = """
memo = {"count": inputs.get("value")}
result = {"doubled": memo["count"] * 2}
"""
        followup_script = """
memo["count"] += inputs.get("value")
result = {"doubled": memo["count"] * 2}
"""

        agent_seed = PythonSynthesis(
            data_model=_IntOut,
            python_script=seed_script,
            default_return_value={"doubled": -1},
        )
        agent_followup = PythonSynthesis(
            data_model=_IntOut,
            python_script=followup_script,
            default_return_value={"doubled": -1},
        )

        sandbox = MontySandbox()
        r1 = await agent_seed(_IntIn(value=3), sandbox=sandbox)
        self.assertEqual(r1.get("doubled"), 6)
        # Second agent sees `memo` because the sandbox persisted.
        r2 = await agent_followup(_IntIn(value=4), sandbox=sandbox)
        self.assertEqual(r2.get("doubled"), 14)  # (3 + 4) * 2

    async def test_default_sandbox_is_fresh_each_call(self):
        """Without an injected sandbox, state from a previous call must
        not leak into the next one."""
        seed_script = """
memo = {"count": inputs.get("value")}
result = {"doubled": memo["count"] * 2}
"""
        followup_script = """
memo["count"] += inputs.get("value")
result = {"doubled": memo["count"] * 2}
"""

        agent_seed = PythonSynthesis(
            data_model=_IntOut,
            python_script=seed_script,
            default_return_value={"doubled": -1},
        )
        agent_followup = PythonSynthesis(
            data_model=_IntOut,
            python_script=followup_script,
            default_return_value={"doubled": -1},
        )

        r1 = await agent_seed(_IntIn(value=3))
        self.assertEqual(r1.get("doubled"), 6)
        # No shared sandbox: the second call builds a fresh one and the
        # reference to `memo` errors — the module falls back to default.
        r2 = await agent_followup(_IntIn(value=4))
        self.assertEqual(r2.get("doubled"), -1)
        self.assertIn("Runtime Error", r2.get("stderr"))
