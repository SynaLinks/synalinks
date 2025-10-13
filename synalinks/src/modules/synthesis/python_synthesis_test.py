from synalinks.src import testing
from synalinks.src.datasets.arcagi import get_arcagi1_evaluation_task_names
from synalinks.src.datasets.arcagi import get_input_data_model
from synalinks.src.datasets.arcagi import get_output_data_model
from synalinks.src.datasets.arcagi import load_data
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.synthesis.python_synthesis import PythonSynthesis
from synalinks.src.programs.program import Program


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

        (x_train, y_train), (x_test, y_test) = load_data(task_name=task_name)

        x = x_train[0]
        y = y_train[0]

        result = await program(x)
        self.assertEqual(result.get("output_grid"), x.get("input_grid"))

    async def test_program_synthesis_with_timeout(self):
        task_names = get_arcagi1_evaluation_task_names()
        task_name = task_names[0]

        default_python_script = """
import time

def transform(inputs):
    time.sleep(15)
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

        (x_train, y_train), (x_test, y_test) = load_data(task_name=task_name)

        x = x_train[0]
        y = y_train[0]

        result = await program(x)
        self.assertEqual(result.get("output_grid"), [[]])