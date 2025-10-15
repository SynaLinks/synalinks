# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import copy
import sys
import traceback
from io import StringIO

import multiprocessing
from multiprocessing import Process, Queue

import jsonschema
from jsonschema import ValidationError

import asyncio

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import Trainable
from synalinks.src.modules.module import Module


class TimeoutException(Exception):
    """Exception raised when script execution times out"""
    pass


class PythonScript(Trainable):
    """The python code to transform a JSON object into another JSON object"""

    python_script: str = Field(
        description="The python script to transform a JSON object into another object"
    )


class PythonConsoleLog(DataModel):
    stdout: str = Field(description="The python console's stdout")
    stderr: str = Field(description="The python console's stderr")


def _execute_script_in_process(python_script, inputs_json, schema, result_queue):
    """Execute the script in a separate process.
    
    This function runs in a separate process and can be forcefully terminated.
    """
    result = None
    
    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    try:
        # Create a local namespace with the inputs
        local_namespace = {"inputs": copy.deepcopy(inputs_json)}
        
        # Execute the entire script
        exec(python_script, local_namespace)
        
        # Look for the result variable in the namespace
        if "result" in local_namespace:
            result = local_namespace["result"]
            
            if result:
                try:
                    jsonschema.validate(result, schema)
                except ValidationError as validation_error:
                    stdout = stdout_capture.getvalue()
                    stderr = stderr_capture.getvalue() + f"Validation Error: {validation_error}\n"
                    result_queue.put((None, stdout, stderr))
                    return
        else:
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue() + "Error: No 'result' variable found after script execution\n"
            result_queue.put((None, stdout, stderr))
            return
            
    except Exception as e:
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue() + f"Error: {str(e)}\n{traceback.format_exc()}"
        result_queue.put((None, stdout, stderr))
        return
    finally:
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    result_queue.put((result, stdout, stderr))


@synalinks_export(
    [
        "synalinks.modules.PythonSynthesis",
        "synalinks.PythonSynthesis",
    ]
)
class PythonSynthesis(Module):
    """A code Python code transformation on JSON data.

    **Note**: This module is **NOT** completly safe (yet) for business applications.
        Its is only provided for reseach purposes on program synthesis.
        Altought the code don't evolve during inference, so it can't be prompt injected.

    This module features a python code as trainable variable, allowing the optimizers
    to refine the code during the training loop based on iterative feedback and
    automatic selection of the best script.

    This module works **ONLY** with advanced optimizers (**NOT** the `RandomFewShot` optimizer).
    
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
        seed_scripts (list): Optional. A list of Python scripts to use as seed for the evolution.
            If not provided, create a seed from the default configuration.
        default_return_value (dict): Default return value.
        timeout (int): Maximum execution time in seconds. (Default 5 seconds).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        schema=None,
        data_model=None,
        python_script=None,
        seed_scripts=None,
        default_return_value=None,
        timeout=5,
        sandbox=False,
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
        self.timeout = timeout

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
        
    async def execute(self, inputs, python_script):
        """Execute the Python script with timeout using multiprocessing.
        """
        result_queue = Queue()
        
        process = Process(
            target=_execute_script_in_process,
            args=(python_script, inputs.get_json(), self.schema, result_queue)
        )
        process.start()
        
        start_time = asyncio.get_event_loop().time()
        timeout_remaining = self.timeout
        
        while process.is_alive() and timeout_remaining > 0:
            await asyncio.sleep(0.1)
            elapsed = asyncio.get_event_loop().time() - start_time
            timeout_remaining = self.timeout - elapsed
        
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            
            if process.is_alive():
                process.kill()
                process.join()
            
            return None, "", f"Timeout Error: Script execution exceeded {self.timeout} second(s)\n"
        
        process.join()
        
        if not result_queue.empty():
            result, stdout, stderr = result_queue.get()
            return result, stdout, stderr
        else:
            return None, "", "Error: Process terminated unexpectedly\n"

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        python_script = self.state.get("python_script")
        result, stdout, stderr = await self.execute(inputs, python_script)
        if training:
            predictions = self.state.get("current_predictions")
            if result:
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
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)