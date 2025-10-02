import copy
import sys
import traceback
from io import StringIO

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


class PythonScript(Trainable):
    """The python code to transform a JSON object into another JSON object"""

    python_script: str = Field(
        description="The python script to transform a JSON object into another object"
    )


class PythonConsoleLog(DataModel):
    stdout: str = Field(description="The python console's stdout")
    stderr: str = Field(description="The python console's stderr")


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
    def transform(inputs: Dict[str, Any]) -> Dict[str, Any]:
        # TODO implement the code
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
            name=self.name + "_state",
        )

    async def execute(self, inputs, python_script):
        value = None
        stdout = ""
        stderr = ""

        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            # Create a local namespace with the inputs
            local_namespace = {"inputs": copy.deepcopy(inputs.get_json())}

            # Execute the entire script
            exec(python_script, local_namespace)

            # Look for the result variable in the namespace
            if "result" in local_namespace:
                result = local_namespace["result"]

                if result:
                    try:
                        jsonschema.validate(result, self.schema)
                    except ValidationError as validation_error:
                        stderr += f"Validation Error: {validation_error}\n"
                        result = None
            else:
                stderr += "Error: No 'result' variable found after script execution\n"
                result = None

        except Exception as e:
            stderr += f"Error: {str(e)}\n"
            stderr += traceback.format_exc()
            result = None
        finally:
            stdout = sys.stdout.getvalue()
            stderr += sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return result, stdout, stderr

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        python_script = self.state.get("python_script")
        result, stdout, stderr = await self.execute(inputs, python_script)
        if training:
            predictions = self.state.get("predictions")
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
