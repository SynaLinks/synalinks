## `InMask`

Bases: `Module`

A module to keep specific fields of the given data models

Example:

```
import synalinks
import asyncio

language_model = synalinks.LanguageModel(
    model="ollama/mistral",
)

class Document(synalinks.DataModel):
    title: str = synalinks.Field(
        description="The title of the document",
    )
    text: str = synalinks.Field(
        description="The content of the document",
    )

class Summary(synalinks.DataModel):
    summary: str = synalinks.Field(
        description="the concise summary of the document",
    )

async def main():
    inputs = Input(data_model=Document)
    summary = synalinks.ChainOfThought(
        data_model=Summary,
        language_model=language_model,
    )(inputs)
    masked_summary = synalinks.InMask(
        # remove the thinking field from the chain of thought
        # by keeping only the summary
        mask=["summary"],
    )(summary)

    program = Program(
        inputs=inputs,
        outputs=masked_summary,
        name="summary_generator",
        description="Generate a summary from a document",
    )
```

Parameters:

| Name          | Type   | Description                                         | Default |
| ------------- | ------ | --------------------------------------------------- | ------- |
| `mask`        | `list` | The list of keys to keep.                           | `None`  |
| `name`        | `str`  | Optional. The name of the module.                   | `None`  |
| `description` | `str`  | Optional. The description of the module.            | `None`  |
| `trainable`   | `bool` | Whether the module's variables should be trainable. | `False` |

Source code in `synalinks/src/modules/masking/in_mask.py`

````
@synalinks_export(
    [
        "synalinks.InMask",
        "synalinks.modules.InMask",
    ]
)
class InMask(Module):
    """A module to keep specific fields of the given data models

    Example:

    ```python
    import synalinks
    import asyncio

    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    class Document(synalinks.DataModel):
        title: str = synalinks.Field(
            description="The title of the document",
        )
        text: str = synalinks.Field(
            description="The content of the document",
        )

    class Summary(synalinks.DataModel):
        summary: str = synalinks.Field(
            description="the concise summary of the document",
        )

    async def main():
        inputs = Input(data_model=Document)
        summary = synalinks.ChainOfThought(
            data_model=Summary,
            language_model=language_model,
        )(inputs)
        masked_summary = synalinks.InMask(
            # remove the thinking field from the chain of thought
            # by keeping only the summary
            mask=["summary"],
        )(summary)

        program = Program(
            inputs=inputs,
            outputs=masked_summary,
            name="summary_generator",
            description="Generate a summary from a document",
        )

    ```

    Args:
        mask (list): The list of keys to keep.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        mask=None,
        name=None,
        description=None,
        trainable=False,
    ):
        if not mask or not isinstance(mask, list):
            raise ValueError("`mask` parameter should be a list of fields to keep")
        super().__init__(
            name=name,
            description=description,
        )
        self.mask = mask

    async def call(self, inputs):
        outputs = tree.map_structure(
            lambda x: x.in_mask(mask=self.mask),
            inputs,
        )
        return outputs
````
