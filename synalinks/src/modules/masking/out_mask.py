# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.modules.module import Module


@synalinks_export(
    [
        "synalinks.OutMask",
        "synalinks.modules.OutMask",
    ]
)
class OutMask(Module):
    """A module to remove specific fields of the given data models

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
        masked_summary = synalinks.OutMask(
            # remove the thinking field from the chain of thought
            mask=["thinking"],
        )(summary)

        program = Program(
            inputs=inputs,
            outputs=masked_summary,
            name="summary_generator",
            description="Generate a summary from a document",
        )
    ```

    Args:
        mask (list): The list of keys to remove.
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
            raise ValueError("`mask` parameter should be a list of fields to remove")
        super().__init__(
            name=name,
            description=description,
        )
        self.mask = mask

    async def call(self, inputs):
        outputs = tree.map_structure(
            lambda x: x.out_mask(mask=self.mask),
            inputs,
        )
        return outputs
