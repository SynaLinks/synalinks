# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

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
        inputs = synalinks.Input(data_model=Document)
        summary = await synalinks.ChainOfThought(
            data_model=Summary,
            language_model=language_model,
        )(inputs)
        masked_summary = await synalinks.OutMask(
            # remove the thinking field
            mask=["thinking"],
        )(summary)

        program = synalinks.Program(
            inputs=inputs,
            outputs=masked_summary,
            name="summary_generator",
            description="Generate a summary from a document",
        )

    asyncio.run(main())
    ```

    Args:
        mask (list): The list of keys to remove.
        pattern (str): Optional. A regex; the fields whose key matches it are
            removed. At least one of `mask` or `pattern` is required.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        *,
        mask=None,
        pattern=None,
        name=None,
        description=None,
        trainable=False,
    ):
        if not (mask or pattern) or (mask and not isinstance(mask, list)):
            raise ValueError(
                "`mask` parameter should be a list of fields to remove, "
                "or a `pattern` should be given"
            )
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.mask = mask
        self.pattern = pattern

    async def call(self, inputs):
        outputs = tree.map_structure(
            lambda x: (
                x.out_mask(mask=self.mask, pattern=self.pattern) if x is not None else x
            ),
            inputs,
        )
        return outputs
