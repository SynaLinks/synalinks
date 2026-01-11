# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from datetime import datetime

from synalinks.src import ops
from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.modules.module import Module
from synalinks.src.utils.async_utils import run_maybe_nested


class Timestamp(DataModel):
    """A data model containing a creation timestamp."""

    created_at: datetime


@synalinks_export(
    [
        "synalinks.modules.StampKnowledge",
        "synalinks.StampKnowledge",
    ]
)
class StampKnowledge(Module):
    """Stamps data models with a creation timestamp.

    This module adds a `created_at` field containing the current datetime
    to each input data model using a logical AND operation.

    Args:
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        name=None,
        description=None,
        trainable=False,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )

    async def _stamp(self, data_model):
        timestamp = JsonDataModel(
            json={"created_at": datetime.now().isoformat()},
            schema=Timestamp.get_schema(),
            name="timestamp_" + data_model.name,
        )
        return await ops.logical_and(
            data_model,
            timestamp,
            name="stamped_" + data_model.name,
        )

    async def call(self, inputs):
        if not inputs:
            return None
        return tree.map_structure(
            lambda x: run_maybe_nested(self._stamp(x)),
            inputs,
        )

    async def compute_output_spec(self, inputs):
        def _stamp_spec(x):
            timestamp = SymbolicDataModel(
                schema=Timestamp.get_schema(),
                name="timestamp_" + x.name,
            )
            return run_maybe_nested(
                ops.logical_and(
                    x,
                    timestamp,
                    name="stamped_" + x.name,
                )
            )

        return tree.map_structure(_stamp_spec, inputs)

    def get_config(self):
        return {
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
