# Modified from: keras/src/metrics/regression_metrics.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.metrics.reduction_metrics import MeanMetricWrapper
from synalinks.src.rewards.cosine_similarity import cosine_similarity
from synalinks.src.saving import serialization_lib


@synalinks_export("synalinks.metrics.CosineSimilarity")
class CosineSimilarity(MeanMetricWrapper):
    """Computes the cosine similarity between the labels and predictions.

    Formula:

    ```python
    metric = (sum(l2_norm(y_true) * l2_norm(y_pred))+1) / 2
    ```

    The formula is similar to the classic cosine similarity used in deep learning,
    but scaled to [0.0, 1.0] and adjusted to have a reward that tend
    towards 1.0 if the two objects are similar (and 0.0 otherwise).

    Args:
        embedding_model (EmbeddingModel): The embedding model to use to compute the
            cosine similarity.
        axis (int): (Optional) Defaults to `-1`. The dimension along which the cosine
            similarity is computed.
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep to compute the metric.
        out_mask (list): (Optional) list of keys to remove to compute the metric.
        in_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are dropped (combined with ``out_mask`` via OR).
    """

    def __init__(
        self,
        name="cosine_similarity",
        axis=-1,
        embedding_model=None,
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            fn=cosine_similarity,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
        self.axis = axis
        self.embedding_model = embedding_model
        self._fn_kwargs = {"axis": axis, "embedding_model": embedding_model}

    def get_config(self):
        config = {
            "axis": self.axis,
            "name": self.name,
            "in_mask": self.in_mask,
            "out_mask": self.out_mask,
            "in_mask_pattern": self.in_mask_pattern,
            "out_mask_pattern": self.out_mask_pattern,
        }
        embedding_model_config = {
            "embedding_model": serialization_lib.serialize_synalinks_object(
                self.embedding_model
            )
        }
        return {**embedding_model_config, **config}

    @classmethod
    def from_config(cls, config):
        embedding_model = serialization_lib.deserialize_synalinks_object(
            config.pop("embedding_model")
        )
        return cls(embedding_model=embedding_model, **config)
