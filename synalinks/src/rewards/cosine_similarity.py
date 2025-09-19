# Modified from: keras/src/losses/losses.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common import numpy as np
from synalinks.src.rewards.reward import Reward
from synalinks.src.rewards.reward import squeeze_or_expand_to_same_rank
from synalinks.src.rewards.reward_wrappers import RewardFunctionWrapper


@synalinks_export("synalinks.rewards.cosine_similarity")
async def cosine_similarity(y_true, y_pred, embedding_model=None, axis=-1):
    """
    Computes the cosine similarity between `y_true` and `y_pred`.

    Formula:

    ```
    reward = (sum(l2_norm(y_true) * l2_norm(y_pred))+1) / 2
    ```

    The formula is similar to the classic cosine similarity used in deep learning,
    but scaled to [0.0, 1.0] and adjusted to have a reward that tend
    towards 1.0 if the two objects are similar (and 0.0 otherwise).

    Args:
        y_true (JsonDataModel): The ground truth JSON data_model.
        y_pred (JsonDataModel): The predicted JSON data_model.
        embedding_model (EmbeddingModel): The embedding model to use to compute the
            cosine similarity.
        axis (int): (Optional) Defaults to `-1`. The dimension along which the cosine
            similarity is computed.

    Returns:
        (float): The reward value, which tend to 1.0 if the values are similar,
            and towards 0.0 otherwise.
    """
    reward = 0.0
    if y_pred is not None:
        y_true = await ops.embedding(y_true, embedding_model=embedding_model)
        y_pred = await ops.embedding(y_pred, embedding_model=embedding_model)
        y_true = np.convert_to_tensor(y_true.get("embeddings"))
        y_pred = np.convert_to_tensor(y_pred.get("embeddings"))
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
        y_pred = np.normalize(y_pred, axis=axis)
        y_true = np.normalize(y_true, axis=axis)
        reward = (np.sum(y_true * y_pred, axis=axis) + 1) / 2
    return reward


@synalinks_export(
    [
        "synalinks.CosineSimilarity",
        "synalinks.rewards.CosineSimilarity",
    ]
)
class CosineSimilarity(RewardFunctionWrapper):
    """
    Computes the cosine similarity between `y_true` and `y_pred`.

    Formula:

    ```
    reward = (sum(l2_norm(y_true) * l2_norm(y_pred))+1) / 2
    ```

    The formula is similar to the classic cosine similarity used in deep learning,
    but scaled to [0.0, 1.0] and adjusted to have a reward that tend
    towards 1.0 if the two objects are similar (and 0.0 otherwise).

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.CosineSimilarity(
            embedding_model=embedding_model
        )
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```

    Args:
        embedding_model (EmbeddingModel): The embedding model to use to compute the
            cosine similarity.
        axis (int): (Optional) Defaults to `-1`. The dimension along which the cosine
            similarity is computed.
        name (str): (Optional) string name of the reward instance.
        in_mask (list): (Optional) list of keys to keep to compute the reward.
        out_mask (list): (Optional) list of keys to remove to compute the reward.
    """

    def __init__(
        self,
        embedding_model=None,
        axis=-1,
        name="cosine_similarity",
        in_mask=None,
        out_mask=None,
    ):
        super().__init__(
            fn=cosine_similarity,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            axis=axis,
            embedding_model=embedding_model,
        )

    def get_config(self):
        config = Reward.get_config()
        from synalinks.src.saving.serialization_lib import serialize_synalinks_object

        embedding_model_config = {
            "embedding_model": serialize_synalinks_object(self.embedding_model)
        }
        return {**config, **embedding_model_config}

    @classmethod
    def from_config(cls, config):
        from synalinks.saving.serialization_lib import deserialize_synalinks_object

        embedding_model = deserialize_synalinks_object(config.pop("embedding_model"))
        return cls(embedding_model=embedding_model, **config)
