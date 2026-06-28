import inspect

from synalinks.src.api_export import synalinks_export
from synalinks.src.metrics.accuracy_metrics import Accuracy
from synalinks.src.metrics.accuracy_metrics import BinaryAccuracy
from synalinks.src.metrics.accuracy_metrics import CategoricalAccuracy
from synalinks.src.metrics.agents_metrics import GapK
from synalinks.src.metrics.agents_metrics import PassAtK
from synalinks.src.metrics.agents_metrics import PassHatK
from synalinks.src.metrics.batch_metric import BatchMetric
from synalinks.src.metrics.em_metrics import AvgEmbeddingCachedTokensPerCall

# Note: `accuracy_metrics` is imported below (after `metric`) to avoid a
# circular import via `synalinks.src.ops -> modules -> module -> metrics`.
from synalinks.src.metrics.em_metrics import AvgEmbeddingCostPerCall
from synalinks.src.metrics.em_metrics import AvgEmbeddingLatency
from synalinks.src.metrics.em_metrics import AvgEmbeddingTokensPerCall
from synalinks.src.metrics.em_metrics import AvgEmbeddingVectorsPerCall
from synalinks.src.metrics.em_metrics import AvgOptimizerEmbeddingCachedTokensPerCall
from synalinks.src.metrics.em_metrics import AvgOptimizerEmbeddingCostPerCall
from synalinks.src.metrics.em_metrics import AvgOptimizerEmbeddingLatency
from synalinks.src.metrics.em_metrics import AvgOptimizerEmbeddingTokensPerCall
from synalinks.src.metrics.em_metrics import AvgOptimizerEmbeddingVectorsPerCall
from synalinks.src.metrics.em_metrics import AvgRewardEmbeddingCachedTokensPerCall
from synalinks.src.metrics.em_metrics import AvgRewardEmbeddingCostPerCall
from synalinks.src.metrics.em_metrics import AvgRewardEmbeddingLatency
from synalinks.src.metrics.em_metrics import AvgRewardEmbeddingTokensPerCall
from synalinks.src.metrics.em_metrics import AvgRewardEmbeddingVectorsPerCall
from synalinks.src.metrics.em_metrics import EmbeddingCachedTokens
from synalinks.src.metrics.em_metrics import EmbeddingCacheHitRate
from synalinks.src.metrics.em_metrics import EmbeddingCost
from synalinks.src.metrics.em_metrics import EmbeddingErrorRate
from synalinks.src.metrics.em_metrics import EmbeddingFailedCalls
from synalinks.src.metrics.em_metrics import EmbeddingFallbackActivations
from synalinks.src.metrics.em_metrics import EmbeddingModelOperationalMetric
from synalinks.src.metrics.em_metrics import EmbeddingModelOptimizersOperationalMetric
from synalinks.src.metrics.em_metrics import EmbeddingModelRewardsOperationalMetric
from synalinks.src.metrics.em_metrics import EmbeddingThroughput
from synalinks.src.metrics.em_metrics import EmbeddingTokens
from synalinks.src.metrics.em_metrics import EmbeddingTokensPerSecond
from synalinks.src.metrics.em_metrics import EmbeddingVectors
from synalinks.src.metrics.em_metrics import EmbeddingVectorsPerSecond
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingCachedTokens
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingCacheHitRate
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingCost
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingErrorRate
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingFailedCalls
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingFallbackActivations
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingThroughput
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingTokens
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingTokensPerSecond
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingVectors
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingVectorsPerSecond
from synalinks.src.metrics.em_metrics import RewardEmbeddingCachedTokens
from synalinks.src.metrics.em_metrics import RewardEmbeddingCacheHitRate
from synalinks.src.metrics.em_metrics import RewardEmbeddingCost
from synalinks.src.metrics.em_metrics import RewardEmbeddingErrorRate
from synalinks.src.metrics.em_metrics import RewardEmbeddingFailedCalls
from synalinks.src.metrics.em_metrics import RewardEmbeddingFallbackActivations
from synalinks.src.metrics.em_metrics import RewardEmbeddingThroughput
from synalinks.src.metrics.em_metrics import RewardEmbeddingTokens
from synalinks.src.metrics.em_metrics import RewardEmbeddingTokensPerSecond
from synalinks.src.metrics.em_metrics import RewardEmbeddingVectors
from synalinks.src.metrics.em_metrics import RewardEmbeddingVectorsPerSecond
from synalinks.src.metrics.f_score_metrics import BinaryF1Score
from synalinks.src.metrics.f_score_metrics import BinaryFBetaScore
from synalinks.src.metrics.f_score_metrics import CategoricalF1Score
from synalinks.src.metrics.f_score_metrics import CategoricalFBetaScore
from synalinks.src.metrics.f_score_metrics import F1Score
from synalinks.src.metrics.f_score_metrics import FBetaScore
from synalinks.src.metrics.lm_metrics import AvgCacheCreationTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgCachedTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgCostPerCall
from synalinks.src.metrics.lm_metrics import AvgInputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgLatency
from synalinks.src.metrics.lm_metrics import AvgOptimizerCacheCreationTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOptimizerCachedTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOptimizerCostPerCall
from synalinks.src.metrics.lm_metrics import AvgOptimizerInputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOptimizerLatency
from synalinks.src.metrics.lm_metrics import AvgOptimizerOutputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOptimizerReasoningTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOptimizerTotalTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOutputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgReasoningTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardCacheCreationTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardCachedTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardCostPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardInputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardLatency
from synalinks.src.metrics.lm_metrics import AvgRewardOutputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardReasoningTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardTotalTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgTotalTokensPerCall
from synalinks.src.metrics.lm_metrics import CacheCreationTokens
from synalinks.src.metrics.lm_metrics import CachedTokens
from synalinks.src.metrics.lm_metrics import CacheHitRate
from synalinks.src.metrics.lm_metrics import Cost
from synalinks.src.metrics.lm_metrics import ErrorRate
from synalinks.src.metrics.lm_metrics import FailedCalls
from synalinks.src.metrics.lm_metrics import FallbackActivations
from synalinks.src.metrics.lm_metrics import InputTokens
from synalinks.src.metrics.lm_metrics import LMOperationalMetric
from synalinks.src.metrics.lm_metrics import LMOptimizersOperationalMetric
from synalinks.src.metrics.lm_metrics import LMRewardsOperationalMetric
from synalinks.src.metrics.lm_metrics import OptimizerCacheCreationTokens
from synalinks.src.metrics.lm_metrics import OptimizerCachedTokens
from synalinks.src.metrics.lm_metrics import OptimizerCacheHitRate
from synalinks.src.metrics.lm_metrics import OptimizerCost
from synalinks.src.metrics.lm_metrics import OptimizerErrorRate
from synalinks.src.metrics.lm_metrics import OptimizerFailedCalls
from synalinks.src.metrics.lm_metrics import OptimizerFallbackActivations
from synalinks.src.metrics.lm_metrics import OptimizerInputTokens
from synalinks.src.metrics.lm_metrics import OptimizerOutputTokens
from synalinks.src.metrics.lm_metrics import OptimizerReasoningTokens
from synalinks.src.metrics.lm_metrics import OptimizerReasoningTokenShare
from synalinks.src.metrics.lm_metrics import OptimizerThroughput
from synalinks.src.metrics.lm_metrics import OptimizerTokensPerSecond
from synalinks.src.metrics.lm_metrics import OptimizerTotalTokens
from synalinks.src.metrics.lm_metrics import OutputTokens
from synalinks.src.metrics.lm_metrics import ReasoningTokens
from synalinks.src.metrics.lm_metrics import ReasoningTokenShare
from synalinks.src.metrics.lm_metrics import RewardCacheCreationTokens
from synalinks.src.metrics.lm_metrics import RewardCachedTokens
from synalinks.src.metrics.lm_metrics import RewardCacheHitRate
from synalinks.src.metrics.lm_metrics import RewardCost
from synalinks.src.metrics.lm_metrics import RewardErrorRate
from synalinks.src.metrics.lm_metrics import RewardFailedCalls
from synalinks.src.metrics.lm_metrics import RewardFallbackActivations
from synalinks.src.metrics.lm_metrics import RewardInputTokens
from synalinks.src.metrics.lm_metrics import RewardOutputTokens
from synalinks.src.metrics.lm_metrics import RewardReasoningTokens
from synalinks.src.metrics.lm_metrics import RewardReasoningTokenShare
from synalinks.src.metrics.lm_metrics import RewardThroughput
from synalinks.src.metrics.lm_metrics import RewardTokensPerSecond
from synalinks.src.metrics.lm_metrics import RewardTotalTokens
from synalinks.src.metrics.lm_metrics import Throughput
from synalinks.src.metrics.lm_metrics import TokensPerSecond
from synalinks.src.metrics.lm_metrics import TotalTokens
from synalinks.src.metrics.metric import Metric
from synalinks.src.metrics.precision_recall_metrics import BinaryPrecision
from synalinks.src.metrics.precision_recall_metrics import BinaryRecall
from synalinks.src.metrics.precision_recall_metrics import CategoricalPrecision
from synalinks.src.metrics.precision_recall_metrics import CategoricalRecall
from synalinks.src.metrics.precision_recall_metrics import Precision
from synalinks.src.metrics.precision_recall_metrics import Recall
from synalinks.src.metrics.program_metrics import ProgramAvgCostPerInvocation
from synalinks.src.metrics.program_metrics import ProgramCalls
from synalinks.src.metrics.program_metrics import ProgramCallsPerSecond
from synalinks.src.metrics.program_metrics import ProgramCost
from synalinks.src.metrics.program_metrics import ProgramElapsedTime
from synalinks.src.metrics.program_metrics import ProgramOperationalMetric
from synalinks.src.metrics.reduction_metrics import Mean
from synalinks.src.metrics.reduction_metrics import MeanMetricWrapper
from synalinks.src.metrics.reduction_metrics import Sum
from synalinks.src.saving import serialization_lib
from synalinks.src.utils.naming import to_snake_case

ALL_OBJECTS = {
    # Base
    Metric,
    Mean,
    MeanMetricWrapper,
    Sum,
    # Accuracy
    Accuracy,
    BinaryAccuracy,
    CategoricalAccuracy,
    # Precision / Recall
    Precision,
    BinaryPrecision,
    CategoricalPrecision,
    Recall,
    BinaryRecall,
    CategoricalRecall,
    # F-score
    FBetaScore,
    F1Score,
    BinaryFBetaScore,
    BinaryF1Score,
    CategoricalFBetaScore,
    CategoricalF1Score,
    # Operational (language model)
    LMOperationalMetric,
    InputTokens,
    OutputTokens,
    TotalTokens,
    AvgInputTokensPerCall,
    AvgOutputTokensPerCall,
    AvgTotalTokensPerCall,
    TokensPerSecond,
    Throughput,
    AvgLatency,
    Cost,
    AvgCostPerCall,
    CachedTokens,
    AvgCachedTokensPerCall,
    CacheCreationTokens,
    AvgCacheCreationTokensPerCall,
    CacheHitRate,
    ReasoningTokens,
    AvgReasoningTokensPerCall,
    ReasoningTokenShare,
    FailedCalls,
    FallbackActivations,
    ErrorRate,
    # Operational (embedding model)
    EmbeddingModelOperationalMetric,
    EmbeddingTokens,
    EmbeddingVectors,
    EmbeddingCost,
    EmbeddingThroughput,
    AvgEmbeddingLatency,
    EmbeddingTokensPerSecond,
    EmbeddingVectorsPerSecond,
    AvgEmbeddingTokensPerCall,
    AvgEmbeddingVectorsPerCall,
    AvgEmbeddingCostPerCall,
    EmbeddingCachedTokens,
    AvgEmbeddingCachedTokensPerCall,
    EmbeddingCacheHitRate,
    EmbeddingFailedCalls,
    EmbeddingFallbackActivations,
    EmbeddingErrorRate,
    # Operational (language model, reward phase)
    LMRewardsOperationalMetric,
    RewardInputTokens,
    RewardOutputTokens,
    RewardTotalTokens,
    AvgRewardInputTokensPerCall,
    AvgRewardOutputTokensPerCall,
    AvgRewardTotalTokensPerCall,
    RewardTokensPerSecond,
    RewardThroughput,
    AvgRewardLatency,
    RewardCost,
    AvgRewardCostPerCall,
    RewardCachedTokens,
    AvgRewardCachedTokensPerCall,
    RewardCacheCreationTokens,
    AvgRewardCacheCreationTokensPerCall,
    RewardCacheHitRate,
    RewardReasoningTokens,
    AvgRewardReasoningTokensPerCall,
    RewardReasoningTokenShare,
    RewardFailedCalls,
    RewardFallbackActivations,
    RewardErrorRate,
    # Operational (language model, optimizer phase)
    LMOptimizersOperationalMetric,
    OptimizerInputTokens,
    OptimizerOutputTokens,
    OptimizerTotalTokens,
    AvgOptimizerInputTokensPerCall,
    AvgOptimizerOutputTokensPerCall,
    AvgOptimizerTotalTokensPerCall,
    OptimizerTokensPerSecond,
    OptimizerThroughput,
    AvgOptimizerLatency,
    OptimizerCost,
    AvgOptimizerCostPerCall,
    OptimizerCachedTokens,
    AvgOptimizerCachedTokensPerCall,
    OptimizerCacheCreationTokens,
    AvgOptimizerCacheCreationTokensPerCall,
    OptimizerCacheHitRate,
    OptimizerReasoningTokens,
    AvgOptimizerReasoningTokensPerCall,
    OptimizerReasoningTokenShare,
    OptimizerFailedCalls,
    OptimizerFallbackActivations,
    OptimizerErrorRate,
    # Operational (embedding model, reward phase)
    EmbeddingModelRewardsOperationalMetric,
    RewardEmbeddingTokens,
    RewardEmbeddingVectors,
    RewardEmbeddingCost,
    RewardEmbeddingThroughput,
    AvgRewardEmbeddingLatency,
    RewardEmbeddingTokensPerSecond,
    RewardEmbeddingVectorsPerSecond,
    AvgRewardEmbeddingTokensPerCall,
    AvgRewardEmbeddingVectorsPerCall,
    AvgRewardEmbeddingCostPerCall,
    RewardEmbeddingCachedTokens,
    AvgRewardEmbeddingCachedTokensPerCall,
    RewardEmbeddingCacheHitRate,
    RewardEmbeddingFailedCalls,
    RewardEmbeddingFallbackActivations,
    RewardEmbeddingErrorRate,
    # Operational (embedding model, optimizer phase)
    EmbeddingModelOptimizersOperationalMetric,
    OptimizerEmbeddingTokens,
    OptimizerEmbeddingVectors,
    OptimizerEmbeddingCost,
    OptimizerEmbeddingThroughput,
    AvgOptimizerEmbeddingLatency,
    OptimizerEmbeddingTokensPerSecond,
    OptimizerEmbeddingVectorsPerSecond,
    AvgOptimizerEmbeddingTokensPerCall,
    AvgOptimizerEmbeddingVectorsPerCall,
    AvgOptimizerEmbeddingCostPerCall,
    OptimizerEmbeddingCachedTokens,
    AvgOptimizerEmbeddingCachedTokensPerCall,
    OptimizerEmbeddingCacheHitRate,
    OptimizerEmbeddingFailedCalls,
    OptimizerEmbeddingFallbackActivations,
    OptimizerEmbeddingErrorRate,
    # Agents (sampling-based: pass@k and friends)
    BatchMetric,
    PassAtK,
    PassHatK,
    GapK,
    # Operational (program-wide)
    ProgramOperationalMetric,
    ProgramCalls,
    ProgramElapsedTime,
    ProgramCallsPerSecond,
    ProgramCost,
    ProgramAvgCostPerInvocation,
}

ALL_OBJECTS_DICT = {cls.__name__.lower(): cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update({to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS})


@synalinks_export("synalinks.metrics.serialize")
def serialize(metric):
    """Serializes metric function or `Metric` instance.

    Args:
        metric: A Synalinks `Metric` instance or a metric function.

    Returns:
        Metric configuration dictionary.
    """
    return serialization_lib.serialize_synalinks_object(metric)


@synalinks_export("synalinks.metrics.deserialize")
def deserialize(config, custom_objects=None):
    """Deserializes a serialized metric class/function instance.

    Args:
        config: Metric configuration.
        custom_objects: Optional dictionary mapping names (strings)
            to custom objects (classes and functions) to be
            considered during deserialization.

    Returns:
        A Synalinks `Metric` instance or a metric function.
    """
    # Make deserialization case-insensitive for built-in metrics.
    if config["class_name"].lower() in ALL_OBJECTS_DICT:
        config["class_name"] = config["class_name"].lower()
    return serialization_lib.deserialize_synalinks_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@synalinks_export("synalinks.metrics.get")
def get(identifier):
    """Retrieves a Synalinks metric as a `function`/`Metric` class instance.

    The `identifier` may be the string name of a metric function or class.

    >>> metric = metrics.get("categorical_crossentropy")
    >>> type(metric)
    <class 'function'>
    >>> metric = metrics.get("CategoricalCrossentropy")
    >>> type(metric)
    <class '...metrics.CategoricalCrossentropy'>

    You can also specify `config` of the metric to this function by passing dict
    containing `class_name` and `config` as an identifier. Also note that the
    `class_name` must map to a `Metric` class

    >>> identifier = {"class_name": "CategoricalCrossentropy",
    ...               "config": {"from_logits": True}}
    >>> metric = metrics.get(identifier)
    >>> type(metric)
    <class '...metrics.CategoricalCrossentropy'>

    Args:
        identifier: A metric identifier. One of None or string name of a metric
            function/class or metric configuration dictionary or a metric
            function or a metric class instance

    Returns:
        A Synalinks metric as a `function`/ `Metric` class instance.
    """
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier.lower(), None)
    else:
        obj = identifier
    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(f"Could not interpret metric identifier: {identifier}")
