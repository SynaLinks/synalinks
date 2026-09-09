# Metrics

A `Metric` is a function that is used to judge the performance of your program.

Metric functions are similar to reward functions, except that the results from evaluating a metric are not used when training the program. Note that you may use any reward function as a metric.

## Metrics Overview

- [Base Metric class](Base Metric class.md)
- [Metric wrappers and reduction metrics](Metric wrappers and reduction metrics.md)
- [Accuracy metrics](Accuracy metrics.md)
- [Precision and Recall metrics](Precision and Recall metrics.md)
- [FScore metrics](FScore metrics.md)
- [Regression metrics](Regression metrics.md)
- [Language Model operational metrics](Language Model operational metrics.md)
- [Embedding Model operational metrics](Embedding Model operational metrics.md)
- [Program operational metrics](Program operational metrics.md)

## All built-in metrics

### Base metrics

- [`Metric`](Metric metric.md)

### Batch metrics

- [`BatchMetric`](BatchMetric metric.md)

### Reduction and wrapper metrics

- [`Mean`](Mean metric.md)
- [`MeanMetricWrapper`](MeanMetricWrapper metric.md)
- [`Sum`](Sum metric.md)

### Accuracy metrics

- [`Accuracy`](Accuracy metric.md)
- [`BinaryAccuracy`](BinaryAccuracy metric.md)
- [`CategoricalAccuracy`](CategoricalAccuracy metric.md)

### Precision and recall metrics

- [`BinaryPrecision`](BinaryPrecision metric.md)
- [`BinaryRecall`](BinaryRecall metric.md)
- [`CategoricalPrecision`](CategoricalPrecision metric.md)
- [`CategoricalRecall`](CategoricalRecall metric.md)
- [`Precision`](Precision metric.md)
- [`Recall`](Recall metric.md)

### F-score metrics

- [`BinaryF1Score`](BinaryF1Score metric.md)
- [`BinaryFBetaScore`](BinaryFBetaScore metric.md)
- [`CategoricalF1Score`](CategoricalF1Score metric.md)
- [`CategoricalFBetaScore`](CategoricalFBetaScore metric.md)
- [`F1Score`](F1Score metric.md)
- [`FBetaScore`](FBetaScore metric.md)
- [`ListF1Score`](ListF1Score metric.md) (alias of `CategoricalF1Score`)
- [`ListFBetaScore`](ListFBetaScore metric.md) (alias of `CategoricalFBetaScore`)

### Regression metrics

- [`CosineSimilarity`](CosineSimilarity metric.md)

### Agent sampling metrics

- [`GapK`](GapK metric.md)
- [`PassAtK`](PassAtK metric.md)
- [`PassHatK`](PassHatK metric.md)

### Language model operational metrics

- [`AvgCacheCreationTokensPerCall`](AvgCacheCreationTokensPerCall metric.md)
- [`AvgCachedTokensPerCall`](AvgCachedTokensPerCall metric.md)
- [`AvgCostPerCall`](AvgCostPerCall metric.md)
- [`AvgInputTokensPerCall`](AvgInputTokensPerCall metric.md)
- [`AvgLatency`](AvgLatency metric.md)
- [`AvgOptimizerCacheCreationTokensPerCall`](AvgOptimizerCacheCreationTokensPerCall metric.md)
- [`AvgOptimizerCachedTokensPerCall`](AvgOptimizerCachedTokensPerCall metric.md)
- [`AvgOptimizerCostPerCall`](AvgOptimizerCostPerCall metric.md)
- [`AvgOptimizerInputTokensPerCall`](AvgOptimizerInputTokensPerCall metric.md)
- [`AvgOptimizerLatency`](AvgOptimizerLatency metric.md)
- [`AvgOptimizerOutputTokensPerCall`](AvgOptimizerOutputTokensPerCall metric.md)
- [`AvgOptimizerReasoningTokensPerCall`](AvgOptimizerReasoningTokensPerCall metric.md)
- [`AvgOptimizerTotalTokensPerCall`](AvgOptimizerTotalTokensPerCall metric.md)
- [`AvgOutputTokensPerCall`](AvgOutputTokensPerCall metric.md)
- [`AvgReasoningTokensPerCall`](AvgReasoningTokensPerCall metric.md)
- [`AvgRewardCacheCreationTokensPerCall`](AvgRewardCacheCreationTokensPerCall metric.md)
- [`AvgRewardCachedTokensPerCall`](AvgRewardCachedTokensPerCall metric.md)
- [`AvgRewardCostPerCall`](AvgRewardCostPerCall metric.md)
- [`AvgRewardInputTokensPerCall`](AvgRewardInputTokensPerCall metric.md)
- [`AvgRewardLatency`](AvgRewardLatency metric.md)
- [`AvgRewardOutputTokensPerCall`](AvgRewardOutputTokensPerCall metric.md)
- [`AvgRewardReasoningTokensPerCall`](AvgRewardReasoningTokensPerCall metric.md)
- [`AvgRewardTotalTokensPerCall`](AvgRewardTotalTokensPerCall metric.md)
- [`AvgTimeToFirstToken`](AvgTimeToFirstToken metric.md)
- [`AvgTimeToLastToken`](AvgTimeToLastToken metric.md)
- [`AvgTotalTokensPerCall`](AvgTotalTokensPerCall metric.md)
- [`AvgTrajectoryTimeToFirstToken`](AvgTrajectoryTimeToFirstToken metric.md)
- [`CacheCreationTokens`](CacheCreationTokens metric.md)
- [`CachedTokens`](CachedTokens metric.md)
- [`CacheHitRate`](CacheHitRate metric.md)
- [`Cost`](Cost metric.md)
- [`ErrorRate`](ErrorRate metric.md)
- [`FailedCalls`](FailedCalls metric.md)
- [`FallbackActivations`](FallbackActivations metric.md)
- [`InputTokens`](InputTokens metric.md)
- [`LMOperationalMetric`](LMOperationalMetric metric.md)
- [`LMOptimizersOperationalMetric`](LMOptimizersOperationalMetric metric.md)
- [`LMRewardsOperationalMetric`](LMRewardsOperationalMetric metric.md)
- [`OptimizerCacheCreationTokens`](OptimizerCacheCreationTokens metric.md)
- [`OptimizerCachedTokens`](OptimizerCachedTokens metric.md)
- [`OptimizerCacheHitRate`](OptimizerCacheHitRate metric.md)
- [`OptimizerCost`](OptimizerCost metric.md)
- [`OptimizerErrorRate`](OptimizerErrorRate metric.md)
- [`OptimizerFailedCalls`](OptimizerFailedCalls metric.md)
- [`OptimizerFallbackActivations`](OptimizerFallbackActivations metric.md)
- [`OptimizerInputTokens`](OptimizerInputTokens metric.md)
- [`OptimizerOutputTokens`](OptimizerOutputTokens metric.md)
- [`OptimizerReasoningTokens`](OptimizerReasoningTokens metric.md)
- [`OptimizerReasoningTokenShare`](OptimizerReasoningTokenShare metric.md)
- [`OptimizerThroughput`](OptimizerThroughput metric.md)
- [`OptimizerTokensPerSecond`](OptimizerTokensPerSecond metric.md)
- [`OptimizerTotalTokens`](OptimizerTotalTokens metric.md)
- [`OutputTokens`](OutputTokens metric.md)
- [`ReasoningTokens`](ReasoningTokens metric.md)
- [`ReasoningTokenShare`](ReasoningTokenShare metric.md)
- [`RewardCacheCreationTokens`](RewardCacheCreationTokens metric.md)
- [`RewardCachedTokens`](RewardCachedTokens metric.md)
- [`RewardCacheHitRate`](RewardCacheHitRate metric.md)
- [`RewardCost`](RewardCost metric.md)
- [`RewardErrorRate`](RewardErrorRate metric.md)
- [`RewardFailedCalls`](RewardFailedCalls metric.md)
- [`RewardFallbackActivations`](RewardFallbackActivations metric.md)
- [`RewardInputTokens`](RewardInputTokens metric.md)
- [`RewardOutputTokens`](RewardOutputTokens metric.md)
- [`RewardReasoningTokens`](RewardReasoningTokens metric.md)
- [`RewardReasoningTokenShare`](RewardReasoningTokenShare metric.md)
- [`RewardThroughput`](RewardThroughput metric.md)
- [`RewardTokensPerSecond`](RewardTokensPerSecond metric.md)
- [`RewardTotalTokens`](RewardTotalTokens metric.md)
- [`Throughput`](Throughput metric.md)
- [`TokensPerSecond`](TokensPerSecond metric.md)
- [`TotalTokens`](TotalTokens metric.md)

### Embedding model operational metrics

- [`AvgEmbeddingCachedTokensPerCall`](AvgEmbeddingCachedTokensPerCall metric.md)
- [`AvgEmbeddingCostPerCall`](AvgEmbeddingCostPerCall metric.md)
- [`AvgEmbeddingLatency`](AvgEmbeddingLatency metric.md)
- [`AvgEmbeddingTokensPerCall`](AvgEmbeddingTokensPerCall metric.md)
- [`AvgEmbeddingVectorsPerCall`](AvgEmbeddingVectorsPerCall metric.md)
- [`AvgOptimizerEmbeddingCachedTokensPerCall`](AvgOptimizerEmbeddingCachedTokensPerCall metric.md)
- [`AvgOptimizerEmbeddingCostPerCall`](AvgOptimizerEmbeddingCostPerCall metric.md)
- [`AvgOptimizerEmbeddingLatency`](AvgOptimizerEmbeddingLatency metric.md)
- [`AvgOptimizerEmbeddingTokensPerCall`](AvgOptimizerEmbeddingTokensPerCall metric.md)
- [`AvgOptimizerEmbeddingVectorsPerCall`](AvgOptimizerEmbeddingVectorsPerCall metric.md)
- [`AvgRewardEmbeddingCachedTokensPerCall`](AvgRewardEmbeddingCachedTokensPerCall metric.md)
- [`AvgRewardEmbeddingCostPerCall`](AvgRewardEmbeddingCostPerCall metric.md)
- [`AvgRewardEmbeddingLatency`](AvgRewardEmbeddingLatency metric.md)
- [`AvgRewardEmbeddingTokensPerCall`](AvgRewardEmbeddingTokensPerCall metric.md)
- [`AvgRewardEmbeddingVectorsPerCall`](AvgRewardEmbeddingVectorsPerCall metric.md)
- [`EmbeddingCachedTokens`](EmbeddingCachedTokens metric.md)
- [`EmbeddingCacheHitRate`](EmbeddingCacheHitRate metric.md)
- [`EmbeddingCost`](EmbeddingCost metric.md)
- [`EmbeddingErrorRate`](EmbeddingErrorRate metric.md)
- [`EmbeddingFailedCalls`](EmbeddingFailedCalls metric.md)
- [`EmbeddingFallbackActivations`](EmbeddingFallbackActivations metric.md)
- [`EmbeddingModelOperationalMetric`](EmbeddingModelOperationalMetric metric.md)
- [`EmbeddingModelOptimizersOperationalMetric`](EmbeddingModelOptimizersOperationalMetric metric.md)
- [`EmbeddingModelRewardsOperationalMetric`](EmbeddingModelRewardsOperationalMetric metric.md)
- [`EmbeddingThroughput`](EmbeddingThroughput metric.md)
- [`EmbeddingTokens`](EmbeddingTokens metric.md)
- [`EmbeddingTokensPerSecond`](EmbeddingTokensPerSecond metric.md)
- [`EmbeddingVectors`](EmbeddingVectors metric.md)
- [`EmbeddingVectorsPerSecond`](EmbeddingVectorsPerSecond metric.md)
- [`OptimizerEmbeddingCachedTokens`](OptimizerEmbeddingCachedTokens metric.md)
- [`OptimizerEmbeddingCacheHitRate`](OptimizerEmbeddingCacheHitRate metric.md)
- [`OptimizerEmbeddingCost`](OptimizerEmbeddingCost metric.md)
- [`OptimizerEmbeddingErrorRate`](OptimizerEmbeddingErrorRate metric.md)
- [`OptimizerEmbeddingFailedCalls`](OptimizerEmbeddingFailedCalls metric.md)
- [`OptimizerEmbeddingFallbackActivations`](OptimizerEmbeddingFallbackActivations metric.md)
- [`OptimizerEmbeddingThroughput`](OptimizerEmbeddingThroughput metric.md)
- [`OptimizerEmbeddingTokens`](OptimizerEmbeddingTokens metric.md)
- [`OptimizerEmbeddingTokensPerSecond`](OptimizerEmbeddingTokensPerSecond metric.md)
- [`OptimizerEmbeddingVectors`](OptimizerEmbeddingVectors metric.md)
- [`OptimizerEmbeddingVectorsPerSecond`](OptimizerEmbeddingVectorsPerSecond metric.md)
- [`RewardEmbeddingCachedTokens`](RewardEmbeddingCachedTokens metric.md)
- [`RewardEmbeddingCacheHitRate`](RewardEmbeddingCacheHitRate metric.md)
- [`RewardEmbeddingCost`](RewardEmbeddingCost metric.md)
- [`RewardEmbeddingErrorRate`](RewardEmbeddingErrorRate metric.md)
- [`RewardEmbeddingFailedCalls`](RewardEmbeddingFailedCalls metric.md)
- [`RewardEmbeddingFallbackActivations`](RewardEmbeddingFallbackActivations metric.md)
- [`RewardEmbeddingThroughput`](RewardEmbeddingThroughput metric.md)
- [`RewardEmbeddingTokens`](RewardEmbeddingTokens metric.md)
- [`RewardEmbeddingTokensPerSecond`](RewardEmbeddingTokensPerSecond metric.md)
- [`RewardEmbeddingVectors`](RewardEmbeddingVectors metric.md)
- [`RewardEmbeddingVectorsPerSecond`](RewardEmbeddingVectorsPerSecond metric.md)

### Program operational metrics

- [`ProgramAvgCostPerInvocation`](ProgramAvgCostPerInvocation metric.md)
- [`ProgramCalls`](ProgramCalls metric.md)
- [`ProgramCallsPerSecond`](ProgramCallsPerSecond metric.md)
- [`ProgramCost`](ProgramCost metric.md)
- [`ProgramElapsedTime`](ProgramElapsedTime metric.md)
- [`ProgramOperationalMetric`](ProgramOperationalMetric metric.md)
