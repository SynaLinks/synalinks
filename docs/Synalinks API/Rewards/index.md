# Rewards

`Reward`s are an essential part of reinforcement learning frameworks. 
They are typically float values (usually between 0.0 and 1.0, but they can be 
negative also) that guide the process into making more efficient decisions or 
predictions. During training, the goal is to maximize the reward function. 
The reward gives the system an indication of how well it performed for that task.

The purpose of a reward function is to compute the quantity that the program should maximize during training.

Synalinks ships two flavors of reward base class:

- `Reward`: scores **one sample at a time**. The trainer iterates the batch and
  calls the reward once per `(y_true, y_pred)` pair. This is what the built-in
  rewards (`ExactMatch`, `CosineSimilarity`, `LMAsJudge`, `AgentAsJudge`,
  `RLMAsJudge`, `DeepAgentAsJudge`, `RubricsAsJudge`, the rubric preset
  rewards, and `ProgramAsJudge`) use.
- `BatchReward`: receives the **whole batch** at once and returns a
  `list[float]` of length `batch_size`, one reward per sample. Use this when
  the score for sample *i* depends on the rest of the batch: group-relative
  rewards (e.g. GRPO-style normalization), pairwise/listwise comparisons, or
  any case where you want a single batched call instead of N parallel calls.

Both flavors share the same masking, reduction, and serialization machinery, so
you can mix them freely (including across multi-output programs).

## Custom rewards from a plain function

`compile` accepts a bare function, no wrapper needed. It is auto-wrapped in a
`RewardFunctionWrapper` named after the function:

```python
async def answer_matches(y_true, y_pred):
    return 1.0 if y_true.get("answer") == y_pred.get("answer") else 0.0

program.compile(
    reward=answer_matches,
    optimizer=synalinks.optimizers.RandomFewShot(),
)
```

The function must be declared with `async def`, since rewards are awaited; a
synchronous one raises a `TypeError` naming it. Wrap it in
`RewardFunctionWrapper` yourself when you need masks, a custom `reduction`, or
extra keyword arguments forwarded to it. Batched functions are never
auto-wrapped (a `batch -> list[float]` signature is indistinguishable from a
per-sample one), so they always go through `BatchRewardFunctionWrapper`.

## Rewards Overview

- [Base Reward class](Base Reward class.md)
- [ExactMatch reward](ExactMatch reward.md)
- [CosineSimilarity reward](CosineSimilarity reward.md)
- [LMAsJudge reward](LMAsJudge reward.md)
- [AgentAsJudge reward](AgentAsJudge reward.md)
- [RLMAsJudge reward](RLMAsJudge reward.md)
- [DeepAgentAsJudge reward](DeepAgentAsJudge reward.md)
- [RubricsAsJudge reward](RubricsAsJudge reward.md)
- [ComposableReward reward](ComposableReward reward.md)
- [Rubric preset rewards](Rubric preset rewards.md)
- [RewardFunctionWrapper reward](RewardFunctionWrapper reward.md)
- [ProgramAsJudge reward](ProgramAsJudge reward.md)
- [BatchReward reward](BatchReward reward.md)
- [BatchRewardFunctionWrapper reward](BatchRewardFunctionWrapper reward.md)

## All built-in rewards

### Core rewards

- [`Reward`](Base Reward class.md)
- [`ExactMatch`](ExactMatch reward.md)
- [`CosineSimilarity`](CosineSimilarity reward.md)
- [`LMAsJudge`](LMAsJudge reward.md)
- [`AgentAsJudge`](AgentAsJudge reward.md)
- [`RLMAsJudge`](RLMAsJudge reward.md)
- [`DeepAgentAsJudge`](DeepAgentAsJudge reward.md)
- [`RubricsAsJudge`](RubricsAsJudge reward.md)
- [`ComposableReward`](ComposableReward reward.md)

### Rubric preset rewards

- [`AnswerRelevancy`](AnswerRelevancy reward.md)
- [`Faithfulness`](Faithfulness reward.md)
- [`Hallucination`](Hallucination reward.md)
- [`Summarization`](Summarization reward.md)
- [`ArgumentCorrectness`](ArgumentCorrectness reward.md)
- [`ContextualRelevancy`](ContextualRelevancy reward.md)
- [`ContextualPrecision`](ContextualPrecision reward.md)
- [`ContextualRecall`](ContextualRecall reward.md)
- [`CitationFaithfulness`](CitationFaithfulness reward.md)
- [`Bias`](Bias reward.md)
- [`Toxicity`](Toxicity reward.md)
- [`PIILeakage`](PIILeakage reward.md)
- [`Misuse`](Misuse reward.md)
- [`NonAdvice`](NonAdvice reward.md)
- [`PromptAlignment`](PromptAlignment reward.md)
- [`TaskCompletion`](TaskCompletion reward.md)
- [`ToolCorrectness`](ToolCorrectness reward.md)
- [`ToolUse`](ToolUse reward.md)
- [`ToolPermission`](ToolPermission reward.md)
- [`GoalAccuracy`](GoalAccuracy reward.md)
- [`RoleAdherence`](RoleAdherence reward.md)
- [`RoleViolation`](RoleViolation reward.md)
- [`PlanQuality`](PlanQuality reward.md)
- [`PlanAdherence`](PlanAdherence reward.md)
- [`StepEfficiency`](StepEfficiency reward.md)
- [`AgentLoopDetection`](AgentLoopDetection reward.md)
- [`ConversationCompleteness`](ConversationCompleteness reward.md)
- [`KnowledgeRetention`](KnowledgeRetention reward.md)
- [`TopicAdherence`](TopicAdherence reward.md)
- [`TurnRelevancy`](TurnRelevancy reward.md)
- [`TurnFaithfulness`](TurnFaithfulness reward.md)

### Wrappers and batch rewards

- [`RewardFunctionWrapper`](RewardFunctionWrapper reward.md)
- [`ProgramAsJudge`](ProgramAsJudge reward.md)
- [`BatchReward`](BatchReward reward.md)
- [`BatchRewardFunctionWrapper`](BatchRewardFunctionWrapper reward.md)
