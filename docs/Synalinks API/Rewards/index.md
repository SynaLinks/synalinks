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
  rewards (`ExactMatch`, `CosineSimilarity`, `LMAsJudge`, `ProgramAsJudge`) use.
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
- [RewardFunctionWrapper & ProgramAsJudge](Reward wrappers.md)
- [BatchReward & BatchRewardFunctionWrapper](Batch reward wrappers.md)