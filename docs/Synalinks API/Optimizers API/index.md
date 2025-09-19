# Optimizers API

The `Optimizer`s are a key element in Synalinks, they updates the variables and backpropagate the rewards.

They are in charge of modifying and optimizing the variables (including the prompts) of each individual module composing a synalinks program.

```mermaid
graph LR
A[Training Data] -->|Provide x:DataModel| B[Program];
B -->|Generate y_pred:JsonDataModel| C[Reward];
A -->|Provide y_true:DataModel| C;
C -->|Compute reward:Float| D[Optimizer];
D -->|Update trainable_variable:Variable| B;
```

This reinforcement loop is what makes possible for the system to learn by
repeatedly making predictions and refining its knowledge/methodology in order 
to maximize the reward.

## Optimizers API overview

- [Base Optimizer class](Base Optimizer class.md)
- [RandomFewShot optimizer](RandomFewShot.md)
- [OMEGAEvolve optimizer](OMEGAEvolve.md)