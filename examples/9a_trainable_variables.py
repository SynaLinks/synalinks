# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Trainable Variables Inside a Custom Module

A companion to `guides/9_custom_modules.py`, this example builds a
small custom module called `HintedAnswerer` that:

- owns a **trainable variable** containing a list of "hints" (short reminders
  the LM should consider when answering),
- owns a **non-trainable variable** that counts how many times the module
  has been called,
- wraps a `synalinks.Generator` and feeds it those hints at call time,
- can be optimized end-to-end with `program.compile(...).fit(...)` so the
  optimizer rewrites the hints in-context based on a reward.

The point of the example is not the hints themselves — it is the *plumbing*:
how to declare trainable state on a `Module`, how to read and update it in
`call()`, how the optimizer picks it up automatically, and how to save and
restore the trained state.

## What you'll see

```mermaid
graph LR
    A[Query] --> B[HintedAnswerer]
    B --> C[Generator]
    C --> D[Answer]
    H["Hints (trainable variable)"] --> B
    S["call_count (non-trainable variable)"] --> B
```

1. The variables are discovered automatically by the program.
2. A first call uses the seeded hints; the counter goes from 0 to 1.
3. A simulated optimizer update mutates the hints via `Variable.update`.
4. The next call reflects the new hints.
5. (Optional) `program.compile(...).fit(...)` lets the **OMEGA** optimizer
   rewrite the hints based on a real reward signal.
6. `program.save(...)` / `program.load(...)` round-trips both variables.

## Choosing the optimizer

- `synalinks.optimizers.RandomFewShot` is the right baseline when the only
  trainable field you have is `examples` (few-shot selection). It will
  *not* mutate the `hints` field by itself.
- `synalinks.optimizers.OMEGA` is the general-purpose, LM-driven mutator.
  It treats every non-bookkeeping field on your `Trainable` data model as
  fair game, and can rewrite the `hints` list directly. That is what we
  use in the optional training block.

## Requirements to run the optional training block

- A configured language model and embedding model. Set them up in a `.env`
  file at the project root (see `examples/0_first_steps.py`).
- The training block is guarded by `RUN_TRAINING = False`. Flip it to
  `True` to exercise the full fit/save/load round-trip.

## API References

- [Module (Base Class)](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Base%20Module%20class/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [OMEGA](https://synalinks.github.io/synalinks/Synalinks%20API/Optimizers%20API/OMEGA%20optimizer/)
- [Trainable](https://synalinks.github.io/synalinks/Synalinks%20API/Backend%20API/Trainable/)
"""

import asyncio
import os
from typing import List
from typing import Optional

from dotenv import load_dotenv

import synalinks

# Set to True to also run the optional fit/save/load round-trip. That block
# needs valid LM and embedding model credentials in your environment.
RUN_TRAINING = False
FOLDER = "examples"


# =============================================================================
# Data models
# =============================================================================


class Query(synalinks.DataModel):
    """A user question."""

    query: str = synalinks.Field(description="The user query")


class Answer(synalinks.DataModel):
    """The model's final answer."""

    answer: str = synalinks.Field(description="The correct answer")


class Hints(synalinks.Trainable):
    """The trainable state held by `HintedAnswerer`.

    Subclassing `synalinks.Trainable` is what makes this DataModel eligible
    for optimization: it adds the bookkeeping fields the optimizer expects
    (`examples`, `predictions`, `candidates`, `best_candidates`,
    `seed_candidates`, `nb_visit`, `cumulative_reward`, `history`).

    Every field must have a default value, otherwise the variable cannot be
    initialized symbolically during graph tracing.
    """

    hints: List[str] = synalinks.Field(
        description=(
            "Short reminders the language model should keep in mind while "
            "answering. The optimizer is allowed to rewrite this list."
        ),
        default=[],
    )


class CallStats(synalinks.DataModel):
    """Plain DataModel (non-trainable) holding per-module runtime counters."""

    count: int = synalinks.Field(
        description="Number of times the module has been called.",
        default=0,
    )


# =============================================================================
# Custom module
# =============================================================================


class HintedAnswerer(synalinks.Module):
    """Answer questions, biased by an evolving list of hints.

    Trainable surface:
        `self.state` — a `Hints` variable. Optimizers may rewrite the
        `hints` list and/or the `examples` few-shot store.

    Non-trainable surface:
        `self.stats` — a `CallStats` variable, mutated from `call()`.

    Internals:
        A wrapped `synalinks.Generator` does the actual LM call. The hints
        from `self.state` are injected into the generator's `instructions`
        each call, so the optimizer's hint mutations directly affect the
        prompt the LM sees.
    """

    def __init__(
        self,
        language_model=None,
        hints: Optional[List[str]] = None,
        seed_hints: Optional[List[List[str]]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        trainable: bool = True,
    ):
        super().__init__(name=name, description=description, trainable=trainable)
        self.language_model = language_model
        self._initial_hints = hints or []
        self._seed_hints = seed_hints or []

        # ---------------------------------------------------------------
        # 1) Trainable variable.
        #
        # `seed_candidates` warm-starts the optimizer with known-good
        # configurations. We translate each seed hint list into a candidate
        # dict shaped like the variable itself.
        # ---------------------------------------------------------------
        seed_candidates = [{"hints": seed} for seed in self._seed_hints]

        self.state = self.add_variable(
            initializer=Hints(
                hints=list(self._initial_hints),
                seed_candidates=seed_candidates,
            ).get_json(),
            data_model=Hints,
            name="hints_" + self.name,
        )

        # ---------------------------------------------------------------
        # 2) Non-trainable variable.
        # ---------------------------------------------------------------
        self.stats = self.add_variable(
            initializer=CallStats().get_json(),
            data_model=CallStats,
            trainable=False,
            name="stats_" + self.name,
        )

        # ---------------------------------------------------------------
        # 3) Sub-module (also tracked automatically). Note that the
        # Generator already has its own trainable variable (`Instructions`).
        # Both will show up in `program.trainable_variables`.
        # ---------------------------------------------------------------
        self.generator = synalinks.Generator(
            data_model=Answer,
            language_model=self.language_model,
            name="generator_" + self.name,
        )

    def _format_instructions(self) -> str:
        hints = self.state.get("hints") or []
        if not hints:
            return "Answer the user's query concisely."
        bullet_hints = "\n".join(f"- {h}" for h in hints)
        return (
            "Answer the user's query concisely. While answering, keep these "
            f"hints in mind:\n{bullet_hints}"
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        # Non-trainable bookkeeping.
        self.stats.update({"count": self.stats.get("count") + 1})

        # The Generator's own state has an `instructions` field that the
        # `Instructions` data model exposes. We refresh it here so the
        # current hints feed straight into the prompt.
        self.generator.state.update({"instructions": self._format_instructions()})

        result = await self.generator(inputs, training=training)

        # Record this prediction on *our* trainable variable so the
        # optimizer can attribute rewards to the hint configuration that
        # produced it. This is the same pattern `Generator.call` follows.
        if training and result is not None:
            predictions = self.state.get("current_predictions")
            predictions.append(
                {
                    "inputs": inputs.get_json(),
                    "outputs": result.get_json(),
                    "reward": None,
                }
            )

        return result

    async def compute_output_spec(self, inputs, training=False):
        return await self.generator(inputs)

    def get_config(self):
        config = {
            "hints": self._initial_hints,
            "seed_hints": self._seed_hints,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        config["language_model"] = synalinks.saving.serialize_synalinks_object(
            self.language_model,
        )
        return config

    @classmethod
    def from_config(cls, config):
        language_model = synalinks.saving.deserialize_synalinks_object(
            config.pop("language_model"),
        )
        return cls(language_model=language_model, **config)


# =============================================================================
# Main
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    language_model = synalinks.LanguageModel(
        model="gemini/gemini-3.1-flash-lite-preview",
    )

    # ---------------------------------------------------------------------
    # Build the program.
    # ---------------------------------------------------------------------
    inputs = synalinks.Input(data_model=Query)
    outputs = await HintedAnswerer(
        language_model=language_model,
        hints=[
            "Prefer plain numbers over scientific notation.",
            "Always state the final answer on its own line.",
        ],
        seed_hints=[
            ["Show the unit when relevant."],
            ["Round to two significant figures unless precision matters."],
        ],
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="hinted_answerer",
        description="A toy module showing how trainable variables work.",
    )

    # ---------------------------------------------------------------------
    # Inspect what the framework discovered.
    # ---------------------------------------------------------------------
    print("=" * 70)
    print("Trainable variables discovered by the program")
    print("=" * 70)
    for v in program.trainable_variables:
        print(f"  - {v.name}: fields = {list(v.keys())}")

    print("\nNon-trainable variables (state, but not optimized):")
    for v in program.non_trainable_variables:
        print(f"  - {v.name}: fields = {list(v.keys())}")

    # ---------------------------------------------------------------------
    # Run once with the seeded hints. Requires an LM.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("First call with the seeded hints")
    print("=" * 70)
    result = await program(Query(query="What is the speed of light?"))
    print(result.prettify_json())

    # ---------------------------------------------------------------------
    # Simulate an optimizer step by manually rewriting the hints.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Simulated optimizer step: rewrite the hints by hand")
    print("=" * 70)
    hint_variable = next(
        v for v in program.trainable_variables if v.name.startswith("hints_")
    )
    hint_variable.update(
        {
            "hints": [
                "Reply in a single short sentence.",
                "Use everyday units (e.g. km/s).",
            ],
        }
    )
    print(f"New hints: {hint_variable.get('hints')}")

    result = await program(Query(query="What is the speed of light?"))
    print(result.prettify_json())

    # ---------------------------------------------------------------------
    # The non-trainable counter has tracked both calls.
    # ---------------------------------------------------------------------
    stats_variable = next(
        v for v in program.non_trainable_variables if v.name.startswith("stats_")
    )
    print(
        f"\nCall counter ({stats_variable.name}): count = {stats_variable.get('count')}"
    )

    # ---------------------------------------------------------------------
    # OPTIONAL: real training loop with OMEGA, then save/load round-trip.
    # Flip `RUN_TRAINING` at the top of the file to True to run this.
    # ---------------------------------------------------------------------
    if not RUN_TRAINING:
        print(
            "\nSet RUN_TRAINING=True at the top of the file to run "
            "fit/save/load with the OMEGA optimizer."
        )
        return

    print("\n" + "=" * 70)
    print("Training with OMEGA — the optimizer will rewrite the hints")
    print("=" * 70)

    embedding_model = synalinks.EmbeddingModel(
        model="text-embedding-3-small",
    )

    # A tiny synthetic dataset. Replace with your real data.
    x_train = [
        Query(query="What is 2 + 2?"),
        Query(query="What is the capital of France?"),
        Query(query="What is 10 / 4?"),
    ]
    y_train = [
        Answer(answer="4"),
        Answer(answer="Paris"),
        Answer(answer="2.5"),
    ]

    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.OMEGA(
            language_model=language_model,
            embedding_model=embedding_model,
        ),
    )

    history = await program.fit(
        x=x_train,
        y=y_train,
        validation_split=0.34,
        epochs=1,
        batch_size=2,
    )
    print(f"history keys: {list(history.history.keys())}")

    print(f"\nHints after training: {hint_variable.get('hints')}")
    print(f"History of best candidates: {len(hint_variable.get('history'))} entries")

    # Save and reload — both variables are persisted by the saving layer.
    save_path = os.path.join(FOLDER, "hinted_answerer.json")
    program.save(save_path)
    print(f"\nSaved trained program to {save_path}")

    reloaded = synalinks.Program.load(save_path)
    reloaded_hint_variable = next(
        v for v in reloaded.trainable_variables if v.name.startswith("hints_")
    )
    print(f"Hints after reload: {reloaded_hint_variable.get('hints')}")


if __name__ == "__main__":
    asyncio.run(main())
