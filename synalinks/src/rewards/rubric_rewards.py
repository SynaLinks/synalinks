# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.rewards.rubrics_as_judge import RubricsAsJudge


class _RubricPreset(RubricsAsJudge):
    """A `RubricsAsJudge` that grades against the built-in rubric preset `preset`.

    Its default `name` is the preset's key, and `rubrics` default to the
    preset's (a reloaded config keeps the rubrics it saved). See
    `RubricsAsJudge` for the arguments.
    """

    preset = None

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name=None,
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
        decision_model=None,
        rubrics=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics=rubrics or self.preset,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name or self.preset,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
            decision_model=decision_model,
        )


@synalinks_export(["synalinks.AnswerRelevancy", "synalinks.rewards.AnswerRelevancy"])
class AnswerRelevancy(_RubricPreset):
    """Judge whether an answer is relevant to the user query.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.AnswerRelevancy(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "answer_relevancy"


@synalinks_export(["synalinks.Faithfulness", "synalinks.rewards.Faithfulness"])
class Faithfulness(_RubricPreset):
    """Judge whether an answer is grounded in the provided context.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.Faithfulness(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "faithfulness"


@synalinks_export(["synalinks.Hallucination", "synalinks.rewards.Hallucination"])
class Hallucination(_RubricPreset):
    """Judge whether an answer avoids unsupported claims.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.Hallucination(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "hallucination"


@synalinks_export(["synalinks.Summarization", "synalinks.rewards.Summarization"])
class Summarization(_RubricPreset):
    """Judge summary coverage, faithfulness and concision.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.Summarization(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "summarization"


@synalinks_export(
    [
        "synalinks.ArgumentCorrectness",
        "synalinks.rewards.ArgumentCorrectness",
    ]
)
class ArgumentCorrectness(_RubricPreset):
    """Judge whether an argument is logically correct and supported.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.ArgumentCorrectness(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "argument_correctness"


@synalinks_export(
    [
        "synalinks.ContextualRelevancy",
        "synalinks.rewards.ContextualRelevancy",
    ]
)
class ContextualRelevancy(_RubricPreset):
    """Judge whether retrieved context is relevant to a query.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.ContextualRelevancy(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "contextual_relevancy"


@synalinks_export(
    [
        "synalinks.ContextualPrecision",
        "synalinks.rewards.ContextualPrecision",
    ]
)
class ContextualPrecision(_RubricPreset):
    """Judge whether relevant retrieved items are ranked early.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.ContextualPrecision(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "contextual_precision"


@synalinks_export(["synalinks.ContextualRecall", "synalinks.rewards.ContextualRecall"])
class ContextualRecall(_RubricPreset):
    """Judge whether context contains the facts needed for the answer.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.ContextualRecall(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "contextual_recall"


@synalinks_export(
    [
        "synalinks.CitationFaithfulness",
        "synalinks.rewards.CitationFaithfulness",
    ]
)
class CitationFaithfulness(_RubricPreset):
    """Judge whether citations support the claims they cite.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.CitationFaithfulness(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "citation_faithfulness"


@synalinks_export(["synalinks.Bias", "synalinks.rewards.Bias"])
class Bias(_RubricPreset):
    """Judge whether an answer avoids bias.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.Bias(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "bias"


@synalinks_export(["synalinks.Toxicity", "synalinks.rewards.Toxicity"])
class Toxicity(_RubricPreset):
    """Judge whether an answer avoids toxic language.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.Toxicity(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "toxicity"


@synalinks_export(["synalinks.PIILeakage", "synalinks.rewards.PIILeakage"])
class PIILeakage(_RubricPreset):
    """Judge whether an answer leaks private personal information.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.PIILeakage(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "pii_leakage"


@synalinks_export(["synalinks.Misuse", "synalinks.rewards.Misuse"])
class Misuse(_RubricPreset):
    """Judge whether an answer safely handles misuse requests.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.Misuse(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "misuse"


@synalinks_export(["synalinks.NonAdvice", "synalinks.rewards.NonAdvice"])
class NonAdvice(_RubricPreset):
    """Judge whether an answer avoids authoritative professional advice.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.NonAdvice(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "non_advice"


@synalinks_export(["synalinks.PromptAlignment", "synalinks.rewards.PromptAlignment"])
class PromptAlignment(_RubricPreset):
    """Judge whether an answer follows instructions and constraints.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.PromptAlignment(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "prompt_alignment"


@synalinks_export(["synalinks.TaskCompletion", "synalinks.rewards.TaskCompletion"])
class TaskCompletion(_RubricPreset):
    """Judge whether an agent completed the requested task.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.TaskCompletion(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "task_completion"


@synalinks_export(["synalinks.ToolCorrectness", "synalinks.rewards.ToolCorrectness"])
class ToolCorrectness(_RubricPreset):
    """Judge whether tool calls use the right tools and arguments.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.ToolCorrectness(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "tool_correctness"


@synalinks_export(["synalinks.ToolUse", "synalinks.rewards.ToolUse"])
class ToolUse(_RubricPreset):
    """Judge whether tool usage is necessary and appropriate.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.ToolUse(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "tool_use"


@synalinks_export(["synalinks.ToolPermission", "synalinks.rewards.ToolPermission"])
class ToolPermission(_RubricPreset):
    """Judge whether tool usage respects permissions.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.ToolPermission(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "tool_permission"


@synalinks_export(["synalinks.GoalAccuracy", "synalinks.rewards.GoalAccuracy"])
class GoalAccuracy(_RubricPreset):
    """Judge whether the result satisfies the stated goal.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.GoalAccuracy(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "goal_accuracy"


@synalinks_export(["synalinks.RoleAdherence", "synalinks.rewards.RoleAdherence"])
class RoleAdherence(_RubricPreset):
    """Judge whether the assistant stays in its assigned role.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.RoleAdherence(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "role_adherence"


@synalinks_export(["synalinks.RoleViolation", "synalinks.rewards.RoleViolation"])
class RoleViolation(_RubricPreset):
    """Judge whether the assistant avoids role violations.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.RoleViolation(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "role_violation"


@synalinks_export(["synalinks.PlanQuality", "synalinks.rewards.PlanQuality"])
class PlanQuality(_RubricPreset):
    """Judge whether an agent plan is sound and complete.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.PlanQuality(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "plan_quality"


@synalinks_export(["synalinks.PlanAdherence", "synalinks.rewards.PlanAdherence"])
class PlanAdherence(_RubricPreset):
    """Judge whether agent actions adhere to the plan.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.PlanAdherence(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "plan_adherence"


@synalinks_export(["synalinks.StepEfficiency", "synalinks.rewards.StepEfficiency"])
class StepEfficiency(_RubricPreset):
    """Judge whether an agent reaches the goal efficiently.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.StepEfficiency(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "step_efficiency"


@synalinks_export(
    [
        "synalinks.AgentLoopDetection",
        "synalinks.rewards.AgentLoopDetection",
    ]
)
class AgentLoopDetection(_RubricPreset):
    """Judge whether an agent avoids repeated unsuccessful loops.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.AgentLoopDetection(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "agent_loop_detection"


@synalinks_export(
    [
        "synalinks.ConversationCompleteness",
        "synalinks.rewards.ConversationCompleteness",
    ]
)
class ConversationCompleteness(_RubricPreset):
    """Judge whether a conversation addresses the user's needs.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.ConversationCompleteness(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "conversation_completeness"


@synalinks_export(
    [
        "synalinks.KnowledgeRetention",
        "synalinks.rewards.KnowledgeRetention",
    ]
)
class KnowledgeRetention(_RubricPreset):
    """Judge whether prior conversation context is retained.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.KnowledgeRetention(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "knowledge_retention"


@synalinks_export(["synalinks.TopicAdherence", "synalinks.rewards.TopicAdherence"])
class TopicAdherence(_RubricPreset):
    """Judge whether the answer stays on topic.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.TopicAdherence(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "topic_adherence"


@synalinks_export(["synalinks.TurnRelevancy", "synalinks.rewards.TurnRelevancy"])
class TurnRelevancy(_RubricPreset):
    """Judge whether a conversation turn is locally relevant.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.TurnRelevancy(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "turn_relevancy"


@synalinks_export(["synalinks.TurnFaithfulness", "synalinks.rewards.TurnFaithfulness"])
class TurnFaithfulness(_RubricPreset):
    """Judge whether a turn is faithful to conversation facts.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.TurnFaithfulness(
            language_model=language_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```
    """

    preset = "turn_faithfulness"
