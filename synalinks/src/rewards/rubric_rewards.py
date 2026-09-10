# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.rewards.rubrics_as_judge import RubricsAsJudge


@synalinks_export(["synalinks.AnswerRelevancy", "synalinks.rewards.AnswerRelevancy"])
class AnswerRelevancy(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="answer_relevancy",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="answer_relevancy",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.Faithfulness", "synalinks.rewards.Faithfulness"])
class Faithfulness(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="faithfulness",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="faithfulness",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.Hallucination", "synalinks.rewards.Hallucination"])
class Hallucination(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="hallucination",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="hallucination",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.Summarization", "synalinks.rewards.Summarization"])
class Summarization(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="summarization",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="summarization",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(
    [
        "synalinks.ArgumentCorrectness",
        "synalinks.rewards.ArgumentCorrectness",
    ]
)
class ArgumentCorrectness(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="argument_correctness",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="argument_correctness",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(
    [
        "synalinks.ContextualRelevancy",
        "synalinks.rewards.ContextualRelevancy",
    ]
)
class ContextualRelevancy(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="contextual_relevancy",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="contextual_relevancy",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(
    [
        "synalinks.ContextualPrecision",
        "synalinks.rewards.ContextualPrecision",
    ]
)
class ContextualPrecision(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="contextual_precision",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="contextual_precision",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.ContextualRecall", "synalinks.rewards.ContextualRecall"])
class ContextualRecall(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="contextual_recall",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="contextual_recall",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(
    [
        "synalinks.CitationFaithfulness",
        "synalinks.rewards.CitationFaithfulness",
    ]
)
class CitationFaithfulness(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="citation_faithfulness",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="citation_faithfulness",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.Bias", "synalinks.rewards.Bias"])
class Bias(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="bias",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="bias",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.Toxicity", "synalinks.rewards.Toxicity"])
class Toxicity(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="toxicity",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="toxicity",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.PIILeakage", "synalinks.rewards.PIILeakage"])
class PIILeakage(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="pii_leakage",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="pii_leakage",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.Misuse", "synalinks.rewards.Misuse"])
class Misuse(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="misuse",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="misuse",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.NonAdvice", "synalinks.rewards.NonAdvice"])
class NonAdvice(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="non_advice",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="non_advice",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.PromptAlignment", "synalinks.rewards.PromptAlignment"])
class PromptAlignment(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="prompt_alignment",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="prompt_alignment",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.TaskCompletion", "synalinks.rewards.TaskCompletion"])
class TaskCompletion(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="task_completion",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="task_completion",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.ToolCorrectness", "synalinks.rewards.ToolCorrectness"])
class ToolCorrectness(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="tool_correctness",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="tool_correctness",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.ToolUse", "synalinks.rewards.ToolUse"])
class ToolUse(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="tool_use",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="tool_use",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.ToolPermission", "synalinks.rewards.ToolPermission"])
class ToolPermission(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="tool_permission",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="tool_permission",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.GoalAccuracy", "synalinks.rewards.GoalAccuracy"])
class GoalAccuracy(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="goal_accuracy",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="goal_accuracy",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.RoleAdherence", "synalinks.rewards.RoleAdherence"])
class RoleAdherence(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="role_adherence",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="role_adherence",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.RoleViolation", "synalinks.rewards.RoleViolation"])
class RoleViolation(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="role_violation",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="role_violation",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.PlanQuality", "synalinks.rewards.PlanQuality"])
class PlanQuality(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="plan_quality",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="plan_quality",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.PlanAdherence", "synalinks.rewards.PlanAdherence"])
class PlanAdherence(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="plan_adherence",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="plan_adherence",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.StepEfficiency", "synalinks.rewards.StepEfficiency"])
class StepEfficiency(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="step_efficiency",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="step_efficiency",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(
    [
        "synalinks.AgentLoopDetection",
        "synalinks.rewards.AgentLoopDetection",
    ]
)
class AgentLoopDetection(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="agent_loop_detection",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="agent_loop_detection",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(
    [
        "synalinks.ConversationCompleteness",
        "synalinks.rewards.ConversationCompleteness",
    ]
)
class ConversationCompleteness(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="conversation_completeness",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="conversation_completeness",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(
    [
        "synalinks.KnowledgeRetention",
        "synalinks.rewards.KnowledgeRetention",
    ]
)
class KnowledgeRetention(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="knowledge_retention",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="knowledge_retention",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.TopicAdherence", "synalinks.rewards.TopicAdherence"])
class TopicAdherence(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="topic_adherence",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="topic_adherence",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.TurnRelevancy", "synalinks.rewards.TurnRelevancy"])
class TurnRelevancy(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="turn_relevancy",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="turn_relevancy",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )


@synalinks_export(["synalinks.TurnFaithfulness", "synalinks.rewards.TurnFaithfulness"])
class TurnFaithfulness(RubricsAsJudge):
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

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="turn_faithfulness",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            language_model=language_model,
            rubrics="turn_faithfulness",
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
