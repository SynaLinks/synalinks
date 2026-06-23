# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
import uuid
from unittest.mock import patch

from synalinks import modules
from synalinks.src import ops
from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import DataModel
from synalinks.src.modules import Generator
from synalinks.src.modules import Input
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program


class GeneratorModuleTest(testing.TestCase):
    def test_format_message(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        msgs = Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        ).format_messages(
            inputs=Query(query="What is the french city of aerospace and robotics?")
        )
        print(msgs.prettify_json())
        self.assertTrue(len(msgs.messages) == 2)

    def test_format_chat_message(self):
        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        msgs = Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        ).format_messages(
            inputs=ChatMessages(
                messages=[
                    ChatMessage(
                        role="user",
                        content="What is the french city of aerospace and robotics?",
                    ),
                ]
            )
        )
        print(msgs.prettify_json())
        self.assertTrue(len(msgs.messages) == 2)

    def test_format_multimodal_chat_message(self):
        import base64

        from synalinks.src.backend import Image

        class Answer(DataModel):
            answer: str

        png = b"\x89PNG\r\n\x1a\n"

        class _Resp:
            content = png
            headers = {"content-type": "image/png"}

            def raise_for_status(self):
                pass

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url):
                return _Resp()

        # The image is fetched and inlined at construction (see backend.media),
        # so the content list reaching the chat path is already a base64
        # `data:` URI alongside the text part.
        with patch("httpx.Client", _Client):
            image = Image(url="http://example.com/cat.png")

        msgs = Generator(
            data_model=Answer,
            language_model=LanguageModel(model="ollama/mistral"),
        ).format_messages(
            inputs=ChatMessages(
                messages=[
                    ChatMessage(
                        role="user",
                        content=["What is in this picture?", image],
                    ),
                ]
            )
        )
        user = msgs.messages[-1]
        encoded = base64.b64encode(png).decode("ascii")
        self.assertEqual(
            user.content,
            [
                {"type": "text", "text": "What is in this picture?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ],
        )

    def test_format_chat_message_with_tools(self):
        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        msgs = Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        ).format_messages(
            inputs=ChatMessages(
                messages=[
                    ChatMessage(
                        role="user",
                        content="What is the french city of aerospace and robotics?",
                    ),
                    ChatMessage(
                        role="tool",
                        tool_call_id=str(uuid.uuid4()),
                        content={"expression": "2+2"},
                    ),
                ]
            )
        )
        print(msgs.prettify_json())
        self.assertTrue(len(msgs.messages) == 3)

    async def test_format_messages_mixed_input_prepends_user_turn(self):
        """A (data + message-trajectory) input renders the inputs as a leading
        user turn before the conversation: [system, user(inputs), *trajectory].

        (Without it the inputs are dropped and a multi-turn tool-call history
        starts with an assistant tool-call, which Gemini rejects.)
        """

        class Query(DataModel):
            query: str

        language_model = LanguageModel(model="ollama/mistral")
        generator = Generator(
            language_model=language_model,
            instructions="be helpful",
        )

        # Program input + a turn-1 tool-call exchange already in the trajectory.
        trajectory = await ops.concat(
            Query(query="sum the numbers"),
            ChatMessages(
                messages=[
                    ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "run", "arguments": {"x": 1}},
                            }
                        ],
                    ).get_json(),
                    ChatMessage(
                        role="tool",
                        tool_call_id="c1",
                        content={"stdout": "ok"},
                    ).get_json(),
                ]
            ),
        )

        msgs = generator.format_messages(trajectory).get("messages")
        roles = [
            m["role"].value if hasattr(m["role"], "value") else m["role"] for m in msgs
        ]

        # system, user(inputs), assistant(tool_calls), tool — the assistant
        # tool-call is never the first turn after the system instruction.
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])
        # The inputs must actually appear in the leading user turn.
        self.assertIn("sum the numbers", str(msgs[1]["content"]))

    async def test_format_messages_mixed_input_empty_trajectory(self):
        """With an empty trajectory, the inputs still render as a leading
        user turn (so turn 1 of an agent shows the task to the LM)."""

        class Query(DataModel):
            query: str

        language_model = LanguageModel(model="ollama/mistral")
        generator = Generator(
            language_model=language_model,
            instructions="be helpful",
        )

        trajectory = await ops.concat(
            Query(query="hello there"),
            ChatMessages(messages=[]),
        )

        msgs = generator.format_messages(trajectory).get("messages")
        roles = [
            m["role"].value if hasattr(m["role"], "value") else m["role"] for m in msgs
        ]
        self.assertEqual(roles, ["system", "user"])
        self.assertIn("hello there", str(msgs[1]["content"]))

    async def test_format_messages_excludes_output_fields_after_messages(self):
        """Fields ordered after `messages` are outputs (e.g. concatenated by an
        upstream generator) and must not leak into the leading input turn."""

        class Query(DataModel):
            query: str

        class Answer(DataModel):
            answer: str

        language_model = LanguageModel(model="ollama/mistral")
        generator = Generator(
            language_model=language_model,
            instructions="be helpful",
        )

        # {query (input), messages, answer (output)} — the output field sits
        # after `messages` because concat appends it.
        trajectory = await ops.concat(
            await ops.concat(
                Query(query="what is 1 + 1?"),
                ChatMessages(messages=[]),
            ),
            Answer(answer="leaked-output"),
        )

        msgs = generator.format_messages(trajectory).get("messages")
        rendered = " ".join(str(m["content"]) for m in msgs)

        # The input field is rendered, the trailing output field is not.
        self.assertIn("what is 1 + 1?", rendered)
        self.assertNotIn("leaked-output", rendered)

    async def test_format_messages_multiple_input_and_output_fields(self):
        """All fields before `messages` render as input; all fields after it
        (the outputs) are excluded, regardless of how many there are."""

        class Inputs(DataModel):
            query: str
            context: str

        class Outputs(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama/mistral")
        generator = Generator(
            language_model=language_model,
            instructions="be helpful",
        )

        trajectory = await ops.concat(
            await ops.concat(
                Inputs(query="capital of France?", context="geography quiz"),
                ChatMessages(messages=[]),
            ),
            Outputs(rationale="should-not-leak", answer="Paris-should-not-leak"),
        )

        msgs = generator.format_messages(trajectory).get("messages")
        rendered = " ".join(str(m["content"]) for m in msgs)

        self.assertIn("capital of France?", rendered)
        self.assertIn("geography quiz", rendered)
        self.assertNotIn("should-not-leak", rendered)
        self.assertNotIn("Paris-should-not-leak", rendered)

    async def test_format_messages_input_turn_precedes_conversation(self):
        """The input turn is inserted right after the system message and before
        the existing conversation, and trailing outputs never appear."""

        class Query(DataModel):
            query: str

        class Answer(DataModel):
            answer: str

        language_model = LanguageModel(model="ollama/mistral")
        generator = Generator(
            language_model=language_model,
            instructions="be helpful",
        )

        trajectory = await ops.concat(
            await ops.concat(
                Query(query="the-input-query"),
                ChatMessages(
                    messages=[
                        ChatMessage(role="user", content="earlier-user-turn").get_json(),
                        ChatMessage(
                            role="assistant", content="earlier-assistant-turn"
                        ).get_json(),
                    ]
                ),
            ),
            Answer(answer="trailing-output"),
        )

        msgs = generator.format_messages(trajectory).get("messages")
        roles = [
            m["role"].value if hasattr(m["role"], "value") else m["role"] for m in msgs
        ]
        contents = [str(m["content"]) for m in msgs]

        # system, user(inputs), then the prior conversation in order.
        self.assertEqual(roles, ["system", "user", "user", "assistant"])
        self.assertIn("the-input-query", contents[1])
        self.assertIn("earlier-user-turn", contents[2])
        self.assertIn("earlier-assistant-turn", contents[3])
        self.assertNotIn("trailing-output", " ".join(contents))

    async def test_format_messages_no_input_fields_before_messages(self):
        """When `messages` is the first field, there are no inputs to render and
        no leading user turn is inserted (only outputs trail it)."""

        class Answer(DataModel):
            answer: str

        language_model = LanguageModel(model="ollama/mistral")
        generator = Generator(
            language_model=language_model,
            instructions="be helpful",
        )

        trajectory = await ops.concat(
            ChatMessages(
                messages=[
                    ChatMessage(role="user", content="just-a-question").get_json(),
                ]
            ),
            Answer(answer="trailing-output"),
        )

        msgs = generator.format_messages(trajectory).get("messages")
        roles = [
            m["role"].value if hasattr(m["role"], "value") else m["role"] for m in msgs
        ]
        rendered = " ".join(str(m["content"]) for m in msgs)

        # No synthetic input turn — just system + the conversation.
        self.assertEqual(roles, ["system", "user"])
        self.assertIn("just-a-question", rendered)
        self.assertNotIn("trailing-output", rendered)

    async def test_format_messages_contained_chat_message_with_extra_fields(self):
        """An input that only *contains* a chat message but carries extra sibling
        fields (and no `messages` list) must not be splatted into a single strict
        ChatMessage — that trips ChatMessage's `extra="forbid"`.

        Mirrors `LMAsJudge`, which concatenates a `gold_`-prefixed reference with
        the prediction: the merged model is `{gold_role, gold_content, role,
        content}` — chat-message-shaped, but not strictly a chat message. It is
        rendered as input data (system + user turn), not sent as a chat message.
        """

        language_model = LanguageModel(model="ollama/mistral")
        generator = Generator(
            language_model=language_model,
            instructions="be helpful",
        )

        merged = await ops.concat(
            await ops.prefix(
                ChatMessage(role="assistant", content="gold-answer"),
                prefix="gold",
            ),
            ChatMessage(role="assistant", content="predicted-answer"),
        )

        # Must not raise (previously: ValidationError — extra inputs forbidden).
        msgs = generator.format_messages(merged).get("messages")
        roles = [
            m["role"].value if hasattr(m["role"], "value") else m["role"] for m in msgs
        ]
        rendered = " ".join(str(m["content"]) for m in msgs)

        self.assertEqual(roles, ["system", "user"])
        # Both the prediction and the gold reference survive into the prompt.
        self.assertIn("predicted-answer", rendered)
        self.assertIn("gold-answer", rendered)

    def test_format_message_with_instructions(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        msgs = Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
            instructions="You are an helpfull assistant",
        ).format_messages(
            Query(query="What is the french city of aerospace and robotics?")
        )
        print(msgs.prettify_json())
        self.assertTrue(len(msgs.messages) == 2)

    def test_format_message_with_examples(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        msgs = Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
            examples=[
                (
                    {"query": "What is the capital of France?"},
                    {
                        "rationale": "The capital of France is well known",
                        "answer": "Paris",
                    },
                )
            ],
        ).format_messages(
            Query(query="What is the french city of aerospace and robotics?")
        )
        print(msgs.prettify_json())
        self.assertTrue(len(msgs.messages) == 2)

    def test_format_message_with_examples_and_instructions(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        msgs = Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
            examples=[
                (
                    {
                        "query": "What is the capital of France?",
                    },
                    {
                        "rationale": "The capital of France is well known",
                        "answer": "Paris",
                    },
                )
            ],
            instructions="You are an helpfull assistant",
        ).format_messages(
            Query(query="What is the french city of aerospace and robotics?")
        )
        print(msgs.prettify_json())
        self.assertTrue(len(msgs.messages) == 2)

    @patch("litellm.acompletion")
    async def test_basic_functional_setup(self, mock_completion):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        expected_string = (
            """{"rationale": "Toulouse hosts numerous research institutions """
            """and universities that specialize in aerospace engineering and """
            """robotics, such as the Institut Supérieur de l'Aéronautique et """
            """de l'Espace (ISAE-SUPAERO) and the French National Centre for """
            """Scientific Research (CNRS)","""
            """ "answer": "Toulouse"}"""
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        x0 = Input(data_model=Query)
        x1 = await Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        )(x0)
        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
        )

        result = await program(
            Query(query="What is the french city of aerospace and robotics?")
        )

        self.assertEqual(result.get_json(), json.loads(expected_string))

    @patch("litellm.acompletion")
    async def test_basic_functional_setup_with_schema(self, mock_completion):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama/mistral")

        expected_string = (
            """{"rationale": "Toulouse hosts numerous research institutions """
            """and universities that specialize in aerospace engineering and """
            """robotics, such as the Institut Supérieur de l'Aéronautique et """
            """de l'Espace (ISAE-SUPAERO) and the French National Centre for """
            """Scientific Research (CNRS)","""
            """ "answer": "Toulouse"}"""
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        x0 = Input(data_model=Query)
        x1 = await Generator(
            schema=AnswerWithRationale.get_schema(),
            language_model=language_model,
        )(x0)
        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
        )

        result = await program(
            Query(query="What is the french city of aerospace and robotics?")
        )

        self.assertEqual(result.get_json(), json.loads(expected_string))

    @patch("litellm.acompletion")
    async def test_basic_subclassing_setup(self, mock_completion):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama/mistral")

        expected_string = (
            """{"rationale": "Toulouse hosts numerous research institutions """
            """and universities that specialize in aerospace engineering and """
            """robotics, such as the Institut Supérieur de l'Aéronautique et """
            """de l'Espace (ISAE-SUPAERO) and the French National Centre for """
            """Scientific Research (CNRS)","""
            """ "answer": "Toulouse"}"""
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        class ChainOfThought(Program):
            def __init__(self, language_model):
                super().__init__()
                self.answer = Generator(
                    data_model=AnswerWithRationale, language_model=language_model
                )

            async def call(self, inputs):
                x = await self.answer(inputs)
                return x

        program = ChainOfThought(language_model=language_model)

        result = await program(
            Query(query="What is the french city of aerospace and robotics?")
        )

        self.assertEqual(result.get_json(), json.loads(expected_string))

    def test_serialization(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama/mistral")

        generator = Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        )

        serialized_dict = modules.serialize(generator)
        new_generator = modules.deserialize(serialized_dict)
        # check that the nested object are good
        self.assertEqual(
            str(new_generator.language_model),
            str(generator.language_model),
        )
