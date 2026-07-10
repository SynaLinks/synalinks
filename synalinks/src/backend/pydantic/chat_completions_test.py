# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from typing import get_args

from synalinks.src import testing
from synalinks.src.backend.pydantic.chat_completions import ChatCompletionFunctionCall
from synalinks.src.backend.pydantic.chat_completions import ChatCompletionMessage
from synalinks.src.backend.pydantic.chat_completions import ChatCompletionToolCall
from synalinks.src.backend.pydantic.chat_completions import from_chat_completion_message
from synalinks.src.backend.pydantic.chat_completions import to_chat_completion_message
from synalinks.src.backend.pydantic.common import ChatMessage
from synalinks.src.backend.pydantic.common import ChatRole
from synalinks.src.backend.pydantic.common import ToolCall
from synalinks.src.backend.pydantic.common import ToolCallFunction


def _assistant_with_tool_call():
    return ChatMessage(
        role=ChatRole.ASSISTANT,
        content="",
        tool_calls=[
            ToolCall(
                id="call_0",
                function=ToolCallFunction(
                    name="calculate",
                    arguments={"expression": "1 + 1"},
                ),
            )
        ],
    )


class ChatCompletionsTest(testing.TestCase):
    def test_message_api_has_exact_key_parity_with_chat_completion(self):
        # The synalinks message API and the chat-completion wire message
        # expose exactly the same keys — parity is on names only; value
        # types may deliberately differ (e.g. parsed-dict tool arguments
        # vs the JSON-encoded wire string).
        self.assertEqual(
            set(ChatMessage.model_fields),
            set(ChatCompletionMessage.model_fields),
        )
        self.assertEqual(
            set(ToolCall.model_fields),
            set(ChatCompletionToolCall.model_fields),
        )
        self.assertEqual(
            set(ToolCallFunction.model_fields),
            set(ChatCompletionFunctionCall.model_fields),
        )
        # Roles too: the ChatRole enum matches the wire role literal.
        wire_roles = get_args(ChatCompletionMessage.model_fields["role"].annotation)
        self.assertEqual({role.value for role in ChatRole}, set(wire_roles))

    def test_tool_call_is_nested_on_the_wire(self):
        wire = to_chat_completion_message(_assistant_with_tool_call())
        # OpenAI nests name/arguments under `function` with a `type` discriminator.
        tc = wire.tool_calls[0]
        self.assertEqual(tc.id, "call_0")
        self.assertEqual(tc.type, "function")
        self.assertEqual(tc.function.name, "calculate")
        # `arguments` is a JSON-encoded string on the wire, not a dict.
        self.assertIsInstance(tc.function.arguments, str)
        self.assertEqual(json.loads(tc.function.arguments), {"expression": "1 + 1"})
        # Assistant message that only calls tools has null content.
        self.assertIsNone(wire.content)

    def test_tool_call_round_trip_is_identity(self):
        original = _assistant_with_tool_call()
        restored = from_chat_completion_message(to_chat_completion_message(original))
        rtc = restored.tool_calls[0]
        self.assertEqual(rtc.id, "call_0")
        self.assertEqual(rtc.type, "function")
        self.assertEqual(rtc.function.name, "calculate")
        # Back to a parsed dict internally.
        self.assertEqual(rtc.function.arguments, {"expression": "1 + 1"})

    def test_reasoning_content_round_trip(self):
        msg = ChatMessage(
            role=ChatRole.ASSISTANT,
            content="done",
            reasoning_content="step-by-step",
            thinking_blocks=[{"type": "thinking", "signature": "abc"}],
        )
        wire = to_chat_completion_message(msg)
        self.assertEqual(wire.reasoning_content, "step-by-step")
        self.assertEqual(wire.thinking_blocks, [{"type": "thinking", "signature": "abc"}])
        restored = from_chat_completion_message(wire)
        self.assertEqual(restored.reasoning_content, "step-by-step")
        self.assertEqual(
            restored.thinking_blocks, [{"type": "thinking", "signature": "abc"}]
        )

    def test_thinking_is_a_deprecated_alias_for_reasoning_content(self):
        # Legacy `thinking` keyword still constructs, mapping onto
        # `reasoning_content`; the canonical key is the chat-completion one.
        with self.assertWarns(DeprecationWarning):
            msg = ChatMessage(role=ChatRole.ASSISTANT, thinking="legacy")
        self.assertEqual(msg.reasoning_content, "legacy")
        # The serialized message exposes only the canonical key.
        self.assertIn("reasoning_content", msg.get_json())
        self.assertNotIn("thinking", msg.get_json())
        # And the schema keys stay in exact parity with the chat-completion
        # message keys (no `thinking` extra).
        self.assertIn("reasoning_content", ChatMessage.get_schema()["properties"])
        self.assertNotIn("thinking", ChatMessage.get_schema()["properties"])
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(msg.thinking, "legacy")

    def test_tool_result_dict_content_is_json_encoded(self):
        msg = ChatMessage(
            role=ChatRole.TOOL,
            tool_call_id="call_0",
            content={"stdout": "ok"},
        )
        wire = to_chat_completion_message(msg)
        self.assertEqual(wire.role, "tool")
        self.assertEqual(wire.tool_call_id, "call_0")
        self.assertEqual(json.loads(wire.content), {"stdout": "ok"})

    def test_plain_user_message_round_trip(self):
        msg = ChatMessage(role=ChatRole.USER, content="hello")
        restored = from_chat_completion_message(to_chat_completion_message(msg))
        self.assertEqual(restored.role, ChatRole.USER)
        self.assertEqual(restored.content, "hello")
        self.assertIsNone(restored.tool_calls)

    def test_author_name_round_trip(self):
        msg = ChatMessage(role=ChatRole.USER, content="hello", name="alice")
        wire = to_chat_completion_message(msg)
        self.assertEqual(wire.name, "alice")
        restored = from_chat_completion_message(wire)
        self.assertEqual(restored.name, "alice")

    def test_refusal_and_audio_round_trip(self):
        audio = {"id": "audio_0", "data": "base64==", "transcript": "hi"}
        msg = ChatMessage(
            role=ChatRole.ASSISTANT,
            refusal="I cannot help with that.",
            audio=audio,
        )
        wire = to_chat_completion_message(msg)
        self.assertEqual(wire.refusal, "I cannot help with that.")
        self.assertEqual(wire.audio, audio)
        restored = from_chat_completion_message(wire)
        self.assertEqual(restored.refusal, "I cannot help with that.")
        self.assertEqual(restored.audio, audio)

    def test_legacy_function_role_name_links_tool_call(self):
        msg = ChatMessage(role=ChatRole.FUNCTION, content="2", name="calculate")
        wire = to_chat_completion_message(msg)
        # OpenAI carries the function name in `name` on the legacy role.
        self.assertEqual(wire.role, "function")
        self.assertEqual(wire.name, "calculate")
        restored = from_chat_completion_message(wire)
        # Mapped to the modern tool role, name kept and used as the link.
        self.assertEqual(restored.role, ChatRole.TOOL)
        self.assertEqual(restored.name, "calculate")
        self.assertEqual(restored.tool_call_id, "calculate")

    def test_string_arguments_pass_through_to_the_wire(self):
        # A ChatMessage may hold arguments in the wire (JSON string) form;
        # the already-encoded string passes through without re-encoding.
        msg = ChatMessage(
            role=ChatRole.ASSISTANT,
            tool_calls=[
                ToolCall(
                    id="call_0",
                    function=ToolCallFunction(
                        name="calculate",
                        arguments='{"expression": "1 + 1"}',
                    ),
                )
            ],
        )
        wire = to_chat_completion_message(msg)
        self.assertEqual(wire.tool_calls[0].function.arguments, '{"expression": "1 + 1"}')
        # A wire string is known to be JSON-encoded, so it's parsed back.
        restored = from_chat_completion_message(wire)
        self.assertEqual(
            restored.tool_calls[0].function.arguments, {"expression": "1 + 1"}
        )

    def test_dict_arguments_on_the_wire_are_kept(self):
        # Type-driven conversion: a dict is already parsed, so it's kept as-is.
        wire = ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                ChatCompletionToolCall(
                    id="call_0",
                    function=ChatCompletionFunctionCall(
                        name="calculate",
                        arguments={"expression": "1 + 1"},
                    ),
                )
            ],
        )
        restored = from_chat_completion_message(wire)
        self.assertEqual(
            restored.tool_calls[0].function.arguments, {"expression": "1 + 1"}
        )
