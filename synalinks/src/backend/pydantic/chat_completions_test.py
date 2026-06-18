# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json

from synalinks.src import testing
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
        # And the schema is a subset of the chat-completion message keys.
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
