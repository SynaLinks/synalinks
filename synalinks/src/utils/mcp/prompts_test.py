# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import AsyncMock

from mcp.types import (
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
    TextResourceContents,
)

from synalinks import ChatMessage, ChatRole
from synalinks.src import testing
from synalinks.src.utils.mcp.prompts import (
    convert_mcp_prompt_message_to_synalinks_message,
    load_mcp_prompt,
)


class MCPPromptsTest(testing.TestCase):
    def test_convert_mcp_prompt_message_to_synalinks_message_with_text_content_assistant(self):
        message = PromptMessage(role="assistant", content=TextContent(type="text", text="Hello"))
        result = convert_mcp_prompt_message_to_synalinks_message(message)
        self.assertIsInstance(result, ChatMessage)
        self.assertEqual(result.role, ChatRole.ASSISTANT)
        self.assertEqual(result.content, "Hello")

    def test_convert_mcp_prompt_message_to_synalinks_message_with_text_content_user(self):
        message = PromptMessage(role="user", content=TextContent(type="text", text="Hello"))
        result = convert_mcp_prompt_message_to_synalinks_message(message)
        self.assertIsInstance(result, ChatMessage)
        self.assertEqual(result.role, ChatRole.USER)
        self.assertEqual(result.content, "Hello")

    def test_convert_mcp_prompt_message_to_synalinks_message_with_resource_content_assistant(self):
        message = PromptMessage(
            role="assistant",
            content=EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri="message://greeting", mimeType="text/plain", text="Hi"
                ),
            ),
        )

        with self.assertRaises(ValueError):
            convert_mcp_prompt_message_to_synalinks_message(message)

    def test_convert_mcp_prompt_message_to_synalinks_message_with_resource_content_user(self):
        message = PromptMessage(
            role="user",
            content=EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri="message://greeting", mimeType="text/plain", text="hi"
                ),
            ),
        )

        with self.assertRaises(ValueError):
            convert_mcp_prompt_message_to_synalinks_message(message)

    def test_convert_mcp_prompt_message_to_synalinks_message_with_image_content_assistant(self):
        message = PromptMessage(
            role="assistant", content=ImageContent(type="image", mimeType="image/png", data="iVBORw0KGgoAA...")
        )

        with self.assertRaises(ValueError):
            convert_mcp_prompt_message_to_synalinks_message(message)

    def test_convert_mcp_prompt_message_to_synalinks_message_with_image_content_user(self):
        message = PromptMessage(
            role="user", content=ImageContent(type="image", mimeType="image/png", data="iVBORw0KGgoAA...")
        )

        with self.assertRaises(ValueError):
            convert_mcp_prompt_message_to_synalinks_message(message)

    async def test_load_mcp_prompt(self):
        session = AsyncMock()
        session.get_prompt = AsyncMock(
            return_value=AsyncMock(
                messages=[
                    PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
                    PromptMessage(role="assistant", content=TextContent(type="text", text="Hi")),
                ]
            )
        )
        
        response = await load_mcp_prompt(session, "test")
        
        self.assertEqual(len(response.messages), 2)
        self.assertIsInstance(response.messages[0], ChatMessage)
        self.assertEqual(response.messages[0].role, ChatRole.USER)
        self.assertEqual(response.messages[0].content, "Hello")
        self.assertIsInstance(response.messages[1], ChatMessage)
        self.assertEqual(response.messages[1].role, ChatRole.ASSISTANT)
        self.assertEqual(response.messages[1].content, "Hi")
