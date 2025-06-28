from typing import Any

from mcp import ClientSession
from mcp.types import PromptMessage

from synalinks import ChatMessage, ChatMessages, ChatRole


def convert_mcp_prompt_message_to_synalinks_message(
    message: PromptMessage,
) -> ChatMessage:
    """Convert an MCP prompt message to a Synalinks message.

    Args:
        message: MCP prompt message to convert

    Returns:
        A Synalinks message
    """
    if message.content.type == "text":
        if message.role == "user":
            return ChatMessage(role=ChatRole.USER, content=message.content.text)
        elif message.role == "assistant":
            return ChatMessage(role=ChatRole.ASSISTANT, content=message.content.text)
        else:
            raise ValueError(f"Unsupported prompt message role: {message.role}")

    raise ValueError(f"Unsupported prompt message content type: {message.content.type}")


async def load_mcp_prompt(
    session: ClientSession, name: str, *, arguments: dict[str, Any] | None = None
) -> ChatMessages:
    """Load MCP prompt and convert to Synalinks messages."""
    response = await session.get_prompt(name, arguments)
    return ChatMessages(
        messages=[convert_mcp_prompt_message_to_synalinks_message(message) for message in response.messages]
    )
