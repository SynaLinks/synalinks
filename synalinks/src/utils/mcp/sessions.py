import os
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import AsyncIterator
from typing import Literal
from typing import Protocol
from typing import TypedDict

import httpx2
from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client
from mcp.client.streamable_http import streamable_http_client

EncodingErrorHandler = Literal["strict", "ignore", "replace"]

DEFAULT_ENCODING = "utf-8"
DEFAULT_ENCODING_ERROR_HANDLER: EncodingErrorHandler = "strict"

DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5

DEFAULT_STREAMABLE_HTTP_TIMEOUT = 30.0
DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT = 60.0 * 5


class McpHttpClientFactory(Protocol):
    """Builds the HTTP client an SSE or Streamable HTTP session talks through.

    The MCP SDK (v2) does its HTTP over `httpx2`, so the factory returns an
    `httpx2.AsyncClient`; `mcp.client.streamable_http.create_mcp_http_client`
    is the default and a good starting point for a custom one.
    """

    def __call__(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx2.Timeout | None = None,
        auth: httpx2.Auth | None = None,
    ) -> httpx2.AsyncClient: ...


def _seconds(value: float | timedelta) -> float:
    """Timeouts are plain seconds in MCP SDK v2; accept the v1 `timedelta` too."""
    return value.total_seconds() if isinstance(value, timedelta) else float(value)


class StdioConnection(TypedDict):
    transport: Literal["stdio"]

    command: str
    """The executable to run to start the server."""

    args: list[str]
    """Command line arguments to pass to the executable."""

    env: dict[str, str] | None
    """The environment to use when spawning the process."""

    cwd: str | Path | None
    """The working directory to use when spawning the process."""

    encoding: str
    """The text encoding used when sending/receiving messages to the server."""

    encoding_error_handler: EncodingErrorHandler
    """
    The text encoding error handler.

    See https://docs.python.org/3/library/codecs.html#codec-base-classes for
    explanations of possible values.
    """

    session_kwargs: dict[str, Any] | None
    """Additional keyword arguments to pass to the ClientSession."""


class SSEConnection(TypedDict):
    transport: Literal["sse"]

    url: str
    """The URL of the SSE endpoint to connect to."""

    headers: dict[str, Any] | None
    """HTTP headers to send to the SSE endpoint."""

    timeout: float
    """HTTP timeout."""

    sse_read_timeout: float
    """SSE read timeout."""

    session_kwargs: dict[str, Any] | None
    """Additional keyword arguments to pass to the ClientSession."""

    httpx_client_factory: McpHttpClientFactory | None
    """Custom factory for httpx2.AsyncClient (optional)."""


class StreamableHttpConnection(TypedDict):
    transport: Literal["streamable_http"]

    url: str
    """The URL of the endpoint to connect to."""

    headers: dict[str, Any] | None
    """HTTP headers to send to the endpoint."""

    timeout: float | timedelta
    """HTTP timeout, in seconds (a `timedelta` is accepted too)."""

    sse_read_timeout: float | timedelta
    """How long (in seconds) the client will wait for a new event before disconnecting.
    All other HTTP operations are controlled by `timeout`."""

    terminate_on_close: bool
    """Whether to terminate the session on close."""

    session_kwargs: dict[str, Any] | None
    """Additional keyword arguments to pass to the ClientSession."""

    httpx_client_factory: McpHttpClientFactory | None
    """Custom factory for httpx2.AsyncClient (optional)."""


Connection = StdioConnection | SSEConnection | StreamableHttpConnection


@asynccontextmanager
async def _create_stdio_session(
    *,
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    encoding: str = DEFAULT_ENCODING,
    encoding_error_handler: Literal[
        "strict", "ignore", "replace"
    ] = DEFAULT_ENCODING_ERROR_HANDLER,
    session_kwargs: dict[str, Any] | None = None,
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server using stdio.

    Args:
        command: Command to execute
        args: Arguments for the command
        env: Environment variables for the command
        cwd: Working directory for the command
        encoding: Character encoding
        encoding_error_handler: How to handle encoding errors
        session_kwargs: Additional keyword arguments to pass to the ClientSession
    """
    # NOTE: execution commands (e.g., `uvx` / `npx`) require PATH envvar to be set.
    # To address this, we automatically inject existing PATH envvar into the `env` value,
    # if it's not already set.
    env = env or {}
    if "PATH" not in env:
        env["PATH"] = os.environ.get("PATH", "")

    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env,
        cwd=cwd,
        encoding=encoding,
        encoding_error_handler=encoding_error_handler,
    )

    # Create and store the connection
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, **(session_kwargs or {})) as session:
            yield session


@asynccontextmanager
async def _create_sse_session(
    *,
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    sse_read_timeout: float = DEFAULT_SSE_READ_TIMEOUT,
    session_kwargs: dict[str, Any] | None = None,
    httpx_client_factory: McpHttpClientFactory | None = None,
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server using SSE.

    Args:
        url: (str) URL of the SSE server
        headers (dict): HTTP headers to send to the SSE endpoint
        timeout: HTTP timeout
        sse_read_timeout: SSE read timeout
        session_kwargs: Additional keyword arguments to pass to the ClientSession
        httpx_client_factory: Custom factory for httpx.AsyncClient (optional)
    """
    # Create and store the connection
    kwargs = {}
    if httpx_client_factory is not None:
        kwargs["httpx_client_factory"] = httpx_client_factory

    async with sse_client(url, headers, timeout, sse_read_timeout, **kwargs) as (
        read,
        write,
    ):
        async with ClientSession(read, write, **(session_kwargs or {})) as session:
            yield session


@asynccontextmanager
async def _create_streamable_http_session(
    *,
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: float | timedelta = DEFAULT_STREAMABLE_HTTP_TIMEOUT,
    sse_read_timeout: float | timedelta = DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT,
    terminate_on_close: bool = True,
    session_kwargs: dict[str, Any] | None = None,
    httpx_client_factory: McpHttpClientFactory | None = None,
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server using Streamable HTTP.

    Args:
        url (str): URL of the endpoint to connect to
        headers (dict): HTTP headers to send to the endpoint
        timeout: HTTP timeout, in seconds
        sse_read_timeout: How long (in seconds) the client will wait for a new event
            before disconnecting
        terminate_on_close: Whether to terminate the session on close
        session_kwargs: Additional keyword arguments to pass to the ClientSession
        httpx_client_factory: Custom factory for httpx2.AsyncClient (optional)
    """
    # MCP SDK v2 no longer takes headers and timeouts itself: they travel on
    # the HTTP client, which the caller builds and owns.
    factory = httpx_client_factory or create_mcp_http_client
    http_client = factory(
        headers=headers,
        timeout=httpx2.Timeout(_seconds(timeout), read=_seconds(sse_read_timeout)),
    )
    async with http_client:
        async with streamable_http_client(
            url, http_client=http_client, terminate_on_close=terminate_on_close
        ) as (read, write):
            async with ClientSession(read, write, **(session_kwargs or {})) as session:
                yield session


@asynccontextmanager
async def create_session(
    connection: Connection,
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server.

    Args:
        connection: Connection config to use to connect to the server

    Raises:
        ValueError: If transport is not recognized
        ValueError: If required parameters for the specified transport are missing

    Yields:
        A ClientSession
    """

    if "transport" not in connection:
        raise ValueError(
            "Configuration error: Missing 'transport' key in server configuration. "
            "Each server must include 'transport' with one of: "
            "'stdio', 'sse', 'streamable_http'. "
        )

    transport = connection["transport"]
    if transport == "sse":
        if "url" not in connection:
            raise ValueError("'url' parameter is required for SSE connection")
        async with _create_sse_session(
            url=connection["url"],
            headers=connection.get("headers"),
            timeout=connection.get("timeout", DEFAULT_HTTP_TIMEOUT),
            sse_read_timeout=connection.get("sse_read_timeout", DEFAULT_SSE_READ_TIMEOUT),
            session_kwargs=connection.get("session_kwargs"),
            httpx_client_factory=connection.get("httpx_client_factory"),
        ) as session:
            yield session
    elif transport == "streamable_http":
        if "url" not in connection:
            raise ValueError("'url' parameter is required for Streamable HTTP connection")
        async with _create_streamable_http_session(
            url=connection["url"],
            headers=connection.get("headers"),
            timeout=connection.get("timeout", DEFAULT_STREAMABLE_HTTP_TIMEOUT),
            sse_read_timeout=connection.get(
                "sse_read_timeout", DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT
            ),
            session_kwargs=connection.get("session_kwargs"),
            httpx_client_factory=connection.get("httpx_client_factory"),
        ) as session:
            yield session
    elif transport == "stdio":
        if "command" not in connection:
            raise ValueError("'command' parameter is required for stdio connection")
        if "args" not in connection:
            raise ValueError("'args' parameter is required for stdio connection")
        async with _create_stdio_session(
            command=connection["command"],
            args=connection["args"],
            env=connection.get("env"),
            cwd=connection.get("cwd"),
            encoding=connection.get("encoding", DEFAULT_ENCODING),
            encoding_error_handler=connection.get(
                "encoding_error_handler", DEFAULT_ENCODING_ERROR_HANDLER
            ),
            session_kwargs=connection.get("session_kwargs"),
        ) as session:
            yield session
    elif transport == "websocket":
        raise ValueError(
            "The 'websocket' transport was removed in MCP SDK v2 and is no longer "
            "supported; connect over 'streamable_http' (or 'sse') instead."
        )
    else:
        raise ValueError(
            f"Unsupported transport: {transport}. "
            f"Must be one of: 'stdio', 'sse', 'streamable_http'"
        )
