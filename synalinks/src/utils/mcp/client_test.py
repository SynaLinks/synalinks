# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import platform
import unittest

import httpx
from mcp.server import FastMCP
from mcp.types import ToolAnnotations

from synalinks.src import testing
from synalinks.src.utils.mcp._test_common import run_streamable_server_multiprocessing
from synalinks.src.utils.mcp.client import MultiServerMCPClient


@unittest.skipUnless(platform.system() == "Linux", "server tests require Linux (multiprocessing/pickling issues on Windows)")
class MCPClientIntegrationTest(testing.TestCase):
    """integration tests for MultiServerMCPClient with actual MCP servers."""

    @classmethod
    def setUpClass(cls):
        """Set up multiple MCP servers for integration testing."""
        # server 1: time server with annotations
        cls.time_server = FastMCP(port=8181)
        
        @cls.time_server.tool(
            annotations=ToolAnnotations(title="Get Time", readOnlyHint=True, idempotentHint=False)
        )
        def get_time() -> str:
            """Get current time"""
            return "5:20:00 PM EST"

        # server 2: status server for custom HTTPX client testing
        cls.status_server = FastMCP(port=8182)
        
        @cls.status_server.tool()
        def get_status() -> str:
            """Get server status"""
            return "Server is running"
            
        @cls.status_server.tool()
        def get_uptime() -> str:
            """Get server uptime"""
            return "24h 30m 15s"

        # server 3: math server for multi-server testing
        cls.math_server = FastMCP(port=8183)
        
        @cls.math_server.tool()
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together"""
            return a + b
            
        @cls.math_server.tool()
        def multiply_numbers(a: int, b: int) -> int:
            """Multiply two numbers together"""
            return a * b

        cls.time_server_context = run_streamable_server_multiprocessing(cls.time_server)
        cls.status_server_context = run_streamable_server_multiprocessing(cls.status_server)
        cls.math_server_context = run_streamable_server_multiprocessing(cls.math_server)
        
        cls.time_server_context.__enter__()
        cls.status_server_context.__enter__()
        cls.math_server_context.__enter__()

        cls.time_connection = {
            "url": "http://localhost:8181/mcp/",
            "transport": "streamable_http",
        }
        
        cls.status_connection = {
            "url": "http://localhost:8182/mcp/",
            "transport": "streamable_http",
        }
        
        cls.math_connection = {
            "url": "http://localhost:8183/mcp/",
            "transport": "streamable_http",
        }

    @classmethod
    def tearDownClass(cls):
        """Tear down all MCP servers."""
        try:
            cls.time_server_context.__exit__(None, None, None)
        except Exception:
            pass

        try:
            cls.status_server_context.__exit__(None, None, None)
        except Exception:
            pass

        try:
            cls.math_server_context.__exit__(None, None, None)
        except Exception:
            pass

    async def test_load_tools_with_annotations(self):
        """Test loading tools from a server that uses tool annotations."""
        client = MultiServerMCPClient({"time": self.time_connection})
        
        tools = await client.get_tools(server_name="time")
        self.assertEqual(len(tools), 1)
        
        tool = tools[0]
        self.assertEqual(tool.name(), "get_time")
        self.assertIn("Get current time", tool.description())

    async def test_load_tools_with_custom_httpx_client_factory(self):
        """Test loading tools with a custom HTTPX client factory."""
        def custom_httpx_client_factory(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                headers=headers,
                timeout=timeout or httpx.Timeout(30.0),
                auth=auth,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )

        status_connection_with_factory = {
            **self.status_connection,
            "httpx_client_factory": custom_httpx_client_factory,
        }
        
        client = MultiServerMCPClient({"status": status_connection_with_factory})

        tools = await client.get_tools(server_name="status")
        self.assertEqual(len(tools), 2)
        
        tool_names = {tool.name() for tool in tools}
        self.assertEqual(tool_names, {"get_status", "get_uptime"})

        status_tool = next(tool for tool in tools if tool.name() == "get_status")
        result = await status_tool.async__call__()
        self.assertEqual(result["response"], "Server is running")

    async def test_load_tools_from_specific_server_no_namespacing(self):
        """Test loading tools from a specific server (no namespacing applied)."""
        client = MultiServerMCPClient({
            "time": self.time_connection,
            "status": self.status_connection,
            "math": self.math_connection,
        })

        # Load tools from specific server - no namespacing
        math_tools = await client.get_tools(server_name="math")
        self.assertEqual(len(math_tools), 2)
        
        tool_names = {tool.name() for tool in math_tools}
        self.assertEqual(tool_names, {"add_numbers", "multiply_numbers"})
        
        add_tool = next(tool for tool in math_tools if tool.name() == "add_numbers")
        result = await add_tool.async__call__(a=5, b=3)
        self.assertEqual(result["response"], "8")

    async def test_load_tools_from_all_servers_with_namespacing(self):
        """Test loading tools from all servers (namespacing applied)."""
        client = MultiServerMCPClient({
            "time": self.time_connection,
            "status": self.status_connection,
            "math": self.math_connection,
        })

        # Load tools from all servers - namespacing applied
        all_tools = await client.get_tools()
        self.assertEqual(len(all_tools), 5)  # 1 + 2 + 2 tools
        
        tool_names = {tool.name() for tool in all_tools}
        expected_names = {
            "time/get_time",
            "status/get_status", 
            "status/get_uptime",
            "math/add_numbers",
            "math/multiply_numbers",
        }
        self.assertEqual(tool_names, expected_names)
        
        # Test calling namespaced tools
        namespaced_add_tool = next(tool for tool in all_tools if tool.name() == "math/add_numbers")
        result = await namespaced_add_tool.async__call__(a=10, b=7)
        self.assertEqual(result["response"], "17")
        
        namespaced_time_tool = next(tool for tool in all_tools if tool.name() == "time/get_time")
        result = await namespaced_time_tool.async__call__()
        self.assertEqual(result["response"], "5:20:00 PM EST")
