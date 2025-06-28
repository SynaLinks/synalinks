# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import base64
from unittest.mock import AsyncMock

from mcp.types import (
    BlobResourceContents,
    ListResourcesResult,
    ReadResourceResult,
    Resource,
    ResourceContents,
    TextResourceContents,
)

from synalinks import GenericOutputs
from synalinks.src import testing
from synalinks.src.utils.mcp.resources import (
    convert_mcp_resource_to_synalinks_generic_resource,
    get_mcp_resource,
    load_mcp_resources,
)


class MCPResourcesTest(testing.TestCase):
    def test_convert_mcp_resource_to_synalinks_generic_resource_with_text(self):
        uri = "file:///test.txt"
        contents = TextResourceContents(uri=uri, mimeType="text/plain", text="Hello, world!")

        resource = convert_mcp_resource_to_synalinks_generic_resource(uri, contents)

        self.assertIsInstance(resource, GenericOutputs)
        self.assertEqual(resource.outputs["data"], "Hello, world!")
        self.assertEqual(resource.outputs["mime_type"], "text/plain")
        self.assertEqual(resource.outputs["metadata"]["uri"], uri)

    def test_convert_mcp_resource_to_synalinks_generic_resource_with_blob(self):
        uri = "file:///test.png"
        original_data = b"binary-image-data"
        base64_blob = base64.b64encode(original_data).decode()

        contents = BlobResourceContents(uri=uri, mimeType="image/png", blob=base64_blob)

        resource = convert_mcp_resource_to_synalinks_generic_resource(uri, contents)

        self.assertIsInstance(resource, GenericOutputs)
        self.assertEqual(resource.outputs["data"], original_data)
        self.assertEqual(resource.outputs["mime_type"], "image/png")
        self.assertEqual(resource.outputs["metadata"]["uri"], uri)

    def test_convert_mcp_resource_to_synalinks_generic_resource_with_invalid_type(self):
        class DummyContent(ResourceContents):
            pass

        with self.assertRaises(ValueError):
            convert_mcp_resource_to_synalinks_generic_resource("file:///dummy", DummyContent())

    async def test_get_mcp_resource_with_contents(self):
        session = AsyncMock()
        uri = "file:///test.txt"

        session.read_resource = AsyncMock(
            return_value=ReadResourceResult(
                contents=[
                    TextResourceContents(uri=uri, mimeType="text/plain", text="Content 1"),
                    TextResourceContents(uri=uri, mimeType="text/plain", text="Content 2"),
                ]
            )
        )

        resources = await get_mcp_resource(session, uri)

        self.assertEqual(len(resources), 2)
        self.assertTrue(all(isinstance(r, GenericOutputs) for r in resources))
        self.assertEqual(resources[0].outputs["data"], "Content 1")
        self.assertEqual(resources[1].outputs["data"], "Content 2")

    async def test_get_mcp_resource_with_text_and_blob(self):
        session = AsyncMock()
        uri = "file:///mixed"

        original_data = b"some-binary-content"
        base64_blob = base64.b64encode(original_data).decode()

        session.read_resource = AsyncMock(
            return_value=ReadResourceResult(
                contents=[
                    TextResourceContents(uri=uri, mimeType="text/plain", text="Hello Text"),
                    BlobResourceContents(
                        uri=uri, mimeType="application/octet-stream", blob=base64_blob
                    ),
                ]
            )
        )

        results = await get_mcp_resource(session, uri)

        self.assertEqual(len(results), 2)

        self.assertIsInstance(results[0], GenericOutputs)
        self.assertEqual(results[0].outputs["data"], "Hello Text")
        self.assertEqual(results[0].outputs["mime_type"], "text/plain")

        self.assertIsInstance(results[1], GenericOutputs)
        self.assertEqual(results[1].outputs["data"], original_data)
        self.assertEqual(results[1].outputs["mime_type"], "application/octet-stream")

    async def test_get_mcp_resource_with_empty_contents(self):
        session = AsyncMock()
        uri = "file:///empty.txt"

        session.read_resource = AsyncMock(return_value=ReadResourceResult(contents=[]))

        resources = await get_mcp_resource(session, uri)

        self.assertEqual(len(resources), 0)
        session.read_resource.assert_called_once_with(uri)

    async def test_load_mcp_resources_with_list_of_uris(self):
        session = AsyncMock()
        uri1 = "file:///test1.txt"
        uri2 = "file:///test2.txt"

        session.read_resource = AsyncMock()
        session.read_resource.side_effect = [
            ReadResourceResult(
                contents=[
                    TextResourceContents(uri=uri1, mimeType="text/plain", text="Content from test1")
                ]
            ),
            ReadResourceResult(
                contents=[
                    TextResourceContents(uri=uri2, mimeType="text/plain", text="Content from test2")
                ]
            ),
        ]

        resources = await load_mcp_resources(session, uris=[uri1, uri2])

        self.assertEqual(len(resources), 2)
        self.assertTrue(all(isinstance(r, GenericOutputs) for r in resources))
        self.assertEqual(resources[0].outputs["data"], "Content from test1")
        self.assertEqual(resources[1].outputs["data"], "Content from test2")
        self.assertEqual(resources[0].outputs["metadata"]["uri"], uri1)
        self.assertEqual(resources[1].outputs["metadata"]["uri"], uri2)
        self.assertEqual(session.read_resource.call_count, 2)

    async def test_load_mcp_resources_with_single_uri_string(self):
        session = AsyncMock()
        uri = "file:///test.txt"

        session.read_resource = AsyncMock(
            return_value=ReadResourceResult(
                contents=[
                    TextResourceContents(uri=uri, mimeType="text/plain", text="Content from test")
                ]
            )
        )

        resources = await load_mcp_resources(session, uris=uri)

        self.assertEqual(len(resources), 1)
        self.assertIsInstance(resources[0], GenericOutputs)
        self.assertEqual(resources[0].outputs["data"], "Content from test")
        self.assertEqual(resources[0].outputs["metadata"]["uri"], uri)
        session.read_resource.assert_called_once_with(uri)

    async def test_load_mcp_resources_with_all_resources(self):
        session = AsyncMock()

        session.list_resources = AsyncMock(
            return_value=ListResourcesResult(
                resources=[
                    Resource(uri="file:///test1.txt", name="test1.txt", mimeType="text/plain"),
                    Resource(uri="file:///test2.txt", name="test2.txt", mimeType="text/plain"),
                ]
            )
        )

        session.read_resource = AsyncMock()
        session.read_resource.side_effect = [
            ReadResourceResult(
                contents=[
                    TextResourceContents(
                        uri="file:///test1.txt", mimeType="text/plain", text="Content from test1"
                    )
                ]
            ),
            ReadResourceResult(
                contents=[
                    TextResourceContents(
                        uri="file:///test2.txt", mimeType="text/plain", text="Content from test2"
                    )
                ]
            ),
        ]

        resources = await load_mcp_resources(session)

        self.assertEqual(len(resources), 2)
        self.assertEqual(resources[0].outputs["data"], "Content from test1")
        self.assertEqual(resources[1].outputs["data"], "Content from test2")
        self.assertTrue(session.list_resources.called)
        self.assertEqual(session.read_resource.call_count, 2)

    async def test_load_mcp_resources_with_error_handling(self):
        session = AsyncMock()
        uri1 = "file:///valid.txt"
        uri2 = "file:///error.txt"

        session.read_resource = AsyncMock()
        session.read_resource.side_effect = [
            ReadResourceResult(
                contents=[TextResourceContents(uri=uri1, mimeType="text/plain", text="Valid content")]
            ),
            Exception("Resource not found"),
        ]

        with self.assertRaises(RuntimeError) as exc_info:
            await load_mcp_resources(session, uris=[uri1, uri2])

        self.assertIn("Error fetching resource", str(exc_info.exception))

    async def test_load_mcp_resources_with_blob_content(self):
        session = AsyncMock()
        uri = "file:///with_blob"
        original_data = b"binary data"
        base64_blob = base64.b64encode(original_data).decode()

        session.read_resource = AsyncMock(
            return_value=ReadResourceResult(
                contents=[
                    BlobResourceContents(uri=uri, mimeType="application/octet-stream", blob=base64_blob)
                ]
            )
        )

        resources = await load_mcp_resources(session, uris=uri)

        self.assertEqual(len(resources), 1)
        self.assertIsInstance(resources[0], GenericOutputs)
        self.assertEqual(resources[0].outputs["data"], original_data)
        self.assertEqual(resources[0].outputs["mime_type"], "application/octet-stream")
