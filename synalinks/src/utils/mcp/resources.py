import base64

from mcp import ClientSession
from mcp.types import BlobResourceContents, ResourceContents, TextResourceContents

from synalinks import GenericOutputs


def convert_mcp_resource_to_synalinks_generic_resource(
    resource_uri: str,
    contents: ResourceContents,
) -> GenericOutputs:
    """Convert an MCP resource content to a Synalinks GenericOutputs resource.

    Args:
        resource_uri: URI of the resource
        contents: The resource contents

    Returns:
        A Synalinks GenericOutputs resource
    """
    if isinstance(contents, TextResourceContents):
        data = contents.text
    elif isinstance(contents, BlobResourceContents):
        data = base64.b64decode(contents.blob)
    else:
        raise ValueError(f"Unsupported content type for URI {resource_uri}")
    
    metadata = {"uri": resource_uri}

    return GenericOutputs(
        outputs=dict(data=data, mime_type=contents.mimeType, metadata=metadata)
    )


async def get_mcp_resource(session: ClientSession, uri: str) -> list[GenericOutputs]:
    """Fetch a single MCP resource and convert it to Synalinks GenericOutputs resources.

    Args:
        session: MCP client session
        uri: URI of the resource to fetch

    Returns:
        A list of Synalinks GenericOutputs resources
    """
    contents_result = await session.read_resource(uri)
    if not contents_result.contents or len(contents_result.contents) == 0:
        return []

    return [
        convert_mcp_resource_to_synalinks_generic_resource(uri, content) for content in contents_result.contents
    ]


async def load_mcp_resources(
    session: ClientSession,
    *,
    uris: str | list[str] | None = None,
) -> list[GenericOutputs]:
    """Load MCP resources and convert them to Synalinks GenericOutputs resources.

    Args:
        session: MCP client session
        uris: List of URIs to load.
            If None, all resources will be loaded.
            NOTE: if you specify None, dynamic resources will NOT be loaded,
            as they need the parameters to be provided,
            and are ignored by MCP SDK's session.list_resources() method.

    Returns:
        A list of Synalinks GenericOutputs resources
    """
    blobs = []

    if uris is None:
        resources_list = await session.list_resources()
        uri_list = [r.uri for r in resources_list.resources]
    elif isinstance(uris, str):
        uri_list = [uris]
    else:
        uri_list = uris

    for uri in uri_list:
        try:
            resource_blobs = await get_mcp_resource(session, uri)
            blobs.extend(resource_blobs)
        except Exception as e:
            raise RuntimeError(f"Error fetching resource {uri}") from e

    return blobs
