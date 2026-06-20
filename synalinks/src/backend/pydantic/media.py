# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Multimodal content types.

A modality is just a *content part* of a `Message`: the OpenAI/litellm
chat-completion `content` is either a string or a list of parts, and an image
or an audio clip is one of those parts. So multimodal input is plain
chat-completion — `Image` and `Audio` are special types you drop straight into
a `ChatMessage`'s content list, mixed with the text that usually goes with
them:

```python
synalinks.ChatMessage(
    role="user",
    content=[
        "What is in this picture? Answer in one sentence.",
        synalinks.Image(url="https://example.com/cat.png"),
    ],
)
```

`ChatMessage` normalizes each element to its wire part (a plain `str` becomes a
`text` part, an `Image`/`Audio` becomes its own part), so the message stays a
strict chat-completion message and flows through the normal generator path.

## Two ways a source becomes a payload

A `url` or `path` is just a *reference*; at some point the actual bytes have to
be read and inlined as a base64 ``data:`` URI so every provider — including
local ones that cannot fetch a URL themselves — receives the same self-contained
payload. There are two moments that can happen, for two different use cases:

1. **At construction**, when you build an `Image(url=...)` / `Audio(path=...)`
   in Python. The bytes are fetched/read immediately, so an unreachable source
   fails loudly right where it is written and the object is reusable without
   re-fetching. This is the right default for hand-written, interactive use.

2. **Per batch, at inference**, for content built from raw JSON rather than the
   Python constructor — most importantly `Dataset` rows, which go through
   `ChatMessages.model_validate_json(...)`. There the `image_url`/`input_audio`
   parts stay as lightweight **references** (a `url`/`path`), so a dataset of a
   million images never inlines a million payloads into memory. Resolution is
   deferred to `resolve_content_media`, which the language model calls on the
   messages it is about to send — i.e. only the current batch's media is ever
   resolved at once, and it is freed as soon as the request goes out.

A payload already inline — base64 `data`, or a ``data:`` URI — is left untouched
by both paths, so the per-batch resolver is a no-op on construction-resolved
content.

Adding a modality (video, documents, ...) is a new `DataModel` here with a
`to_content_part()` method; nothing else in the stack needs to change.
"""

import base64
import mimetypes
import os
from typing import Optional

import httpx
from pydantic import Field
from pydantic import model_validator

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.pydantic.core import DataModel

# Marker key carrying a not-yet-resolved source (url/path) on a content part
# whose wire shape has no natural URL slot (e.g. `input_audio`). The per-batch
# resolver consumes and removes it before the part reaches the provider.
_SOURCE_KEY = "_synalinks_source"


def _fetch_url(url):
    """Fetch an ``http(s)`` source synchronously, returning ``(bytes, mime)``."""
    with httpx.Client(follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        mime = response.headers.get("content-type") or mimetypes.guess_type(url)[0]
        return response.content, (mime.split(";")[0].strip() if mime else None)


def _read_file(path):
    """Read a local file, returning ``(bytes, mime)``."""
    with open(path, "rb") as f:
        return f.read(), mimetypes.guess_type(path)[0]


def _read_source(url, path):
    """Resolve a `url`/`path` source to ``(bytes, mime)`` synchronously.

    Returns `None` when there is nothing to fetch — no source, or a `url` that
    is already an inline ``data:`` URI.
    """
    if path is not None:
        return _read_file(os.path.abspath(path))
    if url and url.startswith(("http://", "https://")):
        return _fetch_url(url)
    if url and url.startswith("file://"):
        return _read_file(url[len("file://") :])
    return None


@synalinks_export(
    [
        "synalinks.backend.Image",
        "synalinks.Image",
    ]
)
class Image(DataModel):
    """An image content part for a chat message.

    Provide exactly one source: a `url` (``http(s)://`` or a ``data:`` URI), a
    local file `path`, or raw base64 `data` (with its `mime_type`). Drop it
    into a `ChatMessage`'s content list next to the accompanying text. A
    `url`/`path` is resolved to the actual payload at construction (see the
    module docstring).
    """

    url: Optional[str] = Field(
        default=None,
        description="An http(s) URL or a `data:` URI pointing to the image.",
    )
    path: Optional[str] = Field(
        default=None,
        description="A local file path to the image; read at construction.",
    )
    data: Optional[str] = Field(
        default=None,
        description="Raw base64-encoded image bytes (used with `mime_type`).",
    )
    mime_type: Optional[str] = Field(
        default=None,
        description="The image MIME type, e.g. `image/png` (used with `data`).",
    )

    @model_validator(mode="after")
    def _inline_source(self):
        """Fetch/read a `url`/`path` source and inline it as base64 `data`.

        Runs once, at construction. A source already inline — base64 `data` or
        a ``data:`` URI in `url` — is left untouched.
        """
        if self.data is not None:
            return self
        read = _read_source(self.url, self.path)
        if read is not None:
            raw, mime = read
            self.data = base64.b64encode(raw).decode("ascii")
            if self.mime_type is None:
                self.mime_type = mime
        return self

    def to_content_part(self):
        """Render this image as an OpenAI/litellm `image_url` content part.

        A resolved image emits an inline base64 ``data:`` URI; a bare ``data:``
        URI `url` is passed through as-is.
        """
        if self.data:
            mime = self.mime_type or "image/png"
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{self.data}"},
            }
        return {"type": "image_url", "image_url": {"url": self.url}}


@synalinks_export(
    [
        "synalinks.backend.Audio",
        "synalinks.Audio",
    ]
)
class Audio(DataModel):
    """An audio content part for a chat message.

    Provide a source — a `url`, a local file `path`, or raw base64 `data` — and
    a container `format` (e.g. ``"wav"`` or ``"mp3"``). The chat-completion
    `input_audio` part carries inline base64 only, so a `url`/`path` is fetched
    and inlined at construction.
    """

    url: Optional[str] = Field(
        default=None,
        description="An http(s) URL pointing to the audio; fetched at construction.",
    )
    path: Optional[str] = Field(
        default=None,
        description="A local file path to the audio; read at construction.",
    )
    data: Optional[str] = Field(
        default=None,
        description="Raw base64-encoded audio bytes.",
    )
    format: Optional[str] = Field(
        default=None,
        description="The audio container format, e.g. `wav` or `mp3`.",
    )

    @model_validator(mode="after")
    def _inline_source(self):
        """Fetch/read a `url`/`path` source and inline it as base64 `data`.

        Runs once, at construction. A payload already given as base64 `data` is
        left untouched.
        """
        if self.data is not None:
            return self
        read = _read_source(self.url, self.path)
        if read is not None:
            raw, _ = read
            self.data = base64.b64encode(raw).decode("ascii")
        return self

    def to_content_part(self):
        """Render this audio as an OpenAI/litellm `input_audio` content part."""
        return {
            "type": "input_audio",
            "input_audio": {"data": self.data, "format": self.format},
        }


def _normalize_content_element(element):
    """Map one content-list element to its chat-completion wire part.

    A `str` becomes a `text` part, anything exposing `to_content_part()` (an
    `Image`/`Audio`) becomes its own part, and an already-shaped `dict` is kept
    verbatim.
    """
    if isinstance(element, str):
        return {"type": "text", "text": element}
    to_part = getattr(element, "to_content_part", None)
    if callable(to_part):
        return to_part()
    return element


def normalize_content(content):
    """Normalize a `ChatMessage.content` value to its chat-completion shape.

    A plain string or a non-list value is returned unchanged; a list has each
    element mapped through `_normalize_content_element`.
    """
    if isinstance(content, list):
        return [_normalize_content_element(e) for e in content]
    return content


async def _read_source_async(source):
    """Fetch a not-yet-inlined source, returning ``(raw_bytes, mime_type)``."""
    if source.startswith("file://"):
        return _read_file(source[len("file://") :])
    if source.startswith(("http://", "https://")):
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(source)
            response.raise_for_status()
            mime = response.headers.get("content-type") or mimetypes.guess_type(source)[0]
            return response.content, (mime.split(";")[0].strip() if mime else None)
    raise ValueError(f"Unsupported media source: {source!r}")


async def _resolve_image_part(part):
    """Inline an `image_url` part whose url is an ``http(s)``/``file://`` ref."""
    url = part.get("image_url", {}).get("url")
    if not url or url.startswith("data:"):
        return part
    raw, mime = await _read_source_async(url)
    encoded = base64.b64encode(raw).decode("ascii")
    part["image_url"]["url"] = f"data:{mime or 'image/png'};base64,{encoded}"
    return part


async def _resolve_audio_part(part):
    """Inline an `input_audio` part carrying a deferred `url`/`path` source.

    The wire `input_audio` shape has only `data`/`format`, so a dataset row
    points at the audio with a `url`, a `path`, or the `_SOURCE_KEY` marker;
    whichever is present is read, base64-encoded into `data`, and removed.
    """
    audio = part.get("input_audio", {})
    if audio.get("data"):
        return part
    path = audio.pop("path", None)
    source = audio.pop("url", None) or audio.pop(_SOURCE_KEY, None)
    if path is not None:
        source = f"file://{os.path.abspath(path)}"
    if not source:
        return part
    raw, _ = await _read_source_async(source)
    audio["data"] = base64.b64encode(raw).decode("ascii")
    return part


async def resolve_content_media(messages):
    """Resolve every deferred media reference in `messages` to an inline payload.

    `messages` is a list of chat-completion wire dicts (one batch's worth). Any
    `image_url` part pointing at an ``http(s)``/``file://`` source, and any
    `input_audio` part carrying a deferred `url`/`path`/marker source, is
    fetched/read and inlined as base64. Parts that are already inline — a
    ``data:`` URI or base64 `data`, e.g. content built from a constructed
    `Image`/`Audio` — are left untouched, so construction-resolved messages and
    text-only conversations pay nothing.
    """
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                await _resolve_image_part(part)
            elif part.get("type") == "input_audio":
                await _resolve_audio_part(part)
    return messages
