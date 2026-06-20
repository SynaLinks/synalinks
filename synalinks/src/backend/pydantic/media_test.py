# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import base64
import os
from unittest import mock

from synalinks.src import testing
from synalinks.src.backend.pydantic.common import ChatMessage
from synalinks.src.backend.pydantic.common import ChatMessages
from synalinks.src.backend.pydantic.media import _SOURCE_KEY
from synalinks.src.backend.pydantic.media import Audio
from synalinks.src.backend.pydantic.media import Image
from synalinks.src.backend.pydantic.media import _read_source_async
from synalinks.src.backend.pydantic.media import normalize_content
from synalinks.src.backend.pydantic.media import resolve_content_media

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SAMPLE_IMAGE = os.path.join(_THIS_DIR, "..", "..", "..", "..", "guides", "traced_qa.png")


def _patched_httpx(content, content_type="image/png"):
    """Patch `httpx.Client` (sync, construction-time) with a fixed-bytes stub."""

    class _Resp:
        def __init__(self):
            self.content = content
            self.headers = {"content-type": content_type}

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

    return mock.patch("httpx.Client", _Client)


def _patched_async_httpx(content, content_type="image/png"):
    """Patch `httpx.AsyncClient` (per-batch resolver) with a fixed-bytes stub."""

    class _Resp:
        def __init__(self):
            self.content = content
            self.headers = {"content-type": content_type}

        def raise_for_status(self):
            pass

    class _AClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            return _Resp()

    return mock.patch("httpx.AsyncClient", _AClient)


class MediaContentPartTest(testing.TestCase):
    def test_image_part_from_data_uri_url_passes_through(self):
        self.assertEqual(
            Image(url="data:image/png;base64,AAAA").to_content_part(),
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        )

    def test_image_part_from_base64(self):
        self.assertEqual(
            Image(data="AAAA", mime_type="image/jpeg").to_content_part(),
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
        )

    def test_image_from_url_is_inlined_at_construction(self):
        with _patched_httpx(b"\x89PNG\r\n\x1a\n"):
            img = Image(url="http://x/y.png")
        expected = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
        self.assertEqual(img.data, expected)
        self.assertEqual(img.mime_type, "image/png")
        self.assertEqual(
            img.to_content_part(),
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{expected}"},
            },
        )

    def test_image_from_path_is_inlined_at_construction(self):
        img = Image(path=_SAMPLE_IMAGE)
        with open(_SAMPLE_IMAGE, "rb") as f:
            expected = base64.b64encode(f.read()).decode("ascii")
        self.assertEqual(img.data, expected)
        self.assertEqual(img.mime_type, "image/png")
        self.assertEqual(
            img.to_content_part()["image_url"]["url"],
            f"data:image/png;base64,{expected}",
        )

    def test_image_from_file_uri_is_inlined_at_construction(self):
        img = Image(url=f"file://{_SAMPLE_IMAGE}")
        with open(_SAMPLE_IMAGE, "rb") as f:
            expected = base64.b64encode(f.read()).decode("ascii")
        self.assertEqual(img.data, expected)
        self.assertEqual(img.mime_type, "image/png")

    def test_image_keeps_explicit_mime_type_over_detected(self):
        # An explicitly-set mime_type must survive the fetch, not be clobbered
        # by the response's content-type.
        with _patched_httpx(b"\x89PNG\r\n\x1a\n", content_type="image/png"):
            img = Image(url="http://x/y.png", mime_type="image/x-custom")
        self.assertEqual(img.mime_type, "image/x-custom")

    def test_audio_part_from_base64(self):
        self.assertEqual(
            Audio(data="AAAA", format="wav").to_content_part(),
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        )

    def test_audio_from_url_is_inlined_at_construction(self):
        with _patched_httpx(b"RIFFxxxx", content_type="audio/wav"):
            audio = Audio(url="http://x/a.wav", format="wav")
        expected = base64.b64encode(b"RIFFxxxx").decode("ascii")
        self.assertEqual(audio.data, expected)
        self.assertEqual(
            audio.to_content_part(),
            {"type": "input_audio", "input_audio": {"data": expected, "format": "wav"}},
        )

    def test_audio_from_path_is_inlined_at_construction(self):
        audio = Audio(path=_SAMPLE_IMAGE, format="wav")
        with open(_SAMPLE_IMAGE, "rb") as f:
            expected = base64.b64encode(f.read()).decode("ascii")
        self.assertEqual(audio.data, expected)


class NormalizeContentTest(testing.TestCase):
    def test_string_stays_string(self):
        self.assertEqual(normalize_content("hello"), "hello")

    def test_mixed_list_maps_each_element(self):
        with _patched_httpx(b"\x89PNG\r\n\x1a\n"):
            out = normalize_content(["question?", Image(url="http://x/y.png")])
        expected = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
        self.assertEqual(
            out,
            [
                {"type": "text", "text": "question?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{expected}"},
                },
            ],
        )

    def test_existing_dict_parts_pass_through(self):
        parts = [{"type": "text", "text": "hi"}]
        self.assertEqual(normalize_content(parts), parts)

    def test_chat_message_normalizes_multimodal_content(self):
        with _patched_httpx(b"\x89PNG\r\n\x1a\n"):
            m = ChatMessage(
                role="user",
                content=["What is in this picture?", Image(url="http://x/y.png")],
            )
        expected = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
        self.assertEqual(
            m.content,
            [
                {"type": "text", "text": "What is in this picture?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{expected}"},
                },
            ],
        )

    def test_chat_message_plain_string_unchanged(self):
        self.assertEqual(ChatMessage(role="user", content="hello").content, "hello")


class ResolveContentMediaTest(testing.TestCase):
    """The per-batch resolver: deferred refs in -> inline payloads out."""

    async def test_text_only_is_untouched(self):
        messages = [{"role": "user", "content": "hello"}]
        self.assertEqual(await resolve_content_media(messages), messages)

    async def test_data_uri_image_is_left_inline(self):
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        await resolve_content_media([{"role": "user", "content": [part]}])
        self.assertEqual(part["image_url"]["url"], "data:image/png;base64,AAAA")

    async def test_constructed_image_part_is_a_noop(self):
        # Content built from a constructed Image() is already inline; the
        # per-batch resolver must leave it byte-for-byte unchanged.
        part = Image(data="AAAA", mime_type="image/png").to_content_part()
        before = dict(part["image_url"])
        await resolve_content_media([{"role": "user", "content": [part]}])
        self.assertEqual(part["image_url"], before)

    async def test_http_image_ref_is_downloaded_and_inlined(self):
        part = {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
        with _patched_async_httpx(b"\x89PNG\r\n\x1a\n"):
            await resolve_content_media([{"role": "user", "content": [part]}])
        expected = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
        self.assertEqual(part["image_url"]["url"], f"data:image/png;base64,{expected}")

    async def test_file_image_ref_is_inlined(self):
        part = {"type": "image_url", "image_url": {"url": f"file://{_SAMPLE_IMAGE}"}}
        await resolve_content_media([{"role": "user", "content": [part]}])
        url = part["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        with open(_SAMPLE_IMAGE, "rb") as f:
            expected = base64.b64encode(f.read()).decode("ascii")
        self.assertEqual(url.split(",", 1)[1], expected)

    async def test_audio_url_ref_is_inlined_and_stripped(self):
        part = {
            "type": "input_audio",
            "input_audio": {"format": "wav", "url": "http://x/a.wav"},
        }
        with _patched_async_httpx(b"RIFFxxxx", content_type="audio/wav"):
            await resolve_content_media([{"role": "user", "content": [part]}])
        audio = part["input_audio"]
        self.assertEqual(sorted(audio.keys()), ["data", "format"])
        self.assertEqual(audio["data"], base64.b64encode(b"RIFFxxxx").decode("ascii"))

    async def test_audio_path_ref_is_inlined_and_stripped(self):
        part = {
            "type": "input_audio",
            "input_audio": {"format": "wav", "path": _SAMPLE_IMAGE},
        }
        await resolve_content_media([{"role": "user", "content": [part]}])
        audio = part["input_audio"]
        self.assertEqual(sorted(audio.keys()), ["data", "format"])
        with open(_SAMPLE_IMAGE, "rb") as f:
            expected = base64.b64encode(f.read()).decode("ascii")
        self.assertEqual(audio["data"], expected)

    async def test_audio_source_marker_ref_is_inlined_and_stripped(self):
        # The deferred-source marker is how an `input_audio` part smuggles a
        # url/path through a wire shape that has no slot for it.
        part = {
            "type": "input_audio",
            "input_audio": {"format": "wav", _SOURCE_KEY: f"file://{_SAMPLE_IMAGE}"},
        }
        await resolve_content_media([{"role": "user", "content": [part]}])
        self.assertEqual(sorted(part["input_audio"].keys()), ["data", "format"])

    async def test_constructed_audio_part_is_a_noop(self):
        part = Audio(data="AAAA", format="wav").to_content_part()
        before = dict(part["input_audio"])
        await resolve_content_media([{"role": "user", "content": [part]}])
        self.assertEqual(part["input_audio"], before)

    async def test_non_dict_content_parts_are_skipped(self):
        # A content list may carry a stray non-dict element; the resolver
        # must skip it rather than crash.
        messages = [{"role": "user", "content": ["raw string", 42]}]
        self.assertEqual(await resolve_content_media(messages), messages)

    async def test_unsupported_source_scheme_raises(self):
        with self.assertRaises(ValueError):
            await _read_source_async("ftp://example.com/clip.wav")


class MultimodalDatasetRoundTripTest(testing.TestCase):
    """A dataset-built message keeps a lightweight ref until the batch runs."""

    async def test_validated_json_keeps_ref_then_resolves_per_batch(self):
        # This is exactly how `Dataset._make_input` builds a row: render JSON,
        # then `model_validate_json` — no Image() constructor, so the source
        # stays a reference rather than being inlined into the dataset.
        rendered = (
            '{"messages":[{"role":"user","content":['
            '{"type":"text","text":"caption"},'
            '{"type":"image_url","image_url":{"url":"http://x/y.png"}}]}]}'
        )
        msgs = ChatMessages.model_validate_json(rendered)
        part = msgs.messages[0].content[1]
        self.assertEqual(part["image_url"]["url"], "http://x/y.png")  # not inlined

        # The batch is resolved only when it is about to be sent to the LM.
        wire = [m.model_dump() for m in msgs.messages]
        with _patched_async_httpx(b"\x89PNG\r\n\x1a\n"):
            await resolve_content_media(wire)
        expected = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
        self.assertEqual(
            wire[0]["content"][1]["image_url"]["url"],
            f"data:image/png;base64,{expected}",
        )
