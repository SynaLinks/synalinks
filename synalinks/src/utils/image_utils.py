# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Fitting the images an agent looks at to what a language model reads."""

import io

import PIL.Image

# The vision models read an image at most this large: Anthropic downscales
# past a 1568 px long edge before the model sees it, OpenAI tiles a
# high-detail image down to a 768 px short side. Sending more only costs
# upload and tokens.
MAX_IMAGE_EDGE = 1568
# Anthropic's 5 MB per-image limit, the strictest of the providers (OpenAI
# and Gemini take 20 MB), counts the base64 payload: 4/3 of the raw bytes.
MAX_IMAGE_BYTES = 5 * 1024 * 1024 * 3 // 4

# The formats every provider takes, in a message and in a tool result (Gemini
# 3's function responses take no GIF, which is re-encoded).
MIME_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}


def fit_image(data: bytes) -> dict:
    """Downscale an image to what a language model reads.

    An image within `MAX_IMAGE_EDGE` and `MAX_IMAGE_BYTES`, in a format all
    the providers take (PNG, JPEG, WebP), is returned as is. Any other image
    Pillow reads is scaled down to fit and re-encoded: as PNG, which keeps
    charts and screenshots sharp, or as JPEG when the source is a JPEG or the
    PNG would still be too large.

    Args:
        data (bytes): The image file's bytes.

    Returns:
        dict: ``data`` (the bytes to send), ``mime_type``, ``width`` and
        ``height``, plus ``original_width`` and ``original_height`` when the
        image was scaled down.

    Raises:
        ValueError: If `data` is not an image Pillow can read.
    """
    try:
        image = PIL.Image.open(io.BytesIO(data))
        image.load()
    except (OSError, PIL.Image.DecompressionBombError) as exc:
        raise ValueError(f"not a readable image: {exc}") from exc
    width, height = image.size
    mime_type = MIME_TYPES.get(image.format)
    if (
        mime_type
        and max(width, height) <= MAX_IMAGE_EDGE
        and len(data) <= MAX_IMAGE_BYTES
    ):
        return {"data": data, "mime_type": mime_type, "width": width, "height": height}
    source_format = image.format
    image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), PIL.Image.LANCZOS)
    output = io.BytesIO()
    if source_format != "JPEG":
        image.save(output, format="PNG", optimize=True)
    if source_format == "JPEG" or output.tell() > MAX_IMAGE_BYTES:
        if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
            # JPEG has no alpha: flatten onto white, like a page.
            image = image.convert("RGBA")
            background = PIL.Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            image = background
        output = io.BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=85, optimize=True)
        mime_type = "image/jpeg"
    else:
        mime_type = "image/png"
    return {
        "data": output.getvalue(),
        "mime_type": mime_type,
        "width": image.size[0],
        "height": image.size[1],
        "original_width": width,
        "original_height": height,
    }
