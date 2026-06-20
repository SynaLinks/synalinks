"""
# Multimodal Inputs

Every guide so far has fed the language model *text*: a `DataModel` in, a
typed `DataModel` out. But the chat protocol modern models speak is not
text-only — a single user turn can carry a picture, an audio clip, and the
words that go with them, all at once. This guide shows how Synalinks lets you
drop an image or a sound straight into a message, with no new module, no
special generator, and no provider-specific plumbing.

If you have not yet read [Guide 1](https://synalinks.github.io/synalinks/guides/Getting%20Started/)
(the `DataModel` / `Generator` / `Program` trio) and
[Guide 2](https://synalinks.github.io/synalinks/guides/Data%20Models/) (data
models in depth), start there — this guide builds directly on the chat-message
shape they introduce.

## A picture is just another content part

In the OpenAI/litellm chat protocol that Synalinks speaks under the hood, a
message's `content` is allowed to be one of two things: a plain string, or a
**list of parts**. A part is a small typed dict — `{"type": "text", ...}`,
`{"type": "image_url", ...}`, and so on. Text and media are not different
*channels*; they are different *parts of the same list*. So "send the model an
image" is really just "put an image part in the content list next to the text
part."

Synalinks gives you two small data models for those non-text parts —
`synalinks.Image` and `synalinks.Audio` — that you drop directly into a
`ChatMessage`'s `content` list, mixed in with the strings:

```python
import synalinks

message = synalinks.ChatMessage(
    role="user",
    content=[
        "What is in this picture? Answer in one sentence.",
        synalinks.Image(url="https://example.com/cat.png"),
    ],
)
```

`ChatMessage` normalizes each element of the list to its wire shape: a plain
`str` becomes a `text` part, an `Image`/`Audio` becomes its own part. The
result is still a strict chat-completion message, so it flows through the exact
same `Generator` path as a text-only one. Nothing downstream needs to know it
was ever multimodal.

```mermaid
graph LR
    A["content list:\n[str, Image]"] --> B["ChatMessage\nnormalizes parts"]
    B --> C["Generator"]
    C --> D["vision-capable\nlanguage model"]
```

## Three ways to point at an image

An `Image` needs exactly one source. Pick whichever you have on hand:

| Field           | Use it when…                                            |
|-----------------|---------------------------------------------------------|
| `url`           | the image lives at an `http(s)://` address (or is already a `data:` URI) |
| `path`          | the image is a file on the local disk                   |
| `data`          | you already hold the raw bytes, base64-encoded (pass `mime_type` too) |

```python
synalinks.Image(url="https://example.com/cat.png")        # remote
synalinks.Image(path="/home/me/photos/cat.png")           # local file
synalinks.Image(data=b64_bytes, mime_type="image/png")    # raw base64
```

`Audio` is the same idea, with a `format` (the container, e.g. `"wav"` or
`"mp3"`) instead of a MIME type:

```python
synalinks.Audio(url="https://example.com/clip.wav", format="wav")
synalinks.Audio(path="/home/me/clips/note.mp3", format="mp3")
```

## Resolution happens at construction

Here is the one behavior worth internalizing, because it differs from what you
might expect. A `url` or `path` is resolved to the **actual bytes the moment
you construct the `Image`/`Audio`** — not lazily at inference. The line

```python
logo = synalinks.Image(url="https://example.com/cat.png")
```

performs the HTTP GET right there, reads the response, and inlines it as a
base64 `data:` URI inside the object. By the time the image reaches a
`Generator`, there is no URL left for the provider to fetch — the payload is
already in hand. (A source that is already inline — raw `data`, or a `data:`
URI in `url` — is kept untouched; no fetch happens.)

This buys three things:

1. **It works across every provider, including local ones.** A model running
   on your laptop via Ollama cannot reach out to `example.com` to download an
   image — but it never has to, because the bytes were inlined before the
   request was built. The same `Image(url=...)` works identically against a
   hosted API and a local model.
2. **A bad source fails loudly where you wrote it.** A typo in a path or an
   unreachable URL raises an exception *on the `Image(...)` line*, with a clear
   stack trace — not three modules deep inside an inference call where it is
   far harder to trace back.
3. **The resolved object is self-contained.** You can build it once and reuse
   it across many calls without re-fetching, and it serializes with the payload
   embedded.

The trade-off to keep in mind: because construction does the I/O, building an
`Image(url=...)` performs a **synchronous** network fetch. If you construct one
inside a hot async path, that GET blocks until it completes. For the common
case — set up your inputs, then run the program — this is exactly what you
want.

## Putting it all together

The example below hands the photo shown here to a small vision-capable model
running locally through Ollama and asks for a one-line caption. Note that the
*only* thing that makes this "multimodal" is the content list in the input
message — the `Generator` and the `Program` are exactly the same as in every
previous guide.

![The sample photo the example captions — a dog.](../assets/multimodal_sample.jpg)

One deliberate choice here: `data_model=None`. Every earlier guide gave the
`Generator` a `DataModel` so the output was a typed schema. Passing `None`
**relaxes that structured-output constraint** — the `Generator` returns a
free-form `ChatMessage` whose `content` is the model's text, with no JSON shape
imposed. For open-ended description that is usually what you want, and it
matters more than it sounds for small models: a tiny vision model captions
noticeably better when it is *not* also being forced to emit valid JSON at the
same time. Reach for a `DataModel` (as in [Guide 2](https://synalinks.github.io/synalinks/guides/Data%20Models/))
when you want typed fields back; reach for `None` when you just want the text.

You need a **vision-capable** model for this to work. The example uses
`ollama/moondream` (a ~1.7 GB local vision model: `ollama pull moondream`).
Any vision model your provider offers — `gpt-4o`, `gemini/gemini-2.0-flash`,
`anthropic/claude-...` — works with no other change.

```python
import asyncio
from dotenv import load_dotenv
import synalinks

# A fixed, freely-usable sample photo (a dog).
PHOTO_URL = "https://picsum.photos/id/237/320/240"

async def main():
    load_dotenv()
    synalinks.clear_session()

    language_model = synalinks.LanguageModel(model="ollama/moondream")

    inputs = synalinks.Input(data_model=synalinks.ChatMessages)
    # data_model=None -> free-form ChatMessage out, no structured-output schema.
    outputs = await synalinks.Generator(
        data_model=None,
        language_model=language_model,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="captioner")

    # The image is fetched and inlined the moment this Image() is constructed.
    result = await program(
        synalinks.ChatMessages(
            messages=[
                synalinks.ChatMessage(
                    role="user",
                    content=[
                        "Describe this image in one short sentence.",
                        synalinks.Image(url=PHOTO_URL),
                    ],
                ),
            ]
        )
    )

    print(f"Caption: {result['content']}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it and you will get something like:

```
Caption: iced coffee cup with a handle, black and white photograph of a dog's
face on a wooden surface.
```

That is **not** a perfect caption — `moondream` is a tiny model and it has
hallucinated a coffee cup that is not there. Two things are worth saying about
that. First, caption quality is a property of the *model*, not of the
multimodal plumbing: swap in `gpt-4o` or `gemini` and the same program returns
a clean one-liner with no code change. Second — and this is the part that makes
Synalinks different from a plain API wrapper — a multimodal pipeline is **just
another program, so it trains like everything else**. The image-bearing
`ChatMessage` flows through the exact same `Generator` you have been
optimizing in every other guide, which means you can `compile()` it with a
reward and `fit()` it on a handful of (image, good-caption) examples: the
in-context optimizers will refine the instructions and accumulate few-shot
examples until even a small local model captions reliably. Nothing about the
image being an image changes that — see [Guide 13](https://synalinks.github.io/synalinks/guides/Rewards/)
(rewards) and [Guide 15](https://synalinks.github.io/synalinks/guides/Training/)
(training) for the mechanics.

## Datasets: many images, resolved one batch at a time

Training implies a *dataset*, and a dataset of images raises an obvious worry:
if constructing an `Image` fetches and inlines its bytes (as we saw above),
does loading a dataset of a million photos inline a million base64 blobs into
memory? No — and the reason is the two-tier resolution from the top of this
guide. A `Dataset` builds each row from JSON via `model_validate_json`, which
never calls the `Image` constructor, so the `image_url` part keeps its
**reference** (a `url` or a `file://` path). The bytes are read and inlined
only when a batch is actually sent to the model — by the same per-batch
`resolve_content_media` step the language model runs on every request — and
freed once the request goes out. **Memory is bounded by one batch, never by the
dataset.** You store paths; only the current batch is ever resolved.

For the common case — a directory of image files on disk — `ImageFolderDataset`
does this for you. It walks the folder, emits one lightweight row per image
(the path as a `file://` ref, plus a `label` taken from the parent
sub-directory, torchvision-style), and never reads a pixel until the batch runs:

```python
import synalinks

# photos/cat/*.jpg, photos/dog/*.jpg, ...  -> label = the sub-folder name
ds = synalinks.ImageFolderDataset(
    root="./photos",
    prompt="Describe this image in one short sentence.",
    output_template='{"role": "assistant", "content": {{ label | tojson }}}',
    batch_size=8,
)

# Each batch's files are read + inlined on demand, then discarded.
program.fit(x=ds(), epochs=3, ...)
```

For images that live behind URLs, or any custom row shape, use one of the
file/HTTP-backed datasets (`JSONDataset`, `CSVDataset`, `HuggingFaceDataset`,
...) with an `input_template` that renders an `image_url` part pointing at the
row's `url` — the resolver treats `http(s)` and `file://` refs identically:

```python
ds = synalinks.JSONDataset(
    path="captions.json",   # [{"image_url": "...", "caption": "..."}, ...]
    input_template=(
        '{"messages":[{"role":"user","content":['
        '{"type":"text","text":"Describe this image."},'
        '{"type":"image_url","image_url":{"url":{{ image_url | tojson }}}}]}]}'
    ),
    output_template='{"role":"assistant","content":{{ caption | tojson }}}',
    batch_size=8,
)
```

The same holds for audio: a dataset row points an `input_audio` part at a
`url`/`path` (audio's wire shape has no URL slot, so the reference rides in a
`url` field the resolver consumes), and it is inlined per batch just like an
image.

## Adding a new modality

`Image` and `Audio` are not special-cased anywhere in the stack — they are just
`DataModel`s that expose a `to_content_part()` method returning their wire
shape. Supporting a new modality (video, documents, …) is a matter of adding
one more such `DataModel` in `synalinks/src/backend/pydantic/media.py`; the
`ChatMessage` normalizer and the `Generator` path pick it up for free.

## Take-Home Summary

- A message's `content` can be a **list of parts** — drop `synalinks.Image` /
  `synalinks.Audio` into it next to your text strings.
- An image has **three possible sources**: `url`, `path`, or raw base64
  `data` (+ `mime_type`). Audio uses a `format` instead of a MIME type.
- **Resolution happens at construction**: the bytes are fetched/read and
  inlined the moment you build the object, so it works across every provider
  (local models included) and a bad source fails right where you wrote it.
- Multimodal input is **plain chat-completion** — the `Generator`, `Program`,
  and output schema are unchanged. You just need a **vision-capable model**.
- A multimodal pipeline is **just a program**, so an imperfect caption is not a
  dead end: `compile()` + `fit()` it with a reward and the optimizers improve
  it exactly as they would any other `Generator`.
- **Datasets store references, not bytes.** A `Dataset` row keeps a `url`/`path`;
  only the current batch's media is read and inlined when it is sent to the
  model. `ImageFolderDataset` turns a directory of images into exactly that.

## API References

- [Multimodal Data Models (`Image`, `Audio`)](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/Multimodal%20Data%20Models/)
- [Generator module](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [`ImageFolderDataset`](https://synalinks.github.io/synalinks/Synalinks%20API/Datasets/ImageFolderDataset/)
"""

import asyncio
import base64
import os
import tempfile

from dotenv import load_dotenv

import synalinks

# A fixed, freely-usable sample photo (a dog). The same image is shown in the
# guide above (docs/assets/multimodal_sample.jpg).
PHOTO_URL = "https://picsum.photos/id/237/320/240"


async def main():
    load_dotenv()
    synalinks.clear_session()
    synalinks.enable_logging()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_31_multimodal",
    # )

    # A tiny local vision model (~1.7 GB: `ollama pull moondream`). Any larger
    # vision model (gpt-4o, gemini, claude) is a drop-in upgrade.
    language_model = synalinks.LanguageModel(model="ollama/moondream")

    inputs = synalinks.Input(data_model=synalinks.ChatMessages)
    # data_model=None -> free-form ChatMessage out, no structured-output schema.
    outputs = await synalinks.Generator(
        data_model=None,
        language_model=language_model,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="captioner")
    program.summary()

    # 1) Interactive path: the image is fetched and inlined the moment this
    #    Image() is constructed, then captioned.
    image = synalinks.Image(url=PHOTO_URL)
    result = await program(
        synalinks.ChatMessages(
            messages=[
                synalinks.ChatMessage(
                    role="user",
                    content=[
                        "Describe this image in one short sentence.",
                        image,
                    ],
                ),
            ]
        )
    )
    print(f"Caption: {result['content']}")

    # 2) Dataset path: a folder of image files, resolved one batch at a time.
    #    To keep the example self-contained we drop the bytes we already have
    #    into a temp folder; in practice this is simply a directory of photos.
    folder = tempfile.mkdtemp()
    raw = base64.b64decode(image.data)
    for name in ("photo_a.jpg", "photo_b.jpg"):
        with open(os.path.join(folder, name), "wb") as f:
            f.write(raw)

    dataset = synalinks.ImageFolderDataset(
        root=folder,
        prompt="Describe this image in one short sentence.",
        batch_size=2,
    )
    # Each row holds only a `file://` reference; the file is read and inlined
    # when its batch is sent to the model, then freed.
    captions = await program.predict(x=dataset(), verbose=0)
    for i, caption in enumerate(captions):
        print(f"Folder caption[{i}]: {caption['content']}")


if __name__ == "__main__":
    asyncio.run(main())
