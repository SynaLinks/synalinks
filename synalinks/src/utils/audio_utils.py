# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Fitting the audio an agent listens to to what a language model takes."""

import io

import numpy as np
import soundfile

# The audio models hear 16 kHz mono (Gemini and OpenAI's audio models
# downsample to it); a higher rate or more channels only cost upload.
SAMPLE_RATE = 16000
# The longest clip sent at once. At 16 kHz 16-bit mono (32 KB/s) five minutes
# is 9.6 MB, 12.8 MB once base64-encoded: under Gemini's 20 MB inline request
# limit, and about 10k tokens at Gemini's 32 tokens per second.
MAX_AUDIO_SECONDS = 300
# A WAV or MP3 file up to this size is sent as it is.
MAX_AUDIO_BYTES = 10 * 1024 * 1024
# The containers every audio model takes (OpenAI's `input_audio` takes only
# these two).
FORMATS = {"WAV": "wav", "MP3": "mp3"}


def fit_audio(data: bytes, offset: float = 0.0, duration: float = 0.0) -> dict:
    """Cut an audio file to a clip a language model takes.

    A whole WAV or MP3 file up to `MAX_AUDIO_SECONDS` and `MAX_AUDIO_BYTES`
    is returned as is. Otherwise the clip from `offset` (at most
    `MAX_AUDIO_SECONDS` long) is decoded, from any format libsndfile reads
    (WAV, FLAC, OGG, MP3, ...), mixed down to mono, resampled to
    `SAMPLE_RATE` and encoded as 16-bit WAV.

    Args:
        data (bytes): The audio file's bytes.
        offset (float): Where the clip starts, in seconds.
        duration (float): The clip's length in seconds; 0 for the rest of the
            file. Capped at `MAX_AUDIO_SECONDS`.

    Returns:
        dict: ``data`` (the bytes to send), ``format`` (``wav`` or ``mp3``),
        ``offset`` and ``duration`` of the clip, ``total_duration`` of the
        file (seconds) and ``truncated`` (whether audio remains after the
        clip).

    Raises:
        ValueError: If `data` is not audio libsndfile can read, or `offset`
            is past its end.
    """
    try:
        info = soundfile.info(io.BytesIO(data))
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"not a readable audio file: {exc}") from exc
    total = info.frames / info.samplerate
    offset = max(offset, 0.0)
    if offset >= total:
        raise ValueError(f"offset {offset:g}s is past the end ({total:.3f}s)")
    remaining = total - offset
    length = min(duration if duration > 0 else remaining, remaining, MAX_AUDIO_SECONDS)
    clip = {
        "offset": offset,
        "duration": round(length, 3),
        "total_duration": round(total, 3),
        "truncated": offset + length < total,
    }
    container = FORMATS.get(info.format)
    if container and offset == 0 and length == total and len(data) <= MAX_AUDIO_BYTES:
        return {"data": data, "format": container, **clip}
    rate = info.samplerate
    with soundfile.SoundFile(io.BytesIO(data)) as audio:
        audio.seek(int(offset * rate))
        samples = audio.read(int(length * rate), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if rate > SAMPLE_RATE and len(mono):
        # Band-limited resampling: keep the spectrum below the new Nyquist.
        count = round(len(mono) * SAMPLE_RATE / rate)
        mono = np.fft.irfft(np.fft.rfft(mono)[: count // 2 + 1], count)
        mono *= count / len(samples)
        rate = SAMPLE_RATE
    output = io.BytesIO()
    soundfile.write(
        output, np.clip(mono, -1.0, 1.0), rate, format="WAV", subtype="PCM_16"
    )
    return {"data": output.getvalue(), "format": "wav", **clip}
