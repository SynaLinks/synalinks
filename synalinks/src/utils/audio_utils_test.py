# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import io

import numpy as np
import soundfile

from synalinks.src import testing
from synalinks.src.utils.audio_utils import MAX_AUDIO_SECONDS
from synalinks.src.utils.audio_utils import SAMPLE_RATE
from synalinks.src.utils.audio_utils import fit_audio


def tone(seconds, rate=44100, channels=1, format="WAV", frequency=440.0):
    """A sine tone encoded as an audio file."""
    t = np.arange(int(seconds * rate)) / rate
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    samples = np.stack([wave] * channels, axis=1)
    output = io.BytesIO()
    soundfile.write(output, samples, rate, format=format)
    return output.getvalue()


class FitAudioTest(testing.TestCase):
    def test_short_wav_is_returned_as_is(self):
        data = tone(2)
        clip = fit_audio(data)
        self.assertEqual(clip["data"], data)
        self.assertEqual(clip["format"], "wav")
        self.assertEqual(clip["duration"], 2.0)
        self.assertEqual(clip["total_duration"], 2.0)
        self.assertFalse(clip["truncated"])

    def test_other_format_becomes_16khz_mono_wav(self):
        clip = fit_audio(tone(2, channels=2, format="FLAC"))
        self.assertEqual(clip["format"], "wav")
        info = soundfile.info(io.BytesIO(clip["data"]))
        self.assertEqual((info.samplerate, info.channels), (SAMPLE_RATE, 1))
        self.assertAlmostEqual(info.duration, 2.0, places=2)
        # The resampled tone keeps its pitch and loudness.
        samples, _ = soundfile.read(io.BytesIO(clip["data"]))
        spectrum = np.abs(np.fft.rfft(samples))
        self.assertAlmostEqual(np.argmax(spectrum) * SAMPLE_RATE / len(samples), 440, 0)
        self.assertAlmostEqual(np.max(np.abs(samples)), 0.5, places=1)

    def test_clip_from_offset(self):
        clip = fit_audio(tone(10), offset=4, duration=3)
        self.assertEqual((clip["offset"], clip["duration"]), (4, 3))
        self.assertTrue(clip["truncated"])
        info = soundfile.info(io.BytesIO(clip["data"]))
        self.assertAlmostEqual(info.duration, 3.0, places=2)
        self.assertFalse(fit_audio(tone(10), offset=8)["truncated"])

    def test_long_file_is_capped(self):
        clip = fit_audio(tone(MAX_AUDIO_SECONDS + 20, rate=8000))
        self.assertEqual(clip["duration"], MAX_AUDIO_SECONDS)
        self.assertTrue(clip["truncated"])

    def test_errors(self):
        with self.assertRaises(ValueError):
            fit_audio(b"not audio")
        with self.assertRaises(ValueError):
            fit_audio(tone(1), offset=5)
