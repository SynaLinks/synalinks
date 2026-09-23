# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import io
import os

import PIL.Image

from synalinks.src import testing
from synalinks.src.utils.image_utils import MAX_IMAGE_BYTES
from synalinks.src.utils.image_utils import MAX_IMAGE_EDGE
from synalinks.src.utils.image_utils import fit_image


def encode(image, format):
    output = io.BytesIO()
    image.save(output, format=format)
    return output.getvalue()


class FitImageTest(testing.TestCase):
    def test_small_image_is_returned_as_is(self):
        data = encode(PIL.Image.new("RGB", (64, 48), "red"), "WEBP")
        fitted = fit_image(data)
        self.assertEqual(fitted["data"], data)
        self.assertEqual(fitted["mime_type"], "image/webp")
        self.assertEqual((fitted["width"], fitted["height"]), (64, 48))
        self.assertNotIn("original_width", fitted)

    def test_large_png_is_scaled_down_as_png(self):
        data = encode(PIL.Image.new("RGBA", (3000, 6000), "green"), "PNG")
        fitted = fit_image(data)
        self.assertEqual(fitted["mime_type"], "image/png")
        self.assertEqual((fitted["width"], fitted["height"]), (784, MAX_IMAGE_EDGE))
        self.assertEqual(
            (fitted["original_width"], fitted["original_height"]), (3000, 6000)
        )
        self.assertEqual(PIL.Image.open(io.BytesIO(fitted["data"])).size, (784, 1568))

    def test_large_jpeg_stays_jpeg(self):
        data = encode(PIL.Image.new("RGB", (2000, 2000), "blue"), "JPEG")
        fitted = fit_image(data)
        self.assertEqual(fitted["mime_type"], "image/jpeg")
        self.assertEqual((fitted["width"], fitted["height"]), (1568, 1568))

    def test_other_format_is_converted(self):
        fitted = fit_image(encode(PIL.Image.new("RGB", (10, 10)), "BMP"))
        self.assertEqual(fitted["mime_type"], "image/png")
        self.assertEqual((fitted["original_width"], fitted["width"]), (10, 10))

    def test_png_too_heavy_even_scaled_down_becomes_jpeg(self):
        noise = PIL.Image.frombytes("RGBA", (1568, 1568), os.urandom(1568 * 1568 * 4))
        data = encode(noise, "PNG")
        self.assertGreater(len(data), MAX_IMAGE_BYTES)
        fitted = fit_image(data)
        self.assertEqual(fitted["mime_type"], "image/jpeg")
        self.assertLessEqual(len(fitted["data"]), MAX_IMAGE_BYTES)

    def test_non_image_raises(self):
        with self.assertRaises(ValueError):
            fit_image(b"hello")
