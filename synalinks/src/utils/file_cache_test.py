# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os

from synalinks.src import testing
from synalinks.src.utils.file_cache import FileCache


class FileCacheTest(testing.TestCase):
    def test_set_get_roundtrip(self):
        cache = FileCache(os.path.join(self.get_temp_dir(), "cache"))
        key = cache.make_key({"model": "m", "input": ["a", "b"]})
        self.assertIsNone(cache.get(key))
        cache.set(key, {"embeddings": [[0.1, 0.2]]})
        self.assertEqual(cache.get(key), {"embeddings": [[0.1, 0.2]]})

    def test_key_is_deterministic_and_order_insensitive(self):
        cache = FileCache(os.path.join(self.get_temp_dir(), "cache"))
        key1 = cache.make_key({"a": 1, "b": 2})
        key2 = cache.make_key({"b": 2, "a": 1})
        key3 = cache.make_key({"a": 1, "b": 3})
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_unserializable_payload_returns_none_key(self):
        cache = FileCache(os.path.join(self.get_temp_dir(), "cache"))
        self.assertIsNone(cache.make_key({"fn": object()}))

    def test_corrupt_entry_is_a_miss(self):
        cache_dir = os.path.join(self.get_temp_dir(), "cache")
        cache = FileCache(cache_dir)
        key = cache.make_key({"x": 1})
        cache.set(key, {"y": 2})
        with open(os.path.join(cache_dir, f"{key}.json"), "wb") as f:
            f.write(b"{not json")
        self.assertIsNone(cache.get(key))

    def test_persists_across_instances(self):
        cache_dir = os.path.join(self.get_temp_dir(), "cache")
        cache = FileCache(cache_dir)
        key = cache.make_key({"x": 1})
        cache.set(key, {"y": 2})
        reopened = FileCache(cache_dir)
        self.assertEqual(reopened.get(key), {"y": 2})
