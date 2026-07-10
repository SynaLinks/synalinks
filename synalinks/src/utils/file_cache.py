# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import hashlib
import os
import tempfile

import orjson


class FileCache:
    """A minimal persistent JSON key/value cache backed by files.

    Each entry is stored as one `<sha256>.json` file under `cache_dir`,
    so the cache survives process restarts and can be shared between
    concurrent processes. Writes are atomic (temp file + rename) so a
    reader never observes a partial entry; corrupt or unreadable entries
    are treated as misses.
    """

    def __init__(self, cache_dir):
        self.cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def make_key(self, payload):
        """Deterministic cache key: sha256 of the JSON-serialized payload.

        Returns None when the payload is not JSON-serializable (the caller
        should then skip caching for that request).
        """
        try:
            data = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
        except (TypeError, orjson.JSONEncodeError):
            return None
        return hashlib.sha256(data).hexdigest()

    def _path(self, key):
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key):
        """Return the cached JSON value for `key`, or None on a miss."""
        try:
            with open(self._path(key), "rb") as f:
                return orjson.loads(f.read())
        except (OSError, orjson.JSONDecodeError):
            return None

    def set(self, key, value):
        """Persist `value` (any JSON-serializable object) under `key`."""
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(orjson.dumps(value))
            os.replace(tmp_path, self._path(key))
        except (TypeError, orjson.JSONEncodeError, OSError):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
