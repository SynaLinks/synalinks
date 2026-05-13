# Modified from: keras/src/utils/file_utils_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import hashlib
import os
import tarfile
import urllib
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.utils import file_utils


def _write(path, content=b"hello"):
    with open(path, "wb") as f:
        f.write(content)


class PathToStringTest(testing.TestCase):
    def test_converts_pathlike(self):
        self.assertEqual(file_utils.path_to_string(Path("/tmp/x")), "/tmp/x")

    def test_passthrough_for_strings(self):
        self.assertEqual(file_utils.path_to_string("/tmp/x"), "/tmp/x")

    def test_passthrough_for_other_types(self):
        obj = object()
        self.assertIs(file_utils.path_to_string(obj), obj)


class RemotePathTest(testing.TestCase):
    def test_remote_path_detection(self):
        for p in ("/gcs/bucket/x", "/cns/abc", "s3://bucket/key", "/hdfs/data"):
            self.assertTrue(file_utils.is_remote_path(p), p)

    def test_local_paths_are_not_remote(self):
        for p in ("/tmp/file.txt", "./local", "relative/path"):
            self.assertFalse(file_utils.is_remote_path(p), p)

    def test_wrappers_reject_remote(self):
        funcs = [
            (file_utils.exists, ("/gcs/x",)),
            (file_utils.File, ("/gcs/x",)),
            (file_utils.join, ("/gcs/x", "y")),
            (file_utils.isdir, ("/gcs/x",)),
            (file_utils.remove, ("/gcs/x",)),
            (file_utils.rmtree, ("/gcs/x",)),
            (file_utils.listdir, ("/gcs/x",)),
            (file_utils.makedirs, ("/gcs/x",)),
        ]
        for fn, args in funcs:
            with self.assertRaisesRegex(ValueError, "remote paths"):
                fn(*args)

    def test_copy_rejects_either_remote(self):
        with self.assertRaisesRegex(ValueError, "remote paths"):
            file_utils.copy("/gcs/src", "/tmp/dst")
        with self.assertRaisesRegex(ValueError, "remote paths"):
            file_utils.copy("/tmp/src", "/gcs/dst")


class LocalFileWrappersTest(testing.TestCase):
    def test_local_filesystem_wrappers(self):
        tmp = self.get_temp_dir()
        work = file_utils.join(tmp, "work")
        os.makedirs(work)

        path = file_utils.join(work, "file.txt")
        with file_utils.File(path, "w") as f:
            f.write("body")

        self.assertTrue(file_utils.exists(path))
        self.assertTrue(file_utils.isdir(work))
        self.assertIn("file.txt", file_utils.listdir(work))

        dst = file_utils.join(work, "copy.txt")
        file_utils.copy(path, dst)
        self.assertTrue(file_utils.exists(dst))

        nested = file_utils.join(work, "a", "b")
        file_utils.makedirs(nested)
        self.assertTrue(file_utils.isdir(nested))

        file_utils.remove(path)
        self.assertFalse(file_utils.exists(path))

        file_utils.rmtree(work)
        self.assertFalse(os.path.exists(work))


class HashFileTest(testing.TestCase):
    def test_hash_file_sha256(self):
        tmp = self.get_temp_dir()
        path = os.path.join(tmp, "f.bin")
        _write(path, b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        self.assertEqual(file_utils.hash_file(path, algorithm="sha256"), expected)

    def test_hash_file_md5(self):
        tmp = self.get_temp_dir()
        path = os.path.join(tmp, "f.bin")
        _write(path, b"hello world")
        expected = hashlib.md5(b"hello world").hexdigest()
        self.assertEqual(file_utils.hash_file(path, algorithm="md5"), expected)

    def test_resolve_hasher_auto_sha256_from_length(self):
        # 64-char hash → sha256
        h = file_utils.resolve_hasher("auto", file_hash="a" * 64)
        self.assertEqual(h.name, "sha256")

    def test_resolve_hasher_auto_md5_default(self):
        # Anything else → md5
        h = file_utils.resolve_hasher("auto", file_hash="a" * 32)
        self.assertEqual(h.name, "md5")

    def test_validate_file_true(self):
        tmp = self.get_temp_dir()
        path = os.path.join(tmp, "f.bin")
        _write(path, b"data")
        expected = hashlib.sha256(b"data").hexdigest()
        self.assertTrue(file_utils.validate_file(path, expected, algorithm="sha256"))

    def test_validate_file_false(self):
        tmp = self.get_temp_dir()
        path = os.path.join(tmp, "f.bin")
        _write(path, b"data")
        self.assertFalse(file_utils.validate_file(path, "0" * 64, algorithm="sha256"))


class ExtractArchiveTest(testing.TestCase):
    def _make_zip(self, tmp, archive_name="data.zip"):
        archive_path = os.path.join(tmp, archive_name)
        inner_dir = os.path.join(tmp, "inner")
        os.makedirs(inner_dir)
        member = os.path.join(inner_dir, "hello.txt")
        _write(member, b"hi")
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.write(member, arcname="hello.txt")
        return archive_path

    def _make_tar(self, tmp, archive_name="data.tar"):
        archive_path = os.path.join(tmp, archive_name)
        inner_dir = os.path.join(tmp, "inner_t")
        os.makedirs(inner_dir)
        member = os.path.join(inner_dir, "world.txt")
        _write(member, b"world")
        with tarfile.open(archive_path, "w") as tf:
            tf.add(member, arcname="world.txt")
        return archive_path

    def test_extract_zip(self):
        tmp = self.get_temp_dir()
        archive = self._make_zip(tmp)
        dest = os.path.join(tmp, "out_zip")
        os.makedirs(dest)
        self.assertTrue(file_utils.extract_archive(archive, dest))
        self.assertTrue(os.path.exists(os.path.join(dest, "hello.txt")))

    def test_extract_tar(self):
        tmp = self.get_temp_dir()
        archive = self._make_tar(tmp)
        dest = os.path.join(tmp, "out_tar")
        os.makedirs(dest)
        self.assertTrue(file_utils.extract_archive(archive, dest))
        self.assertTrue(os.path.exists(os.path.join(dest, "world.txt")))

    def test_extract_none_format_returns_false(self):
        self.assertFalse(file_utils.extract_archive("/tmp/x", "/tmp", archive_format=None))

    def test_extract_unsupported_format_raises(self):
        tmp = self.get_temp_dir()
        path = os.path.join(tmp, "x.bin")
        _write(path)
        with self.assertRaises(NotImplementedError):
            file_utils.extract_archive(path, tmp, archive_format="rar")

    def test_extract_non_archive_returns_false(self):
        tmp = self.get_temp_dir()
        path = os.path.join(tmp, "plain.bin")
        _write(path, b"not an archive")
        self.assertFalse(file_utils.extract_archive(path, tmp))


class SafePathFilterTest(testing.TestCase):
    def test_filter_safe_paths_warns_on_escape(self):
        tmp = self.get_temp_dir()
        archive_path = os.path.join(tmp, "evil.tar")
        member_path = os.path.join(tmp, "ok.txt")
        _write(member_path, b"x")
        with tarfile.open(archive_path, "w") as tf:
            tf.add(member_path, arcname="ok.txt")
            # Inject a member with an absolute-style escape path.
            info = tarfile.TarInfo(name="../escape.txt")
            info.size = 0
            tf.addfile(info, fileobj=None)

        with tarfile.open(archive_path) as tf:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                kept = list(file_utils.filter_safe_paths(tf.getmembers()))
        names = {m.name for m in kept}
        self.assertIn("ok.txt", names)
        self.assertNotIn("../escape.txt", names)
        self.assertTrue(any("Skipping invalid path" in str(x.message) for x in w))


class GetFileTest(testing.TestCase):
    def test_get_file_requires_origin(self):
        with self.assertRaisesRegex(ValueError, '"origin"'):
            file_utils.get_file(fname="x")

    def test_get_file_rejects_paths_in_fname(self):
        with self.assertRaisesRegex(ValueError, "no longer accepted"):
            file_utils.get_file(
                fname=os.path.join("sub", "x"), origin="http://example/x"
            )

    def test_get_file_requires_parseable_fname(self):
        with self.assertRaisesRegex(ValueError, "parse the file name"):
            # URL with a trailing slash yields an empty path → unparseable.
            file_utils.get_file(origin="http://example/")

    def test_get_file_uses_cache_when_present(self):
        tmp = self.get_temp_dir()
        target = os.path.join(tmp, "datasets", "hello.txt")
        os.makedirs(os.path.dirname(target))
        _write(target, b"cached")
        # Pre-existing file with a matching hash → no download attempted.
        digest = hashlib.sha256(b"cached").hexdigest()

        with patch("synalinks.src.utils.file_utils.urlretrieve") as mock_dl:
            out = file_utils.get_file(
                fname="hello.txt",
                origin="http://example/hello.txt",
                file_hash=digest,
                cache_dir=tmp,
                progbar=False,
            )
        mock_dl.assert_not_called()
        self.assertEqual(out, target)

    def test_get_file_redownloads_on_hash_mismatch(self):
        tmp = self.get_temp_dir()
        target = os.path.join(tmp, "datasets", "x.txt")
        os.makedirs(os.path.dirname(target))
        _write(target, b"stale")

        good_digest = hashlib.sha256(b"fresh").hexdigest()

        def fake_dl(origin, dest):
            _write(dest, b"fresh")

        with patch(
            "synalinks.src.utils.file_utils.urlretrieve", side_effect=fake_dl
        ) as mock_dl:
            out = file_utils.get_file(
                fname="x.txt",
                origin="http://example/x.txt",
                file_hash=good_digest,
                cache_dir=tmp,
                progbar=False,
            )
        mock_dl.assert_called_once()
        self.assertEqual(out, target)
        with open(target, "rb") as f:
            self.assertEqual(f.read(), b"fresh")

    def test_get_file_http_error_is_wrapped(self):
        tmp = self.get_temp_dir()

        def fake_dl(origin, dest, *args, **kwargs):
            raise urllib.error.HTTPError(origin, 404, "Not Found", None, None)

        with patch(
            "synalinks.src.utils.file_utils.urlretrieve", side_effect=fake_dl
        ):
            with self.assertRaisesRegex(Exception, "URL fetch failure"):
                file_utils.get_file(
                    fname="x.bin",
                    origin="http://example/x.bin",
                    cache_dir=tmp,
                    progbar=False,
                )

    def test_get_file_corruption_raises(self):
        tmp = self.get_temp_dir()

        def fake_dl(origin, dest, *args, **kwargs):
            _write(dest, b"bad")

        with patch(
            "synalinks.src.utils.file_utils.urlretrieve", side_effect=fake_dl
        ):
            with self.assertRaisesRegex(ValueError, "Incomplete or corrupted"):
                file_utils.get_file(
                    fname="x.bin",
                    origin="http://example/x.bin",
                    file_hash="0" * 64,
                    cache_dir=tmp,
                    progbar=False,
                )

    def test_get_file_md5_legacy_argument(self):
        tmp = self.get_temp_dir()
        target = os.path.join(tmp, "datasets", "y.bin")
        os.makedirs(os.path.dirname(target))
        _write(target, b"hello")
        # md5_hash is still accepted as a legacy alias for file_hash.
        digest = hashlib.md5(b"hello").hexdigest()

        with patch("synalinks.src.utils.file_utils.urlretrieve") as mock_dl:
            file_utils.get_file(
                fname="y.bin",
                origin="http://example/y.bin",
                md5_hash=digest,
                cache_dir=tmp,
                progbar=False,
            )
        mock_dl.assert_not_called()

    def test_get_file_force_download_overrides_cache(self):
        tmp = self.get_temp_dir()
        target = os.path.join(tmp, "datasets", "z.bin")
        os.makedirs(os.path.dirname(target))
        _write(target, b"cached")

        def fake_dl(origin, dest, *args, **kwargs):
            _write(dest, b"new")

        with patch(
            "synalinks.src.utils.file_utils.urlretrieve", side_effect=fake_dl
        ) as mock_dl:
            file_utils.get_file(
                fname="z.bin",
                origin="http://example/z.bin",
                cache_dir=tmp,
                force_download=True,
                progbar=False,
            )
        mock_dl.assert_called_once()

    def test_get_file_with_extract(self):
        tmp = self.get_temp_dir()
        # Pre-stage a zip archive at the expected download target.
        archive_dir = os.path.join(tmp, "datasets")
        os.makedirs(archive_dir)

        # The downloader writes a zip file; verify extract path is returned.
        def fake_dl(origin, dest, *args, **kwargs):
            inner = os.path.join(archive_dir, "_inner.txt")
            _write(inner, b"content")
            with zipfile.ZipFile(dest, "w") as zf:
                zf.write(inner, arcname="inner.txt")

        with patch(
            "synalinks.src.utils.file_utils.urlretrieve", side_effect=fake_dl
        ):
            out = file_utils.get_file(
                fname="bundle.zip",
                origin="http://example/bundle.zip",
                cache_dir=tmp,
                extract=True,
                progbar=False,
            )
        # The function returns the extraction directory.
        self.assertTrue(out.endswith("bundle_extracted"))
        self.assertTrue(os.path.exists(os.path.join(out, "inner.txt")))
