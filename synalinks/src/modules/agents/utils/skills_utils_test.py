# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Tests for Agent Skills helpers in ``skills_utils``, which delegate the
spec logic to the official ``skills-ref`` reference library: lenient
``parse_skill``/``load_skill``, ``validate_skill`` returning a problems list,
and the ``<available_skills>`` ``skills_prompt``."""

import os
import tempfile

from synalinks.src import testing
from synalinks.src.modules.agents.utils.skills_utils import Skill
from synalinks.src.modules.agents.utils.skills_utils import SkillParseError
from synalinks.src.modules.agents.utils.skills_utils import SkillValidationError
from synalinks.src.modules.agents.utils.skills_utils import discover_skills
from synalinks.src.modules.agents.utils.skills_utils import find_skill_md
from synalinks.src.modules.agents.utils.skills_utils import load_skill
from synalinks.src.modules.agents.utils.skills_utils import parse_skill
from synalinks.src.modules.agents.utils.skills_utils import skills_prompt
from synalinks.src.modules.agents.utils.skills_utils import validate_skill

_FULL = """\
---
name: pdf-processing
description: Extract PDF text, fill forms, merge files. Use when handling PDFs.
license: Apache-2.0
compatibility: Requires python 3.12+
allowed-tools: Bash(git:*) Bash(jq:*) Read
metadata:
  author: example-org
  version: "1.0"
---

# PDF Processing

Use pdfplumber to extract text.
"""

_MINIMAL = """\
---
name: hello
description: A minimal skill.
---
Body here.
"""


class ParseSkillTest(testing.TestCase):
    def test_parses_all_frontmatter_fields(self):
        skill = parse_skill(_FULL)
        self.assertEqual(skill.name, "pdf-processing")
        self.assertEqual(
            skill.description,
            "Extract PDF text, fill forms, merge files. Use when handling PDFs.",
        )
        self.assertEqual(skill.license, "Apache-2.0")
        self.assertEqual(skill.compatibility, "Requires python 3.12+")
        # allowed-tools kept as the raw spec string; split via the helper
        self.assertEqual(skill.allowed_tools, "Bash(git:*) Bash(jq:*) Read")
        self.assertEqual(
            skill.allowed_tools_list(), ["Bash(git:*)", "Bash(jq:*)", "Read"]
        )
        # metadata coerced to a str->str map
        self.assertEqual(skill.metadata, {"author": "example-org", "version": "1.0"})
        # only spec frontmatter is modelled — the Markdown body is not stored
        self.assertFalse(hasattr(skill, "instructions"))

    def test_minimal_skill(self):
        skill = parse_skill(_MINIMAL)
        self.assertEqual(skill.name, "hello")
        self.assertEqual(skill.allowed_tools_list(), [])
        self.assertEqual(skill.metadata, {})
        self.assertIsNone(skill.license)

    def test_requires_frontmatter(self):
        with self.assertRaises(SkillParseError):
            parse_skill("# no frontmatter\njust text")

    def test_unterminated_frontmatter(self):
        with self.assertRaises(SkillParseError):
            parse_skill("---\nname: x\ndescription: y\n")

    def test_missing_required_field_raises_validation(self):
        with self.assertRaises(SkillValidationError):
            parse_skill("---\nname: ok\n---\nbody")

    def test_parse_is_lenient_about_name_format(self):
        # parse_skill is lenient (required fields only); a non-conforming but
        # present name still loads — validate_skill is the strict checker.
        skill = parse_skill("---\nname: NotKebab\ndescription: d\n---\n")
        self.assertEqual(skill.name, "NotKebab")

    def test_skill_is_spec_only_entity(self):
        # Skill is an `Entity` DataModel whose fields are exactly the spec
        # frontmatter (plus `label`, the Entity type pin) — no body / locators.
        from synalinks.src.backend import is_entity

        skill = parse_skill(_FULL)
        self.assertIsInstance(skill, Skill)
        self.assertTrue(is_entity(skill))
        self.assertEqual(skill.label, "Skill")
        self.assertEqual(
            set(Skill.get_schema()["properties"]),
            {
                "label",
                "name",
                "description",
                "license",
                "compatibility",
                "allowed_tools",
                "metadata",
            },
        )


class ValidateSkillTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)
        super().tearDown()

    def _write(self, name, text):
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as fh:
            fh.write(text)
        return d

    def test_valid_skill_has_no_problems(self):
        d = self._write("pdf-processing", _FULL)
        self.assertEqual(validate_skill(d), [])

    def test_flags_uppercase_and_consecutive_hyphen_names(self):
        d = self._write("Bad-Name", _MINIMAL.replace("hello", "Bad-Name"))
        problems = validate_skill(d)
        self.assertTrue(any("lowercase" in p for p in problems))

    def test_flags_directory_name_mismatch(self):
        d = self._write("wrongdir", _MINIMAL)  # SKILL.md says name: hello
        problems = validate_skill(d)
        self.assertTrue(any("must match" in p for p in problems))

    def test_flags_unexpected_frontmatter_field(self):
        d = self._write("hello", _MINIMAL.replace("---\nBody", "extra: nope\n---\nBody"))
        problems = validate_skill(d)
        self.assertTrue(any("Unexpected" in p for p in problems))

    def test_accepts_unicode_lowercase_name(self):
        # The spec allows i18n (Unicode lowercase) letters in names.
        d = self._write("café", _MINIMAL.replace("hello", "café"))
        self.assertEqual(validate_skill(d), [])

    def test_missing_skill_md(self):
        os.makedirs(os.path.join(self.root, "empty"))
        self.assertEqual(
            validate_skill(os.path.join(self.root, "empty")),
            ["Missing required file: SKILL.md"],
        )


class FindAndDiscoverTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)
        super().tearDown()

    def _write(self, name, text, filename="SKILL.md"):
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, filename), "w", encoding="utf-8") as fh:
            fh.write(text)
        return d

    def test_find_skill_md_prefers_uppercase_accepts_lowercase(self):
        d = self._write("low", _MINIMAL.replace("hello", "low"), filename="skill.md")
        # On a case-insensitive filesystem (the default on macOS and Windows)
        # "SKILL.md" and "skill.md" are the same file, so the uppercase
        # preference cannot be observed — skip rather than assert a name the
        # filesystem folds away.
        if os.path.exists(os.path.join(d, "SKILL.md")):
            self.skipTest("filesystem is case-insensitive")
        found = find_skill_md(d)
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "skill.md")

    def test_discovers_and_sorts_skills(self):
        self._write("hello", _MINIMAL)
        self._write("pdf-processing", _FULL)
        # discover_skills returns the skill *directories*, sorted by name.
        dirs = discover_skills(self.root)
        self.assertEqual([d.name for d in dirs], ["hello", "pdf-processing"])
        self.assertTrue(str(dirs[0]).endswith("/hello"))
        self.assertEqual(find_skill_md(dirs[0]).name, "SKILL.md")

    def test_missing_root_returns_empty(self):
        self.assertEqual(discover_skills(os.path.join(self.root, "nope")), [])
        self.assertEqual(discover_skills(None), [])

    def test_skips_dir_without_skill_md(self):
        os.makedirs(os.path.join(self.root, "notaskill"))
        self._write("hello", _MINIMAL)
        self.assertEqual([d.name for d in discover_skills(self.root)], ["hello"])

    def test_malformed_skipped_by_default_strict_raises(self):
        self._write("hello", _MINIMAL)
        self._write("bad", "no frontmatter at all")  # unparseable
        self.assertEqual([d.name for d in discover_skills(self.root)], ["hello"])
        with self.assertRaises(SkillParseError):
            discover_skills(self.root, strict=True)

    def test_load_skill_missing(self):
        with self.assertRaises(SkillParseError):
            load_skill(os.path.join(self.root, "absent"))


class SkillsPromptTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # realpath: on macOS tempfile returns /var/folders/... (a symlink to
        # /private/var/...) while the rendered <location> is canonicalized, so
        # pin the canonical form to keep the path assertions portable.
        self.root = os.path.realpath(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)
        super().tearDown()

    def _write(self, name, text):
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as fh:
            fh.write(text)
        return d

    def test_empty(self):
        self.assertEqual(skills_prompt([]), "<available_skills>\n</available_skills>")

    def test_available_skills_xml(self):
        # skills_prompt renders from skill *directories* (via skills_ref.to_prompt).
        d = self._write("pdf", "---\nname: pdf\ndescription: Handle PDFs.\n---\n")
        out = skills_prompt([d])
        self.assertIn("<available_skills>", out)
        self.assertIn("<skill>", out)
        self.assertIn("<name>\npdf\n</name>", out)
        self.assertIn("<description>\nHandle PDFs.\n</description>", out)
        self.assertIn(f"<location>\n{d}/SKILL.md\n</location>", out)

    def test_root_overrides_location_prefix(self):
        d = self._write("pdf", _MINIMAL.replace("hello", "pdf"))
        out = skills_prompt([d], root="/work/skills")
        self.assertIn("/work/skills/pdf/SKILL.md", out)
        self.assertNotIn(self.root, out)

    def test_html_escapes_description(self):
        d = self._write("x", "---\nname: x\ndescription: a & b <c>\n---\n")
        out = skills_prompt([d])
        self.assertIn("a &amp; b &lt;c&gt;", out)
