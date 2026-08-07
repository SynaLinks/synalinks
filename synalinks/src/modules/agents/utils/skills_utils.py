# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Helpers for the open Agent Skills standard (https://agentskills.io).

A *skill* is a directory with a ``SKILL.md`` (YAML frontmatter + Markdown
instructions) and optional ``scripts/`` / ``references/`` / ``assets/``. The
model follows **progressive disclosure**: only each skill's ``name`` +
``description`` are surfaced up front (the ``<available_skills>`` prompt block);
the full ``SKILL.md`` body and bundled files are read on demand through the
agent's own file / bash tools, so a code agent (Deep Agent / RLM) consumes
skills with no extra machinery.

All spec-governed logic (YAML frontmatter parsing, the naming / length /
allowed-field validation rules, and the ``<available_skills>`` XML) is
delegated to Anthropic's official reference library ``skills-ref``
(https://github.com/agentskills/agentskills); this module only adds the layer
``skills-ref`` does not provide: a `Skill` `Entity` DataModel (the spec
frontmatter as a synalinks data model), multi-root directory discovery
(`discover_skills` / `discover_skills_in_roots`), root path resolution
(`resolve_skills_paths`), and a `skills_prompt` that adds a sandbox ``root``
override on top of ``skills_ref.to_prompt``. Like ``skills-ref``, discovery and
the prompt work in terms of skill *directories*; a skill's Markdown body and
bundled files are not modelled here; they are read on demand. The general
agent helpers (workdir, AGENTS.md, tool merging, input summaries) live in
``agents_utils``.
"""

from pathlib import Path
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

# Spec primitives come straight from the official reference library so the
# rules (frontmatter schema, naming conventions, length limits, discovery file
# names) never drift from agentskills.io.
from skills_ref import SkillError
from skills_ref import find_skill_md as _find_skill_md
from skills_ref import read_properties
from skills_ref import to_prompt
from skills_ref import validate as _validate_skill_dir
from skills_ref.errors import ParseError
from skills_ref.errors import ValidationError
from skills_ref.parser import parse_frontmatter
from skills_ref.validator import validate_metadata as _validate_metadata

from synalinks.src.backend import Entity
from synalinks.src.backend import Field

# Back-compat aliases: synalinks historically exposed these names. They are the
# official exceptions (``ParseError`` / ``ValidationError`` both subclass
# ``SkillError``), so existing ``except SkillError`` / ``isinstance`` checks and
# ``SkillValidationError.errors`` keep working unchanged.
SkillParseError = ParseError
SkillValidationError = ValidationError

__all__ = [
    "Skill",
    "SkillError",
    "SkillParseError",
    "SkillValidationError",
    "find_skill_md",
    "parse_skill",
    "load_skill",
    "validate_skill",
    "validate_skill_metadata",
    "discover_skills",
    "discover_skills_in_roots",
    "resolve_skills_paths",
    "skills_prompt",
    "READ_SKILL_TOOL_NAME",
    "build_read_skill_tool",
]


class Skill(Entity):
    """A parsed Agent Skill's frontmatter properties.

    A knowledge-graph `Entity` (``label`` pinned to ``"Skill"``) so skills are
    first-class synalinks data models (schema-bearing, JSON-serializable via
    ``get_json``, and storable in a `KnowledgeBase`) rather than opaque
    dataclasses. The fields are exactly the spec frontmatter (the six fields in
    https://agentskills.io/specification, mirroring ``skills-ref``'s
    ``SkillProperties``); ``label`` is the only addition, required structurally
    by `Entity`. Locations (the skill directory / ``SKILL.md`` path) and the
    Markdown body are *not* spec frontmatter and are deliberately not stored
    here: discovery works in terms of directories and the body is read on
    demand (progressive disclosure). Per the spec model ``allowed_tools`` is the
    raw space-separated string; use `allowed_tools_list` to split it.
    """

    label: Literal["Skill"] = "Skill"
    name: str = Field(description="The skill name (its directory name)")
    description: str = Field(description="When the skill should be used")
    license: Optional[str] = Field(default=None, description="SPDX license id")
    compatibility: Optional[str] = Field(
        default=None, description="Runtime/environment compatibility note"
    )
    allowed_tools: Optional[str] = Field(
        default=None, description="Raw space-separated ``allowed-tools`` patterns"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict, description="Free-form ``metadata`` frontmatter map"
    )

    def allowed_tools_list(self) -> List[str]:
        """The ``allowed-tools`` string split into individual tool patterns."""
        return self.allowed_tools.split() if self.allowed_tools else []


def find_skill_md(skill_dir) -> Optional[Path]:
    """Return the skill's ``SKILL.md`` (preferred) or ``skill.md``, else ``None``.

    Thin wrapper over ``skills_ref.find_skill_md`` that accepts any path-like.
    """
    return _find_skill_md(Path(skill_dir))


def parse_skill(content: str) -> Skill:
    """Parse ``SKILL.md`` text into a `Skill` (lenient: required fields only).

    Frontmatter is parsed by ``skills_ref.parser.parse_frontmatter`` (the
    official strictyaml-backed parser); this only checks that ``name`` and
    ``description`` are present and non-empty; use `validate_skill` for the full
    naming / length / allowed-field rules. The Markdown body is discarded (it is
    not spec frontmatter; agents read it on demand). Raises `SkillParseError`
    (bad frontmatter) / `SkillValidationError` (missing required field).
    """
    metadata, _ = parse_frontmatter(content)
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not name.strip():
        raise SkillValidationError("skill `name` is required and must be non-empty")
    if not isinstance(description, str) or not description.strip():
        raise SkillValidationError(
            "skill `description` is required and must be non-empty"
        )
    # ``parse_frontmatter`` already coerces the ``metadata`` block to a str->str
    # map; default to an empty dict when absent.
    skill_metadata = metadata.get("metadata") or {}
    if not isinstance(skill_metadata, dict):
        raise SkillValidationError("skill `metadata` must be a mapping")
    return Skill(
        name=name.strip(),
        description=description.strip(),
        license=metadata.get("license"),
        compatibility=metadata.get("compatibility"),
        allowed_tools=metadata.get("allowed-tools"),
        metadata={str(k): str(v) for k, v in skill_metadata.items()},
    )


def load_skill(directory) -> Skill:
    """Load and parse the ``SKILL.md`` in a skill ``directory`` into a `Skill`."""
    skill_dir = Path(directory)
    skill_md = find_skill_md(skill_dir)
    if skill_md is None:
        raise SkillParseError(f"no SKILL.md in {directory}")
    return parse_skill(skill_md.read_text(encoding="utf-8"))


def validate_skill_metadata(metadata: dict, *, dir_name=None) -> List[str]:
    """Return spec problems in already-parsed skill ``metadata`` (``[]`` = valid).

    Delegates to ``skills_ref.validator.validate_metadata``: flags unexpected
    frontmatter fields, missing/invalid required fields, name-format violations
    (Unicode-aware, lowercase, hyphen rules, directory match) and over-length
    fields.
    """
    skill_dir = Path(dir_name) if dir_name is not None else None
    return _validate_metadata(metadata, skill_dir)


def validate_skill(directory) -> List[str]:
    """Validate a skill ``directory`` against the spec; return a list of problems.

    ``[]`` means valid. Never raises for spec violations: delegates directly to
    ``skills_ref.validate`` (a parse failure comes back as a single-item list).
    """
    return _validate_skill_dir(Path(directory))


def discover_skills(root, *, strict: bool = False) -> List[Path]:
    """Discover Agent Skill *directories* under ``root`` (each ``<root>/<name>``).

    Returns the skill directories (those holding a parseable ``SKILL.md``) sorted
    by name. Directories are the unit of work, mirroring ``skills-ref`` (whose
    ``validate`` / ``read_properties`` / ``to_prompt`` all take paths); call
    `load_skill` on one for its `Skill` metadata, or read its files directly
    (progressive disclosure). A directory whose ``SKILL.md`` fails to parse is
    skipped by default; with ``strict=True`` the first `SkillError` propagates.
    Returns ``[]`` when ``root`` is missing or not a directory.
    """
    base = Path(root) if root else None
    if base is None or not base.is_dir():
        return []
    dirs: List[Path] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or find_skill_md(entry) is None:
            continue
        try:
            load_skill(entry)  # parse-check only (drop the result)
        except SkillError:
            if strict:
                raise
            continue
        dirs.append(entry)
    return sorted(dirs, key=lambda d: d.name)


def resolve_skills_paths(skills) -> List[str]:
    """Validate and resolve skill *root* directories to absolute paths.

    ``skills`` is an iterable of folder paths, each a *root* under which skills
    live as ``<root>/<name>/SKILL.md`` (the layout `discover_skills` expects).
    Each path must exist and be a directory: a typo'd root that silently
    yielded no skills would be a poor UX, so this raises instead (matching
    ``resolve_workdir``).

    Returns:
        The resolved absolute paths as a list (``[]`` when ``skills`` is falsy).

    Raises:
        ValueError: If a path is missing or is not a directory.
    """
    if not skills:
        return []
    resolved: List[str] = []
    for path in skills:
        p = Path(path).resolve()
        if not p.exists():
            raise ValueError(f"skills path does not exist: {path}")
        if not p.is_dir():
            raise ValueError(f"skills path is not a directory: {path}")
        resolved.append(str(p))
    return resolved


def discover_skills_in_roots(roots, *, strict: bool = False) -> List[Path]:
    """Discover Agent Skill directories across multiple root directories.

    Runs `discover_skills` on each root in ``roots`` and merges the results,
    deduped by directory name (**first root wins**: earlier roots take
    precedence over later ones for a shared name; the spec requires a skill's
    ``name`` to equal its directory name). Returns the directories sorted by
    name. With ``strict=True`` a malformed skill propagates its `SkillError`.
    """
    seen: dict = {}
    for root in roots or []:
        for skill_dir in discover_skills(root, strict=strict):
            if skill_dir.name not in seen:
                seen[skill_dir.name] = skill_dir
    return sorted(seen.values(), key=lambda d: d.name)


def skills_prompt(skill_dirs, *, root: Optional[str] = None) -> str:
    """Render the ``<available_skills>`` prompt block (Agent Skills *level 1*).

    ``skill_dirs`` is an iterable of skill directories (as from `discover_skills`
    / `discover_skills_in_roots`). Without ``root`` this delegates to
    ``skills_ref.to_prompt``: the exact XML Anthropic recommends, one
    ``<skill>`` per entry with ``<name>`` / ``<description>`` / ``<location>``
    (the absolute path to ``SKILL.md``). ``root`` overrides the ``<location>``
    prefix (e.g. the skills directory as the *sandbox* sees it, when the host
    path differs), a remap ``skills_ref.to_prompt`` cannot do, so that case is
    rendered locally to byte-identical XML using ``skills_ref.read_properties``
    for each skill's name / description.
    """
    skill_dirs = [Path(d) for d in skill_dirs]
    if not root:
        return to_prompt(skill_dirs)

    import html

    if not skill_dirs:
        return "<available_skills>\n</available_skills>"
    lines = ["<available_skills>"]
    for skill_dir in skill_dirs:
        props = read_properties(skill_dir)
        location = f"{root.rstrip('/')}/{skill_dir.name}/SKILL.md"
        lines += [
            "<skill>",
            "<name>",
            html.escape(props.name),
            "</name>",
            "<description>",
            html.escape(props.description),
            "</description>",
            "<location>",
            location,
            "</location>",
            "</skill>",
        ]
    lines.append("</available_skills>")
    return "\n".join(lines)


#: The reserved name of the built-in skill-reading tool (see
#: `build_read_skill_tool`).
READ_SKILL_TOOL_NAME = "read_skill"


def build_read_skill_tool(roots, *, max_chars: int = 50_000):
    """Build the built-in ``read_skill`` tool over Agent Skill *root* dirs.

    The ``<available_skills>`` block surfaces only each skill's name and
    description (level 1 of progressive disclosure); the spec assumes the
    agent reads the rest through its own file tools. A tool-calling agent
    whose tools are domain tools (SQL, retrieval, ...) has none — its skills
    would be advertised but unreadable. This tool completes the disclosure
    chain for those agents, with the activation shape the ecosystem settled
    on (Claude Code's Skill tool, the Claude API's container skills):
    **one name-like argument**. The bare skill name returns the ``SKILL.md``
    body plus the bundled file listing (level 2 — activation); the name
    prefixing one of the relative paths skills use to reference their own
    files (``pdf-processing/references/api.md``) returns that bundled file
    (level 3).

    Reads are confined to the discovered skill directories: the requested
    path is resolved and must stay inside the skill's own directory, so
    neither ``..`` traversal nor absolute paths nor symlinks escaping the
    skill can reach anything else. Errors come back as observations (an
    ``"error"`` key with the available names/files), never as exceptions —
    a wrong guess costs the agent one round, not the run.

    Args:
        roots: Skill root directories, as accepted by `discover_skills_in_roots`
            (each root holds ``<name>/SKILL.md`` folders; first root wins on a
            shared name — matching what the listing shows).
        max_chars: Truncation budget for a returned file, so a large bundled
            asset cannot flood the context window.

    Returns:
        The ``read_skill`` `Tool`.
    """
    # Deferred: `Tool` builds on `Module`, and importing it at module scope
    # would cycle through `modules/__init__` while `modules.agents.utils` is
    # itself being imported.
    from synalinks.src.modules.core.tool import Tool

    resolved = list(roots or [])

    async def read_skill(skill: str) -> dict:
        """Read an installed Agent Skill.

        The available skills (name + description) are listed in the
        <available_skills> block of your context; this tool returns the rest
        on demand. Read a skill BEFORE following it. When its instructions
        reference a bundled file, read that file the same way, prefixing the
        skill name to the file's relative path.

        Args:
            skill (str): The skill name from <available_skills>
                (e.g. 'pdf-processing') to read its instructions — the
                response also lists the skill's bundled files. Or a bundled
                file the instructions mention, as the skill name followed by
                the file's relative path (e.g.
                'pdf-processing/references/api.md').
        """
        name, _, relative = (skill or "").strip().strip("/").partition("/")
        skill_dirs = {d.name: d for d in discover_skills_in_roots(resolved)}
        skill_dir = skill_dirs.get(name)
        if skill_dir is None:
            return {
                "error": f"Unknown skill {name!r}.",
                "available_skills": sorted(skill_dirs),
            }
        relative = relative or "SKILL.md"
        root = skill_dir.resolve()
        target = (root / relative).resolve()
        if root != target and root not in target.parents:
            return {
                "error": (
                    f"{skill!r} is outside the {name!r} skill directory; only "
                    f"the skill's own files can be read."
                )
            }
        files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        if not target.is_file():
            return {
                "error": f"No file {relative!r} in skill {name!r}.",
                "available_files": files,
            }
        content = target.read_text(encoding="utf-8", errors="replace")
        report = {"skill": name, "file": relative}
        if relative == "SKILL.md" and len(files) > 1:
            # Activation also surfaces what else the skill bundles, so the
            # agent knows what it can read next (level 3).
            report["files"] = files
        if len(content) > max_chars:
            report["note"] = (
                f"Truncated to the first {max_chars} of {len(content)} characters."
            )
            content = content[:max_chars]
        report["content"] = content
        return report

    return Tool(read_skill, name=READ_SKILL_TOOL_NAME)
