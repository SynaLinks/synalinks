# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Shared helpers used by both SQL and graph KB adapters.

Keeping these in one place stops the DuckDB and Ladybug adapters from
drifting on shared concerns — output formatting and per-query keyword
alignment — that have nothing to do with the underlying engine.
"""

import io
import os
import re
from typing import List
from typing import Optional
from typing import Union

import pyarrow as pa
import pyarrow.csv as pa_csv

from synalinks.src.backend.config import synalinks_home
from synalinks.src.utils.naming import to_pascal_case
from synalinks.src.utils.naming import to_snake_case

SEARCH_OUTPUT_FORMATS = ("json", "csv")


# Shared identifier shape for SQL columns/tables and Cypher labels/properties.
# Both engines reject (or, for Cypher catalog procs, refuse to bind) anything
# that isn't a bare ASCII identifier, so the regex doubles as an injection
# gate for the few spots where we have to interpolate names rather than bind
# them as parameters.
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sanitize_identifier(name: str) -> str:
    """Validate that ``name`` is a bare ASCII identifier.

    The base check used by both the SQL and graph adapters whenever a
    name has to be interpolated into a query (table/column for SQL,
    node/relation label and property name for Cypher) rather than
    bound as a parameter.

    Raises:
        ValueError: when ``name`` isn't a string or doesn't match
            :data:`IDENTIFIER_RE`.
    """
    if not isinstance(name, str) or not IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid identifier: {name!r}")
    return name


def to_pascal_identifier(name: str) -> str:
    """PascalCase-normalize ``name`` and validate as an identifier.

    Used for table names (DuckDB) and node/relation labels (Ladybug)
    so a KB that mixes SQL and graph backends ends up with the same
    capitalization regardless of how the caller spelled the input.
    The pascal-case step also strips separators and punctuation, so
    injection-shaped inputs (``"Doc; DROP TABLE x"``) collapse to a
    plain identifier (``"DocDropTableX"``) before the regex check.

    Raises:
        ValueError: when the normalized result doesn't match
            :data:`IDENTIFIER_RE` (empty / leading digit / etc.).
    """
    return sanitize_identifier(to_pascal_case(name))


def to_snake_identifier(name: str, *, strict: bool = True) -> Optional[str]:
    """snake_case-normalize ``name`` and validate as an identifier.

    Used for column names (DuckDB) and property names (Ladybug). The
    ``strict`` flag picks the failure mode:

      * ``strict=True`` (default): raise ``ValueError`` on a
        non-recoverable name. Right for SQL call sites where keys
        come from a registered schema we trust.
      * ``strict=False``: return ``None`` so callers can warn-and-drop.
        Right for the graph adapter where property dicts can arrive
        from external entity/relation payloads with arbitrary keys.
    """
    normalized = to_snake_case(name)
    if not IDENTIFIER_RE.match(normalized):
        if strict:
            raise ValueError(f"Invalid identifier: {name!r}")
        return None
    return normalized


def resolve_db_path(
    uri: Optional[str],
    *,
    scheme: str,
    extension: str,
    name: Optional[str] = None,
    default_stem: str = "database",
) -> str:
    """Strip ``scheme://`` from ``uri`` or build a default path under ``.synalinks``.

    Shared by every KB adapter so a no-args ``KnowledgeBase()`` lands
    every store under the same ``synalinks_home()`` directory and
    survives across processes — SQL goes to ``database.db``, the
    graph store sits next to it as ``database.lb``. Recognized inputs:

      * ``None`` → ``{synalinks_home()}/{name or default_stem}.{extension}``
      * ``"<scheme>://:memory:"`` or bare ``":memory:"`` → ``":memory:"``
        (in-memory; explicit opt-in).
      * ``"<scheme>://<path>"`` or bare ``<path>`` → the path verbatim.

    ``name`` (when supplied for the default-path branch) is validated
    as an identifier first: it interpolates into the file path, so a
    traversal-shaped value (``"../escape"``) would create a file
    outside ``synalinks_home`` and is rejected up-front.

    Args:
        uri: Caller-supplied URI, ``":memory:"`` sentinel, or ``None``
            for the synalinks-home default.
        scheme: URI scheme to strip (``"duckdb"`` / ``"ladybug"``).
        extension: File extension (no leading dot) used in the
            default-path branch.
        name: Optional adapter/KB name. Replaces ``default_stem`` in
            the default-path branch; must be identifier-shaped.
        default_stem: Filename stem when ``name`` is ``None``.

    Raises:
        ValueError: ``name`` isn't identifier-shaped (path traversal
            guard).
    """
    if uri:
        return uri.replace(f"{scheme}://", "", 1)
    if name is not None:
        sanitize_identifier(name)
    stem = name if name else default_stem
    return os.path.join(synalinks_home(), f"{stem}.{extension}")


def format_search_results(arrow_or_records, output_format: str):
    """Render a search result set as ``json`` (list of dicts) or ``csv`` (text).

    Accepts either a PyArrow ``Table`` (the DuckDB adapter's native
    result shape) or a list of dicts (what the Ladybug adapter
    produces from row-iteration). PyArrow handles the CSV encoding
    for both — it's faster than Python's ``csv`` module and gets
    quoting right out of the box.
    """
    if output_format not in SEARCH_OUTPUT_FORMATS:
        raise ValueError(
            f"Unknown output_format {output_format!r}; "
            f"expected one of {SEARCH_OUTPUT_FORMATS}."
        )

    if output_format == "json":
        if isinstance(arrow_or_records, pa.Table):
            return arrow_or_records.to_pylist()
        return arrow_or_records

    if isinstance(arrow_or_records, pa.Table):
        arrow_table = arrow_or_records
    else:
        if not arrow_or_records:
            return ""
        arrow_table = pa.Table.from_pylist(arrow_or_records)
    buf = io.BytesIO()
    pa_csv.write_csv(arrow_table, buf)
    return buf.getvalue().decode("utf-8")


def minmax_normalize_scores(records: List[dict], *, key: str = "score") -> List[dict]:
    """Min-max scale ``records[i][key]`` into ``[0, 1]`` in place (higher = better).

    Full-text/BM25 scores live on engine-specific ranges (DuckDB's
    ``match_bm25`` and LanceDB's Tantivy ``_score`` differ in magnitude
    and shape), so the raw numbers aren't comparable across adapters.
    Rescaling each returned result set so the best hit maps to ``1.0``
    and the worst to ``0.0`` gives a bounded, backend-independent
    relevance score: the endpoints and the ranking match across engines
    (the spacing in between stays engine-dependent).

    Edge cases: a single row, or an all-equal set, maps every score to
    ``1.0`` (every row is "the best"); ``None`` scores map to ``0.0``.

    Returns the same ``records`` list for convenient chaining.
    """
    values = [r.get(key) for r in records if r.get(key) is not None]
    if not values:
        return records
    lo, hi = min(values), max(values)
    span = hi - lo
    for r in records:
        v = r.get(key)
        if v is None:
            r[key] = 0.0
        elif span == 0:
            r[key] = 1.0
        else:
            r[key] = (v - lo) / span
    return records


def align_keywords(
    text_or_texts: Union[str, List[str]],
    keywords: Optional[Union[str, List[str]]],
    *,
    text_arg_name: str = "text_or_texts",
    keyword_arg_name: str = "keywords",
) -> tuple:
    """Pair vector-side texts with FTS-side keywords for hybrid search.

    Hybrid methods take ``text_or_texts`` for the vector branch and a
    parallel ``keywords`` argument for the BM25 branch — they look at
    different signals (semantic vs lexical) so the natural-language
    query that drives the vectors often isn't the keyword set you'd
    hand to BM25. When ``keywords`` is omitted the helper falls back
    to reusing ``text_or_texts`` so existing call sites keep working.

    Returns ``(texts: List[str], keyword_list: List[str])`` both of
    equal length so the caller can ``zip`` them through the per-query
    loop.

    Raises:
        ValueError: ``keywords`` and ``text_or_texts`` are both lists
            but of different lengths — the per-query pairing would be
            ambiguous.
    """
    texts: List[str] = (
        [text_or_texts] if not isinstance(text_or_texts, list) else list(text_or_texts)
    )
    if keywords is None:
        return texts, list(texts)
    keyword_list: List[str] = (
        [keywords] if not isinstance(keywords, list) else list(keywords)
    )
    if len(keyword_list) != len(texts):
        raise ValueError(
            f"`{keyword_arg_name}` must align with `{text_arg_name}`: "
            f"got {len(keyword_list)} keyword(s) vs {len(texts)} text(s)."
        )
    return texts, keyword_list
