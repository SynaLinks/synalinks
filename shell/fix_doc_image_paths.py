"""Post-build fix for asset links in the Zensical-built ``site/``.

Zensical (through mkdocstrings) miscomputes the relative depth of image links
that come from Python **docstrings** rendered by ``:::`` reference blocks: it
emits one ``../`` too many for ``examples.*`` pages and two too many for the
deeper ``synalinks.*`` API pages, deriving the count from the object's dotted
identifier instead of the page's URL. The page's own CSS/JS links are correct,
so the bug is specific to docstring-injected images.

At a root deploy the extra ``../`` is harmlessly clamped at ``/``, but the docs
publish under the ``/synalinks/`` project-pages sub-path (see ``site_url`` in
``mkdocs.yml``), so the surplus ``../`` climbs *past* ``/synalinks/`` to the
domain root and every such image 404s.

Every asset lives at ``site/assets/``, so the correct number of ``../`` for any
page is exactly its directory depth below ``site/``. This pass rewrites the
leading ``./``/``../`` run of every ``assets/...`` link (and only those — it
leaves ``http(s)://`` and root-absolute ``/assets`` links untouched) to match
that depth. Links that were already correct (CSS, nav) normalize to themselves.

Run after ``zensical build`` and before publishing.
"""

import pathlib
import re
import sys

# Matches the leading dot-run of a relative ``assets/...`` URL in src/href, e.g.
# ``src="../../../assets/examples/x.png"`` -> groups: attr, dots, rest.
_LINK_RE = re.compile(r'((?:src|href)=")((?:\.\./)*(?:\./)?)(assets/[^"]*")')


def _depth(html_path: pathlib.Path, site_root: pathlib.Path) -> int:
    """Directory depth of a built page below ``site/`` (its ``../`` count)."""
    return len(html_path.relative_to(site_root).parent.parts)


def fix_file(html_path: pathlib.Path, site_root: pathlib.Path) -> int:
    prefix = "../" * _depth(html_path, site_root)
    text = html_path.read_text(encoding="utf-8")
    new_text, n = _LINK_RE.subn(lambda m: f"{m.group(1)}{prefix}{m.group(3)}", text)
    changed = sum(
        1
        for m in _LINK_RE.finditer(text)
        if m.group(2) != prefix  # count only links whose dot-run actually moved
    )
    if changed:
        html_path.write_text(new_text, encoding="utf-8")
    return changed


def main(argv: list[str]) -> int:
    site_root = pathlib.Path(argv[1] if len(argv) > 1 else "site").resolve()
    if not site_root.is_dir():
        print(f"fix_doc_image_paths: no such site dir: {site_root}", file=sys.stderr)
        return 1
    pages = total = 0
    for html_path in site_root.rglob("*.html"):
        fixed = fix_file(html_path, site_root)
        if fixed:
            pages += 1
            total += fixed
    print(f"fix_doc_image_paths: corrected {total} asset link(s) across {pages} page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
