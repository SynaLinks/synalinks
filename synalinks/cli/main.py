"""Command line interface for Synalinks.

Exposes the ``synalinks`` console script (see ``[project.scripts]`` in
``pyproject.toml``). The ``init`` command scaffolds a new project from one of
the bundled templates, which ship inside the wheel so it works fully offline.
"""

import importlib.resources
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

import click
import inquirer

# Templates live next to this module (``synalinks/cli/templates``) and are
# packaged with the wheel so ``synalinks init`` works without network access.
_TEMPLATES_PACKAGE = "synalinks.cli.templates"

# Template selected by default (when the user accepts the prompt or runs
# non-interactively without ``--template``).
_DEFAULT_TEMPLATE = "script"

# Files/directories that should never be copied into a generated project even
# if they sneak into the packaged templates.
_COPY_IGNORE = shutil.ignore_patterns(
    ".venv", "venv", "__pycache__", "*.pyc", "uv.lock", "*.egg-info"
)


def _templates_root(stack: ExitStack) -> Path:
    """Return a concrete filesystem path to the bundled templates directory.

    Uses ``importlib.resources`` so it resolves correctly whether Synalinks is
    installed as a regular package or extracted from a wheel/zip. The
    ``ExitStack`` keeps any temporary extraction alive for the caller.
    """
    resource = importlib.resources.files(_TEMPLATES_PACKAGE)
    return stack.enter_context(importlib.resources.as_file(resource))


def _available_templates() -> list[str]:
    with ExitStack() as stack:
        root = _templates_root(stack)
        names = sorted(p.name for p in root.iterdir() if p.is_dir())
    # Surface the default template first so it heads the selection menu.
    if _DEFAULT_TEMPLATE in names:
        names.remove(_DEFAULT_TEMPLATE)
        names.insert(0, _DEFAULT_TEMPLATE)
    return names


@click.group()
@click.version_option(package_name="synalinks", message="%(version)s")
def cli():
    """Synalinks command line interface."""


@cli.command()
@click.argument("project_name", required=False)
@click.option(
    "-n",
    "--name",
    "name",
    default=None,
    help="Project name (and generated package name). Pass this with "
    "--template to scaffold non-interactively.",
)
@click.option(
    "-d",
    "--description",
    "description",
    default=None,
    help="Project description written into the generated pyproject.toml.",
)
@click.option(
    "-t",
    "--template",
    "template",
    default=None,
    help="Template to scaffold from. If omitted, you'll pick from a menu "
    f"(default: {_DEFAULT_TEMPLATE}).",
)
@click.option(
    "-l",
    "--list",
    "list_templates",
    is_flag=True,
    help="List the available templates and exit.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Allow scaffolding into an existing non-empty directory.",
)
def init(project_name, name, description, template, list_templates, force):
    """Create a new Synalinks project from a bundled template.

    The project name may be given as a positional argument or via ``-n``.
    Pass ``-n`` and ``-t`` together to scaffold without any prompts. Examples:

        synalinks init my-agent
        synalinks init my-agent --template autotrain
        synalinks init -n my-agent -t autotrain -d "My research agent"
        synalinks init --list
    """
    templates = _available_templates()
    if not templates:
        raise click.ClickException(
            "No templates are bundled with this Synalinks installation."
        )

    if list_templates:
        click.echo("Available templates:")
        for name_ in templates:
            click.echo(f"  - {name_}")
        return

    # ``-n`` takes precedence over the positional argument.
    project_name = (name or project_name or "").strip()
    if not project_name:
        if sys.stdin.isatty():
            project_name = click.prompt("Project name", type=str).strip()
        if not project_name:
            raise click.ClickException(
                "A project name is required (pass it positionally or with -n)."
            )

    default = _DEFAULT_TEMPLATE if _DEFAULT_TEMPLATE in templates else templates[0]
    if template is None:
        if sys.stdin.isatty():
            # Arrow-key menu so the user picks instead of typing a name.
            template = inquirer.list_input(
                "Select a template",
                choices=templates,
                default=default,
            )
        else:
            template = default
    if template not in templates:
        raise click.ClickException(
            f"Unknown template {template!r}. " f"Available: {', '.join(templates)}."
        )

    target = Path(project_name).resolve()
    if target.exists() and any(target.iterdir()) and not force:
        raise click.ClickException(
            f"Directory '{target}' already exists and is not empty. "
            "Use --force to scaffold into it anyway."
        )

    with ExitStack() as stack:
        source = _templates_root(stack) / template
        shutil.copytree(source, target, dirs_exist_ok=True, ignore=_COPY_IGNORE)

    _rewrite_pyproject(target, project_name, description)

    click.echo(f"\n✨ Created '{project_name}' from the '{template}' template.")
    click.echo("\nNext steps:")
    click.echo(f"  cd {project_name}")
    click.echo("  uv sync")


def _rewrite_pyproject(target: Path, project_name: str, description: str | None) -> None:
    """Best-effort rewrite of name/description in a generated pyproject.toml."""
    pyproject = target / "pyproject.toml"
    if not pyproject.is_file():
        return
    lines = pyproject.read_text(encoding="utf-8").splitlines(keepends=True)
    name_done = False
    desc_done = description is None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if not name_done and stripped.startswith("name ") and "=" in line:
            lines[i] = f'{indent}name = "{project_name}"\n'
            name_done = True
        elif not desc_done and stripped.startswith("description ") and "=" in line:
            lines[i] = f'{indent}description = "{description}"\n'
            desc_done = True
        if name_done and desc_done:
            break
    pyproject.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    cli()
