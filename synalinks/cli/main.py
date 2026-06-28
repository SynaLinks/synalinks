"""Command line interface for Synalinks.

Exposes the ``synalinks`` console script (see ``[project.scripts]`` in
``pyproject.toml``). The ``init`` command scaffolds a new project from one of
the bundled templates, which ship inside the wheel so it works fully offline.
"""

import importlib.metadata
import importlib.resources
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

import click
import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Synalinks brand amber (matches the logo gradient #92400E → #D97706).
_BRAND_COLOR = "#D97706"
# Terminal stand-in for the Möbius-strip logo.
_BRAND_ICON = "∞"

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


def _synalinks_version() -> str:
    """Read the installed version without importing the (heavy) package."""
    try:
        return importlib.metadata.version("synalinks")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _print_banner() -> None:
    """Print an amber icon + wordmark + version banner."""
    console = Console()
    logo = Text()
    logo.append(f"{_BRAND_ICON} ", style=f"bold {_BRAND_COLOR}")
    logo.append("synalinks", style=f"bold {_BRAND_COLOR}")
    console.print(logo)
    console.print(Text(f"  v{_synalinks_version()}", style="dim"))


def _version_callback(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    _print_banner()
    ctx.exit()


@click.group(invoke_without_command=True)
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_version_callback,
    help="Show the version and exit.",
)
@click.pass_context
def cli(ctx: click.Context):
    """Synalinks command line interface."""
    if ctx.invoked_subcommand is None:
        _print_banner()
        click.echo()
        click.echo(ctx.get_help())

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
    _print_banner()
    click.echo()
    templates = _available_templates()
    if not templates:
        _fail(
            Text(
                "No templates are bundled with this Synalinks installation.",
                style="default",
            )
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
            _fail(
                Text.assemble(
                    ("A project name is required ", "default"),
                    ("(pass it positionally or with ", "default"),
                    ("-n", "bold cyan"),
                    (").", "default"),
                )
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
        _fail(
            Text.assemble(
                ("Unknown template ", "default"),
                (f"{template!r}", "bold yellow"),
                (".\nAvailable: ", "default"),
                (", ".join(templates), "bold cyan"),
                (".", "default"),
            )
        )

    target = Path(project_name).resolve()
    if target.exists() and any(target.iterdir()) and not force:
        _fail(
            Text.assemble(
                ("Directory ", "default"),
                (f"{target}", "bold yellow"),
                (" already exists and is not empty.", "default"),
                ("\nPass ", "default"),
                ("--force", "bold cyan"),
                (" to scaffold into it anyway.", "default"),
            )
        )

    with ExitStack() as stack:
        source = _templates_root(stack) / template
        # ``symlinks=True`` keeps CLAUDE.md as a symlink to AGENTS.md rather
        # than dereferencing it into a duplicate file.
        shutil.copytree(
            source, target, dirs_exist_ok=True, ignore=_COPY_IGNORE, symlinks=True
        )

    _rewrite_pyproject(target, project_name, description)
    _link_claude_md(target)

    console = Console()
    console.print(
        f"\n✨ Created '{project_name}' from the '{template}' template."
    )
    _print_next_steps(
        console,
        [
            f"cd {project_name}",
            "uv sync",
            "npx skills add -y SynaLinks/synalinks-skills --skill synalinks",
        ],
    )
    hint = Text("\n")
    hint.append("Then start your coding agent ", style="default")
    hint.append("(Claude Code, Cursor, Copilot, …)", style="dim")
    hint.append(" in the project folder to get started.", style="default")
    console.print(hint)


def _fail(message: Text) -> None:
    """Print a colored error to stderr and exit with a non-zero status."""
    console = Console(stderr=True)
    body = Text.assemble(("✗ ", "bold red"), ("Error  ", "bold red"))
    body.append(message)
    console.print(
        Panel(
            body,
            border_style="red",
            padding=(0, 1),
            expand=False,
        )
    )
    raise SystemExit(1)


def _print_next_steps(console: Console, commands: list[str]) -> None:
    """Render the next-step commands inside a faux terminal window."""
    body = Text()
    for i, command in enumerate(commands):
        if i:
            body.append("\n")
        # A dim prompt sign followed by the command, mimicking a shell.
        body.append("$ ", style=f"bold {_BRAND_COLOR}")
        body.append(command, style="bold white")
    # macOS-style "traffic light" dots make the panel read as a terminal.
    title = Text()
    title.append("● ", style="#FF5F56")
    title.append("● ", style="#FFBD2E")
    title.append("●", style="#27C93F")
    title.append("  Next steps", style="dim")
    console.print(
        Panel(
            body,
            title=title,
            title_align="left",
            border_style=_BRAND_COLOR,
            padding=(1, 2),
            expand=False,
        )
    )


def _link_claude_md(target: Path) -> None:
    """Make CLAUDE.md a symlink to AGENTS.md in the generated project.

    The template ships CLAUDE.md as a symlink, but ``importlib.resources``
    extracts packaged templates to a temp dir (and wheels store plain files),
    which dereferences the link — so we recreate it on the generated project.
    """
    agents = target / "AGENTS.md"
    claude = target / "CLAUDE.md"
    if not agents.is_file():
        return
    try:
        if claude.is_symlink() or claude.exists():
            claude.unlink()
        claude.symlink_to("AGENTS.md")
    except OSError:
        # Symlinks may be unavailable (e.g. Windows without privileges);
        # leave whatever was copied so CLAUDE.md still exists.
        pass


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
