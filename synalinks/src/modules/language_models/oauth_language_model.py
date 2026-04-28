# License Apache 2.0: (c) 2026 Synalinks contributors

"""OAuth :class:`LanguageModel` — claude / codex / gemini subprocess bridge.

Wraps the locally-installed interactive CLIs (``claude``, ``codex``,
``gemini``) as a drop-in :class:`LanguageModel`, for users on OAuth-only
subscriptions (Claude Max, ChatGPT Plus/Pro, Google account) who cannot
reach the model through a litellm-style API key.

In-place replacement: same constructor signature as :class:`LanguageModel`
(``model``, ``api_base``, ``timeout``, ``retry``, ``fallback``, ``caching``).
The provider is encoded in the ``model`` string as ``provider/model``,
matching the existing convention. Only ``claude``, ``codex``, ``gemini``
are accepted as providers.

.. code-block:: python

    import synalinks

    lm = synalinks.OAuthLanguageModel(model="codex/gpt-5.2")
    lm = synalinks.OAuthLanguageModel(model="claude/claude-sonnet-4-6")
    lm = synalinks.OAuthLanguageModel(model="gemini/gemini-2.0-flash")

Latency-critical decisions:

* **codex** uses ``--output-schema`` when a schema is given so the Responses
  API enforces it server-side (no parse retries on our side).
* **claude** does NOT use ``--bare`` (requires API key, incompatible with
  OAuth) nor ``--json-schema`` (measurably slower); we fall back to
  prompt-based JSON extraction.
* **gemini** runs against an isolated minimal HOME at
  ``~/.synalinks_gemini_home`` so user hooks/skills/agents do not bloat
  cold-start (drops a ~63s baseline to 8–16s). Override via
  ``SYNALINKS_GEMINI_HOME``.

``reasoning_effort`` is consumed at call time (same kwarg as the parent),
mapped to codex ``model_reasoning_effort`` and claude ``--effort``.

Cost: OAuth subscriptions are flat-rate; ``cumulated_cost`` stays at 0.0.
Streaming: not supported (subprocess buffering makes it brittle).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatRole
from synalinks.src.language_models.language_model import LanguageModel
from synalinks.src.saving.object_registration import register_synalinks_serializable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_OUTPUT_BYTES = 512 * 1024

# Env vars that must be scrubbed from the CLI subprocess environment: the CLIs
# auto-detect these and will prefer API-key auth over OAuth if present. On an
# OAuth-only subscription account this usually fails loudly (wrong auth) or
# silently (wrong workspace).
_API_KEY_VARS = {
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
}


# ---------------------------------------------------------------------------
# Gemini minimal HOME (isolated from user's ~/.gemini hooks + skills)
# ---------------------------------------------------------------------------

_GEMINI_HOME_DIRNAME = ".synalinks_gemini_home"
_GEMINI_CRED_FILES = ("oauth_creds.json", "google_accounts.json", "installation_id")
_GEMINI_MINIMAL_SETTINGS = {
    "security": {"auth": {"selectedType": "oauth-personal"}},
    "ide": {"enabled": False},
    "experimental": {"enableAgents": False},
}
_GEMINI_HOME_PATH: str | None = None


def _gemini_home_root() -> Path:
    """Default parent dir for the isolated gemini HOME (user's home)."""
    return Path(os.path.expanduser("~"))


def ensure_minimal_gemini_home() -> str:
    """Create (idempotent) an isolated HOME for the ``gemini`` CLI.

    The user's real ``~/.gemini/`` may load hooks, skills, agents, and
    ``enableAgents=true``, adding tens of seconds of cold-start overhead per
    invocation. For headless LLM calls we only need OAuth credentials.

    Returns absolute path to use as ``HOME`` when running ``gemini``.
    Override location via ``SYNALINKS_GEMINI_HOME`` env var.
    """
    global _GEMINI_HOME_PATH
    if _GEMINI_HOME_PATH is not None:
        return _GEMINI_HOME_PATH

    target = os.environ.get("SYNALINKS_GEMINI_HOME") or str(
        _gemini_home_root() / _GEMINI_HOME_DIRNAME
    )

    home = Path(target)
    gem_dir = home / ".gemini"
    gem_dir.mkdir(parents=True, exist_ok=True)

    real = Path(os.path.expanduser("~/.gemini"))
    for name in _GEMINI_CRED_FILES:
        src = real / name
        dst = gem_dir / name
        if not src.exists():
            continue
        if dst.is_symlink() or dst.exists():
            try:
                if dst.resolve() == src.resolve():
                    continue
                dst.unlink()
            except OSError:
                pass
        try:
            dst.symlink_to(src)
        except OSError:
            dst.write_bytes(src.read_bytes())

    settings = gem_dir / "settings.json"
    if not settings.exists():
        settings.write_text(json.dumps(_GEMINI_MINIMAL_SETTINGS, indent=2))

    _GEMINI_HOME_PATH = str(home)
    return _GEMINI_HOME_PATH


# ---------------------------------------------------------------------------
# Per-provider command builders
# ---------------------------------------------------------------------------


def _build_codex_cmd(
    model: str,
    schema_path: str | None,
    output_path: str,
    reasoning: str = "low",
) -> list[str]:
    # `reasoning=low` is the floor: `minimal` is rejected by the Responses API
    # whenever web_search is attached (always true for ChatGPT accounts).
    cmd: list[str] = [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "-c",
        'personality="none"',
        "-c",
        f'model_reasoning_effort="{reasoning}"',
        "-o",
        output_path,
    ]
    if model:
        cmd += ["-m", model]
    if schema_path:
        cmd += ["--output-schema", schema_path]
    cmd += ["-"]  # prompt from stdin
    return cmd


def _build_claude_cmd(
    model: str,
    effort: str = "low",
    output_format: str = "text",
) -> list[str]:
    # No `--bare`: it requires ANTHROPIC_API_KEY, incompatible with OAuth.
    cmd = [
        "claude",
        "--print",
        "--no-session-persistence",
        "--tools",
        "",
        "--disable-slash-commands",
        "--effort",
        effort,
        "--output-format",
        output_format,
    ]
    if model:
        cmd += ["--model", model]
    return cmd


def _build_gemini_cmd(model: str) -> list[str]:
    # `-m` is mandatory — omitting it triples cold-start latency.
    # `-e ""` disables extensions; `--approval-mode yolo` skips prompts.
    cmd = [
        "gemini",
        "-p",
        "",
        "-o",
        "text",
        "--approval-mode",
        "yolo",
        "-e",
        "",
    ]
    if model:
        cmd += ["-m", model]
    return cmd


# ---------------------------------------------------------------------------
# JSON extraction fallback (claude + gemini don't have --output-schema)
# ---------------------------------------------------------------------------


def _make_strict_schema(schema: dict) -> dict:
    """Return a copy of ``schema`` with strict-mode guards (codex only).

    Codex ``--output-schema`` reaches the OpenAI Responses API with
    ``strict=true``, which mandates:

    * every object node declares ``additionalProperties=false``
    * every object node lists EVERY property key in ``required``

    Synalinks-emitted schemas often omit optional fields from ``required``
    and sometimes set ``additionalProperties=true`` for open param bags.
    We fix both recursively, including inside ``$defs``, without mutating
    the input.

    Note: ``additionalProperties`` that is already a dict (schema for
    dynamic-key objects) is left untouched — only the boolean ``True`` is
    overridden to ``False``.
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if "$defs" in out and isinstance(out["$defs"], dict):
        out["$defs"] = {k: _make_strict_schema(v) for k, v in out["$defs"].items()}
    if out.get("type") == "object":
        props = out.get("properties")
        if isinstance(props, dict):
            out["properties"] = {k: _make_strict_schema(v) for k, v in props.items()}
            out["required"] = list(props.keys())
        if out.get("additionalProperties") is True:
            out["additionalProperties"] = False
        out.setdefault("additionalProperties", False)
    elif out.get("type") == "array":
        items = out.get("items")
        if isinstance(items, dict):
            out["items"] = _make_strict_schema(items)
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(out.get(key), list):
            out[key] = [_make_strict_schema(s) for s in out[key]]
    return out


def _extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON dict extraction from CLI text output."""
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    balanced = _extract_balanced_json(text)
    if balanced:
        try:
            parsed = json.loads(balanced)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def _format_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(x.get("text", "") if isinstance(x, dict) else x) for x in content
            )
        content = str(content).strip()
        if content:
            lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Core async call
# ---------------------------------------------------------------------------


async def ask_llm_via_cli(
    provider: str,
    prompt: str,
    model: str = "",
    timeout: int = 180,
    schema: dict | None = None,
    reasoning: str = "medium",
    effort: str = "low",
) -> tuple[str, dict | None]:
    """Run one CLI model call.

    Returns ``(raw_text, parsed_dict_or_None)``:

    * ``raw_text`` is the stdout/file content of the CLI call.
    * ``parsed_dict`` is the extracted JSON if ``schema`` was provided,
      else None.

    On error, ``raw_text`` starts with the error glyph and ``parsed_dict``
    is ``None``.
    """
    provider = provider.strip().lower()
    tmp_files: list[str] = []

    try:
        if provider == "codex":
            schema_path: str | None = None
            if schema:
                strict = _make_strict_schema(schema)
                f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
                f.write(json.dumps(strict))
                f.close()
                schema_path = f.name
                tmp_files.append(schema_path)
            out_file = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
            out_file.close()
            tmp_files.append(out_file.name)
            cmd = _build_codex_cmd(model, schema_path, out_file.name, reasoning)

        elif provider == "claude":
            cmd = _build_claude_cmd(model, effort=effort)
            if schema:
                prompt = (
                    "Return ONLY one valid JSON object matching this schema. "
                    "No markdown fences, no explanations.\n\n"
                    f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
                    f"Conversation:\n{prompt}\n"
                )

        elif provider == "gemini":
            ensure_minimal_gemini_home()
            cmd = _build_gemini_cmd(model)
            if schema:
                prompt = (
                    "Return ONLY one valid JSON object matching this schema. "
                    "No markdown fences, no explanations.\n\n"
                    f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
                    f"Conversation:\n{prompt}\n"
                )
        else:
            return (
                f"❌ Unknown provider '{provider}' (expected: claude|codex|gemini)",
                None,
            )

        env = {k: v for k, v in os.environ.items() if k not in _API_KEY_VARS}
        if provider == "gemini":
            env["HOME"] = ensure_minimal_gemini_home()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError:
            return f"❌ CLI '{cmd[0]}' not found — check PATH", None

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=float(timeout),
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.communicate()
            except Exception:
                pass
            return f"❌ CLI '{cmd[0]}' timed out after {timeout}s", None
        except asyncio.CancelledError:
            try:
                proc.kill()
                await proc.communicate()
            except Exception:
                pass
            raise

        # Read codex output from file (cleaner than stdout)
        if provider == "codex":
            try:
                raw = (
                    Path(out_file.name)
                    .read_text(encoding="utf-8", errors="replace")
                    .strip()
                )
            except Exception:
                raw = ""
        else:
            raw = stdout_b[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace").strip()
            # claude --output-format json wraps the response; peel it
            if provider == "claude" and raw.startswith('{"type":"result"'):
                try:
                    j = json.loads(raw)
                    raw = j.get("result", raw)
                except Exception:
                    pass

        if proc.returncode != 0:
            err = stderr_b[:500].decode("utf-8", errors="replace").strip()
            detail = err or raw[:300] or "(no output)"
            return f"❌ CLI '{cmd[0]}' exited {proc.returncode}: {detail}", None

        if not raw:
            return "❌ Empty output from CLI", None

        parsed = _extract_json(raw) if schema else None
        return raw, parsed

    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Synalinks LanguageModel adapter
# ---------------------------------------------------------------------------


_VALID_PROVIDERS = ("claude", "codex", "gemini")


@synalinks_export(
    [
        "synalinks.OAuthLanguageModel",
        "synalinks.language_models.OAuthLanguageModel",
    ]
)
@register_synalinks_serializable(package="synalinks")
class OAuthLanguageModel(LanguageModel):
    """Drop-in :class:`LanguageModel` backed by a local OAuth CLI subprocess.

    In-place replacement for :class:`LanguageModel` when the user authenticates
    via an OAuth subscription (Claude Max, ChatGPT Plus/Pro, Google account)
    rather than an API key. The provider is encoded in the ``model`` string
    as ``provider/model``, matching the existing convention. Only ``claude``,
    ``codex``, and ``gemini`` are accepted as providers.

    Reasoning effort is consumed at call time as the standard
    ``reasoning_effort`` kwarg (``low``|``medium``|``high``), mapped to codex
    ``model_reasoning_effort`` and claude ``--effort``. Defaults to ``low``.

    Caveats:

    * No streaming (subprocess stdout buffering makes token-by-token streaming
      brittle — ``streaming=True`` raises explicitly).
    * Per-call cost is reported as ``0.0`` (OAuth subscriptions are flat-rate;
      we deliberately do not fake a number).
    * Each call forks a subprocess — for high-QPS workloads the litellm-backed
      :class:`LanguageModel` is preferable.

    Args:
        model (str): ``provider/model`` string, e.g. ``codex/gpt-5.2``,
            ``claude/claude-sonnet-4-6``, ``gemini/gemini-2.0-flash``.
            Provider must be one of ``claude``, ``codex``, ``gemini``.
        api_base (str): Accepted for signature compatibility with
            :class:`LanguageModel`; not used by the OAuth CLI path.
        timeout (int): Subprocess timeout in seconds.
        retry (int): Number of attempts on subprocess/parse failure.
        fallback (LanguageModel): Optional fallback delegated to after retries
            are exhausted.
        caching (bool): Forwarded to the parent :class:`LanguageModel`.
    """

    def __init__(
        self,
        model=None,
        api_base=None,
        timeout=600,
        retry=5,
        fallback=None,
        caching=False,
    ):
        if model is None:
            raise ValueError("You need to set the `model` argument for any LanguageModel")
        if "/" not in model:
            raise ValueError(
                f"OAuthLanguageModel expects model='<provider>/<model>', got {model!r}"
            )
        provider, _, cli_model = model.partition("/")
        provider = provider.strip().lower()
        if provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"OAuthLanguageModel provider must be one of "
                f"{_VALID_PROVIDERS}, got {provider!r} (from model={model!r})"
            )
        self.provider = provider
        self.cli_model = cli_model.strip()
        super().__init__(
            model=model,
            api_base=api_base,
            timeout=timeout,
            retry=max(1, retry),
            fallback=fallback,
            caching=caching,
        )

    def _obj_type(self):
        return "OAuthLanguageModel"

    async def __call__(self, messages, schema=None, streaming=False, **kwargs):
        if streaming:
            raise ValueError("OAuthLanguageModel does not support streaming")

        reasoning_effort = kwargs.pop("reasoning_effort", "low")
        if reasoning_effort in ("none", "disable"):
            reasoning_effort = "low"

        prompt = _format_messages(messages.get_json().get("messages", []))
        prompt_tokens = len(prompt) // 4
        last_error = ""

        for _ in range(self.retry):
            t0 = time.time()
            raw, parsed = await ask_llm_via_cli(
                provider=self.provider,
                prompt=prompt,
                model=self.cli_model,
                timeout=self.timeout,
                schema=schema,
                reasoning=reasoning_effort,
                effort=reasoning_effort,
            )
            dt = time.time() - t0

            if raw.startswith("❌"):
                last_error = raw
                logger.warning("[oauth-lm] %s failed (%.1fs): %s", self.model, dt, raw)
                continue

            if schema:
                if parsed is not None:
                    return parsed
                last_error = f"Could not parse JSON: {raw[:200]}"
                logger.warning("[oauth-lm] %s unparseable (%.1fs)", self.model, dt)
                continue

            return {
                "role": ChatRole.ASSISTANT,
                "content": raw,
                "tool_call_id": None,
                "tool_calls": [],
                "created_at": int(t0),
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(raw) // 4,
                    "total_tokens": prompt_tokens + len(raw) // 4,
                },
            }

        raise RuntimeError(
            f"OAuthLanguageModel({self.model}) failed after {self.retry} retries: "
            f"{last_error or 'unknown'}"
        )
