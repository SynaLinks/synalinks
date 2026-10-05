# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import contextvars
import copy
import json
import os
import tempfile
import warnings

from synalinks.src.api_export import synalinks_export
from synalinks.src.modules.language_models.language_model import DeterministicStopError
from synalinks.src.modules.language_models.language_model import LanguageModel
from synalinks.src.saving.object_registration import register_synalinks_serializable

# The providers reachable through a locally authenticated CLI, i.e. the
# values accepted before the `/` in `model`.
OAUTH_PROVIDERS = ("codex",)

# The only environment variables handed to the CLI subprocess. An allowlist
# rather than a denylist: the parent process usually holds unrelated secrets
# (API keys, tokens) that a prompt-injected model must never be able to read,
# and an `OPENAI_API_KEY` in the environment would make the CLI bill the API
# account instead of the subscription.
_ENV_ALLOWLIST = (
    "HOME",
    "PATH",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "CODEX_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    # Windows
    "APPDATA",
    "LOCALAPPDATA",
    "USERPROFILE",
    "SYSTEMROOT",
)

# Codex features that give the model a way to act (shell, web, apps, plugins,
# sub-agents...). They are switched off through `-c features.<name>=false`
# rather than `--disable <name>`: the latter aborts on a name the installed
# CLI does not know, the former only emits a warning, so the list survives
# CLI upgrades. `_ALLOWED_ITEM_TYPES` below is what actually enforces that no
# tool ran.
_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "plugins",
    "remote_plugin",
    "multi_agent",
    "skill_search",
    "tool_suggest",
    "view_image",
    "sleep_tool",
    "goals",
    "image_generation",
    "browser_use",
    "computer_use",
    "in_app_browser",
    "hooks",
    "memories",
)

# The JSONL item types a plain completion may produce. Anything else (a
# command execution, a web search, a file change, an MCP tool call...) means
# the model acted instead of answering: the call is failed.
_ALLOWED_ITEM_TYPES = frozenset({"agent_message", "reasoning", "error"})

# `reasoning_effort` values Synalinks accepts that the Codex CLI does not.
# "none" sends nothing (model default); "disable"/"minimal" fall back to the
# lowest effort the ChatGPT backend accepts.
_EFFORT_ALIASES = {"disable": "low", "minimal": "low"}

# Request parameters with no Codex CLI equivalent, dropped with a warning.
_IGNORED_PARAMS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "api_base",
)

_MAX_ERROR_CHARS = 500

# The reasoning effort of the call running in the current task. A ContextVar
# rather than a request parameter: the parameters of a call are handed
# unchanged to the `fallback`, which must not receive anything private to
# this class, and concurrent calls on one instance never see each other's.
_REASONING_EFFORT = contextvars.ContextVar("synalinks_oauth_reasoning_effort")


def _strict_schema(schema):
    """Returns a copy of `schema` valid for OpenAI strict structured output.

    The ChatGPT backend only accepts strict JSON schemas: every object must
    list all its properties in `required` and set `additionalProperties` to
    `False`. Pydantic leaves the fields that have a default out of
    `required`, so the schema is rewritten recursively (`$defs`, `items`,
    `anyOf`...) without mutating the input.

    Raises:
        ValueError: If the schema contains a free-form object (one without
            `properties`, or accepting additional properties), which strict
            mode cannot express.
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    for key in ("properties", "$defs", "definitions"):
        if isinstance(out.get(key), dict):
            out[key] = {k: _strict_schema(v) for k, v in out[key].items()}
    for key in ("items", "not"):
        if isinstance(out.get(key), dict):
            out[key] = _strict_schema(out[key])
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if isinstance(out.get(key), list):
            out[key] = [_strict_schema(item) for item in out[key]]
    properties = out.get("properties")
    if out.get("type") == "object" or isinstance(properties, dict):
        if not isinstance(properties, dict) or out.get("additionalProperties"):
            raise ValueError(
                "`OAuthLanguageModel` enforces the output schema in strict mode, "
                "which cannot express a free-form object (no `properties`, or "
                "`additionalProperties` enabled). Declare every field of the "
                f"data model explicitly. Received: {schema}"
            )
        out["required"] = list(properties.keys())
        out["additionalProperties"] = False
    return out


def _message_text(message):
    """Returns the text of a wire-format chat message (non-text parts dropped)."""
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
                else:
                    warnings.warn(
                        "`OAuthLanguageModel` only sends text: a "
                        f"`{part.get('type')}` content part was dropped."
                    )
            else:
                parts.append(str(part))
        content = "\n".join(parts)
    return str(content or "").strip()


def _split_messages(formatted_messages):
    """Splits wire messages into `(instructions, prompt)`.

    The system messages become the instructions of the model (they replace
    the coding-agent prompt the CLI ships with). A lone user message is sent
    verbatim; a longer conversation is rendered as a role-tagged transcript,
    since the CLI takes a single prompt.
    """
    instructions = []
    turns = []
    for message in formatted_messages:
        text = _message_text(message)
        if not text:
            continue
        role = str(message.get("role") or "user")
        if role in ("system", "developer"):
            instructions.append(text)
        else:
            turns.append((role, text))
    if len(turns) == 1 and turns[0][0] == "user":
        prompt = turns[0][1]
    else:
        prompt = "\n\n".join(f"[{role.upper()}]\n{text}" for role, text in turns)
    return "\n\n".join(instructions), prompt


def _build_codex_command(
    model,
    workdir,
    instructions_path=None,
    schema_path=None,
    reasoning_effort=None,
):
    """Builds the `codex exec` command line of one completion.

    The CLI is an agent harness; every flag below strips it down to a plain
    completion endpoint:

    - `--ephemeral`: no session file written to disk.
    - `--ignore-user-config` / `--ignore-rules`: the user's `config.toml`
        (MCP servers, profiles, hooks...) and exec rules are not loaded; the
        login is still read from `CODEX_HOME`.
    - `-s read-only` + `-C <empty dir>`: nothing to read, nothing writable.
    - `web_search="disabled"` and `features.*=false`: no tool is offered.
    - `project_doc_max_bytes=0`: no `AGENTS.md` is injected.
    - `model_instructions_file`: the system message replaces the built-in
        coding-agent prompt.
    """
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--json",
        "--color",
        "never",
        "-s",
        "read-only",
        "-C",
        workdir,
        "-c",
        'web_search="disabled"',
        "-c",
        "project_doc_max_bytes=0",
    ]
    for feature in _DISABLED_FEATURES:
        command += ["-c", f"features.{feature}=false"]
    if instructions_path:
        command += ["-c", f"model_instructions_file={json.dumps(instructions_path)}"]
    if reasoning_effort:
        command += ["-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}"]
    if model:
        command += ["-m", model]
    if schema_path:
        command += ["--output-schema", schema_path]
    # The prompt is read from stdin.
    command.append("-")
    return command


def _parse_codex_events(stdout):
    """Parses the JSONL event stream of `codex exec --json`.

    Returns:
        (tuple): `(content, usage, errors, forbidden)` where `content` is the
            last agent message (None if there is none), `usage` the token
            usage dict of the turn, `errors` the list of error messages and
            `forbidden` the sorted list of item types that reveal a tool ran.
    """
    content = None
    usage = {}
    errors = []
    forbidden = set()
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        event_type = event.get("type")
        item = event.get("item")
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type not in _ALLOWED_ITEM_TYPES:
                forbidden.add(str(item_type))
            elif item_type == "agent_message" and event_type == "item.completed":
                content = item.get("text")
            # An `error` item is a non-fatal notice (e.g. an unknown feature
            # flag): the turn carries on, so it is not collected.
        elif event_type == "turn.completed":
            usage = event.get("usage") or {}
        elif event_type == "error":
            errors.append(str(event.get("message")))
        elif event_type == "turn.failed":
            errors.append(str((event.get("error") or {}).get("message")))
    return content, usage, errors, sorted(forbidden)


def _to_model_response(content, usage):
    """Shapes a Codex turn like a LiteLLM chat completion response."""
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {
                "cached_tokens": int(usage.get("cached_input_tokens") or 0),
            },
            "completion_tokens_details": {
                "reasoning_tokens": int(usage.get("reasoning_output_tokens") or 0),
            },
        },
    }


@synalinks_export(
    [
        "synalinks.OAuthLanguageModel",
        "synalinks.language_models.OAuthLanguageModel",
    ]
)
@register_synalinks_serializable(package="synalinks")
class OAuthLanguageModel(LanguageModel):
    """A language model reached through a subscription login instead of an API key.

    Flat-rate subscriptions (ChatGPT Plus/Pro/Business) authenticate with
    OAuth and come without an API key, so they cannot be used through
    LiteLLM. This class is a drop-in replacement for `LanguageModel` that
    sends each request through the vendor's own CLI, which holds the login.

    Only the Codex CLI (`codex/` prefix) is supported: it is the only one
    that enforces a JSON schema server-side, which Synalinks relies on for
    constrained structured output.

    Install the CLI and sign in once (interactive, opens a browser):

    ```bash
    npm install -g @openai/codex
    codex login
    ```

    Then use it like any other language model:

    ```python
    import synalinks

    language_model = synalinks.OAuthLanguageModel(
        model="codex/gpt-5.5",
        reasoning_effort="low",
    )
    ```

    `synalinks.language_models.get("codex/gpt-5.5")`, and therefore
    `language_model="codex/gpt-5.5"` on any module, returns this class too.
    Use `"codex/default"` to let the CLI pick the default model of the
    account, and `codex debug models` to list the available ones.

    **How it works**

    Only the transport differs from `LanguageModel`: message formatting,
    cache, retry, `fallback`, usage counters and response parsing are
    inherited. Each call spawns `codex exec` stripped down to a plain
    completion endpoint:

    - The output schema is passed with `--output-schema` and enforced by the
        backend in strict mode: every field is required and returned.
    - The system message replaces the built-in coding-agent prompt.
    - The model gets no tool: the shell, web search, apps and plugins are
        disabled, the sandbox is read-only and the working directory is an
        empty temporary directory. If a tool runs anyway, the call fails.
    - The subprocess only inherits an allowlist of environment variables,
        so the API keys of the parent process are not exposed to it.

    **Limitations**

    - No tool calling and no streaming: `tools`, `tool_schemas` and
        `streaming=True` raise a `ValueError`. Use an API-key provider for
        the tool-calling generator of an agent.
    - Text only: images and audio parts are dropped.
    - No sampling control: the CLI exposes neither `temperature` nor
        `max_tokens`, they are ignored with a warning. Do not rely on this
        class where a deterministic `temperature=0.0` judge is required.
    - A free-form `dict` field cannot be expressed in strict mode and
        raises a `ValueError`.
    - The cost is not tracked (flat-rate subscription); token usage is.
        Each call carries a few thousand tokens of harness overhead and
        counts against the usage limits of the subscription.

    `reasoning_effort` is supported (`"low"`, `"medium"`, `"high"`...);
    `"none"` keeps the default of the model.

    Args:
        model (str): The model to use, as `codex/<model>` (e.g.
            `codex/gpt-5.5`), or `codex/default` for the account default.
        api_base (str): Unused, accepted for compatibility with
            `LanguageModel`.
        timeout (int): The timeout in seconds of one call, after which the
            subprocess is killed (Default to 600).
        retry (int): The number of attempts per call (Default to 5).
        retry_max_wait (int): The maximum wait in seconds between two
            attempts (Default to 60).
        fallback (LanguageModel): The language model to use if every
            attempt failed (Optional).
        caching (bool): Unused, accepted for compatibility with
            `LanguageModel`. Use `cache_dir` instead.
        cache_dir (str): The directory of the persistent response cache
            (Optional).
        name (str): The name of the language model (Optional).
        description (str): The description of the language model (Optional).
        hooks (list): The hooks of the language model (Optional).
        **default_kwargs (keyword arguments): The default parameters of
            every call (e.g. `reasoning_effort`).
    """

    def __init__(self, model=None, **kwargs):
        if model is None:
            raise ValueError("You need to set the `model` argument for any LanguageModel")
        provider, separator, cli_model = model.partition("/")
        if not separator or provider not in OAUTH_PROVIDERS or not cli_model:
            raise ValueError(
                "`OAuthLanguageModel` expects `model` as '<provider>/<model>' with "
                f"a provider among {list(OAUTH_PROVIDERS)} (e.g. 'codex/gpt-5.5'). "
                f"Received: model={model!r}"
            )
        super().__init__(model=model, **kwargs)

    async def call(
        self,
        messages,
        schema=None,
        tools=None,
        tool_schemas=None,
        streaming=False,
        **kwargs,
    ):
        """
        Call method to generate a response using the language model.

        Args:
            messages (dict): A formatted dict of chat messages.
            schema (dict): The target JSON schema for structed output (optional).
                If None, output a ChatMessage-like answer.
            tools (list | dict): Not supported, must be empty.
            tool_schemas (list): Not supported, must be empty.
            streaming (bool): Not supported, must be False.
            **kwargs (keyword arguments): The additional keywords arguments
                forwarded to the LM call.
        Returns:
            (dict): The generated structured response.
        """
        if tools or tool_schemas:
            raise ValueError(
                "`OAuthLanguageModel` does not support tool calling: the CLI is "
                "used as a plain completion endpoint. Use a `LanguageModel` for "
                "the modules that pass `tools`."
            )
        if streaming:
            raise ValueError("`OAuthLanguageModel` does not support streaming.")
        if schema:
            schema = _strict_schema(schema)
        # The base class drops `reasoning_effort` for the models LiteLLM does
        # not know to reason (see `supports_reasoning`); it is read here and
        # handed to the transport out of band, leaving `kwargs` untouched for
        # the `fallback`.
        reasoning_effort = kwargs.get(
            "reasoning_effort", self.default_kwargs.get("reasoning_effort", "none")
        )
        token = _REASONING_EFFORT.set(
            None
            if not reasoning_effort or reasoning_effort == "none"
            else _EFFORT_ALIASES.get(reasoning_effort, reasoning_effort)
        )
        try:
            return await super().call(
                messages,
                schema=schema,
                streaming=False,
                **kwargs,
            )
        finally:
            _REASONING_EFFORT.reset(token)

    async def _acompletion(self, formatted_messages, **kwargs):
        kwargs = copy.copy(kwargs)
        reasoning_effort = _REASONING_EFFORT.get(None)
        response_format = kwargs.pop("response_format", None)
        schema = None
        if response_format:
            schema = (response_format.get("json_schema") or {}).get("schema")
        ignored = sorted(k for k in kwargs if k in _IGNORED_PARAMS)
        if ignored:
            warnings.warn(
                f"{self} ignores the parameters {ignored}: the Codex CLI does not "
                "expose them."
            )
        instructions, prompt = _split_messages(formatted_messages)
        cli_model = self.model.split("/", 1)[1]
        if cli_model == "default":
            cli_model = None

        with tempfile.TemporaryDirectory(prefix="synalinks_oauth_") as tmp:
            # The model runs in an empty directory; the request files live
            # next to it, out of its working root.
            workdir = os.path.join(tmp, "workdir")
            os.mkdir(workdir)
            instructions_path = None
            if instructions:
                instructions_path = os.path.join(tmp, "instructions.md")
                with open(instructions_path, "w", encoding="utf-8") as f:
                    f.write(instructions)
            schema_path = None
            if schema:
                schema_path = os.path.join(tmp, "schema.json")
                with open(schema_path, "w", encoding="utf-8") as f:
                    json.dump(schema, f)
            command = _build_codex_command(
                cli_model,
                workdir,
                instructions_path=instructions_path,
                schema_path=schema_path,
                reasoning_effort=reasoning_effort,
            )
            stdout, stderr, returncode = await self._run(command, prompt, cwd=workdir)

        content, usage, errors, forbidden = _parse_codex_events(stdout)
        if forbidden:
            raise DeterministicStopError(
                f"The model used tools ({', '.join(forbidden)}) instead of "
                "answering; the response is discarded.",
                "tool_use",
            )
        if returncode != 0 or errors:
            # The same failure is reported as an `error` and a `turn.failed`.
            detail = "; ".join(dict.fromkeys(errors)) or stderr.strip() or stdout.strip()
            message = (
                f"`{command[0]}` failed (exit code {returncode}): "
                f"{detail[-_MAX_ERROR_CHARS:] or '(no output)'}"
            )
            if "invalid_request_error" in detail:
                # A rejected request (unknown model, invalid schema...) fails
                # identically on every attempt.
                raise DeterministicStopError(message, "invalid_request")
            raise RuntimeError(message)
        return _to_model_response(content, usage)

    async def _run(self, command, prompt, cwd=None):
        """Runs `command` with `prompt` on stdin.

        Returns:
            (tuple): `(stdout, stderr, returncode)`.
        """
        env = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )
        except FileNotFoundError:
            raise DeterministicStopError(
                f"The `{command[0]}` CLI was not found in the PATH. Install it "
                f"and sign in with `{command[0]} login` to use {self}.",
                "cli_not_found",
            )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode("utf-8")),
                timeout=self.timeout,
            )
        except BaseException as e:
            # Timeout or cancellation: never leave the subprocess behind.
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass
            if isinstance(e, asyncio.TimeoutError):
                raise RuntimeError(
                    f"`{command[0]}` timed out after {self.timeout}s."
                ) from e
            raise
        return (
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            process.returncode,
        )

    @classmethod
    def supported_providers(cls):
        """Returns the provider prefixes this class accepts.

        Returns:
            (list): The sorted list of supported provider prefixes.
        """
        return list(OAUTH_PROVIDERS)

    def supports_reasoning(self):
        """Whether the base class forwards `reasoning_effort` (always False).

        The effort is handed to the CLI by this class instead, and the
        `thinking` field of a schema, if any, is generated like any other.
        """
        return False

    def supports_vision(self):
        """Whether the model accepts images in its input (always False)."""
        return False

    def supports_audio(self):
        """Whether the model accepts audio in its input (always False)."""
        return False

    def _obj_type(self):
        return "OAuthLanguageModel"

    def __repr__(self):
        return f"<OAuthLanguageModel model={self.model}>"
