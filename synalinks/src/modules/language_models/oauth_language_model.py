# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import json
import os
import shutil
import tempfile
import warnings

from synalinks.src.api_export import synalinks_export
from synalinks.src.modules.language_models.language_model import DeterministicStopError
from synalinks.src.modules.language_models.language_model import LanguageModel
from synalinks.src.saving.object_registration import register_synalinks_serializable
from synalinks.src.utils import model_catalog
from synalinks.src.version import __version__

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
# sub-agents...).
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

# Everything the harness would add to a request on its own, switched off with
# `-c <key>=<value>`: the tools above and the text it injects around the
# prompt (sandbox rules, collaboration mode, working directory and date,
# skills, `AGENTS.md`). `-c features.<name>=false` is used rather than
# `--disable <name>`: the latter aborts on a name the installed CLI does not
# know, while an unknown key only emits a notice, so the list survives CLI
# upgrades.
_CONFIG_OVERRIDES = (
    'web_search="disabled"',
    "project_doc_max_bytes=0",
    "include_permissions_instructions=false",
    "include_collaboration_mode_instructions=false",
    "include_environment_context=false",
    "skills.include_instructions=false",
    # The analytics upload holds the process about 0.4 s after the turn has
    # completed (measured with codex-cli 0.162: 0.37 to 0.41 s with it,
    # 0.01 s without), about a tenth of a short completion.
    "analytics.enabled=false",
    *(f"features.{name}=false" for name in _DISABLED_FEATURES),
)

# What the model catalog of the CLI gives a model whatever the configuration
# says: the "code mode" JavaScript tool (the only way the gpt-6 models call
# tools), sub-agents, the patch tool and the experimental tools. With code
# mode on, gpt-6.1-sol spends two requests listing its tools before it
# answers (measured with codex-cli 0.162), and none of it shows in the JSONL
# stream.
_CATALOG_TOOL_FIELDS = {
    "tool_mode": None,
    "multi_agent_version": None,
    "apply_patch_tool_type": None,
    "experimental_supported_tools": [],
}

# The JSONL item types a plain completion may produce. Anything else (a
# command execution, a web search, a file change, an MCP tool call...) means
# the model acted instead of answering: the call is failed. This is a second
# line of defence: what runs through code mode is not reported as an item,
# which is why the tools are taken out of the catalog in the first place.
_ALLOWED_ITEM_TYPES = frozenset({"agent_message", "reasoning", "error"})

# `reasoning_effort` values Synalinks accepts that the ChatGPT backend does not
# take on every model. Measured with codex-cli 0.162: gpt-6.1-sol accepts
# low/medium/high/xhigh/max and rejects "none" and "minimal" (gpt-5.5 still
# took "none"), so "low" is the lowest effort every listed model accepts.
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

# The `model_catalog.PROVIDERS`-like entry of `codex/`: the login lives in
# `CODEX_HOME` (default `~/.codex`) and the models are listed by the CLI.
_CODEX_PROVIDER = {
    "env": [model_catalog.env_var("CODEX_HOME", required=False, secret=False)],
    "server": "codex",
}

_INSTALL_HINT = (
    "Install it with `pip install 'synalinks[codex]'` (or `npm install -g "
    "@openai/codex`), then sign in once with `codex login`."
)

# The system message of a completion that has none: without one, the CLI
# would send its own coding-agent prompt (about 3,500 tokens) instead.
_DEFAULT_INSTRUCTIONS = "Answer the user's message."

# A turn that declares tools, or whose history holds tool calls, goes through
# `codex app-server` instead of `codex exec` (see `_AppServer`). The protocol
# has no `--ignore-user-config`: these keys neutralise what the user's
# `config.toml` could add to a request on top of `_CONFIG_OVERRIDES`. The
# `mcp_servers` it declares are disabled per thread, by name (`config/read`).
_APP_SERVER_OVERRIDES = (
    'model_provider="openai"',
    "notify=[]",
)

# Whatever the catalog says, the harness declares one function to the model,
# `request_user_input` ("only available in Plan mode"), in the
# `additional_tools` item of the request. The gpt-6 models call it, the CLI
# refuses it, and each attempt costs a request that is not reported. This key
# takes it out (captured with codex-cli 0.164 on a local provider), but only a
# turn that declares tools gets it: when the request declares no function at
# all, the CLI sends the instructions twice (measured live: 365 -> 105 input
# tokens on a short system message, 2,435 -> 4,245 on a 2,100-token one).
_NO_HARNESS_FUNCTION = "tools.experimental_request_user_input.enabled=false"

# What the client says it is in the `initialize` handshake.
_APP_SERVER_CLIENT = {"name": "synalinks", "version": __version__}

# The thread items a completion with tools may produce: the user message, the
# answer, the reasoning, and the calls to the tools the request declared.
# Anything else means the model acted through the CLI: the call is failed.
# How long a turn is left running after the model's first tool call before
# it is interrupted, so that the response's stream ends and its usage is
# reported (see `_run_turn`).
_STREAM_DRAIN_S = 1.0

_ALLOWED_APP_SERVER_ITEMS = frozenset(
    {"userMessage", "agentMessage", "reasoning", "dynamicToolCall"}
)


def _strict_schema(schema):
    """Returns a copy of `schema` valid for OpenAI strict structured output.

    The ChatGPT backend only accepts strict JSON schemas: every object must
    list all its properties in `required` and set `additionalProperties` to
    `False`. Pydantic leaves the fields that have a default out of
    `required`, so the schema is rewritten recursively (`$defs`, `items`,
    `anyOf`...) without mutating the input. A `$ref` is also left alone in
    its node (see `_bare_refs`).

    Raises:
        ValueError: If the schema contains a free-form object (one without
            `properties`, or accepting additional properties), which strict
            mode cannot express.
    """
    if isinstance(schema, dict):
        schema = _bare_refs(schema, schema.get("$defs") or {})
    return _close_objects(schema)


def _bare_refs(node, defs, inlining=()):
    """Returns a copy of `node` where no `$ref` has a sibling keyword.

    Pydantic writes `{"$ref": ..., "default": ...}` for a nested data model
    with a default, and `{"$ref": ..., "description": ...}` for a described
    one. About half of the ChatGPT backend's calls reject both ("$ref cannot
    have keywords"), measured with gpt-6.1-sol. The `default` is dropped (every
    field is required in strict mode); a `description` is kept by inlining
    the referenced definition, unless it is recursive.
    """
    if isinstance(node, list):
        return [_bare_refs(item, defs, inlining) for item in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str) and len(node) > 1:
        name = ref.rsplit("/", 1)[-1]
        if "description" in node and name in defs and name not in inlining:
            target = _bare_refs(defs[name], defs, inlining + (name,))
            return {**target, "description": node["description"]}
        return {"$ref": ref}
    return {key: _bare_refs(value, defs, inlining) for key, value in node.items()}


def _close_objects(schema):
    """Closes every object of `schema` and requires all its properties."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    for key in ("properties", "$defs", "definitions"):
        if isinstance(out.get(key), dict):
            out[key] = {k: _close_objects(v) for k, v in out[key].items()}
    for key in ("items", "not"):
        if isinstance(out.get(key), dict):
            out[key] = _close_objects(out[key])
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if isinstance(out.get(key), list):
            out[key] = [_close_objects(item) for item in out[key]]
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


def _without_tools(catalog):
    """Returns a copy of the CLI's model catalog where no model is given tools.

    Only the fields an entry already has are overridden, so that an entry of
    another catalog version is left as the CLI wrote it.
    """
    return {
        **catalog,
        "models": [
            {**model, **{k: v for k, v in _CATALOG_TOOL_FIELDS.items() if k in model}}
            for model in catalog["models"]
        ],
    }


def _codex_executable():
    """Returns the Codex CLI to run: the one of `synalinks[codex]`, else `codex`.

    Raises:
        ImportError: If neither is installed.
    """
    try:
        import codex_cli_bin

        return str(codex_cli_bin.bundled_codex_path())
    except ImportError:
        pass
    executable = shutil.which("codex")
    if executable is None:
        raise ImportError(f"`OAuthLanguageModel` needs the Codex CLI. {_INSTALL_HINT}")
    return executable


def _write(directory, name, text):
    """Writes `text` to `name` in `directory`.

    Returns:
        (str): The path of the file, or None if there is nothing to write.
    """
    if not text:
        return None
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


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
    catalog_path=None,
    reasoning_effort=None,
    executable="codex",
):
    """Builds the `codex exec` command line of one completion.

    The CLI is an agent harness; every flag below strips it down to a plain
    completion endpoint, whose request holds the system message, the prompt
    and the schema, and nothing else:

    - `--ephemeral`: no session file written to disk.
    - `--ignore-user-config` / `--ignore-rules`: the user's `config.toml`
        (MCP servers, profiles, hooks...) and exec rules are not loaded; the
        login is still read from `CODEX_HOME`.
    - `-s read-only` + `-C <empty dir>`: nothing to read, nothing writable.
    - `_CONFIG_OVERRIDES`: no tool is offered and no text is injected.
    - `model_catalog_json`: the catalog of the CLI without the tools it
        gives a model (see `_CATALOG_TOOL_FIELDS`).
    - `model_instructions_file`: the system message replaces the built-in
        coding-agent prompt.
    """
    command = [
        executable,
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
    ]
    for override in _CONFIG_OVERRIDES:
        command += ["-c", override]
    if catalog_path:
        command += ["-c", f"model_catalog_json={json.dumps(catalog_path)}"]
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


def _to_model_response(content, usage, *, tool_calls=None):
    """Shapes a Codex turn like a LiteLLM chat completion response.

    `tool_calls`, in the OpenAI wire shape, are set on the message as a
    provider sets native ones, with the `tool_calls` finish reason.
    """
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": message,
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


# Tool calling, native, through `codex app-server`: the tools of a request are
# declared to the thread (`dynamicTools`), the earlier turns are appended to
# its history as the Responses API items they are (`thread/inject_items`), and
# the model's calls come back as `item/tool/call` requests. Synalinks keeps
# the loop: the turn is interrupted at the first call, which is returned to
# the caller as a native tool call; the CLI runs nothing. One process per
# completion, like `codex exec`.


def _tool_function(tool):
    """Returns the `function` of a wire tool declaration."""
    if not isinstance(tool, dict):
        raise ValueError(f"A tool declaration must be a dict. Received: {tool!r}")
    function = tool.get("function") if "function" in tool else tool
    if not isinstance(function, dict) or not function.get("name"):
        raise ValueError(f"A tool declaration has no `name`. Received: {tool!r}")
    return function


def _tool_choice_mode(tool_choice):
    """Normalizes `tool_choice` into `(mode, tool name)`.

    `mode` is `"auto"` (None or `"auto"`), `"none"`, `"required"` (also
    Anthropic's `"any"`) or `"named"` (a dict naming one tool, in the OpenAI
    `{"type": "function", "function": {"name": ...}}` or the
    `{"type": "tool", "name": ...}` shape).

    Raises:
        ValueError: If `tool_choice` is none of these.
    """
    if tool_choice is None or tool_choice == "auto":
        return "auto", None
    if tool_choice == "none":
        return "none", None
    if tool_choice in ("required", "any"):
        return "required", None
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
        if name:
            return "named", name
    raise ValueError(
        "`tool_choice` must be None, 'auto', 'none', 'required' or a dict naming "
        f"a tool. Received: {tool_choice!r}"
    )


def _offered_tools(tools, tool_choice=None):
    """The `function`s of the wire tools the model may call under `tool_choice`.

    Raises:
        ValueError: If a tool is malformed, or `tool_choice` names a tool
            that is not declared.
    """
    functions = [_tool_function(tool) for tool in tools or []]
    mode, name = _tool_choice_mode(tool_choice)
    if mode == "none":
        return []
    if mode == "named":
        functions = [f for f in functions if f["name"] == name]
        if not functions:
            raise ValueError(
                f"`tool_choice` names `{name}`, which is not a declared tool."
            )
    return functions


def _dynamic_tools(functions):
    """The `dynamicTools` of a thread, from wire `function` declarations."""
    return [
        {
            "type": "function",
            "name": function["name"],
            "description": str(function.get("description") or ""),
            "inputSchema": function.get("parameters")
            or {"type": "object", "properties": {}},
        }
        for function in functions
    ]


def _tool_choice_instructions(tool_choice, parallel_tool_calls=None):
    """The lines added to the instructions for what the protocol cannot say.

    The app-server declares tools but takes no `tool_choice`: `"required"`
    and a named tool are asked for in the instructions and checked on the
    answer (`_app_server_completion` raises, and the base class retries).
    """
    mode, name = _tool_choice_mode(tool_choice)
    lines = []
    if mode == "required":
        lines.append("Call at least one of the tools before answering.")
    elif mode == "named":
        lines.append(f"Call the tool `{name}` now.")
    if parallel_tool_calls is False:
        lines.append("Call at most one tool at a time.")
    return "\n".join(lines)


def _has_tool_history(formatted_messages):
    """Whether the messages hold a tool call or a tool result."""
    return any(
        message.get("role") in ("tool", "function") or message.get("tool_calls")
        for message in formatted_messages
    )


def _history_items(formatted_messages):
    """Splits wire messages into `(instructions, items)`.

    The system and developer messages become the instructions, as in
    `_split_messages`. The other messages become the Responses API items the
    thread's history takes as they are: a `message` per user or assistant
    text, a `function_call` per tool call of an assistant message (its id
    kept, its arguments as the JSON string the model wrote), and a
    `function_call_output` per tool result, naming the call it answers.
    """
    instructions = []
    items = []
    for message in formatted_messages:
        text = _message_text(message)
        role = str(message.get("role") or "user")
        if role in ("system", "developer"):
            if text:
                instructions.append(text)
        elif role == "assistant":
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id"),
                        "name": function.get("name"),
                        "arguments": arguments,
                    }
                )
        elif role in ("tool", "function"):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or message.get("name"),
                    "output": text,
                }
            )
        elif text:
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                }
            )
    return "\n\n".join(instructions), items


def _build_app_server_command(executable="codex", catalog_path=None, tools=False):
    """Builds the `codex app-server` command line of one completion.

    The same `-c` overrides as `codex exec` (`_build_codex_command`), plus
    `_APP_SERVER_OVERRIDES`: the app-server has no `--ignore-user-config`,
    so what the user's `config.toml` could add is overridden key by key. The
    rest of the lockdown (read-only sandbox, empty working directory, no
    approval, ephemeral thread) is set on the thread. With `tools`, the
    harness's own function is taken out (`_NO_HARNESS_FUNCTION`).
    """
    command = [executable]
    for override in (*_CONFIG_OVERRIDES, *_APP_SERVER_OVERRIDES):
        command += ["-c", override]
    if tools:
        command += ["-c", _NO_HARNESS_FUNCTION]
    if catalog_path:
        command += ["-c", f"model_catalog_json={json.dumps(catalog_path)}"]
    command += ["app-server", "--listen", "stdio://"]
    return command


def _default_effort(catalog, cli_model):
    """The default reasoning effort of `cli_model` in the CLI's catalog, or None.

    `codex exec --ignore-user-config` leaves a call without effort at the
    model's default; the app-server would apply the `model_reasoning_effort`
    of the user's `config.toml` instead, so the default is sent explicitly.
    """
    try:
        models = json.loads(catalog or "{}").get("models") or []
    except ValueError:
        return None
    for model in models:
        if model.get("slug") == cli_model:
            return model.get("default_reasoning_level") or None
    return None


def _turn_usage(token_usage):
    """The usage of the last request of a turn, in the `codex exec` keys."""
    last = (token_usage or {}).get("last") or {}
    return {
        "input_tokens": last.get("inputTokens"),
        "cached_input_tokens": last.get("cachedInputTokens"),
        "output_tokens": last.get("outputTokens"),
        "reasoning_output_tokens": last.get("reasoningOutputTokens"),
    }


class _AppServer:
    """One `codex app-server` process, spoken to in JSON-RPC over stdio.

    One JSON object per line, each way. `request` awaits the server's answer;
    the notifications and the requests the server makes to the client
    (`item/tool/call`...) are read with `next_event`, in order. The process
    is killed if it outlives the context.
    """

    def __init__(self, command, env, cwd):
        self._command = command
        self._env = env
        self._cwd = cwd
        self._process = None
        self._next_id = 1
        self._pending = {}
        self._events = asyncio.Queue()
        self._reader = None

    async def __aenter__(self):
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._env,
                cwd=self._cwd,
            )
        except FileNotFoundError:
            raise DeterministicStopError(
                f"The Codex CLI `{self._command[0]}` was not found. {_INSTALL_HINT}",
                "cli_not_found",
            )
        self._reader = asyncio.create_task(self._read())
        return self

    async def __aexit__(self, *exc_info):
        if self._reader is not None:
            self._reader.cancel()
        if self._process.returncode is None:
            # The server exits once its input closes; killed if it lingers.
            try:
                self._process.stdin.close()
                await asyncio.wait_for(self._process.wait(), timeout=2)
            except (asyncio.TimeoutError, OSError):
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
                await self._process.wait()

    async def _read(self):
        while True:
            line = await self._process.stdout.readline()
            if not line:
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(
                            RuntimeError("`codex app-server` closed its output.")
                        )
                await self._events.put(None)
                return
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if "method" in message:
                await self._events.put(message)
                continue
            future = self._pending.pop(message.get("id"), None)
            if future is None or future.done():
                continue
            if "error" in message:
                future.set_exception(
                    RuntimeError(json.dumps(message["error"], ensure_ascii=False))
                )
            else:
                future.set_result(message.get("result"))

    async def _write(self, payload):
        self._process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await self._process.stdin.drain()

    async def request(self, method, params=None):
        """Sends a request and returns its result.

        Raises:
            RuntimeError: If the server answers with an error.
        """
        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        return await future

    async def notify(self, method, params=None):
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def next_event(self):
        """The next notification or server request, None once the output closed."""
        return await self._events.get()


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

    Install the CLI with the `codex` extra (or with
    `npm install -g @openai/codex`) and sign in once (interactive, opens a
    browser):

    ```bash
    pip install 'synalinks[codex]'
    "$(python -c 'import codex_cli_bin; print(codex_cli_bin.bundled_codex_path())')" login
    ```

    The CLI of the extra is used when installed, else `codex` from the
    `PATH`; without either, the first call raises an `ImportError`.

    Then use it like any other language model:

    ```python
    import synalinks

    language_model = synalinks.OAuthLanguageModel(
        model="codex/gpt-6.1-sol",
        reasoning_effort="low",
    )
    ```

    `synalinks.language_models.get("codex/gpt-6.1-sol")`, and therefore
    `language_model="codex/gpt-6.1-sol"` on any module, returns this class too.
    Use `"codex/default"` to let the CLI pick the default model of the
    account, and `codex debug models` to list the available ones.

    **How it works**

    Only the transport differs from `LanguageModel`: message formatting,
    cache, retry, `fallback`, usage counters and response parsing are
    inherited. Each call spawns `codex exec` stripped down to a plain
    completion endpoint:

    - The output schema is passed with `--output-schema` and enforced by the
        backend in strict mode: every field is required and returned.
    - The system message replaces the built-in coding-agent prompt (a short
        default one is sent when there is none), and the harness adds
        nothing to the request: it holds the messages and the schema.
    - The model gets no tool: the shell, web search, apps and plugins are
        disabled, and the tools the model catalog of the CLI turns on for a
        model (the "code mode" of the gpt-6 models, sub-agents, the patch
        tool) are taken out of it. The catalog is read once with
        `codex debug models`; if the CLI has none, a warning says so.
    - Whatever the model is given, the sandbox is read-only, without
        network, and the working directory is an empty temporary directory.
        A tool call the CLI reports fails the call.
    - The subprocess only inherits an allowlist of environment variables,
        so the API keys of the parent process are not exposed to it.

    **Tool calling**

    `tools` and `tool_schemas` are accepted as with `LanguageModel`, so a
    `FunctionCallingAgent` (and the agents built on it) runs unchanged on a
    `codex/` model. A turn that declares tools, or whose history holds tool
    calls, goes through `codex app-server` (one process per completion too),
    where tool calling is native:

    - The tools are declared to the thread (`dynamicTools`): the model
        sees them as the functions of the request, as on the API.
    - The earlier turns are appended to the thread's history as the
        Responses API items they are (`thread/inject_items`): user and
        assistant messages, `function_call`, `function_call_output`. The
        last user message starts the turn.
    - The model's first call comes back as an `item/tool/call` request.
        The turn is interrupted there and the call returned as a native
        tool call, which the caller runs: Synalinks keeps the loop, the CLI
        runs nothing. `tool_choice="none"` declares no tool; a structured
        answer over a tool history uses the turn's `outputSchema`.
    - The same lockdown applies: read-only sandbox without network, empty
        working directory, no approval, ephemeral thread, tool-free catalog,
        allowlisted environment. A thread item that is not a message, a
        reasoning or a call to a declared tool fails the call. The one
        function the harness declares on its own, `request_user_input`, is
        taken out of a turn with tools: the gpt-6 models call it, the CLI
        refuses it, and each attempt costs an unreported request.

    **Limitations**

    - The messages are not sent as a chat. The system and developer
        messages become the instructions of the CLI, as one block; the rest
        becomes a single prompt: a lone user message verbatim, a longer
        conversation as a role-tagged transcript (`[USER]`, `[ASSISTANT]`).
        Earlier assistant turns are therefore not sent as assistant
        messages, and what reaches the backend is shaped by the CLI, which
        can change between releases (`codex debug prompt-input` shows it).
        Results are not directly comparable with the same program on an
        API-key provider.
    - No streaming: `streaming=True` raises a `ValueError`.
    - A turn with tools goes through `codex app-server`, which has no
        `--ignore-user-config`: the user's `config.toml` is loaded, and what
        it could add to a request is overridden key by key (its MCP servers
        are disabled by name, its `developer_instructions`, `notify` and
        `model_provider` overridden, the effort set to the model's default
        of the catalog when the call sets none). `codex/default` takes the
        model of that configuration there. `dynamicTools` and
        `thread/inject_items` are experimental in the app-server protocol
        of codex-cli 0.164.
    - `tool_choice="required"` or a named tool have no native form: they
        are asked for in the instructions, and an answer without the call is
        retried. `parallel_tool_calls` is not controlled: the CLI sends
        `parallel_tool_calls=false`, so a turn carries one call.
    - Text only: images and audio parts are dropped, including those a
        tool returns.
    - No sampling control: the CLI exposes neither `temperature` nor
        `max_tokens`, they are ignored with a warning. Do not rely on this
        class where a deterministic `temperature=0.0` judge is required.
    - A free-form `dict` field cannot be expressed in strict mode and
        raises a `ValueError`.
    - The cost is not tracked (flat-rate subscription); token usage is, and
        each call counts against the usage limits of the subscription. The
        backend also caps concurrent requests per account (about five in
        flight, measured with codex-cli 0.162); beyond that, requests are
        refused and retried.
    - One process per call. The Codex Python SDK (`openai-codex`, a
        persistent `codex app-server`) sends the same request for the same
        settings and was not measurably faster.
    - OpenAI recommends API-key authentication for programmatic Codex
        workflows; check the terms of your subscription before unattended
        use.

    `reasoning_effort` behaves as with `LanguageModel`: a value (`"low"`,
    `"medium"`, `"high"`...) is sent to the backend, `"none"` and `None`
    keep the default of the model. `"disable"` and `"minimal"` send `"low"`,
    the lowest effort every model accepts (most of them cannot turn
    reasoning off). A `thinking` field of the output schema is written by
    the model.

    Args:
        model (str): The model to use, as `codex/<model>` (e.g.
            `codex/gpt-6.1-sol`), or `codex/default` for the account default.
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
                f"a provider among {list(OAUTH_PROVIDERS)} (e.g. 'codex/gpt-6.1-sol'). "
                f"Received: model={model!r}"
            )
        super().__init__(model=model, **kwargs)
        # The ignored parameters already reported, so that a parameter set on
        # every call (e.g. the `temperature` of a `Generator`) warns once.
        self._reported_ignored_params = set()
        # The tool-free model catalog, as JSON, read from the CLI on the
        # first call ("" if the CLI has none).
        self._catalog = None
        # The path of the CLI, looked up on the first call.
        self._codex = None

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
            tools (list | dict): Optional `synalinks.modules.Tool`s the LM may
                call, as with `LanguageModel` (native, through the app-server:
                see the class docstring). Mutually exclusive with `schema`.
            tool_schemas (list): Optional wire-format tool declarations, as
                with `LanguageModel`.
            streaming (bool): Not supported, must be False.
            **kwargs (keyword arguments): The additional keywords arguments
                forwarded to the LM call (e.g. `tool_choice`,
                `reasoning_effort`).
        Returns:
            (dict): The generated structured response.
        """
        if streaming:
            raise ValueError("`OAuthLanguageModel` does not support streaming.")
        if self._codex is None:
            self._codex = _codex_executable()
        if schema:
            schema = _strict_schema(schema)
        return await super().call(
            messages, schema=schema, tools=tools, tool_schemas=tool_schemas, **kwargs
        )

    async def _acompletion(self, formatted_messages, **kwargs):
        # As for any provider, `None` (what a module passes when it sets no
        # effort) and "none" (not forwarded by the base class) leave the model
        # at its default: no effort is sent.
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        reasoning_effort = _EFFORT_ALIASES.get(reasoning_effort, reasoning_effort)
        response_format = kwargs.pop("response_format", None)
        schema = None
        if response_format:
            schema = (response_format.get("json_schema") or {}).get("schema")
        ignored = sorted(
            k
            for k in kwargs
            if k in _IGNORED_PARAMS and k not in self._reported_ignored_params
        )
        if ignored:
            self._reported_ignored_params.update(ignored)
            warnings.warn(
                f"{self} ignores the parameters {ignored}: the Codex CLI does not "
                "expose them."
            )
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        parallel_tool_calls = kwargs.pop("parallel_tool_calls", None)
        cli_model = self.model.split("/", 1)[1]
        if cli_model == "default":
            cli_model = None
        catalog = await self._model_catalog()
        if tools or _has_tool_history(formatted_messages):
            # Declared tools, or earlier calls and results to send as the
            # items they are: the app-server, the only transport with both.
            return await self._app_server_completion(
                formatted_messages,
                cli_model,
                catalog,
                schema=schema,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                reasoning_effort=reasoning_effort,
            )
        instructions, prompt = _split_messages(formatted_messages)

        with tempfile.TemporaryDirectory(prefix="synalinks_oauth_") as tmp:
            # The model runs in an empty directory; the request files live
            # next to it, out of its working root.
            workdir = os.path.join(tmp, "workdir")
            os.mkdir(workdir)
            command = _build_codex_command(
                cli_model,
                workdir,
                instructions_path=_write(
                    tmp, "instructions.md", instructions or _DEFAULT_INSTRUCTIONS
                ),
                schema_path=_write(tmp, "schema.json", schema and json.dumps(schema)),
                catalog_path=_write(tmp, "models.json", catalog),
                reasoning_effort=reasoning_effort,
                executable=self._codex,
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

    def _app_server(self, command, cwd):
        """The app-server process of one completion (a seam for the tests)."""
        env = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
        return _AppServer(command, env, cwd)

    async def _app_server_completion(
        self,
        formatted_messages,
        cli_model,
        catalog,
        *,
        schema=None,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
        reasoning_effort=None,
    ):
        """One completion through `codex app-server`.

        The conversation with the server: `initialize`, `config/read` (the
        MCP servers of the user's configuration, to disable), `thread/start`
        (instructions, tools, lockdown), `thread/inject_items` (the earlier
        turns), `turn/start` (the last user message, the output schema), then
        the events of the turn until it completes; it is interrupted at the
        model's first tool call. The calls received, if any, are returned as
        native tool calls; else the answer.
        """
        try:
            functions = _offered_tools(tools, tool_choice)
        except ValueError as e:
            raise DeterministicStopError(str(e), "invalid_request") from e
        if functions and schema:
            raise DeterministicStopError(
                "`OAuthLanguageModel` cannot combine a `response_format` with "
                "tools that may be called.",
                "invalid_request",
            )
        instructions, items = _history_items(formatted_messages)
        instructions = "\n\n".join(
            text
            for text in (
                instructions or _DEFAULT_INSTRUCTIONS,
                _tool_choice_instructions(tool_choice, parallel_tool_calls)
                if functions
                else "",
            )
            if text
        )
        # The last user message is the input of the turn; whatever precedes
        # it is the history of the thread.
        turn_input = []
        if items and items[-1]["type"] == "message" and items[-1]["role"] == "user":
            last = items.pop()
            turn_input = [{"type": "text", "text": last["content"][0]["text"]}]
        mode, _ = _tool_choice_mode(tool_choice)
        reasoning_effort = reasoning_effort or _default_effort(catalog, cli_model)

        with tempfile.TemporaryDirectory(prefix="synalinks_oauth_") as tmp:
            workdir = os.path.join(tmp, "workdir")
            os.mkdir(workdir)
            command = _build_app_server_command(
                executable=self._codex,
                catalog_path=_write(tmp, "models.json", catalog),
                tools=bool(functions),
            )
            async with self._app_server(command, workdir) as app:
                try:
                    turn = await asyncio.wait_for(
                        self._run_turn(
                            app,
                            workdir,
                            cli_model,
                            instructions,
                            items,
                            turn_input,
                            dynamic_tools=_dynamic_tools(functions),
                            schema=schema,
                            reasoning_effort=reasoning_effort,
                        ),
                        timeout=self.timeout,
                    )
                except asyncio.TimeoutError as e:
                    raise RuntimeError(
                        f"`{command[0]} app-server` timed out after {self.timeout}s."
                    ) from e

        content, usage, calls, errors, forbidden, status = turn
        if forbidden:
            raise DeterministicStopError(
                f"The model used tools ({', '.join(forbidden)}) instead of "
                "answering; the response is discarded.",
                "tool_use",
            )
        if errors or status == "failed":
            detail = "; ".join(dict.fromkeys(errors)) or f"turn {status}"
            message = f"`{command[0]} app-server` failed: {detail[-_MAX_ERROR_CHARS:]}"
            if "invalid_request_error" in detail:
                raise DeterministicStopError(message, "invalid_request")
            raise RuntimeError(message)
        tool_calls = [
            {
                "id": call["callId"],
                "type": "function",
                "function": {
                    "name": call["tool"],
                    "arguments": json.dumps(call.get("arguments") or {}),
                },
            }
            for call in calls
        ]
        if mode in ("required", "named") and not tool_calls:
            # Retried by the base class, as an empty answer is.
            raise ValueError(
                f"The model called no tool although `tool_choice={tool_choice!r}`."
            )
        return _to_model_response(content, usage, tool_calls=tool_calls)

    async def _run_turn(
        self,
        app,
        workdir,
        cli_model,
        instructions,
        items,
        turn_input,
        *,
        dynamic_tools,
        schema,
        reasoning_effort,
    ):
        """Runs one turn on `app` and returns what it produced.

        Returns:
            (tuple): `(content, usage, calls, errors, forbidden, status)`:
                the answer (None if the model called tools), the usage of its
                last request, the `item/tool/call` params received, the error
                messages, the sorted item types that reveal the model acted,
                and the status of the turn.
        """
        await app.request(
            "initialize",
            {"clientInfo": _APP_SERVER_CLIENT, "capabilities": {"experimentalApi": True}},
        )
        await app.notify("initialized")
        # The app-server loads the user's `config.toml`; its MCP servers would
        # give the model tools. Disabled by name on this thread.
        config = (await app.request("config/read", {}) or {}).get("config") or {}
        mcp_servers = config.get("mcp_servers") or {}
        thread_params = {
            "model": cli_model,
            "baseInstructions": instructions,
            "developerInstructions": "",
            "ephemeral": True,
            "cwd": workdir,
            "sandbox": "read-only",
            "approvalPolicy": "never",
            "config": {"mcp_servers": {name: {"enabled": False} for name in mcp_servers}},
        }
        if dynamic_tools:
            thread_params["dynamicTools"] = dynamic_tools
        thread = await app.request("thread/start", thread_params)
        thread_id = thread["thread"]["id"]
        if items:
            await app.request(
                "thread/inject_items", {"threadId": thread_id, "items": items}
            )
        turn_params = {"threadId": thread_id, "input": turn_input}
        if schema:
            turn_params["outputSchema"] = schema
        if reasoning_effort:
            turn_params["effort"] = reasoning_effort
        turn = await app.request("turn/start", turn_params)
        turn_id = turn["turn"]["id"]

        content = None
        usage = {}
        calls = []
        errors = []
        forbidden = set()
        status = None
        interrupted = False
        while True:
            event = await app.next_event()
            if event is None:
                errors.append(
                    "`codex app-server` closed its output before the turn ended."
                )
                break
            method = event.get("method")
            params = event.get("params") or {}
            if "id" in event:
                # A request of the server. A tool call is what the turn is
                # for: it is kept, never answered, and the turn is interrupted
                # so that the caller runs the tool. Anything else (an approval,
                # a question to the user) means the model acted.
                if method == "item/tool/call":
                    calls.append(params)
                else:
                    forbidden.add(str(method))
                if not interrupted:
                    interrupted = True
                    # The call is dispatched as soon as its item is streamed,
                    # before the response ends: interrupted at once, the
                    # stream is dropped and the usage of the response with it
                    # (measured live: 8 of 10 tool-calling turns reported no
                    # usage). The stream ends within the second.
                    await asyncio.sleep(_STREAM_DRAIN_S)
                    await app.request(
                        "turn/interrupt", {"threadId": thread_id, "turnId": turn_id}
                    )
            elif method in ("item/started", "item/completed"):
                item = params.get("item") or {}
                item_type = item.get("type")
                if item_type not in _ALLOWED_APP_SERVER_ITEMS:
                    forbidden.add(str(item_type))
                elif item_type == "agentMessage" and method == "item/completed":
                    content = item.get("text")
            elif method == "thread/tokenUsage/updated":
                usage = _turn_usage(params.get("tokenUsage"))
            elif method == "error":
                errors.append(str(params.get("message")))
            elif method == "turn/completed":
                turn = params.get("turn") or {}
                status = turn.get("status")
                error = turn.get("error")
                if error and status != "interrupted":
                    errors.append(str(error.get("message") or error))
                break
        return content, usage, calls, errors, sorted(forbidden), status

    async def _model_catalog(self):
        """Returns the tool-free model catalog of the CLI, as JSON.

        The catalog holds the request settings of each model (reasoning
        levels, verbosity, context window...), so the one of the CLI is kept
        and only its tools are removed. It is read once per instance.

        Returns:
            (str): The catalog, or an empty string if the CLI printed none.
        """
        if self._catalog is None:
            stdout, _, _ = await self._run([self._codex, "debug", "models"], "")
            try:
                self._catalog = json.dumps(_without_tools(json.loads(stdout)))
            except (ValueError, TypeError, KeyError):
                self._catalog = ""
                warnings.warn(
                    f"`codex debug models` printed no model catalog: {self} keeps "
                    "the tools the CLI gives the model by default (the code mode "
                    "of the gpt-6 models), whose calls cost requests and are not "
                    "reported."
                )
        return self._catalog

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
                f"The Codex CLI `{command[0]}` was not found. {_INSTALL_HINT}",
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

    @classmethod
    def provider_info(cls, provider):
        """Returns what a frontend needs to set a provider up.

        Args:
            provider (str): A provider prefix from `supported_providers()`.

        Returns:
            (dict): `provider`, `env`, `default_api_base` (`None`) and
                `lists_models_from` (`"server"`: the CLI lists them).

        Raises:
            ValueError: If the provider is not supported.
        """
        model_catalog.check_provider(
            provider, cls.supported_providers(), "language model"
        )
        return model_catalog.provider_info(provider, _CODEX_PROVIDER)

    @classmethod
    async def supported_models(cls, provider, api_base=None):
        """Returns the models the Codex CLI offers (those of its model picker).

        Args:
            provider (str): A provider prefix from `supported_providers()`.
            api_base (str): Unused, accepted for compatibility with
                `LanguageModel`.

        Returns:
            (list): The sorted model IDs, ready to pass as `model`
                (e.g. `"codex/gpt-6.1-sol"`).

        Raises:
            ValueError: If the provider is not supported.
            ImportError: If the Codex CLI is not installed.
            ConnectionError: If the CLI prints no model catalog.
        """
        model_catalog.check_provider(
            provider, cls.supported_providers(), "language model"
        )
        language_model = cls(model=f"{provider}/default")
        stdout, stderr, _ = await language_model._run(
            [_codex_executable(), "debug", "models"], ""
        )
        try:
            models = json.loads(stdout)["models"]
        except (ValueError, TypeError, KeyError) as e:
            raise ConnectionError(
                f"`codex debug models` printed no model catalog: {stderr.strip()}"
            ) from e
        return sorted(
            f"{provider}/{model['slug']}"
            for model in models
            if model.get("visibility", "list") == "list"
        )

    def supports_reasoning(self):
        """Whether the model takes a reasoning effort (always True).

        The effort reaches the CLI as `model_reasoning_effort`.
        """
        return True

    def supports_reasoning_trace(self):
        """Whether the reasoning trace fills the `thinking` field (always False).

        The CLI only returns a one-line summary of the reasoning, and not on
        every turn: the model writes the `thinking` field itself.
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
