# Sandboxes API

A `Sandbox` is a **stateful**, **restricted** Python execution environment. Subsequent `run_code` calls see variables, imports, and function definitions from previous runs; state can be captured via `dump()` and restored via `load()`, and sandboxes round-trip through `get_config()` / `from_config()` so they flow through Synalinks' normal serialization pipeline.

Ownership is the caller's responsibility: construct a sandbox, hand it to a code-executing module (such as a recursive agent) across successive interactive turns, and build a new one for a fresh conversation. The consuming module stays stateless.

```python
import synalinks
import asyncio


async def main():
    sandbox = synalinks.MirageSandbox(timeout=5.0)

    execution = await sandbox.run_code("x = 21\nx * 2")
    print(execution.text)  # 42

    await sandbox.kill()


if __name__ == "__main__":
    asyncio.run(main())
```

## E2B-compatible API

Method names **and return types** follow the [E2B](https://e2b.dev/docs) `AsyncSandbox` SDK, so code written for E2B reads results the same way:

| Method | Returns |
|---|---|
| `await Sandbox.create(**kwargs)` | a new sandbox |
| `await sandbox.run_code(code)` | `Execution`: `results` (the last expression as the main `Result`, with `text` = its repr and `json` = its value), `logs.stdout` / `logs.stderr` (lists of chunks), `error` (`ExecutionError` with `name`, `value`, `traceback`), `execution_count`, and `.text` |
| `await sandbox.commands.run(cmd, timeout=None)` | `CommandResult` (`stdout`, `stderr`, `exit_code`, `error`); raises `CommandExitException` on a non-zero exit and `TimeoutException` on timeout |
| `await sandbox.files.read(path, format="text")` | `str` (or `bytearray` with `format="bytes"`); raises `NotFoundException` |
| `await sandbox.files.write(path, data)` | `WriteInfo` (`name`, `type`, `path`) |
| `await sandbox.files.list(path="/", depth=1)` | `list[EntryInfo]` (`name`, `type`: `FileType`, `path`, `size`, `mode`, `permissions`, `owner`, `group`, `modified_time`) |
| `await sandbox.files.exists(path)` | `bool` |
| `await sandbox.files.get_info(path)` | `EntryInfo`; raises `NotFoundException` |
| `await sandbox.files.remove(path)` | `None` |
| `await sandbox.files.rename(old_path, new_path)` | `EntryInfo` |
| `await sandbox.files.make_dir(path)` | `bool` (`False` if it already existed) |
| `await sandbox.is_running()` | `bool` |
| `await sandbox.kill()` | `True` |

Differences from E2B:

- `timeout` is a per-call execution budget in seconds, not the sandbox's lifetime. `commands.run(..., timeout=0)` means no limit, as in E2B.
- Rich outputs (`html`, `png`, charts, ...) are not produced: the main result carries `text` and `json` only.
- The virtual filesystem has no real owners or permission bits for most mounts; `EntryInfo` then reports E2B's defaults (`user`, `0o644` / `0o755`).
- `sandbox.run(...)` still works as a deprecated alias returning the old flat `ExecutionResult`.

## Sandboxes API overview

- [Mirage Sandbox](Mirage Sandbox.md)
