# Sandboxes API

A `Sandbox` is a **stateful**, **restricted** Python execution environment. Subsequent `run_code` calls see variables, imports, and function definitions from previous runs; state can be captured via `dump()` and restored via `load()`, and sandboxes round-trip through `get_config()` / `from_config()` so they flow through Synalinks' normal serialization pipeline.

Ownership is the caller's responsibility: construct a sandbox, hand it to a code-executing module (such as a recursive agent) across successive interactive turns, and build a new one for a fresh conversation. The consuming module stays stateless.

```python
import synalinks
import asyncio


async def main():
    sandbox = synalinks.MirageSandbox(timeout=5.0)

    result = await sandbox.run_code("x = 21\nx * 2")
    print(result.result)  # 42

    await sandbox.kill()


if __name__ == "__main__":
    asyncio.run(main())
```

## E2B-compatible methods

Method names follow the [E2B](https://e2b.dev/docs) `AsyncSandbox` SDK, so code written for E2B ports with few changes:

| Method | Returns |
|---|---|
| `await Sandbox.create(**kwargs)` | a new sandbox |
| `await sandbox.run_code(code)` | `ExecutionResult` (`stdout`, `stderr`, `result`, `error`) |
| `await sandbox.commands.run(cmd, timeout=None)` | `CommandResult` (`stdout`, `stderr`, `exit_code`, `error`) |
| `await sandbox.files.read(path, format="text")` | `str` (or `bytes` with `format="bytes"`) |
| `await sandbox.files.write(path, data)` | `WriteInfo` |
| `await sandbox.files.list(path="/", depth=1)` | `list[EntryInfo]` |
| `await sandbox.files.exists(path)` | `bool` |
| `await sandbox.files.get_info(path)` | `EntryInfo` |
| `await sandbox.files.remove(path)` | `None` |
| `await sandbox.files.rename(old_path, new_path)` | `EntryInfo` |
| `await sandbox.files.make_dir(path)` | `bool` (`False` if it already existed) |
| `await sandbox.is_running()` | `bool` |
| `await sandbox.kill()` | `True` |

One difference from E2B: `timeout` is a per-call execution budget in seconds, not the sandbox's lifetime. `sandbox.run(...)` still works as a deprecated alias of `run_code`.

## Sandboxes API overview

- [Mirage Sandbox](Mirage Sandbox.md)
