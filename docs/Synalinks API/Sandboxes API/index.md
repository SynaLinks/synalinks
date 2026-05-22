# Sandboxes API

A `Sandbox` is a **stateful**, **restricted** Python execution environment. Subsequent `run` calls see variables, imports, and function definitions from previous runs; state can be captured via `dump()` and restored via `load()`, and sandboxes round-trip through `get_config()` / `from_config()` so they flow through Synalinks' normal serialization pipeline.

Ownership is the caller's responsibility: construct a sandbox, hand it to a code-executing module (such as a recursive agent) across successive interactive turns, and build a new one for a fresh conversation. The consuming module stays stateless.

```python
import synalinks
import asyncio

async def main():
    sandbox = synalinks.MontySandbox(timeout=5.0)

    result = await sandbox.run("x = 21\nx * 2")
    print(result.result)  # 42

if __name__ == "__main__":
    asyncio.run(main())
```

## Sandboxes API overview

- [Monty Sandbox](Monty Sandbox.md)
