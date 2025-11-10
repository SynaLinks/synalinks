# Callbacks API

A callback is an object that can perform various actions at multiple stages of the program's training.
For example, at the start or end of an epoch, before or after a single batch, etc.

## How to use Callbacks

You can pass a list of callbacks to the `.fit()` method of a program.

```python
import synalinks
import asyncio

async def main():
    # ... you program declaration here

    callbacks = [
        synalinks.callbacks.CSVLogger(filepath="training_log.csv"),
        synalinks.callbacks.ProgramCheckpoint(
            filepath="program.{epoch:02d}-{val_loss:.2f}.json"
        ),
    ]

    history = await program.fit(
        x=x_train,
        y=y_train,
        epochs=10,
        callbacks=callbacks,
    )

if __main__ == "__main__":
    asyncio.run(main())
```

## Callbacks Overview

- [Base Callback class](Base Callback class.md)
- [CSVLogger callback](CSVLogger.md)
- [ProgramCheckPoint callback](ProgramCheckpoint.md)
- [Monitor callback](Monitor.md)