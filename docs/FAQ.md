# FAQ

## General questions

- [What makes Synalinks revolutionary compared to DSPy?](#what-makes-synalinks-revolutionary-compared-to-dspy)
- [Why do you focus on in-context techniques?](#why-do-you-focus-on-in-context-techniques)
- [I already use structured output, why would I use Synalinks?](#i-already-use-structured-output-why-would-i-use-synalinks)
- [Can Synalinks be used for non-LMs applications](#can-synalinks-be-used-for-non-lms-applications)
- [What are my options for saving programs?](#what-are-my-options-for-saving-programs)
- [How to do hyperparameter tuning with Synalinks?](#how-to-do-hyperparameter-tuning-with-synalinks)
- [Where is the Synalinks configuration file stored?](#where-is-the-synalinks-configuration-file-stored)
- [How should I cite Synalinks?](#how-should-i-cite-synalinks)
- [Do you provide help or support?](#do-you-provide-help-or-support)

## Training related questions

- [What do "sample", "batch", and "epoch" mean?](#what-do-sample-batch-and-epoch-mean)
- [What's the difference between the `training` argument in `call()` and the `trainable` attribute?](#whats-the-difference-between-the-training-argument-in-call-and-the-trainable-attribute)
- [In `fit()`, how is the validation split computed?](#in-fit-how-is-the-validation-split-computed)
- [In `fit()`, is the data shuffled during training?](#in-fit-is-the-data-shuffled-during-training)
- [What's the difference between Program methods `predict()` and `__call__()`?](#whats-the-difference-between-program-methods-predict-and-__call__)

---

### What makes Synalinks revolutionary compared to DSPy?

While DSPy wrestles with PyTorch complexity, Synalinks delivers the elegant simplicity of Keras with enterprise-grade power. We're the only framework featuring logical flows inspired by logical circuits and comprehensive Knowledge Graph support. Synalinks transforms AI workflow design into an intuitive, natural process that accelerates development cycles and reduces implementation complexity.

---

### Why do you focus on in-context techniques?

In the real-world, most problems that companies face, are not labelled in public datasets for ML engineers to train their networks against. Meaning that their is a big discrepency between the results annonced on public benchmarks and real-world problems. Making adoption difficult when companies face the reality and complexity of real-world.

Training a whole Language Model (LM) from scratch is out of the reach of most companies, so adapting them to real-world tasks is essential.

LMs have the capability to leverage their prompt to mimick the examples given, but it means that one have to update the examples each time you change the pipelines as you experiment. Making it cumberstone, but even with that, their is no guaranty that the examples you gave yield to the best results.

To select the best examples and instructions to give to the LMs, it needs a complex system like Synalinks that automate the generation and selection.

---

### I already use structured output, why would I use Synalinks?

While structured output ensure a correct format, ready to parse, it doesn't guaranty the content of the LMs answers. Synalinks use *constrained structured output in conjunction with in-context techniques* to ensure **both** format and content correctness.

---

### Can Synalinks be used for non-LMs applications?

While Synalinks provide everything to work with LMs, we emphasize that you can create modules (or entire pipelines) that don't use them. In fact, many neuro-symbolic systems use a conjunction of LMs modules with non-LMs modules. Synalinks backend can suit any algorithm that works with any kind of data-structure as long as they are formalized in JSON.

---

### What are my options for saving programs?

**1. Whole-program saving (configuration + variables)**

Whole program saving means creating a file that will contain:

- The architecture of the program, allowing you to re-create the program.
- The variables of the program
- The training configuration (reward, optimizer, metrics)
- The state of the optimizer, allowing you to resume the training exactly where you left off.

The default and recommended way to save the whole program is to do:

`program.save("my_program.json")`

After saving a program you can re-instanciate it via:

`program = synalinks.Program.load("my_program.json")`

**2. Variables-only saving**

If you need to save the variables of a program, you can it using:

`program.save_variables("my_program.variables.json")`

You can then load the variables into a program *with the same architecture*:

`program.load_variables("my_program.variables.json")`

**Note:** All programs and/or variables are saved in JSON format, we removed the pickle format for obvious security concerns.

---

### How to do hyperparameter tuning with Synalinks?

Synalinks is compatible with [KerasTuner](https://keras.io/keras_tuner/) which allows you to find hyperparameters.

---

### Where is the Synalinks configuration file stored?

The default directory where all Synalinks data is stored is:

```bash
$HOME/.synalinks/
```

Note that Windows users should replace `$HOME` with `%USERPROFILE%`.

In case Synalinks cannot create the above directory (e.g. due to permission issues), `/tmp/.synalinks/` is used as a backup.

The Synalinks configuration file is a JSON file stored at $HOME/.synalinks/synalinks.json. The default configuration file looks like this:

```json
{
    "backend": "pydantic",
    "floatx": "float32",
    "epsilon": 1e-07
}
```

Likewise, cached dataset files, such as those downloaded with `get_file()`, are stored by default in `$HOME/.synalinks/datasets/`.

---

### How should I cite Synalinks?

Please cite Synalinks if it is useful in your research. Here is the bibtex entry to use:

```bibtex
@misc{sallami2025synalinks,
  title={Synalinks},
  author={Sallami, Yoan and Chollet, Fran\c{c}ois},
  year={2025},
  howpublished={\url{https://github.com/SynaLinks/Synalinks}},
}
```

---

### Do you provide help or support?

We provide consulting, development and technical support for companies that want to implement any neuro-symbolic systems. Using a framework is one thing, having a complete view of possible neuro-symbolic applications and the knowledge to create such complex systems is another. If you can't afford our services, you can find help in our public [Discord channel](https://discord.gg/82nt97uXcM).

---

### What do "sample", "batch", and "epoch" mean?

- **Sample**: A sample is one element of a dataset. For example, one DataModel is one sample.
- **Batch**: A batch is a set of N samples. The samples in a batch are processed independently, in parallel. During training, a batch result in only one program update. A batch approximates the input distribution better than a single input. The larger the batch, the better the approximation; however a larger batch will take longer to process and still result in only one update.
- **Epochs**: A epochs is an arbitrarly cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation. When using `validation_split` or `validation_data` with the `fit` method of Synalinks programs, evaluation will be run at the end of every epoch.

---

### What's the difference between the `training` argument in `call()` and the `trainable` attribute?

`training` is a boolean argument in `call` that determines whether the call should be run in inference mode or training mode. For example, in training mode, a `Generator` module save each LM prediction for later backpropagation. In inference mode, the `Generator` doesn't save the predictions.

`trainable` is a boolean module attribute that determines the trainable variables of the module should be updated to maximize the reward during training. If `module.trainable` is set to False, then `module.trainable_variables` will always be an empty list. 

Example:

```python
import synalinks
import asyncio

async def main():
    class ThinkingWithAnswer(synalinks.DataModel):
        thinking: str
        answer: str

    language_model = synalinks.LanguageModel(
        "ollama_chat/deepseek-r1",
    )

    program = synalinks.Sequential(
        [
            synalinks.Generator(
                data_model=ThinkingWithAnswer,
                language_model=language_model,
            ),
            synalinks.Generator(
                data_model=ThinkingWithAnswer,
                language_model=language_model,
            ),
        ]
    )

    program.modules[0].trainable = False # Freeze the first generator

    assert program.modules[0].trainable_variables == []

    program.compile(...)
    history = await program.fit(...) # Train only the second generator

if __main__ == "__name__":
    asyncio.run(main())
```

In essence, "inference mode vs training mode" and "module variable trainability" are two very different concepts.

---

### In `fit()`, how is the validation split computed?

If you set the `validation_split` argument in `program.fit` to e.g. 0.1, then the validation data used will be the *last 10%* of the data. If you set it to 0.25, it will be the *last 25%* of the data, etc. Note that the data isn't shuffled before extracting the validation split, so the validation is literally just the *last x%* of samples in the input you passed.

The same validation set is used for all epochs (within the same call to fit).

Note that the validation_split option is only available if your data is passed as Numpy arrays.

---

### In `fit()`, is the data shuffled during training?

If you pass your data as NumPy arrays and if the `shuffle` argument in `program.fit()` is set to True (which is the default), the training data will be globally randomly shuffled at each epoch.

Validation data is never shuffled.

---

### What's the difference between `Program` methods `predict()` and `__call__()`?

`predict()` loops over the data in batches (you can specify the batch size via `predict(x, batch_size=64)`) and returns a Numpy array of predictions.

While `program(x)` perform a single prediction, and is used to create APIs that process a single user request at a time.

This means that `predict()` calls can perform prediction on a dataset. While `program(x)` perform a single prediction.

---