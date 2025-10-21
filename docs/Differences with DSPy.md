# Differences with DSPy
## A Complete Guide

This document highlights the key differences between **DSPy** and **Synalinks**. While both frameworks enable in-context learning and neuro-symbolic programming, they differ significantly in design philosophy, reliability, and production readiness. We assume you have a basic understanding of in-context learning frameworks to be able to compare them.

---

## Fundamental Differences

### DSPy: PyTorch-Inspired

- **Purpose**: Build intelligent applications combining LMs with symbolic reasoning.
- **Memory**: Natively supports vector-only databases.
- **Reliability**: DSPy relies on brittle parsing logic in `Adapter` classes. While optimization reduces errors, exceptions due to LM output format failures remain common in production.
- **Async**: Offers both async and sync code, which can lead to inconsistent practices in production environments.
- **String Variables**: Like TextGrad and other in-context learning frameworks, DSPy represents variables as strings. This limits the ability to handle complex structured variables (e.g., graphs, plans), making them less suitable for advanced neuro-symbolic systems that require learning to plan or structure working memory.

### Synalinks: Keras-Inspired

- **Purpose**: Build intelligent applications combining LMs with symbolic reasoning.
- **Memory**: Natively supports hybrid graph + vector databases, enabling richer data relationships and more flexible memory structures.
- **Reliability**: Uses constrained structured output by default, eliminating brittle parsing and ensuring robust, predictable behavior.
- **Async**: Async by default, enforcing production-ready practices and consistent performance.
- **Strict Module Typing**: Modules in Synalinks are strictly typed using JSON schemas (defined in `compute_output_spec()`). This allows the system to compute output contracts end-to-end before any computation, ensuring type safety and clarity.
- **JSON Variables**: Variables are JSON objects with associated schemas. The optimizer uses constrained structured output to guarantee 100% correct variable structure, enabling complex, nested, and graph-like data handling.
- **Arithmetic & Logical Operators**: Implements JSON-level concatenation and logical operators (OR, AND, XOR) via Python operators, allowing rapid architecture changes without additional class implementations.
- **Robust Branching and Merging**: Dynamically creates schemas on the fly for branching, and handles merging via JSON operators. This enables complex workflows without requiring custom classes.
- **Observable by Default**: Every LM call within a Module can be returned, allowing reward computation based on internal processes, not just outputs, enabling finer-grained optimization and debugging.

---

## Key Concept Mapping

| **DSPy Concept**         | **Synalinks Equivalent**         | **Key Difference**                                                                 |
|--------------------------|----------------------------------|-------------------------------------------------------------------------------------|
| `Adapter`                | -                         | No brittle parsing; uses JSON schemas for robust I/O                                |
| `GEPA`                | `OMEGA`                         | Use a SOTA algorithm (2025) in evolutionary AI instead of a 10 years old method                                |
| String-based variables   | JSON-based variables             | Supports complex structures (graphs, plans) and strict validation                  |
| Sync/Async choice         | Async by default                 | Enforces production best practices                                                  |
| Vector-only memory       | Hybrid graph + vector memory    | Enables richer data relationships and more flexible memory structures              |
| Custom branching logic   | JSON operators for branching    | Dynamic schema creation and merging; no need for custom classes                    |
| Limited observability    | Observable by default            | Full visibility into LM calls for reward computation and debugging                  |

---

## When to Use Each Framework

### Use DSPy when:
- You are in a research environment and need rapid prototyping.
- Your use case is simple and does not require complex structured variables.
- You prefer the flexibility of choosing between sync and async code.
- You are comfortable managing parsing logic and potential LM output format issues (and don't mind about failures in production).

### Use Synalinks when:
- You need a production-ready, reliable system with robust error handling.
- Your application requires complex structured variables (e.g., graphs, plans).
- You want strict typing and end-to-end contract validation.
- You need hybrid memory (graph + vector) for richer data relationships.
- You want to observe and optimize internal LM processes, not just outputs.
- You need to rapidly change architectures using built-in JSON operators.

---

## Summary

While **DSPy** is a powerful research tool inspired by PyTorch’s flexibility, **Synalinks** is designed for production use, inspired by Keras user-friendliness and reliability. Synalinks use of JSON schemas, strict typing, async-by-default design, robust branching/merging makes it ideal for building complex, reliable neuro-symbolic systems that can learn, plan, and reason with structured data.