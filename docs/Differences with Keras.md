# Differences with Keras
## A Complete Guide

This document provides a comprehensive guide for translating Keras concepts into Synalinks. While Keras is designed for building traditional neural networks with tensor operations, Synalinks is a framework for creating **neuro-symbolic programs** that combine language models with structured reasoning.

---

## Fundamental Paradigm Shift

### Keras: Neural Network Framework
- **Purpose**: Build and train deep neural networks using tensor operations
- **Core abstraction**: Mathematical tensors flowing through differentiable layers
- **Training**: Gradient-based optimization (backpropagation)
- **Computation**: Matrix multiplications and activation functions
- **Use cases**: Computer vision, time series, traditional ML tasks

### Synalinks: Neuro-Symbolic LM Framework
- **Purpose**: Build intelligent applications combining LMs with symbolic reasoning
- **Core abstraction**: Structured JSON data flowing through modular programs
- **Training**: Reinforcement learning with LM-based optimizers
- **Computation**: Language model inference + symbolic operations
- **Use cases**: AI agents, reasoning systems, structured generation, API orchestration

---

## Quick Concept Mapping

| **Keras Concept** | **Synalinks Equivalent** | **Key Difference** |
|------------------|------------------------|-------------------|
| `Layer` | `Module` | Processes JSON instead of tensors |
| `Model` | `Program` | DAG of modules with conditional logic |
| `Tensor` | `Data Model` | JSON object with schema validation |
| Tensor shape | JSON schema | Explicit structure definition |
| Weights/biases | `Trainable Variable` | JSON objects, not floating-point arrays |
| Loss function | Reward function | Maximize reward vs minimize loss |
| Backpropagation | LM-based optimization | No gradients; uses language model reasoning |
| `model.compile()` | `program.compile()` | Sets up LM optimizer instead of SGD/Adam |
| `model.fit()` | `program.fit()` | Reinforcement learning loop |

---

## Core Concepts Explained

### Module (Layer Equivalent)
A **Module** is a self-contained computational unit that:
- **Receives**: JSON Data Models conforming to input schemas
- **Processes**: Via LM calls, symbolic operations, or hybrid approaches
- **Outputs**: JSON Data Models conforming to output schemas

#### Key Module Properties:
- **Keras Layer**: Receives tensors, performs matrix operations, outputs tensors
- **Synalinks Module**: Receives JSON, performs LM/symbolic operations, outputs JSON with schema validation

### Program (Model Equivalent)
A **Program** orchestrates multiple Modules into a directed acyclic graph (DAG):

#### Keras Model vs Synalinks Program:
- **Keras**: Fixed computation graph of tensor operations
- **Synalinks**: Dynamic graph with conditional branching based on LM decisions

In Keras, you build sequential or functional models with fixed layer connections. In Synalinks, you create programs that can include conditional branches, where different modules execute based on LM-driven decisions or data conditions.

### Data Models & JSON Schemas

Instead of implicit tensor shapes, Synalinks uses **explicit JSON schemas**:

- **Keras**: Input shape defined as dimensions (e.g., 784-dimensional vector)
- **Synalinks**: Input structure defined as JSON schema with explicit fields, types, and validation rules

**Why JSON?**
- **Interpretability**: Human-readable intermediate states
- **Validation**: Built-in schema validation
- **Interoperability**: Native compatibility with APIs and web services
- **Debugging**: Easy to inspect and modify
- **Structured generation**: Natural fit for LM constrained output

---

## Training Paradigm Differences

### Keras: Gradient Descent
- Defines differentiable loss function (e.g., categorical crossentropy)
- Computes gradients via automatic differentiation (backpropagation)
- Updates weights using gradient information through optimizers like Adam or SGD
- Requires continuous, differentiable operations throughout

### Synalinks: LM-Based Optimization
- Defines reward function (higher values indicate better performance)
- No gradient computation—uses language models to reason about improvements
- LM analyzes current performance and proposes better configurations
- Updates trainable variables (JSON objects) based on LM suggestions
- Can incorporate discrete decisions and non-differentiable operations

### Key Training Differences:

| **Aspect** | **Keras** | **Synalinks** |
|-----------|----------|--------------|
| **Objective** | Minimize loss | Maximize reward |
| **Optimization** | Gradient descent | Reinforcement learning |
| **Update mechanism** | Mathematical derivatives | LM-generated improvements |
| **Trainable params** | Float tensors | Structured JSON objects |

---

## When to Use Each Framework

### Use Keras when:
- Working with numerical data (images, signals, time series)
- Need fast, vectorized computations
- Have large labeled datasets
- Problem has clear differentiable objectives
- Deploying to edge devices with limited resources

### Use Synalinks when:
- Building LM-powered applications
- Need symbolic reasoning and logic
- Working with structured/semi-structured text data
- Require interpretable intermediate steps
- Building API orchestration or agent systems
- Need adaptive, context-aware processing
- Want to combine multiple LMs and tools

---

## Summary

While Keras excels at building traditional neural networks with tensor operations and gradient-based training, Synalinks is designed for a fundamentally different purpose: creating **neuro-symbolic applications** that combine the reasoning capabilities of language models with structured, validated data processing.

The shift from tensors to JSON, from gradients to LM-based optimization, and from fixed architectures to adaptive programs represents not just a technical change, but a paradigm shift in how we think about building intelligent systems. Synalinks is ideal when you need interpretability, flexibility, and the ability to integrate language understanding with symbolic reasoning—making it perfect for next-generation AI applications.