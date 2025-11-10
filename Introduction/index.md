# Introduction

______________________________________________________________________

## What is Synalinks?

Synalinks is an open-source framework that makes it easy to create, evaluate, train, and deploy industry-standard Language Models (LMs) applications. Synalinks follows the principle of *progressive disclosure of complexity*: meaning that simple workflows should be quick and easy, while arbitrarily advanced ones should be possible via a clear path that builds upon what you've already learned.

Synalinks is an *adaptation of Keras 3* focused on neuro-symbolic systems and in-context reinforcement learning, an ensemble of techniques that enhance the LMs predictions and accuracy without changing the weights of the model. The goal of Synalinks is to facilitate the rapid setup of simple applications while providing the flexibility for researchers and advanced users to develop sophisticated systems.

______________________________________________________________________

Info

You can use the [`llms.txt`](/synalinks/llms.txt) or [`llms-full.txt`](/synalinks/llms-full.txt) to feed your favorite LMs with Synalinks documentation

## Who is Synalinks for?

Synalinks is designed for a diverse range of users, from professionals and AI researchers to students, independent developers, and hobbyists. It is suitable for anyone who wants to learn about AI by building/composing blocks or build solid foundations for enterprise-grade products. While a background in Machine Learning and Deep Learning can be advantageous — as Synalinks leverages design patterns from Keras, one of the most user-friendly and popular Deep Learning frameworks — it is not a prerequisite. Synalinks is designed to be accessible to anyone with programming skills in Python, making it a versatile and inclusive platform for AI development.

______________________________________________________________________

## Why use Synalinks?

Developping a successful LM application in a profesional context, beyond stateless chatbots, is difficult and typically include:

- **Building optimized prompts with examples/instructions at each step**: Synalinks uses advanced In-Context Reinforcement Learning techniques to optimize each prompt.
- **Pipelines that change over time**: Easily edit your pipelines, re-run your training, and you're good to go.
- **Ensuring the correctness of the LMs output**: Synalinks combines constrained structured output with In-Context RL to ensure both format and content correctness.
- **Async Optimization**: Synalinks automatically optimizes your pipelines by detecting parallel processes.
- **Assessing the performance of your application**: Synalinks provides built-in metrics and rewards to evaluate your workflows.
- **Configuring Language & Embedding Models**: Seamlessly integrate multiple LM providers like Ollama, OpenAI, Anthropic, Mistral or Groq.
- **Documenting your ML workflows**: Plot your workflows, training history, and evaluations; document everything.
- **Versioning the prompts/pipelines**: Each program is serializable into JSON so you can version it with git.
- **Deploying REST APIs**: Compatible out-of-the-box with FastAPI so your Data Scientists and Web Developers can stop tearing each other apart.

Synalinks can help you simplify these tasks by leveraging decade old practices in Deep Learning frameworks. We provide a comprehensive suite of tools and features designed to streamline the development process, making it easier to create, evaluate, train, document and deploy robust neuro-symbolic LMs applications.

______________________________________________________________________
