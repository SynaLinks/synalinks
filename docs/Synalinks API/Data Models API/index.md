# Data Models API

Synalinks features four distinct types of data models, each serving a unique purpose within the framework:

- **[DataModel](The DataModel class.md)**: This is the backend-dependent data model, built on Pydantic's `BaseModel`. It is the primary model most users will interact with. It allows for schema and variable declarations, and is used to format datasets. When entering a workflow, this data model is automatically converted into a backend-independent format.

- **[JsonDataModel](The JsonDataModel class.md)**: This is the backend-independent data model that flows through the pipelines. It holds both a JSON schema and a JSON value, enabling it to perform computations. Unlike the backend-dependent model, this one is dynamically created and modified.

- **[SymbolicDataModel](The SymbolicDataModel class.md)**: This is the symbolic data model used during the functional API declaration to infer the pipeline's edges and nodes. It only holds a JSON schema, allowing the system to compute the pipeline from inputs and outputs without performing actual computations.

- **[Variable](The Variable class.md)**: This data model holds the module's state and can be updated during training. It includes a JSON schema and JSON value, enabling computations, and also contains metadata about training. `Optimizer`s can update it during the training process.

## Data Models API Overview

- [The DataModel Class](The DataModel class.md): The backend-dependent data models.
- [The JsonDataModel Class](The JsonDataModel class.md): The backend-independent data models.
- [The SymbolicDataModel Class](The SymbolicDataModel class.md): The symbolic data models.
- [The Variable Class](The Variable class.md): The variable data models that the optimizers can act upon.
- [The Base DataModels](The Base DataModels.md): A collection of basic backend-dependent data models.
- [Score Data Models](Score Data Models.md): Discretized score scales (`Score`, `FineScore`, `Rating`, `Rating10`, `Rating20`) the language model can pick from, and the helpers to resolve and normalize them.