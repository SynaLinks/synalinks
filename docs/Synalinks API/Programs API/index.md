# Programs API

Synalinks offers 4 methods to create programs, each tailored to different levels of complexity and use cases:

- **[Sequential Program](The%20Sequential%20class.md)**: This is the simplest method, involving a straightforward list of modules. Ideal for single-input, single-output stacks of modules. However, it is limited in flexibility compared to other methods.

- **[Functional Program](The%20Program%20class.md)**: This is a fully-featured API that supports arbitrary program architectures. Easy to use and suitable for most users, offering greater flexibility than the Sequential program.

- **[Mixing Subclassing and Functional API](The%20Program%20class.md)**: This make easier to encapsulate your programs made with the functional API into bigger systems.

- **[Program Subclassing](The%20Program%20class.md)**: This method allows you to implement everything from scratch. Ideal for complex or research use cases. It is also the preferred method for contributing.

## Programs API Overview

### The Program class

- [Program class](The Program class.md)
- [summary method](The Program class.md)
- [get_module method](The Program class.md)

### The Sequential class

- [Sequential class](The Sequential class.md)
- [add method](The Sequential class.md)
- [pop method](The Sequential class.md)

### Program training APIs

- [compile method](Program training API.md)
- [fit method](Program training API.md)
- [evaluate method](Program training API.md)
- [predict method](Program training API.md)
- [train_on_batch method](Program training API.md)
- [test_on_batch method](Program training API.md)
- [predict_on_batch method](Program training API.md)

### Saving & Serialization

- [Whole program saving and loading](Program Saving API/Program saving and loading.md)
- [Variables-only saving and loading](Program Saving API/Variable saving and loading.md)