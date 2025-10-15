import synalinks
import asyncio

synalinks.enable_logging()


class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(
        description="The correct answer",
    )


class BacktrackingOfThought(synalinks.Module):
    def __init__(
        self,
        schema=None,
        data_model=None,
        language_model=None,
        backtracking_threshold=0.5,
        stop_threshold=0.9,
        max_iterations=5,
        return_inputs=False,
        prompt_template=None,
        examples=None,
        instructions=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema
        self.language_model = language_model
        self.backtracking_threshold = backtracking_threshold
        self.stop_threshold = stop_threshold
        self.max_iterations = max_iterations
        self.return_inputs = return_inputs
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema

        self.thinking = []
        for i in range(self.max_iterations):
            self.thinking.append(
                synalinks.ChainOfThought(
                    schema=self.schema,
                    language_model=self.language_model,
                    prompt_template=self.prompt_template,
                    examples=self.examples,
                    return_inputs=False,
                    instructions=self.instructions,
                    use_inputs_schema=self.use_inputs_schema,
                    use_outputs_schema=self.use_outputs_schema,
                    name="thinking_generator_" + self.name,
                )
            )
        self.critique = []
        for i in range(self.max_iterations):
            self.critique.append(
                synalinks.SelfCritique(
                    language_model=self.language_model,
                    prompt_template=self.prompt_template,
                    examples=self.examples,
                    return_inputs=True,
                    instructions=self.instructions,
                    use_inputs_schema=self.use_inputs_schema,
                    use_outputs_schema=self.use_outputs_schema,
                    name="critique_generator_" + self.name,
                )
            )
        # This is going to be the final generator
        self.generator = synalinks.Generator(
            schema=self.schema,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            return_inputs=self.return_inputs,
            instructions=self.instructions,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            name="generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            # This is to allow logical flows
            # (e.g. don't run the module if no inputs provided)
            return None
        for i in range(self.max_iterations):
            thinking = await self.thinking[i](
                inputs,
                training=training,
            )
            critique = await self.critique[i](
                thinking,
                training=training,
            )
            reward = critique.get("reward")
            if reward > self.backtracking_threshold:
                inputs = await synalinks.ops.concat(
                    inputs,
                    critique,
                    name=f"inputs_with_thinking_{i}_" + self.name,
                )
                if reward > self.stop_threshold:
                    break
        return await self.generator(
            inputs,
            training=training,
        )

    async def compute_output_spec(self, inputs, training=False):
        for i in range(self.max_iterations):
            inputs = await self.thinking[i](inputs)
            inputs = await self.critique[i](inputs)
        return await self.generator(inputs)

    def get_config(self):
        config = {
            "schema": self.schema,
            "backtracking_threshold": self.backtracking_threshold,
            "stop_threshold": self.stop_threshold,
            "max_iterations": self.max_iterations,
            "return_inputs": self.return_inputs,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = {
            "language_model": synalinks.saving.serialize_synalinks_object(
                self.language_model,
            )
        }
        return {**language_model_config, **config}

    @classmethod
    def from_config(cls, config):
        language_model = synalinks.saving.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(
            language_model=language_model,
            **config,
        )


async def main():
    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    inputs = synalinks.Input(data_model=Query)
    outputs = await BacktrackingOfThought(
        language_model=language_model,
        data_model=Answer,
        return_inputs=True,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="backtracking_of_thought",
        description="A Backtracking of Thought algorithm",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/implementing_custom_modules_and_programs_via_subclassing",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )

    result = await program(
        Query(
            query=(
                "How can we develop a scalable, fault-tolerant, and secure quantum"
                " computing system that can solve problems intractable for classical"
                " computers, and what are the practical implications for cryptography"
                " and data security?"
            )
        )
    )

    print(result.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
