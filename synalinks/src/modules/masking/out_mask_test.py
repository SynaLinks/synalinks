# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.modules import Input
from synalinks.src.modules.masking.out_mask import OutMask
from synalinks.src.programs import Program


class Document(DataModel):
    title: str
    text: str


class InMaskTest(testing.TestCase):
    async def test_in_mask_single_data_model(self):
        inputs = Input(data_model=Document)

        outputs = await OutMask(
            mask=["title"],
        )(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
            name="masking_program",
            description="A program to remove fields",
        )

        doc = Document(title="Test document", text="Hello world")

        result = await program(doc)

        self.assertTrue(len(result.keys()) == 1)

    async def test_in_mask_multiple_data_models(self):
        inputs = [Input(data_model=Document), Input(data_model=Document)]

        outputs = await OutMask(
            mask=["title"],
        )(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
            name="masking_program",
            description="A program to remove fields",
        )

        doc1 = Document(title="Test document 1", text="Hello world")
        doc2 = Document(title="Test document 2", text="Hello world")

        results = await program([doc1, doc2])

        self.assertTrue(len(results[0].keys()) == 1)
        self.assertTrue(len(results[1].keys()) == 1)
