# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import numpy as np

from synalinks.src.backend import DataModel


class Query(DataModel):
    query: str


class AnswerWithRationale(DataModel):
    rationale: str
    answer: str


def load_test_data():
    x_train = np.array(
        [
            Query(query="What is the capital of France?"),
            Query(query="What is the French city of aeronautics?"),
            Query(query="Which city is known as the fashion capital of the world?"),
            Query(query="What is the highest mountain in France?"),
            Query(query="Which river flows through Paris?"),
            Query(query="What is the capital of Italy?"),
            Query(query="Which mountain range separates France and Spain?"),
            Query(query="What is the largest lake in France?"),
            Query(query="Which French city is famous for its wine production?"),
            Query(query="What is the currency used in France?"),
            Query(query="Which ocean borders western France?"),
            Query(query="What is the official language of France?"),
            Query(query="Which French region is known for lavender fields?"),
            Query(query="What is the most visited monument in Paris?"),
            Query(query="Which French city hosted the 1968 Winter Olympics?"),
        ],
        dtype="object",
    )
    y_train = np.array(
        [
            AnswerWithRationale(
                rationale="""The capital of France is well-known and is the seat of """
                """the French government.""",
                answer="Paris",
            ),
            AnswerWithRationale(
                rationale="""Toulouse is known as the French city of aeronautics due"""  # noqa: E501
                """ to its significant contributions to the aerospace industry.""",
                answer="Toulouse",
            ),
            AnswerWithRationale(
                rationale="""Paris is widely recognized as the fashion capital of the world, """  # noqa: E501
                """hosting major fashion weeks and luxury brands.""",
                answer="Paris",
            ),
            AnswerWithRationale(
                rationale="""Mont Blanc is the highest mountain in France, standing at """  # noqa: E501
                """4,807 meters above sea level.""",
                answer="Mont Blanc",
            ),
            AnswerWithRationale(
                rationale="""The Seine River flows through Paris, dividing the city """
                """and serving as a major waterway.""",
                answer="Seine",
            ),
            AnswerWithRationale(
                rationale="""Rome is the capital city of Italy and has been the center """  # noqa: E501
                """of Italian government and culture for centuries.""",
                answer="Rome",
            ),
            AnswerWithRationale(
                rationale="""The Pyrenees mountain range forms a natural border """
                """between France and Spain.""",
                answer="Pyrenees",
            ),
            AnswerWithRationale(
                rationale="""Lake Geneva (Lac Léman) is the largest lake in France, """  # noqa: E501
                """shared with Switzerland.""",
                answer="Lake Geneva",
            ),
            AnswerWithRationale(
                rationale="""Bordeaux is renowned worldwide for its wine production """  # noqa: E501
                """and vineyard regions.""",
                answer="Bordeaux",
            ),
            AnswerWithRationale(
                rationale="""France uses the Euro as its official currency, """
                """adopted in 1999 as part of the European Union.""",
                answer="Euro",
            ),
            AnswerWithRationale(
                rationale="""The Atlantic Ocean borders the western coast of France, """  # noqa: E501
                """providing important maritime access.""",
                answer="Atlantic Ocean",
            ),
            AnswerWithRationale(
                rationale="""French is the official language of France, spoken """
                """by the vast majority of the population.""",
                answer="French",
            ),
            AnswerWithRationale(
                rationale="""Provence is famous for its extensive lavender fields, """
                """particularly around the Valensole plateau.""",
                answer="Provence",
            ),
            AnswerWithRationale(
                rationale="""The Eiffel Tower is the most visited paid monument """
                """in the world and Paris's most iconic landmark.""",
                answer="Eiffel Tower",
            ),
            AnswerWithRationale(
                rationale="""Grenoble hosted the 1968 Winter Olympics, becoming """
                """an important winter sports destination.""",
                answer="Grenoble",
            ),
        ],
        dtype="object",
    )

    x_test = np.array(
        [
            Query(query="What is the largest city in France?"),  # noqa: E501
            Query(query="What is the capital of Germany?"),  # noqa: E501
            Query(query="Which French city is known for its film festival?"),  # noqa: E501
            Query(query="What is the longest river in France?"),  # noqa: E501
            Query(query="Which French region is famous for champagne?"),  # noqa: E501
            Query(query="What is the capital of Spain?"),  # noqa: E501
            Query(query="Which sea borders southern France?"),  # noqa: E501
            Query(query="What is the national symbol of France?"),  # noqa: E501
            Query(query="Which French city is home to the European Parliament?"),  # noqa: E501
            Query(query="What is the highest point in the French Alps?"),  # noqa: E501
        ],
        dtype="object",
    )
    y_test = np.array(
        [
            AnswerWithRationale(
                rationale="Paris is the largest city in France by population and area.",  # noqa: E501
                answer="Paris",
            ),
            AnswerWithRationale(
                rationale="The capital of Germany is well-known and is the seat of the German government.",  # noqa: E501
                answer="Berlin",
            ),
            AnswerWithRationale(
                rationale="Cannes is famous for hosting the prestigious Cannes Film Festival annually.",  # noqa: E501
                answer="Cannes",
            ),
            AnswerWithRationale(
                rationale="The Loire River is the longest river in France, flowing for 1,012 kilometers.",  # noqa: E501
                answer="Loire",
            ),
            AnswerWithRationale(
                rationale="Champagne region is world-famous for producing the sparkling wine of the same name.",  # noqa: E501
                answer="Champagne",
            ),
            AnswerWithRationale(
                rationale="Madrid is the capital and largest city of Spain, located in the center of the country.",  # noqa: E501
                answer="Madrid",
            ),
            AnswerWithRationale(
                rationale="The Mediterranean Sea borders the southern coast of France along the French Riviera.",  # noqa: E501
                answer="Mediterranean Sea",
            ),
            AnswerWithRationale(
                rationale="The rooster (coq) is the national symbol of France, representing vigilance and pride.",  # noqa: E501
                answer="Rooster",
            ),
            AnswerWithRationale(
                rationale="Strasbourg is home to the European Parliament and serves as a major EU political center.",  # noqa: E501
                answer="Strasbourg",
            ),
            AnswerWithRationale(
                rationale="Mont Blanc is the highest point in the French Alps and in Western Europe at 4,807 meters.",  # noqa: E501
                answer="Mont Blanc",
            ),
        ],
        dtype="object",
    )
    return (x_train, y_train), (x_test, y_test)


def mock_completion_data():
    # Training responses - expanded to match new training data
    responses = [
        """{"rationale":"The capital of France is well-known and is the seat of the French government.", "answer": "Paris"}""",  # noqa: E501
        """{"rationale":"Toulouse is known as the French city of aeronautics due to its significant contributions to the aerospace industry.", "answer": "Toulouse"}""",  # noqa: E501
        """{"rationale":"Paris is widely recognized as the fashion capital of the world, hosting major fashion weeks and luxury brands.", "answer": "Paris"}""",  # noqa: E501
        """{"rationale":"Mont Blanc is the highest mountain in France, standing at 4,807 meters above sea level.", "answer": "Mont Blanc"}""",  # noqa: E501
        """{"rationale":"The Seine River flows through Paris, dividing the city and serving as a major waterway.", "answer": "Seine"}""",  # noqa: E501
        """{"rationale":"Rome is the capital city of Italy and has been the center of Italian government and culture for centuries.", "answer": "Rome"}""",  # noqa: E501
        """{"rationale":"The Pyrenees mountain range forms a natural border between France and Spain.", "answer": "Pyrenees"}""",  # noqa: E501
        """{"rationale":"Lake Geneva (Lac Léman) is the largest lake in France, shared with Switzerland.", "answer": "Lake Geneva"}""",  # noqa: E501
        """{"rationale":"Bordeaux is renowned worldwide for its wine production and vineyard regions.", "answer": "Bordeaux"}""",  # noqa: E501
        """{"rationale":"France uses the Euro as its official currency, adopted in 1999 as part of the European Union.", "answer": "Euro"}""",  # noqa: E501
        """{"rationale":"The Atlantic Ocean borders the western coast of France, providing important maritime access.", "answer": "Atlantic Ocean"}""",  # noqa: E501
        """{"rationale":"French is the official language of France, spoken by the vast majority of the population.", "answer": "French"}""",  # noqa: E501
        """{"rationale":"Provence is famous for its extensive lavender fields, particularly around the Valensole plateau.", "answer": "Provence"}""",  # noqa: E501
        """{"rationale":"The Eiffel Tower is the most visited paid monument in the world and Paris's most iconic landmark.", "answer": "Eiffel Tower"}""",  # noqa: E501
        """{"rationale":"Grenoble hosted the 1968 Winter Olympics, becoming an important winter sports destination.", "answer": "Grenoble"}""",  # noqa: E501
    ]
    return [{"choices": [{"message": {"content": response}}]} for response in responses]


def mock_incorrect_completion_data():
    # Training responses with some incorrect answers
    responses = [
        """{"rationale":"The capital of France is well-known and is the seat of the French government.", "answer": "Paris"}""",  # noqa: E501
        """{"rationale":"Paris is known as the French city of aeronautics due to its significant contributions to the aerospace industry.", "answer": "Paris"}""",  # noqa: E501
        """{"rationale":"Milan is widely recognized as the fashion capital of the world, hosting major fashion weeks.", "answer": "Milan"}""",  # noqa: E501
        """{"rationale":"Mount Everest is the highest mountain in France, standing at great height.", "answer": "Mount Everest"}""",  # noqa: E501
        """{"rationale":"The Thames River flows through Paris, dividing the city.", "answer": "Thames"}""",  # noqa: E501
        """{"rationale":"Paris is the capital city of Italy and has been the center of Italian government.", "answer": "Paris"}""",  # noqa: E501
        """{"rationale":"The Alps mountain range forms a natural border between France and Spain.", "answer": "Alps"}""",  # noqa: E501
        """{"rationale":"Lake Superior is the largest lake in France, shared with other countries.", "answer": "Lake Superior"}""",  # noqa: E501
        """{"rationale":"Paris is renowned worldwide for its wine production and vineyard regions.", "answer": "Paris"}""",  # noqa: E501
        """{"rationale":"France uses the Dollar as its official currency in modern times.", "answer": "Dollar"}""",  # noqa: E501
        """{"rationale":"The Pacific Ocean borders the western coast of France.", "answer": "Pacific Ocean"}""",  # noqa: E501
        """{"rationale":"English is the official language of France, spoken by most people.", "answer": "English"}""",  # noqa: E501
        """{"rationale":"Paris is famous for its extensive lavender fields in the countryside.", "answer": "Paris"}""",  # noqa: E501
        """{"rationale":"The Statue of Liberty is the most visited monument in Paris.", "answer": "Statue of Liberty"}""",  # noqa: E501
        """{"rationale":"Paris hosted the 1968 Winter Olympics, becoming a winter sports destination.", "answer": "Paris"}""",  # noqa: E501
    ]
    return [{"choices": [{"message": {"content": response}}]} for response in responses]


def mock_test_completion_data():
    # Test responses - expanded to match new test data
    responses = [
        """{"rationale":"Paris is the largest city in France by population and area.", "answer": "Paris"}""",  # noqa: E501
        """{"rationale":"The capital of Germany is well-known and is the seat of the German government.", "answer": "Berlin"}""",  # noqa: E501
        """{"rationale":"Cannes is famous for hosting the prestigious Cannes Film Festival annually.", "answer": "Cannes"}""",  # noqa: E501
        """{"rationale":"The Loire River is the longest river in France, flowing for 1,012 kilometers.", "answer": "Loire"}""",  # noqa: E501
        """{"rationale":"Champagne region is world-famous for producing the sparkling wine of the same name.", "answer": "Champagne"}""",  # noqa: E501
        """{"rationale":"Madrid is the capital and largest city of Spain, located in the center of the country.", "answer": "Madrid"}""",  # noqa: E501
        """{"rationale":"The Mediterranean Sea borders the southern coast of France along the French Riviera.", "answer": "Mediterranean Sea"}""",  # noqa: E501
        """{"rationale":"The rooster (coq) is the national symbol of France, representing vigilance and pride.", "answer": "Rooster"}""",  # noqa: E501
        """{"rationale":"Strasbourg is home to the European Parliament and serves as a major EU political center.", "answer": "Strasbourg"}""",  # noqa: E501
        """{"rationale":"Mont Blanc is the highest point in the French Alps and in Western Europe at 4,807 meters.", "answer": "Mont Blanc"}""",  # noqa: E501
    ]
    return [{"choices": [{"message": {"content": response}}]} for response in responses]


def mock_incorrect_test_completion_data():
    # Incorrect test responses, wrong answers for x_test queries
    responses = [
        """{"rationale":"Lyon is the largest city in France by population and area.", "answer": "Lyon"}""",  # noqa: E501
        """{"rationale":"The capital of Germany is Munich, which is a major city in southern Germany.", "answer": "Munich"}""",  # noqa: E501
        """{"rationale":"Paris is famous for hosting the prestigious film festival annually.", "answer": "Paris"}""",  # noqa: E501
        """{"rationale":"The Seine River is the longest river in France, flowing through major cities.", "answer": "Seine"}""",  # noqa: E501
        """{"rationale":"Bordeaux region is world-famous for producing the sparkling wine.", "answer": "Bordeaux"}""",  # noqa: E501
        """{"rationale":"Barcelona is the capital and largest city of Spain, located on the coast.", "answer": "Barcelona"}""",  # noqa: E501
        """{"rationale":"The Atlantic Ocean borders the southern coast of France.", "answer": "Atlantic Ocean"}""",  # noqa: E501
        """{"rationale":"The eagle is the national symbol of France, representing strength and freedom.", "answer": "Eagle"}""",  # noqa: E501
        """{"rationale":"Paris is home to the European Parliament and serves as a major EU center.", "answer": "Paris"}""",  # noqa: E501
        """{"rationale":"The Eiffel Tower is the highest point in the French Alps.", "answer": "Eiffel Tower"}""",  # noqa: E501
    ]
    return [{"choices": [{"message": {"content": response}}]} for response in responses]


def named_product(*args, **kwargs):
    """Utility to generate the cartesian product of parameters values and
    generate a test case names for each combination.

    The result of this function is to be used with the
    `@parameterized.named_parameters` decorator. It is a replacement for
    `@parameterized.product` which adds explicit test case names.

    For example, this code:
    ```
    class NamedExample(parameterized.TestCase):
        @parameterized.named_parameters(
            named_product(
                [
                    {'testcase_name': 'negative', 'x': -1},
                    {'testcase_name': 'positive', 'x': 1},
                    {'testcase_name': 'zero', 'x': 0},
                ],
                numeral_type=[float, int],
            )
        )
        def test_conversion(self, x, numeral_type):
            self.assertEqual(numeral_type(x), x)
    ```
    produces six tests (note that absl will reorder them by name):
    - `NamedExample::test_conversion_negative_float`
    - `NamedExample::test_conversion_positive_float`
    - `NamedExample::test_conversion_zero_float`
    - `NamedExample::test_conversion_negative_int`
    - `NamedExample::test_conversion_positive_int`
    - `NamedExample::test_conversion_zero_int`

    This function is also useful in the case where there is no product to
    generate test case names for one argument:
    ```
    @parameterized.named_parameters(named_product(numeral_type=[float, int]))
    ```

    Args:
        *args: Each positional parameter is a sequence of keyword arg dicts.
            Every test case generated will include exactly one dict from each
            positional parameter. These will then be merged to form an overall
            list of arguments for the test case. Each dict must contain a
            `"testcase_name"` key whose value is combined with others to
            generate the test case name.
        **kwargs: A mapping of parameter names and their possible values.
            Possible values should given as either a list or a tuple. A string
            representation of each value is used to generate the test case name.

    Returns:
        A list of maps for the test parameters combinations to pass to
        `@parameterized.named_parameters`.
    """

    def value_to_str(value):
        if hasattr(value, "__name__"):
            return value.__name__.lower()
        return str(value).lower()

    # Convert the keyword arguments in the same dict format as the args
    all_test_dicts = args + tuple(
        tuple({"testcase_name": value_to_str(v), key: v} for v in values)
        for key, values in kwargs.items()
    )

    # The current list of tests, start with one empty test
    tests = [{}]
    for test_dicts in all_test_dicts:
        new_tests = []
        for test_dict in test_dicts:
            for test in tests:
                # Augment the testcase name by appending
                testcase_name = test.get("testcase_name", "")
                testcase_name += "_" if testcase_name else ""
                testcase_name += test_dict["testcase_name"]
                new_test = test.copy()
                # Augment the test by adding all the parameters
                new_test.update(test_dict)
                new_test["testcase_name"] = testcase_name
                new_tests.append(new_test)
        # Overwrite the list of tests with the product obtained so far
        tests = new_tests

    return tests
