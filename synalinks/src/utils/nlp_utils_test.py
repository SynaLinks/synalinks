# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.utils.nlp_utils import add_suffix
from synalinks.src.utils.nlp_utils import is_plural
from synalinks.src.utils.nlp_utils import normalize_and_tokenize
from synalinks.src.utils.nlp_utils import normalize_text
from synalinks.src.utils.nlp_utils import remove_articles
from synalinks.src.utils.nlp_utils import remove_numerical_suffix
from synalinks.src.utils.nlp_utils import remove_punctuation
from synalinks.src.utils.nlp_utils import to_plural
from synalinks.src.utils.nlp_utils import to_plural_property
from synalinks.src.utils.nlp_utils import to_plural_without_numerical_suffix
from synalinks.src.utils.nlp_utils import to_singular
from synalinks.src.utils.nlp_utils import to_singular_property
from synalinks.src.utils.nlp_utils import to_singular_without_numerical_suffix


class NLPUtilsTest(testing.TestCase):
    def test_to_plural(self):
        # Test regular plurals
        self.assertEqual(to_plural("cat"), "cats")
        self.assertEqual(to_plural("dog"), "dogs")
        self.assertEqual(to_plural("car"), "cars")

        # Test words ending in 'y'
        self.assertEqual(to_plural("city"), "cities")
        self.assertEqual(to_plural("baby"), "babies")

        # Test words ending in 's', 'sh', 'ch', 'x', 'z'
        self.assertEqual(to_plural("bus"), "buses")
        self.assertEqual(to_plural("watch"), "watches")
        self.assertEqual(to_plural("box"), "boxes")
        self.assertEqual(to_plural("quiz"), "quizzes")

        # Test irregular plurals
        self.assertEqual(to_plural("child"), "children")
        self.assertEqual(to_plural("man"), "men")
        self.assertEqual(to_plural("person"), "people")
        self.assertEqual(to_plural("mouse"), "mice")
        self.assertEqual(to_plural("tooth"), "teeth")
        self.assertEqual(to_plural("foot"), "feet")

        # Test words that are the same in singular and plural
        self.assertEqual(to_plural("fish"), "fish")
        self.assertEqual(to_plural("sheep"), "sheep")
        self.assertEqual(to_plural("series"), "series")

        # Test words not in the dictionary
        self.assertEqual(to_plural("robot"), "robots")
        self.assertEqual(to_plural("algorithm"), "algorithms")

    def test_to_singular(self):
        # Test regular singulars
        self.assertEqual(to_singular("cats"), "cat")
        self.assertEqual(to_singular("dogs"), "dog")
        self.assertEqual(to_singular("cars"), "car")

        # Test words ending in 'ies'
        self.assertEqual(to_singular("cities"), "city")
        self.assertEqual(to_singular("babies"), "baby")

        # Test words ending in 'es'
        self.assertEqual(to_singular("buses"), "bus")
        self.assertEqual(to_singular("watches"), "watch")
        self.assertEqual(to_singular("boxes"), "box")
        self.assertEqual(to_singular("quizzes"), "quiz")

        # Test irregular singulars
        self.assertEqual(to_singular("children"), "child")
        self.assertEqual(to_singular("men"), "man")
        self.assertEqual(to_singular("people"), "person")
        self.assertEqual(to_singular("mice"), "mouse")
        self.assertEqual(to_singular("teeth"), "tooth")
        self.assertEqual(to_singular("feet"), "foot")

        # Test words that are the same in singular and plural
        self.assertEqual(to_singular("fish"), "fish")
        self.assertEqual(to_singular("sheep"), "sheep")
        self.assertEqual(to_singular("series"), "series")

        # Test words not in the dictionary
        self.assertEqual(to_singular("robots"), "robot")
        self.assertEqual(to_singular("algorithms"), "algorithm")

    def test_to_plural_property(self):
        self.assertEqual(to_plural_property("user_account"), "user_accounts")
        self.assertEqual(to_plural_property("product_category"), "product_categories")
        self.assertEqual(to_plural_property("city"), "cities")

    def test_to_singular_property(self):
        self.assertEqual(to_singular_property("user_accounts"), "user_account")
        self.assertEqual(to_singular_property("product_categories"), "product_category")
        self.assertEqual(to_singular_property("cities"), "city")

    def test_remove_numerical_suffix(self):
        self.assertEqual(remove_numerical_suffix("property_key_1"), "property_key")
        self.assertEqual(remove_numerical_suffix("property_key_123"), "property_key")
        self.assertEqual(remove_numerical_suffix("property_key"), "property_key")

    def test_add_suffix(self):
        self.assertEqual(add_suffix("property_key", 1), "property_key_1")
        self.assertEqual(add_suffix("property_key", 123), "property_key_123")

    def test_to_singular_without_numerical_suffix(self):
        self.assertEqual(
            to_singular_without_numerical_suffix("property_key_1"), "property_key"
        )
        self.assertEqual(
            to_singular_without_numerical_suffix("user_accounts_2"), "user_account"
        )

    def test_to_plural_without_numerical_suffix(self):
        self.assertEqual(
            to_plural_without_numerical_suffix("property_key_1"), "property_keys"
        )
        self.assertEqual(
            to_plural_without_numerical_suffix("user_account_2"), "user_accounts"
        )

    def test_is_plural(self):
        self.assertTrue(is_plural("cities"))
        self.assertFalse(is_plural("city"))
        self.assertTrue(is_plural("user_accounts"))
        self.assertFalse(is_plural("user_account"))

    def test_remove_articles(self):
        self.assertEqual(remove_articles("the quick brown fox"), "quick brown fox")
        self.assertEqual(remove_articles("an apple a day"), "apple day")

    def test_remove_punctuation(self):
        self.assertEqual(remove_punctuation("Hello, world!"), "Hello world")
        self.assertEqual(remove_punctuation("Test: 1, 2, 3..."), "Test 1 2 3")

    def test_normalize_text(self):
        self.assertEqual(normalize_text("The Quick Brown Fox!"), "quick brown fox")
        self.assertEqual(normalize_text("An Apple a Day..."), "apple day")

    def test_normalize_and_tokenize(self):
        self.assertEqual(
            normalize_and_tokenize("The Quick Brown Fox!"), ["quick", "brown", "fox"]
        )
        self.assertEqual(normalize_and_tokenize("An Apple a Day..."), ["apple", "day"])
