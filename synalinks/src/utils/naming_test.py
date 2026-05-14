# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.utils.naming import auto_name
from synalinks.src.utils.naming import get_object_name
from synalinks.src.utils.naming import get_uid
from synalinks.src.utils.naming import reset_uids
from synalinks.src.utils.naming import to_pascal_case
from synalinks.src.utils.naming import to_pkg_name
from synalinks.src.utils.naming import to_snake_case
from synalinks.src.utils.naming import uniquify


class NamingTest(testing.TestCase):
    def test_to_snake_case_camel(self):
        self.assertEqual(to_snake_case("CamelCase"), "camel_case")

    def test_to_snake_case_already_snake(self):
        self.assertEqual(to_snake_case("already_snake"), "already_snake")

    def test_to_snake_case_mixed(self):
        self.assertEqual(to_snake_case("HTTPRequest"), "http_request")

    def test_to_snake_case_with_special_chars(self):
        self.assertEqual(to_snake_case("Hello World!"), "hello_world")

    def test_to_pascal_case_separators(self):
        # Common separator styles collapse into PascalCase.
        self.assertEqual(to_pascal_case("my-docs"), "MyDocs")
        self.assertEqual(to_pascal_case("my_docs"), "MyDocs")
        self.assertEqual(to_pascal_case("my docs"), "MyDocs")
        self.assertEqual(to_pascal_case("my.docs"), "MyDocs")
        self.assertEqual(to_pascal_case("docs"), "Docs")

    def test_to_pascal_case_case_boundaries(self):
        # Already-cased input gets re-segmented at case boundaries so
        # "myDocs" / "MyDocs" / "XMLParser" all canonicalise the same.
        self.assertEqual(to_pascal_case("myDocs"), "MyDocs")
        self.assertEqual(to_pascal_case("MyDocs"), "MyDocs")
        self.assertEqual(to_pascal_case("XMLParser"), "XmlParser")
        self.assertEqual(to_pascal_case("HTTPSConnection"), "HttpsConnection")

    def test_to_pascal_case_edge_cases(self):
        # Empty / whitespace / pure-separator inputs return "" so the
        # caller can detect "no identifier survived" and reject.
        self.assertEqual(to_pascal_case(""), "")
        self.assertEqual(to_pascal_case("   "), "")
        self.assertEqual(to_pascal_case("---"), "")
        # Digits stay where they are — note this means a leading digit
        # survives, so callers that need SQL-identifier safety must
        # still validate afterwards.
        self.assertEqual(to_pascal_case("2024_articles"), "2024Articles")
        self.assertEqual(to_pascal_case("test123"), "Test123")

    def test_to_pkg_name_dashes(self):
        self.assertEqual(to_pkg_name("my-package"), "my_package")

    def test_to_pkg_name_spaces(self):
        self.assertEqual(to_pkg_name("my package"), "my_package")

    def test_to_pkg_name_camel(self):
        self.assertEqual(to_pkg_name("MyPackage"), "my_package")

    def test_uniquify_first_call(self):
        name = uniquify("test_unique_xyz")
        self.assertEqual(name, "test_unique_xyz")

    def test_uniquify_subsequent_calls(self):
        # First call creates the name
        uniquify("test_dup_abc")
        # Second call should add suffix
        name2 = uniquify("test_dup_abc")
        self.assertEqual(name2, "test_dup_abc_1")

    def test_auto_name(self):
        name = auto_name("MyModule")
        self.assertTrue(name.startswith("my_module"))

    def test_get_uid(self):
        uid1 = get_uid("test_uid_prefix")
        uid2 = get_uid("test_uid_prefix")
        self.assertEqual(uid2, uid1 + 1)

    def test_reset_uids(self):
        get_uid("test_reset_prefix")
        reset_uids()
        uid = get_uid("test_reset_prefix")
        self.assertEqual(uid, 1)

    def test_get_object_name_with_name_attr(self):
        class Obj:
            name = "my_object"

        self.assertEqual(get_object_name(Obj()), "my_object")

    def test_get_object_name_with_function(self):
        def my_function():
            pass

        self.assertEqual(get_object_name(my_function), "my_function")

    def test_get_object_name_with_class_instance(self):
        class MyClassForNaming:
            pass

        obj = MyClassForNaming()
        # Remove 'name' if present by using object without it
        if hasattr(obj, "name"):
            del obj.name
        result = get_object_name(obj)
        self.assertEqual(result, "my_class_for_naming")
