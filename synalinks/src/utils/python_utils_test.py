# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.utils.python_utils import default
from synalinks.src.utils.python_utils import func_dump
from synalinks.src.utils.python_utils import func_load
from synalinks.src.utils.python_utils import is_default
from synalinks.src.utils.python_utils import pythonify_logs
from synalinks.src.utils.python_utils import remove_by_id
from synalinks.src.utils.python_utils import remove_long_seq
from synalinks.src.utils.python_utils import removeprefix
from synalinks.src.utils.python_utils import removesuffix
from synalinks.src.utils.python_utils import to_list


class PythonUtilsTest(testing.TestCase):
    def test_default_decorator(self):
        @default
        def my_method():
            pass

        self.assertTrue(is_default(my_method))

    def test_is_default_false(self):
        def my_method():
            pass

        self.assertFalse(is_default(my_method))

    def test_to_list_with_list(self):
        result = to_list([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

    def test_to_list_with_scalar(self):
        result = to_list(5)
        self.assertEqual(result, [5])

    def test_to_list_with_none(self):
        result = to_list(None)
        self.assertEqual(result, [None])

    def test_to_list_with_string(self):
        result = to_list("hello")
        self.assertEqual(result, ["hello"])

    def test_remove_long_seq(self):
        seq = [[1, 2], [1, 2, 3, 4, 5], [1]]
        label = ["a", "b", "c"]
        new_seq, new_label = remove_long_seq(3, seq, label)
        self.assertEqual(new_seq, [[1, 2], [1]])
        self.assertEqual(new_label, ["a", "c"])

    def test_remove_long_seq_all_short(self):
        seq = [[1], [2]]
        label = ["a", "b"]
        new_seq, new_label = remove_long_seq(10, seq, label)
        self.assertEqual(new_seq, [[1], [2]])
        self.assertEqual(new_label, ["a", "b"])

    def test_remove_long_seq_all_long(self):
        seq = [[1, 2, 3], [4, 5, 6]]
        label = ["a", "b"]
        new_seq, new_label = remove_long_seq(2, seq, label)
        self.assertEqual(new_seq, [])
        self.assertEqual(new_label, [])

    def test_removeprefix(self):
        self.assertEqual(removeprefix("hello_world", "hello_"), "world")

    def test_removeprefix_no_match(self):
        self.assertEqual(removeprefix("hello_world", "foo"), "hello_world")

    def test_removeprefix_empty(self):
        self.assertEqual(removeprefix("hello", ""), "hello")

    def test_removesuffix(self):
        self.assertEqual(removesuffix("hello_world", "_world"), "hello")

    def test_removesuffix_no_match(self):
        self.assertEqual(removesuffix("hello_world", "foo"), "hello_world")

    def test_removesuffix_empty(self):
        self.assertEqual(removesuffix("hello", ""), "hello")

    def test_remove_by_id(self):
        a = object()
        b = object()
        c = object()
        lst = [a, b, c]
        remove_by_id(lst, b)
        self.assertEqual(len(lst), 2)
        self.assertIs(lst[0], a)
        self.assertIs(lst[1], c)

    def test_remove_by_id_not_found(self):
        a = object()
        b = object()
        lst = [a]
        remove_by_id(lst, b)
        self.assertEqual(len(lst), 1)

    def test_pythonify_logs_empty(self):
        result = pythonify_logs(None)
        self.assertEqual(result, {})

    def test_pythonify_logs_basic(self):
        result = pythonify_logs({"loss": 1.5, "acc": 0.9})
        self.assertEqual(result, {"acc": 0.9, "loss": 1.5})

    def test_pythonify_logs_nested(self):
        result = pythonify_logs({"outer": {"inner_a": 1, "inner_b": 2}})
        self.assertEqual(result, {"inner_a": 1.0, "inner_b": 2.0})

    def test_pythonify_logs_non_numeric(self):
        result = pythonify_logs({"key": "not_a_number"})
        self.assertEqual(result, {"key": "not_a_number"})

    def test_func_dump_and_load(self):
        def add(x, y):
            return x + y

        code, defaults, closure = func_dump(add)
        restored = func_load(code, defaults, closure)
        self.assertEqual(restored(2, 3), 5)

    def test_func_dump_and_load_with_defaults(self):
        def add(x, y=10):
            return x + y

        code, defaults, closure = func_dump(add)
        restored = func_load(code, defaults, closure)
        self.assertEqual(restored(5), 15)

    def test_func_dump_and_load_with_closure(self):
        factor = 3

        def multiply(x):
            return x * factor

        code, defaults, closure = func_dump(multiply)
        restored = func_load(code, defaults, closure)
        self.assertEqual(restored(4), 12)

    def test_func_load_from_tuple(self):
        def sub(x, y):
            return x - y

        dumped = func_dump(sub)
        restored = func_load(dumped)
        self.assertEqual(restored(10, 4), 6)

    def test_func_load_from_list(self):
        def sub(x, y):
            return x - y

        dumped = list(func_dump(sub))
        restored = func_load(dumped)
        self.assertEqual(restored(10, 4), 6)

    def test_func_load_defaults_as_list(self):
        def add(x, y=5):
            return x + y

        code, defaults, closure = func_dump(add)
        # Pass defaults as list to test conversion
        restored = func_load([code, list(defaults), closure])
        self.assertEqual(restored(3), 8)
