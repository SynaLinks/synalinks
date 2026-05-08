# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json as _json
import os
import tempfile
from unittest.mock import patch

import pytest

from synalinks.src import testing
from synalinks.src.datasets.built_in import arcagi as arcagi_module
from synalinks.src.datasets.built_in.arcagi import ARCAGIInput
from synalinks.src.datasets.built_in.arcagi import ARCAGIOutput
from synalinks.src.datasets.built_in.arcagi import default_instructions
from synalinks.src.datasets.built_in.arcagi import get_arcagi1_evaluation_task_names
from synalinks.src.datasets.built_in.arcagi import get_arcagi1_training_task_names
from synalinks.src.datasets.built_in.arcagi import get_arcagi2_evaluation_task_names
from synalinks.src.datasets.built_in.arcagi import get_arcagi2_training_task_names
from synalinks.src.datasets.built_in.arcagi import get_input_data_model
from synalinks.src.datasets.built_in.arcagi import get_output_data_model
from synalinks.src.datasets.built_in.arcagi import load_data
from synalinks.src.datasets.built_in.arcagi import plot_task

# A minimal task with two train pairs and one test pair, kept small so the
# 80/20 / leave-one-out / permutation paths are easy to count.
_TASK_JSON = {
    "train": [
        {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
        {"input": [[2, 2]], "output": [[2, 2]]},
    ],
    "test": [
        {"input": [[3, 3]], "output": [[3, 3]]},
    ],
}


def _write_task(tmp_dir, payload=_TASK_JSON):
    path = os.path.join(tmp_dir, "task.json")
    with open(path, "w") as f:
        f.write(_json.dumps(payload))
    return path


class GettersTest(testing.TestCase):
    def test_default_instructions_is_nonempty(self):
        self.assertIn("input grid", default_instructions())

    def test_data_model_getters(self):
        self.assertIs(get_input_data_model(), ARCAGIInput)
        self.assertIs(get_output_data_model(), ARCAGIOutput)

    def test_arcagi1_task_name_lists(self):
        # These return module-level constants imported from arcagi1_tasks.
        self.assertIsInstance(get_arcagi1_training_task_names(), list)
        self.assertIsInstance(get_arcagi1_evaluation_task_names(), list)

    def test_arcagi2_task_names_via_url(self):
        # Mocked get_file returns a path to a local file with a few names.
        with tempfile.TemporaryDirectory() as tmp:
            train_path = os.path.join(tmp, "training.txt")
            eval_path = os.path.join(tmp, "evaluation.txt")
            with open(train_path, "w") as f:
                f.write("task_a\ntask_b\n")
            with open(eval_path, "w") as f:
                f.write("task_c\n")

            def _fake_get_file(origin, **kwargs):
                if origin.endswith("training.txt"):
                    return train_path
                if origin.endswith("evaluation.txt"):
                    return eval_path
                raise AssertionError(f"Unexpected origin: {origin}")

            with patch.object(
                arcagi_module.file_utils, "get_file", side_effect=_fake_get_file
            ):
                self.assertEqual(get_arcagi2_training_task_names(), ["task_a", "task_b"])
                self.assertEqual(get_arcagi2_evaluation_task_names(), ["task_c"])


class LoadDataTest(testing.TestCase):
    def test_filepath_one_leave_out_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_task(tmp)
            (x_train, y_train), (x_test, y_test) = load_data(
                task_name="ignored", filepath=path
            )
        # 2 train rows × 1 leave-one-out variant each = 2 training pairs.
        self.assertEqual(len(x_train), 2)
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(x_test), 1)
        # The "examples" leaf on each train input is the *other* training row.
        first_examples = x_train[0].examples
        self.assertEqual(len(first_examples), 1)

    def test_filepath_one_leave_out_with_permutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_task(tmp)
            (x_train, _), _ = load_data(
                task_name="ignored",
                filepath=path,
                permutation=True,
                curriculum_learning=False,
            )
        # 2 train rows; for each, the "other" examples list has 1 element,
        # so permutations of 1 element = 1 permutation. So still 2 outputs.
        self.assertEqual(len(x_train), 2)

    def test_filepath_no_leave_out_with_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_task(tmp)
            (x_train, y_train), _ = load_data(
                task_name="ignored",
                filepath=path,
                one_leave_out=False,
                repeat=2,
            )
        # The ``one_leave_out=False`` branch has a nested loop over trainset
        # for each outer i, so we get len(trainset) × len(trainset) × repeat
        # = 2 × 2 × 2 = 8 entries. (Quirk of the source; flagged here so any
        # future de-duplication is a deliberate change.)
        self.assertEqual(len(x_train), 8)
        self.assertEqual(len(y_train), 8)

    def test_filepath_curriculum_learning_off(self):
        # Just exercise the early-exit branch.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_task(tmp)
            (x_train, _), _ = load_data(
                task_name="ignored", filepath=path, curriculum_learning=False
            )
        self.assertEqual(len(x_train), 2)

    def test_filepath_not_found_raises(self):
        with pytest.raises(ValueError, match="Could not find"):
            load_data(task_name="ignored", filepath="/no/such/file.json")

    def test_invalid_arc_version_raises(self):
        with pytest.raises(ValueError, match="`arc_version`"):
            load_data(task_name="anything", arc_version=99)

    def test_arcagi1_unknown_task_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            load_data(task_name="not-a-real-task-id", arc_version=1)

    def test_arcagi2_unknown_task_raises(self):
        # Mock the task-name listings so we can prove the unknown-task path.
        with patch.object(
            arcagi_module, "get_arcagi2_training_task_names", return_value=["a", "b"]
        ), patch.object(
            arcagi_module, "get_arcagi2_evaluation_task_names", return_value=["c"]
        ):
            with pytest.raises(ValueError, match="not recognized"):
                load_data(task_name="not-listed", arc_version=2)

    def test_arcagi1_via_url_with_mocked_fetch(self):
        # Pick the first known training task and stub get_file to return our
        # local JSON.
        names = get_arcagi1_training_task_names()
        self.assertGreater(len(names), 0)
        task_name = names[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_task(tmp)
            with patch.object(
                arcagi_module.file_utils, "get_file", return_value=path
            ):
                (x_train, _), (x_test, _) = load_data(task_name=task_name)
        self.assertEqual(len(x_train), 2)
        self.assertEqual(len(x_test), 1)

    def test_arcagi1_evaluation_task_via_url(self):
        # Force the evaluation branch.
        eval_names = get_arcagi1_evaluation_task_names()
        self.assertGreater(len(eval_names), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_task(tmp)
            with patch.object(
                arcagi_module.file_utils, "get_file", return_value=path
            ):
                (x_train, _), _ = load_data(task_name=eval_names[0], arc_version=1)
        self.assertEqual(len(x_train), 2)

    def test_arcagi2_via_url_with_mocked_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_task(tmp)
            with patch.object(
                arcagi_module, "get_arcagi2_training_task_names", return_value=["TASK"]
            ), patch.object(
                arcagi_module, "get_arcagi2_evaluation_task_names", return_value=[]
            ), patch.object(
                arcagi_module.file_utils, "get_file", return_value=path
            ):
                (x_train, _), _ = load_data(task_name="TASK", arc_version=2)
        self.assertEqual(len(x_train), 2)

    def test_arcagi2_evaluation_via_url_with_mocked_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_task(tmp)
            with patch.object(
                arcagi_module, "get_arcagi2_training_task_names", return_value=[]
            ), patch.object(
                arcagi_module,
                "get_arcagi2_evaluation_task_names",
                return_value=["TASK"],
            ), patch.object(
                arcagi_module.file_utils, "get_file", return_value=path
            ):
                (x_train, _), _ = load_data(task_name="TASK", arc_version=2)
        self.assertEqual(len(x_train), 2)


class PlotTaskTest(testing.TestCase):
    """Exercise plot_task end-to-end by writing to a temp PNG.

    We don't assert pixel content — just that each branch runs without
    raising and produces a file.
    """

    def _x(self):
        return {
            "examples": [
                {"input_grid": [[0, 1], [1, 0]], "output_grid": [[1, 0], [0, 1]]},
            ],
            "input_grid": [[2, 2], [2, 2]],
        }

    def _y(self):
        return {"output_grid": [[3, 3], [3, 3]]}

    def test_plot_with_y_true_and_y_pred(self):
        with tempfile.TemporaryDirectory() as tmp:
            plot_task(
                x=self._x(),
                y_true=self._y(),
                y_pred=self._y(),
                to_folder=tmp,
                task_name="t",
            )
            self.assertTrue(os.path.exists(os.path.join(tmp, "t.png")))

    def test_plot_with_y_true_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            plot_task(x=self._x(), y_true=self._y(), to_folder=tmp)
            self.assertTrue(
                os.path.exists(os.path.join(tmp, "arc_agi_task.png"))
            )

    def test_plot_with_y_pred_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            plot_task(x=self._x(), y_pred=self._y(), to_folder=tmp)

    def test_plot_with_neither_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="y_true or y_pred"):
                plot_task(x=self._x(), to_folder=tmp)

    def test_plot_accepts_custom_cmap(self):
        # Hits the ``cmap is None`` False branch.
        from matplotlib import colors as mpl_colors

        cmap = mpl_colors.ListedColormap(["#000000"] * 10)
        with tempfile.TemporaryDirectory() as tmp:
            plot_task(
                x=self._x(),
                y_true=self._y(),
                cmap=cmap,
                to_folder=tmp,
            )

    def test_curriculum_learning_handles_empty_output_grid(self):
        # Triggers the ``return 0`` branch of get_output_grid_size when an
        # output grid is empty (curriculum_learning sorts by output area).
        payload = {
            "train": [
                {"input": [[0]], "output": []},
                {"input": [[1]], "output": [[1, 2]]},
            ],
            "test": [{"input": [[2]], "output": [[2]]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "task.json")
            with open(path, "w") as f:
                f.write(_json.dumps(payload))
            (x_train, y_train), _ = load_data(
                task_name="ignored", filepath=path, one_leave_out=True
            )
        # The empty-grid example sorts to the front (size 0).
        self.assertEqual(y_train[0].output_grid, [])

    def test_plot_accepts_data_model_inputs(self):
        # Hits the to_json_data_model conversion branches.
        from synalinks.src.datasets.built_in.arcagi import ARCAGITask

        x = ARCAGIInput(
            examples=[
                ARCAGITask(
                    input_grid=[[0, 1]], output_grid=[[1, 0]],
                ),
            ],
            input_grid=[[2, 2]],
        )
        y = ARCAGIOutput(output_grid=[[3, 3]])
        with tempfile.TemporaryDirectory() as tmp:
            plot_task(x=x, y_true=y, y_pred=y, to_folder=tmp, task_name="dm")
            self.assertTrue(os.path.exists(os.path.join(tmp, "dm.png")))
