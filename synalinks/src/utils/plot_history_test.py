# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import numpy as np

from synalinks.src import testing
from synalinks.src.callbacks.history import History
from synalinks.src.utils.plot_history import plot_history
from synalinks.src.utils.plot_history import plot_history_comparison
from synalinks.src.utils.plot_history import plot_history_comparison_with_mean_and_std
from synalinks.src.utils.plot_history import plot_history_with_mean_and_std


class PlotHistoryTest(testing.TestCase):
    def generate_random_history(self, num_epochs=20, metrics=None):
        """Generate random history data with realistic logarithmic training patterns."""
        if metrics is None:
            metrics = ["reward", "reward_val"]

        history = History()
        for metric in metrics:
            x = np.linspace(1, num_epochs, num_epochs)
            log_base = np.log(x + 1) / np.log(num_epochs + 1)
            base_values = 0.1 + (0.8 * log_base)
            noise_scale = np.linspace(0.08, 0.02, num_epochs)
            noise = np.random.normal(0, noise_scale)
            values = base_values + noise

            if np.random.random() > 0.7:
                drop_positions = np.random.choice(
                    range(max(3, num_epochs // 4), num_epochs),
                    size=min(3, num_epochs // 8),
                    replace=False,
                )
                for pos in drop_positions:
                    values[pos] -= np.random.uniform(0.02, 0.06)

            values = np.clip(values, 0.0, 1.0)
            history.history[metric] = values.tolist()

        return history

    def generate_random_history_list(self, num_runs=3, num_epochs=20, metrics=None):
        """Generate a list of random history objects for multiple runs."""
        return [
            self.generate_random_history(num_epochs, metrics) for _ in range(num_runs)
        ]

    def generate_comparison_histories(self, conditions=None, num_epochs=20, metrics=None):
        """Generate comparison histories for multiple conditions."""
        if conditions is None:
            conditions = ["Program A", "Program B", "Program C"]
        if metrics is None:
            metrics = ["reward", "reward_val", "f1_score", "accuracy"]

        return {
            condition: self.generate_random_history(num_epochs, metrics)
            for condition in conditions
        }

    def generate_comparison_histories_with_runs(
        self, conditions=None, num_runs=3, num_epochs=20, metrics=None
    ):
        """Generate comparison histories with multiple runs for each condition."""
        if conditions is None:
            conditions = ["Program A", "Program B"]
        if metrics is None:
            metrics = ["reward", "reward_val", "f1_score", "accuracy"]

        return {
            condition: self.generate_random_history_list(num_runs, num_epochs, metrics)
            for condition in conditions
        }

    def test_plot_history(self):
        """Test basic plot_history function."""
        metrics = ["reward", "reward_val", "f1_score", "accuracy"]
        history = self.generate_random_history(num_epochs=30, metrics=metrics)
        plot_history(history, to_folder="/tmp/")

    def test_plot_history_with_mean_and_std(self):
        """Test plot_history_with_mean_and_std function."""
        metrics = ["reward", "reward_val", "f1_score", "accuracy"]
        history = self.generate_random_history(num_epochs=30, metrics=metrics)
        history1 = self.generate_random_history(num_epochs=30, metrics=metrics)
        plot_history_with_mean_and_std([history, history1], to_folder="/tmp/")

    def test_plot_history_comparison(self):
        """Test plot_history_comparison function."""
        metrics = ["reward", "reward_val", "f1_score", "accuracy"]
        conditions = ["Program A", "Program B", "Program C"]
        comparison_histories = self.generate_comparison_histories(
            conditions=conditions, num_epochs=25, metrics=metrics
        )
        plot_history_comparison(comparison_histories, to_folder="/tmp/")

    def test_plot_history_comparison_with_mean_and_std(self):
        """Test plot_history_comparison_with_mean_and_std function."""
        metrics = ["reward", "reward_val", "f1_score", "accuracy"]
        conditions = ["Program A", "Program B"]
        comparison_histories = self.generate_comparison_histories_with_runs(
            conditions=conditions, num_runs=4, num_epochs=25, metrics=metrics
        )
        plot_history_comparison_with_mean_and_std(comparison_histories, to_folder="/tmp/")

    def test_plot_history_with_filter(self):
        """Test plot_history with metrics filter."""
        metrics = [
            "reward",
            "reward_val",
            "f1_score",
            "accuracy",
            "precision",
            "recall",
        ]
        history = self.generate_random_history(num_epochs=20, metrics=metrics)
        metrics_filter = ["reward", "f1_score", "accuracy"]
        plot_history(history, metrics_filter=metrics_filter, to_folder="/tmp/")

    def test_plot_history_comparison_with_filter(self):
        """Test plot_history_comparison with metrics filter."""
        metrics = [
            "reward",
            "reward_val",
            "f1_score",
            "accuracy",
            "precision",
            "recall",
        ]
        conditions = ["Program A", "Program B"]
        comparison_histories = self.generate_comparison_histories(
            conditions=conditions, num_epochs=20, metrics=metrics
        )
        metrics_filter = ["reward", "f1_score", "accuracy"]
        plot_history_comparison(
            comparison_histories, metrics_filter=metrics_filter, to_folder="/tmp/"
        )

    def test_plot_history_with_high_values(self):
        """Test plot_history with values exceeding 1.0 to verify y-axis scaling."""
        history = History()
        # Create history with some values exceeding 1.0
        epochs = 15
        history.history["reward"] = [
            0.1 + i * 0.08 for i in range(epochs)
        ]  # Goes up to ~1.22
        history.history["f1_score"] = [
            0.2 + i * 0.06 for i in range(epochs)
        ]  # Goes up to ~1.04
        history.history["accuracy"] = [
            0.15 + i * 0.05 for i in range(epochs)
        ]  # Goes up to ~0.85

        plot_history(history, to_folder="/tmp/", to_file="test_history_high_values.png")

    def test_plot_history_comparison_with_high_values(self):
        """Test comparison plot with values exceeding 1.0."""
        history_a = History()
        history_b = History()

        epochs = 12
        # Program A with high values
        history_a.history["reward"] = [
            0.2 + i * 0.08 for i in range(epochs)
        ]  # Goes up to ~1.08
        history_a.history["f1_score"] = [
            0.1 + i * 0.07 for i in range(epochs)
        ]  # Goes up to ~0.87

        # Program B with different pattern
        history_b.history["reward"] = [
            0.15 + i * 0.06 for i in range(epochs)
        ]  # Goes up to ~0.81
        history_b.history["f1_score"] = [
            0.3 + i * 0.09 for i in range(epochs)
        ]  # Goes up to ~1.29

        comparison_histories = {"Program A": history_a, "Program B": history_b}
        plot_history_comparison(
            comparison_histories,
            to_folder="/tmp/",
            to_file="test_history_comparison_high_values.png",
        )

    def test_plot_history_with_different_epochs(self):
        """Test plot_history_comparison_with_mean_and_std with different epoch lengths."""
        # Create histories with different lengths to test min_epochs handling
        comparison_histories = {}

        # Short training runs (10 epochs each)
        comparison_histories["Short Training"] = [
            self.generate_random_history(num_epochs=10, metrics=["reward", "accuracy"]),
            self.generate_random_history(
                num_epochs=12, metrics=["reward", "accuracy"]
            ),  # Different length
            self.generate_random_history(
                num_epochs=11, metrics=["reward", "accuracy"]
            ),  # Different length
        ]

        # Long training runs (20 epochs each)
        comparison_histories["Long Training"] = [
            self.generate_random_history(
                num_epochs=18, metrics=["reward", "accuracy"]
            ),  # Different length
            self.generate_random_history(num_epochs=20, metrics=["reward", "accuracy"]),
            self.generate_random_history(
                num_epochs=19, metrics=["reward", "accuracy"]
            ),  # Different length
        ]

        plot_history_comparison_with_mean_and_std(
            comparison_histories,
            to_folder="/tmp/",
            to_file="test_history_different_epochs.png",
        )

    def test_plot_history_with_custom_linestyles(self):
        """Test plot_history_comparison with custom line styles."""
        metrics = ["reward", "accuracy"]
        conditions = ["Program A", "Program B", "Program C", "Program D"]
        comparison_histories = self.generate_comparison_histories(
            conditions=conditions, num_epochs=15, metrics=metrics
        )

        custom_linestyles = ["-", ":", "--", "-."]
        plot_history_comparison(
            comparison_histories,
            linestyle_cycle=custom_linestyles,
            to_folder="/tmp/",
            to_file="test_history_custom_linestyles.png",
        )

    def test_plot_history_with_alpha_transparency(self):
        """Test plot_history_comparison_with_mean_and_std with different alpha values."""
        metrics = ["reward", "f1_score"]
        conditions = ["Program A", "Program B"]
        comparison_histories = self.generate_comparison_histories_with_runs(
            conditions=conditions, num_runs=3, num_epochs=20, metrics=metrics
        )

        plot_history_comparison_with_mean_and_std(
            comparison_histories,
            alpha=0.4,  # Higher transparency
            to_folder="/tmp/",
            to_file="test_history_alpha_transparency.png",
        )
