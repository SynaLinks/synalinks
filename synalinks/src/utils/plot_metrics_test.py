# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import numpy as np

from synalinks.src import testing
from synalinks.src.utils.plot_metrics import plot_metrics
from synalinks.src.utils.plot_metrics import plot_metrics_comparison
from synalinks.src.utils.plot_metrics import plot_metrics_comparison_with_mean_and_std
from synalinks.src.utils.plot_metrics import plot_metrics_with_mean_and_std


class PlotMetricsTest(testing.TestCase):
    def generate_random_metrics(self, metrics=None):
        """Generate random metrics values."""
        if metrics is None:
            metrics = ["reward", "reward_val"]
        return {metric: np.random.uniform(0.0, 1.0) for metric in metrics}

    def generate_random_metrics_list(self, metrics=None, num_runs=3):
        """Generate a list of random metrics for multiple runs."""
        return [self.generate_random_metrics(metrics) for _ in range(num_runs)]

    def generate_comparison_metrics(self, conditions=None, metrics=None):
        """Generate comparison metrics for multiple conditions."""
        if conditions is None:
            conditions = ["Program A", "Program B", "Program C"]
        if metrics is None:
            metrics = ["reward", "reward_val", "f1_score", "accuracy"]

        return {
            condition: self.generate_random_metrics(metrics) for condition in conditions
        }

    def generate_comparison_metrics_with_runs(
        self, conditions=None, metrics=None, num_runs=3
    ):
        """Generate comparison metrics with multiple runs for each condition."""
        if conditions is None:
            conditions = ["Program A", "Program B"]
        if metrics is None:
            metrics = ["reward", "reward_val", "f1_score", "accuracy"]

        return {
            condition: self.generate_random_metrics_list(metrics, num_runs)
            for condition in conditions
        }

    def test_plot_metrics(self):
        """Test basic plot_metrics function."""
        metrics = ["reward", "reward_val", "f1_score", "accuracy"]
        metrics = self.generate_random_metrics(metrics=metrics)
        plot_metrics(metrics, to_folder="/tmp/")

    def test_plot_metrics_with_mean_and_std(self):
        """Test plot_metrics_with_mean_and_std function."""
        metrics = ["reward", "reward_val", "f1_score", "accuracy"]
        metrics = self.generate_random_metrics(metrics=metrics)
        metrics1 = self.generate_random_metrics(metrics=metrics)
        plot_metrics_with_mean_and_std([metrics, metrics1], to_folder="/tmp/")

    def test_plot_metrics_comparison(self):
        """Test plot_metrics_comparison function."""
        metrics = ["reward", "reward_val", "f1_score", "accuracy"]
        conditions = ["Program A", "Program B"]
        comparison_metrics = self.generate_comparison_metrics(
            conditions=conditions, metrics=metrics
        )
        plot_metrics_comparison(comparison_metrics, to_folder="/tmp/")

    def test_plot_metrics_comparison_with_mean_and_std(self):
        """Test plot_metrics_comparison_with_mean_and_std function."""
        metrics = ["reward", "reward_val", "f1_score", "accuracy"]
        conditions = ["Program A", "Program B"]
        comparison_metrics = self.generate_comparison_metrics_with_runs(
            conditions=conditions, metrics=metrics, num_runs=5
        )
        plot_metrics_comparison_with_mean_and_std(comparison_metrics, to_folder="/tmp/")

    def test_plot_metrics_with_filter(self):
        """Test plot_metrics with metrics filter."""
        metrics = [
            "reward",
            "reward_val",
            "f1_score",
            "accuracy",
            "precision",
            "recall",
        ]
        metrics_dict = self.generate_random_metrics(metrics=metrics)
        metrics_filter = ["reward", "f1_score", "accuracy"]
        plot_metrics(metrics_dict, metrics_filter=metrics_filter, to_folder="/tmp/")

    def test_plot_metrics_comparison_with_filter(self):
        """Test plot_metrics_comparison with metrics filter."""
        metrics = [
            "reward",
            "reward_val",
            "f1_score",
            "accuracy",
            "precision",
            "recall",
        ]
        conditions = ["Program A", "Program B"]
        comparison_metrics = self.generate_comparison_metrics(
            conditions=conditions, metrics=metrics
        )
        metrics_filter = ["reward", "f1_score", "accuracy"]
        plot_metrics_comparison(
            comparison_metrics, metrics_filter=metrics_filter, to_folder="/tmp/"
        )

    def test_plot_metrics_with_high_values(self):
        """Test plot_metrics with values exceeding 1.0 to verify y-axis scaling."""
        metrics = {
            "reward": 0.85,
            "reward_val": 1.25,  # Value exceeding 1.0
            "f1_score": 0.92,
            "accuracy": 1.15,  # Value exceeding 1.0
        }
        plot_metrics(metrics, to_folder="/tmp/", to_file="test_high_values.png")

    def test_plot_metrics_comparison_with_high_values(self):
        """Test comparison plot with values exceeding 1.0."""
        comparison_metrics = {
            "Program A": {"reward": 0.85, "f1_score": 1.25, "accuracy": 0.92},
            "Program B": {"reward": 1.15, "f1_score": 0.88, "accuracy": 1.05},
        }
        plot_metrics_comparison(
            comparison_metrics,
            to_folder="/tmp/",
            to_file="test_comparison_high_values.png",
        )

    def test_plot_metrics_with_many_metrics(self):
        """Test plot_metrics with many metrics to verify x-axis rotation."""
        metrics = ["metric_" + str(i) for i in range(8)]  # 8 metrics to trigger rotation
        metrics_dict = self.generate_random_metrics(metrics=metrics)
        plot_metrics(metrics_dict, to_folder="/tmp/", to_file="test_many_metrics.png")

    def test_plot_metrics_comparison_with_show_values(self):
        """Test comparison plot with mean and std showing values on bars."""
        metrics = ["reward", "f1_score", "accuracy"]
        conditions = ["Program A", "Program B"]
        comparison_metrics = self.generate_comparison_metrics_with_runs(
            conditions=conditions, metrics=metrics, num_runs=4
        )
        plot_metrics_comparison_with_mean_and_std(
            comparison_metrics,
            show_values=True,
            to_folder="/tmp/",
            to_file="test_comparison_with_values.png",
        )
