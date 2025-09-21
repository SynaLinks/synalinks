# Modified from: keras/src/trainers/compile_utils.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from collections import namedtuple

from synalinks.src import metrics as metrics_module
from synalinks.src import ops
from synalinks.src import rewards as rewards_module
from synalinks.src import tree
from synalinks.src.backend.common import numpy
from synalinks.src.backend.common.symbolic_data_model import SymbolicDataModel
from synalinks.src.utils.naming import get_object_name
from synalinks.src.utils.tracking import Tracker


class MetricsList(metrics_module.Metric):
    def __init__(self, metrics, name="metrics_list", output_name=None):
        super().__init__(name=name)
        self.metrics = metrics
        self.output_name = output_name

    async def update_state(self, y_true, y_pred):
        for m in self.metrics:
            await m.update_state(y_true, y_pred)

    def reset_state(self):
        for m in self.metrics:
            m.reset_state()

    def get_result(self):
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError


def is_function_like(value):
    if value is None:
        return True
    if isinstance(value, str):
        return True
    if callable(value):
        return True
    return False


def get_metric(identifier, y_true, y_pred):
    if identifier is None:
        return None  # Ok to have no metric for an output.

    metric_obj = metrics_module.get(identifier)

    if isinstance(identifier, str):
        metric_name = identifier
    else:
        metric_name = get_object_name(metric_obj)

    if not isinstance(metric_obj, metrics_module.Metric):
        metric_obj = metrics_module.MeanMetricWrapper(metric_obj)

    metric_obj.name = metric_name
    return metric_obj


def get_reward(identifier, y_true, y_pred):
    if identifier is None:
        return None  # Ok to have no reward for an output.

    reward_obj = rewards_module.get(identifier)

    if not isinstance(reward_obj, rewards_module.Reward):
        if isinstance(identifier, str):
            reward_name = identifier
        else:
            reward_name = get_object_name(reward_obj)
        reward_obj = rewards_module.RewardFunctionWrapper(reward_obj, name=reward_name)
    return reward_obj


class CompileMetrics(metrics_module.Metric):
    def __init__(
        self,
        metrics,
        name="compile_metric",
        output_names=None,
    ):
        super().__init__(name=name)
        if metrics and not isinstance(metrics, (list, tuple, dict)):
            raise ValueError(
                "Expected `metrics` argument to be a list, tuple, or dict. "
                f"Received instead: metrics={metrics} of type {type(metrics)}"
            )
        self._user_metrics = metrics
        self.built = False
        self.name = "compile_metrics"
        self.output_names = output_names

    @property
    def metrics(self):
        if not self.built:
            return []
        metrics = []
        for m in self._flat_metrics:
            if isinstance(m, MetricsList):
                metrics.extend(m.metrics)
            elif m is not None:
                metrics.append(m)
        return metrics

    @property
    def variables(self):
        # Avoiding relying on implicit tracking since
        # CompileMetrics may be instantiated or built in a no tracking scope.
        if not self.built:
            return []
        vars = []
        for m in self.metrics:
            if m is not None:
                vars.extend(m.variables)
        return vars

    def build(self, y_true, y_pred):
        num_outputs = 1  # default
        if self.output_names:
            output_names = self.output_names
        elif isinstance(y_pred, dict):
            output_names = sorted(list(y_pred.keys()))
        elif isinstance(y_pred, (list, tuple)):
            num_outputs = len(y_pred)
            if all(hasattr(x, "_synalinks_history") for x in y_pred):
                output_names = [x._synalinks_history.operation.name for x in y_pred]
            else:
                output_names = None
        else:
            output_names = None
        if output_names:
            num_outputs = len(output_names)

        y_pred = self._flatten_y(y_pred)
        y_true = self._flatten_y(y_true)

        metrics = self._user_metrics
        self._flat_metrics = self._build_metrics_set(
            metrics,
            num_outputs,
            output_names,
            y_true,
            y_pred,
            argument_name="metrics",
        )
        self.built = True

    def _build_metrics_set(
        self, metrics, num_outputs, output_names, y_true, y_pred, argument_name
    ):
        flat_metrics = []
        if isinstance(metrics, dict):
            for name in metrics.keys():
                if name not in output_names:
                    raise ValueError(
                        f"In the dict argument `{argument_name}`, key "
                        f"'{name}' does not correspond to any program "
                        f"output. Received:\n{argument_name}={metrics}"
                    )
        if num_outputs == 1:
            if not metrics:
                flat_metrics.append(None)
            else:
                if isinstance(metrics, dict):
                    metrics = tree.flatten(metrics)
                if not isinstance(metrics, list):
                    metrics = [metrics]
                if not all(is_function_like(m) for m in metrics):
                    raise ValueError(
                        f"Expected all entries in the `{argument_name}` list "
                        f"to be metric objects. Received instead:\n"
                        f"{argument_name}={metrics}"
                    )
                flat_metrics.append(
                    MetricsList(
                        [
                            get_metric(m, y_true[0], y_pred[0])
                            for m in metrics
                            if m is not None
                        ]
                    )
                )
        else:
            if isinstance(metrics, (list, tuple)):
                if len(metrics) != len(y_pred):
                    raise ValueError(
                        "For a program with multiple outputs, "
                        f"when providing the `{argument_name}` argument as a "
                        "list, it should have as many entries as the program has "
                        f"outputs. Received:\n{argument_name}={metrics}\nof "
                        f"length {len(metrics)} whereas the program has "
                        f"{len(y_pred)} outputs."
                    )
                for idx, (mls, yt, yp) in enumerate(zip(metrics, y_true, y_pred)):
                    if not isinstance(mls, list):
                        mls = [mls]
                    name = output_names[idx] if output_names else None
                    if not all(is_function_like(e) for e in mls):
                        raise ValueError(
                            f"All entries in the sublists of the "
                            f"`{argument_name}` list should be metric objects. "
                            f"Found the following sublist with unknown "
                            f"types: {mls}"
                        )
                    flat_metrics.append(
                        MetricsList(
                            [get_metric(m, yt, yp) for m in mls if m is not None],
                            output_name=name,
                        )
                    )
            elif isinstance(metrics, dict):
                if output_names is None:
                    raise ValueError(
                        f"Argument `{argument_name}` can only be provided as a "
                        "dict when the program also returns a dict of outputs. "
                        f"Received {argument_name}={metrics}"
                    )
                for name in metrics.keys():
                    if not isinstance(metrics[name], list):
                        metrics[name] = [metrics[name]]
                    if not all(is_function_like(e) for e in metrics[name]):
                        raise ValueError(
                            f"All entries in the sublists of the "
                            f"`{argument_name}` dict should be metric objects. "
                            f"At key '{name}', found the following sublist "
                            f"with unknown types: {metrics[name]}"
                        )
                for name, yt, yp in zip(output_names, y_true, y_pred):
                    if name in metrics:
                        flat_metrics.append(
                            MetricsList(
                                [
                                    get_metric(m, yt, yp)
                                    for m in metrics[name]
                                    if m is not None
                                ],
                                output_name=name,
                            )
                        )
                    else:
                        flat_metrics.append(None)
        return flat_metrics

    def _flatten_y(self, y):
        if isinstance(y, dict) and self.output_names:
            result = []
            for name in self.output_names:
                if name in y:
                    result.append(y[name])
            return result
        return tree.flatten(y)

    async def update_state(self, y_true, y_pred):
        if not self.built:
            self.build(y_true, y_pred)
        y_true = self._flatten_y(y_true)
        y_pred = self._flatten_y(y_pred)
        for m, y_t, y_p in zip(self._flat_metrics, y_true, y_pred):
            if m is not None:
                await m.update_state(y_t, y_p)

    def reset_state(self):
        if not self.built:
            return
        for m in self._flat_metrics:
            if m:
                m.reset_state()

    def result(self):
        if not self.built:
            raise ValueError(
                "Cannot get result() since the metric has not yet been built."
            )
        results = {}
        unique_name_counters = {}
        for mls in self._flat_metrics:
            if not mls:
                continue
            for m in mls.metrics:
                name = m.name
                if mls.output_name:
                    name = f"{mls.output_name}_{name}"
                if name not in unique_name_counters:
                    results[name] = m.result()
                    unique_name_counters[name] = 1
                else:
                    index = unique_name_counters[name]
                    unique_name_counters[name] += 1
                    name = f"{name}_{index}"
                    results[name] = m.result()

        return results

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError


class CompileReward(rewards_module.Reward):
    Reward = namedtuple("Reward", ["path", "reward", "reward_weights", "name"])

    def __init__(
        self,
        reward,
        reward_weights=None,
        reduction="mean",
        output_names=None,
    ):
        if reward_weights and not isinstance(reward_weights, (list, tuple, dict, float)):
            raise ValueError(
                "Expected `reward_weights` argument to be a float "
                "(single output case) or a list, tuple, or "
                "dict (multiple output case). "
                f"Received instead: reward_weights={reward_weights} "
                f"of type {type(reward_weights)}"
            )
        self._user_reward = reward
        self._user_reward_weights = reward_weights
        self.built = False
        self.output_names = output_names
        super().__init__(name="compile_reward", reduction=reduction)

        # Use `Tracker` to track metrics for individual rewards.
        self._metrics = []
        self._tracker = Tracker(
            {
                "metrics": (
                    lambda x: isinstance(x, metrics_module.Metric),
                    self._metrics,
                )
            }
        )
        self._flat_rewards = None
        self._y_pred_build_structure = None
        self._y_true_build_structure = None

    @property
    def metrics(self):
        return self._metrics

    @property
    def variables(self):
        vars = []
        for m in self.metrics:
            vars.extend(m.variables)
        return vars

    def _build_nested(self, y_true, y_pred, reward, output_names, current_path):
        flat_y_pred = tree.flatten(y_pred)
        if not tree.is_nested(reward):
            _reward = reward.reward
            if _reward is None:
                return
            reward_weight = reward.weight
            resolved_reward = get_reward(_reward, y_true, y_pred)
            name_path = current_path
            if not tree.is_nested(output_names):
                if output_names is not None:
                    output_name = output_names
                else:
                    output_name = resolved_reward.name
                if len(name_path) == 0:
                    name_path = (output_name,)
                elif isinstance(name_path[-1], int):
                    name_path = name_path[:-1] + (output_name,)
            name = "/".join([str(path) for path in name_path])
            if name == "":
                if isinstance(output_names, dict):
                    flat_output_names = list(output_names.keys())
                else:
                    flat_output_names = tree.flatten(output_names)
                name = "_".join(flat_output_names)
            self._flat_rewards.append(
                CompileReward.Reward(current_path, resolved_reward, reward_weight, name)
            )
            return
        elif (
            issubclass(type(reward), (list, tuple))
            and all([not tree.is_nested(_reward) for _reward in reward])
            and len(reward) == len(flat_y_pred)
        ):
            reward = tree.pack_sequence_as(y_pred, reward)
        elif issubclass(type(reward), (list, tuple)) and not isinstance(
            y_pred, type(reward)
        ):
            for _reward in reward:
                self._build_nested(
                    y_true,
                    y_pred,
                    _reward,
                    output_names,
                    current_path,
                )
            return

        if not tree.is_nested(reward):
            return self._build_nested(y_true, y_pred, reward, output_names, current_path)

        if not isinstance(reward, type(y_pred)):
            raise KeyError(
                f"The path: {current_path} in "
                "the `reward` argument, can't be found in "
                "the program's output (`y_pred`)."
            )

        # shallow traverse the reward config
        if isinstance(reward, dict):
            iterator = reward.items()

            def key_check_fn(key, objs):
                return all([isinstance(obj, dict) and key in obj for obj in objs])

        elif issubclass(type(reward), (list, tuple)):
            iterator = enumerate(reward)

            def key_check_fn(key, objs):
                return all(
                    [
                        issubclass(type(obj), (list, tuple)) and key < len(obj)
                        for obj in objs
                    ]
                )

        else:
            raise TypeError(
                f"Unsupported type {type(reward)} in the `reward` configuration."
            )

        for key, _reward in iterator:
            if _reward is None:
                continue
            if not key_check_fn(key, (y_true, y_pred)):
                raise KeyError(
                    f"The path: {current_path + (key,)} in "
                    "the `reward` argument, can't be found in "
                    "either the program's output (`y_pred`) or in the "
                    "labels (`y_true`)."
                )

            self._build_nested(
                y_true[key],
                y_pred[key],
                _reward,
                output_names[key],
                current_path + (key,),
            )

    def build(self, y_true, y_pred):
        reward = self._user_reward
        reward_weights = self._user_reward_weights
        flat_output_names = self.output_names
        if (
            self.output_names
            and isinstance(self._user_reward, dict)
            and not isinstance(y_pred, dict)
        ):
            if set(self.output_names) == set(self._user_reward.keys()):
                reward = [self._user_reward[name] for name in self.output_names]
                if isinstance(self._user_reward_weights, dict):
                    reward_weights = [
                        self._user_reward_weights[name] for name in self.output_names
                    ]
            else:
                raise ValueError(
                    f"Expected keys {self.output_names} in reward dict, but "
                    f"found reward.keys()={list(self._user_reward.keys())}"
                )

        # Pytree leaf container
        class WeightedReward:
            def __new__(cls, reward, weight):
                if reward is None:
                    return None
                return object.__new__(cls)

            def __init__(self, reward, weight):
                self.reward = reward
                self.weight = weight

        # pack the rewards and the weights together
        if reward_weights is not None:
            try:
                tree.assert_same_structure(reward, reward_weights)
            except ValueError:
                flat_reward_weights = tree.flatten(reward_weights)
                if len(tree.flatten(reward)) != len(flat_reward_weights):
                    raise ValueError(
                        f"`reward_weights` must match the number of rewards, "
                        f"got {len(tree.flatten(reward))} rewards "
                        f"and {len(reward_weights)} weights."
                    )
                reward_weights = tree.pack_sequence_as(reward, flat_reward_weights)
            reward = tree.map_structure(
                lambda _reward, _weight: WeightedReward(_reward, _weight),
                reward,
                reward_weights,
            )
        else:
            reward = tree.map_structure(
                lambda _reward: WeightedReward(_reward, None), reward
            )

        self._flat_rewards = []

        if (
            isinstance(reward, dict)
            and issubclass(type(y_pred), (list, tuple))
            and set(reward.keys()) == set(flat_output_names)
            and len(y_pred) == len(flat_output_names)
        ):
            y_pred = {name: y_p for name, y_p in zip(flat_output_names, y_pred)}
            y_true = {name: y_t for name, y_t in zip(flat_output_names, y_true)}
        elif (
            isinstance(reward, dict)
            and not tree.is_nested(y_pred)
            and set(reward.keys()) == set(flat_output_names)
            and len(flat_output_names) == 1
        ):
            y_pred = {name: y_p for name, y_p in zip(flat_output_names, [y_pred])}
            y_true = {name: y_t for name, y_t in zip(flat_output_names, [y_true])}

        try:
            output_names = tree.pack_sequence_as(y_pred, flat_output_names)
        except:
            inferred_flat_output_names = self._get_y_pred_output_names(y_pred)
            output_names = tree.pack_sequence_as(y_pred, inferred_flat_output_names)

        if not tree.is_nested(reward):
            reward = tree.map_structure(lambda x: reward, y_pred)

        self._build_nested(y_true, y_pred, reward, output_names, ())

        # Add `Mean` metric to the tracker for each reward.
        if len(self._flat_rewards) > 1:
            for _reward in self._flat_rewards:
                name = _reward.name + "_reward"
                self._tracker.add_to_store("metrics", metrics_module.Mean(name=name))

        self._y_pred_build_structure = tree.map_structure(lambda x: None, y_pred)
        self._y_true_build_structure = tree.map_structure(lambda x: None, y_true)
        self.built = True

    def _get_y_pred_output_names(self, y_pred):
        flat_y_pred = tree.flatten(y_pred)
        if all((isinstance(x, SymbolicDataModel) for x in flat_y_pred)):
            output_names = []
            for data_model in flat_y_pred:
                if hasattr(data_model, "_synalinks_history"):
                    output_names.append(data_model._synalinks_history.operation.name)
                else:
                    output_names.append(data_model.name)
        else:
            output_names = [None] * len(flat_y_pred)
        return output_names

    async def __call__(self, y_true, y_pred):
        with ops.name_scope(self.name):
            return await self.call(y_true, y_pred)

    async def call(self, y_true, y_pred):
        if not tree.is_nested(y_true) and not tree.is_nested(y_pred):
            # Fast path: single output case / no reward-tracking metric.
            if not self.built:
                self.build(y_true, y_pred)
            _, reward_fn, _, _ = self._flat_rewards[0]
            reward_value = numpy.convert_to_tensor(await reward_fn(y_true, y_pred))
            # if reward_weight is not None:
            #     reward_value = numpy.multiply(reward_value, reward_weight)
            return reward_value

        try:
            tree.assert_same_structure(y_pred, y_true)
        except ValueError:
            # Check case where y_true is either flat or leaf
            if (
                not tree.is_nested(y_true)
                and hasattr(y_pred, "__len__")
                and len(y_pred) == 1
            ):
                y_true = [y_true]

            # Check case where y_pred is list/tuple and y_true is dict
            elif isinstance(y_pred, (list, tuple)) and isinstance(y_true, dict):
                if set(self.output_names) == set(y_true.keys()):
                    y_true = [y_true[name] for name in self.output_names]

            try:
                y_true = tree.pack_sequence_as(y_pred, y_true)
            except:
                # Check case where y_true has the same structure but uses
                # different (but reconcilable) container types,
                # e.g `list` vs `tuple`.
                try:
                    tree.assert_same_paths(y_true, y_pred)
                    y_true = tree.pack_sequence_as(y_pred, tree.flatten(y_true))
                except:
                    try:
                        # Check case where reward is partially defined over y_pred
                        flat_y_true = tree.flatten(y_true)
                        flat_reward = tree.flatten(self._user_reward)
                        flat_reward_non_nones = [
                            (i, reward)
                            for i, reward in enumerate(flat_reward)
                            if reward is not None
                        ]
                        assert len(flat_y_true) == len(flat_reward_non_nones)
                        y_true = [None] * len(flat_reward)
                        for y_t, (i, reward) in zip(flat_y_true, flat_reward_non_nones):
                            y_true[i] = y_t
                        y_true = tree.pack_sequence_as(self._user_reward, y_true)
                    except:
                        y_true_struct = tree.map_structure(lambda _: "*", y_true)
                        y_pred_struct = tree.map_structure(lambda _: "*", y_pred)
                        raise ValueError(
                            "y_true and y_pred have different structures.\n"
                            f"y_true: {y_true_struct}\n"
                            f"y_pred: {y_pred_struct}\n"
                        )

        if not self.built:
            self.build(y_true, y_pred)
        try:
            tree.assert_same_structure(self._y_pred_build_structure, y_pred)
        except ValueError:
            y_pred = tree.pack_sequence_as(
                self._y_pred_build_structure, tree.flatten(y_pred)
            )
        try:
            tree.assert_same_structure(self._y_true_build_structure, y_true)
        except ValueError:
            y_true = tree.pack_sequence_as(
                self._y_true_build_structure, tree.flatten(y_true)
            )
        # We need to add a dummy `None` if the program has only a single output.
        metrics = [None] if len(self.metrics) == 0 else self.metrics

        # Iterate all rewards in flat form.
        reward_values = []

        def resolve_path(path, object):
            for _path in path:
                object = object[_path]
            return object

        for (path, reward_fn, reward_weight, _), metric in zip(
            self._flat_rewards, metrics
        ):
            y_t, y_p = resolve_path(path, y_true), resolve_path(path, y_pred)

            value = numpy.convert_to_tensor(reward_fn(y_t, y_p))
            # Record *unweighted* individual rewards.
            if metric:
                metric.update_state(value)
            reward_values.append(value)

        if reward_values:
            total_reward = sum(reward_values)
            return total_reward
        return None

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
