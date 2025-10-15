# Modified from: keras/src/ops/optimizer.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import random
import warnings

import docstring_parser
import numpy as np

from synalinks.src import backend
from synalinks.src.backend import DataModel
from synalinks.src.backend import Trainable
from synalinks.src.backend.common.json_utils import out_mask_json
from synalinks.src.initializers import Empty
from synalinks.src.metrics import Metric
from synalinks.src.modules import Module
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils import tracking
from synalinks.src.utils.naming import auto_name


class Iterations(DataModel):
    iterations: int = 0
    epochs: int = 0


class Optimizer(SynalinksSaveable):
    """Optimizer base class: all Synalinks optimizers inherit from this class.

    This abstract base class provides the common infrastructure for all optimizers in Synalinks.

    Concrete optimizer implementations must inherit from this class and implement
    the `propose_new_candidates()` method with their specific optimization logic.

    Args:
        population_size (int): The maximum number of best candidates to keep
            during the optimization process.
        name (str): Optional. The name of the optimizer.
        description (str): Optional. The description of the optimizer.
    """

    def __init__(
        self,
        merging_rate=0.02,
        population_size=10,
        name=None,
        description=None,
        **kwargs,
    ):
        """Initialize the base optimizer.

        Sets up the optimizer's internal state, variable tracking, and naming.

        Args:
            population_size (int): The maximum number of best candidates to keep
                during the optimization process.
            name (str): Optional name for the optimizer instance
            description (str): Optional description for the optimizer
            **kwargs (keyword params): Additional arguments (will raise error if provided)

        Raises:
            ValueError: If unexpected keyword arguments are provided
        """
        self._lock = False

        if kwargs:
            raise ValueError(f"Argument(s) not recognized: {kwargs}")

        self.merging_rate = merging_rate
        self.population_size = population_size

        if name is None:
            name = auto_name(self.__class__.__name__)
        self.name = name

        if description is None:
            if self.__class__.__doc__:
                description = docstring_parser.parse(
                    self.__class__.__doc__
                ).short_description
            else:
                description = ""
        self.description = description

        self.built = False
        self._program = None

        self._initialize_tracker()

        with backend.name_scope(self.name, caller=self):
            iterations = backend.Variable(
                initializer=Empty(data_model=Iterations),
                data_model=Iterations,
                trainable=False,
                name="iterations_"+self.name
            )
        self._iterations = iterations

    @property
    def iterations(self):
        """Get the current iteration count.

        Returns:
            (int): Number of optimization iterations performed
        """
        return self._iterations.get("iterations")

    @property
    def epochs(self):
        """Get the current epoch number.

        Returns:
            (int): Number of epochs performed
        """
        return self._iterations.get("epochs")

    def increment_iterations(self):
        """Increment the iteration counter by 1.

        This method is called after each optimization step to track progress.
        """
        iterations = self._iterations.get("iterations")
        self._iterations.update({"iterations": iterations + 1})

    def increment_epochs(self):
        """Increment the epoch counter by 1.

        This method is called after each epoch step to track progress.
        """
        iterations = self._iterations.get("epochs")
        self._iterations.update({"epochs": iterations + 1})

    def set_program(self, program):
        """Set the program that this optimizer will optimize.

        The program contains the model/pipeline that the optimizer will work on.

        Args:
            program (Program): The Synalinks program to optimize
        """
        self._program = program

    @property
    def program(self):
        """Get the program associated with this optimizer.

        Returns:
            (Program): The Synalinks program being optimized, or None if not set
        """
        return self._program

    @property
    def reward_tracker(self):
        """Get the reward tracker from the associated program.

        The reward tracker monitors the performance/rewards during optimization.

        Returns:
            (RewardTracker): The reward tracker from the program, or None if no program is set
        """
        if self._program:
            return self._program._reward_tracker
        return None

    @tracking.no_automatic_dependency_tracking
    def _initialize_tracker(self):
        if hasattr(self, "_tracker"):
            return

        trainable_variables = []
        non_trainable_variables = []
        modules = []
        self._tracker = tracking.Tracker(
            {
                "trainable_variables": (
                    lambda x: isinstance(x, backend.Variable) and x.trainable,
                    trainable_variables,
                ),
                "non_trainable_variables": (
                    lambda x: isinstance(x, backend.Variable) and not x.trainable,
                    non_trainable_variables,
                ),
                "modules": (
                    lambda x: isinstance(x, Module) and not isinstance(x, Metric),
                    modules,
                ),
            },
            exclusions={"non_trainable_variables": ["trainable_variables"]},
        )
        self._trainable_variables = trainable_variables
        self._non_trainable_variables = non_trainable_variables
        self._modules = modules

    def __setattr__(self, name, value):
        # Track Variables, Modules, Metrics.
        if name != "_tracker":
            if not hasattr(self, "_tracker"):
                self._initialize_tracker()
            value = self._tracker.track(value)
        return super().__setattr__(name, value)

    @property
    def variables(self):
        return self._non_trainable_variables[:] + self._trainable_variables[:]

    @property
    def non_trainable_variables(self):
        return self._non_trainable_variables[:]

    @property
    def trainable_variables(self):
        variables = []
        for module in self._modules:
            variables.extend(module.trainable_variables)
        return variables

    def save_own_variables(self, store):
        """Get the state of this optimizer object."""
        for i, variable in enumerate(self.variables):
            store[str(i)] = variable.numpy()

    def load_own_variables(self, store):
        """Set the state of this optimizer object."""
        if len(store.keys()) != len(self.variables):
            msg = (
                f"Skipping variable loading for optimizer '{self.name}', "
                f"because it has {len(self.variables)} variables whereas "
                f"the saved optimizer has {len(store.keys())} variables. "
            )
            if len(self.variables) == 0:
                msg += (
                    "This is likely because the optimizer has not been called/built yet."
                )
            warnings.warn(msg, stacklevel=2)
            return
        for i, variable in enumerate(self.variables):
            variable.assign(store[str(i)])

    def _check_super_called(self):
        if not hasattr(self, "_lock"):
            raise RuntimeError(
                f"In optimizer '{self.__class__.__name__}', you forgot to call "
                "`super().__init__()` as the first statement "
                "in the `__init__()` method. "
                "Go add it!"
            )

    async def select_variable_name_to_update(self, trainable_variables):
        rewards = []
        for trainable_variable in trainable_variables:
            nb_visit = trainable_variable.get("nb_visit")
            cumulative_reward = trainable_variable.get("cumulative_reward")
            if nb_visit == 0:
                variable_reward = 100000
            else:
                variable_reward = cumulative_reward / nb_visit
            rewards.append(variable_reward)
        rewards = np.array(rewards)
        inverted_rewards = -rewards
        scaled_rewards = inverted_rewards / self.sampling_temperature
        exp_rewards = np.exp(scaled_rewards - np.max(scaled_rewards))
        probabilities = exp_rewards / np.sum(exp_rewards)
        selected_variable = np.random.choice(
            trainable_variables,
            size=1,
            replace=False,
            p=probabilities,
        ).tolist()[0]
        return selected_variable.name

    async def select_evolving_strategy(self):
        rand = random.random()
        if rand > (self.merging_rate * self.epochs):
            return "mutation"
        else:
            return "crossover"

    async def select_candidate_to_merge(
        self,
        step,
        trainable_variable,
    ):
        best_candidates = trainable_variable.get("best_candidates")
        if len(best_candidates) > 0:
            selected_candidate = random.choice(best_candidates)
            return selected_candidate
        return None

    async def on_train_begin(
        self,
        trainable_variables,
    ):
        """Called at the beginning of the training

        Args:
            trainable_variables (list): The list of trainable variables.
        """
        mask = list(Trainable.keys())
        mask.remove("examples")

        for trainable_variable in trainable_variables:
            seed_candidates = trainable_variable.get("seed_candidates")
            masked_variable = out_mask_json(
                trainable_variable.get_json(),
                mask=mask,
            )
            seed_candidates.append(
                {
                    **masked_variable,
                }
            )

    async def on_train_end(
        self,
        trainable_variables,
    ):
        """Called at the end of the training

        Args:
            trainable_variables (list): The list of trainable variables
        """
        for variable in trainable_variables:
            candidates = variable.get("candidates")
            best_candidates = variable.get("best_candidates")
            all_candidates = candidates + best_candidates
            sorted_candidates = sorted(
                all_candidates,
                key=lambda x: x.get("reward"),
                reverse=True,
            )
            best_candidate = sorted_candidates[0]
            best_candidate = out_mask_json(
                best_candidate,
                mask=["reward"],
            )
            variable.update(
                {
                    **best_candidate,
                },
            )

    async def on_epoch_begin(
        self,
        epoch,
        trainable_variables,
    ):
        """Called at the beginning of an epoch

        Args:
            epoch (int): The epoch number
            trainable_variables (list): The list of trainable variables
        """
        for trainable_variable in trainable_variables:
            trainable_variable.update(
                {
                    "predictions": [],
                    "candidates": [],
                }
            )

    async def on_epoch_end(
        self,
        epoch,
        trainable_variables,
    ):
        """Called at the end of an epoch

        Args:
            epoch (int): The epoch number
            trainable_variables (list): The list of trainable variables
        """
        mask = list(Trainable.keys())
        mask.remove("examples")
        
        for trainable_variable in trainable_variables:
            candidates = trainable_variable.get("candidates")
            best_candidates = trainable_variable.get("best_candidates")
            all_candidates = candidates + best_candidates
            sorted_candidates = sorted(
                all_candidates,
                key=lambda x: x.get("reward"),
                reverse=True,
            )
            selected_candidates = sorted_candidates[: self.population_size]
            trainable_variable.update(
                {
                    "best_candidates": selected_candidates,
                }
            )
            best_candidate = selected_candidates[0]
            best_candidate = out_mask_json(
                best_candidate,
                mask=["reward"],
            )
            trainable_variable.update(
                {
                    **best_candidate,
                },
            )
            history = trainable_variable.get("history")
            if len(history) > 0:
                last_candidate = history[-1]
                if last_candidate != best_candidate:
                    history.append(best_candidate)
            else:
                history.append(best_candidate)
        self.increment_epochs()

    async def on_batch_begin(
        self,
        step,
        epoch,
        trainable_variables,
    ):
        """Called at the beginning of a batch

        Args:
            step (int): The batch number
            epoch (int): The epoch number
            trainable_variables (list): The list of trainable variables
        """
        for trainable_variable in trainable_variables:
            best_candidates = trainable_variable.get("best_candidates")
            if epoch == 0:
                seed_candidates = trainable_variable.get("seed_candidates")
                if len(seed_candidates) > 0:
                    seed_candidate = random.choice(seed_candidates)
                    trainable_variable.update(
                        {
                            **seed_candidate,
                        },
                    )
            else:
                if len(best_candidates) > 0:
                    best_candidate = random.choice(best_candidates)
                    best_candidate = out_mask_json(
                        best_candidate,
                        mask=["reward"],
                    )
                    trainable_variable.update(
                        {
                            **best_candidate,
                        },
                    )
                else:
                    seed_candidates = trainable_variable.get("seed_candidates")
                    if len(seed_candidates) > 0:
                        seed_candidate = random.choice(seed_candidates)
                        trainable_variable.update(
                            {
                                **seed_candidate,
                            },
                        )
            trainable_variable.update(
                {
                    "nb_visit": 0,
                    "cumulative_reward": 0.0,
                },
            )

    async def on_batch_end(
        self,
        step,
        epoch,
        trainable_variables,
    ):
        """Called at the end of a batch

        Args:
            step (int): The batch number
            epoch (int): The epoch number
            trainable_variables (list): The list of trainable variables
        """
        for trainable_variable in trainable_variables:
            candidates = trainable_variable.get("candidates")
            best_candidates = trainable_variable.get("best_candidates")
            all_candidates = candidates + best_candidates
            if len(best_candidates) > 0:
                sorted_candidates = sorted(
                    all_candidates,
                    key=lambda x: x.get("reward"),
                    reverse=True,
                )
                best_candidate = sorted_candidates[0]
                best_candidate = out_mask_json(
                    best_candidate,
                    mask=["reward"],
                )
                trainable_variable.update(
                    {
                        **best_candidate,
                    },
                )
        self.increment_iterations()

    async def optimize(
        self,
        step,
        trainable_variables,
        x=None,
        y=None,
        val_x=None,
        val_y=None,
    ):
        """Method for performing optimization.

        Args:
            step (int): The training step.
            trainable_variables (list): Variables to be optimized
            x (np.ndarray): Training batch input data. Must be array-like.
            y (np.ndarray): Training batch target data. Must be array-like.
            val_x (np.ndarray): Input validation data. Must be array-like.
            val_y (np.ndarray): Target validation data. Must be array-like.
        """
        self._check_super_called()
        if not self.built:
            await self.build(trainable_variables)

        y_pred = await self.program.predict_on_batch(
            x=x,
            training=True,
        )

        reward = await self.program.compute_reward(
            x=x,
            y=y,
            y_pred=y_pred,
        )

        await self.assign_reward_to_predictions(
            trainable_variables,
            reward=reward,
        )

        await self.propose_new_candidates(
            step,
            trainable_variables,
            x=x,
            y=y,
            y_pred=y_pred,
        )

        y_pred = await self.program.predict_on_batch(
            x=val_x,
            training=False,
        )

        reward = await self.program.compute_reward(
            x=val_x,
            y=val_y,
            y_pred=y_pred,
        )

        for trainable_variable in trainable_variables:
            await self.maybe_add_candidate(
                step,
                trainable_variable,
                reward=reward,
            )

        await self.reward_tracker.update_state(reward)
        metrics = await self.program.compute_metrics(val_x, val_y, y_pred)
        return metrics

    async def propose_new_candidates(
        self,
        step,
        trainable_variables,
        x=None,
        y=None,
        y_pred=None,
    ):
        raise NotImplementedError(
            "Optimizer subclasses must implement the `propose_new_candidates()` method."
        )

    async def assign_reward_to_predictions(
        self,
        trainable_variables,
        reward=None,
    ):
        """Assign rewards to predictions that don't have them yet.

        This method updates all predictions in trainable variables that have
        None as their reward value. It's typically called after computing
        rewards for a batch of predictions.

        Args:
            trainable_variables (list): Variables containing predictions
            reward (float): Reward value to assign (defaults to 0.0 if None/False)
        """
        if not reward:
            reward = 0.0
        for trainable_variable in trainable_variables:
            current_predictions = trainable_variable.get("current_predictions")
            predictions = trainable_variable.get("predictions")
            for p in current_predictions:
                if p["reward"] is None:
                    p["reward"] = reward
                    nb_visit = trainable_variable.get("nb_visit")
                    cumulative_reward = trainable_variable.get("cumulative_reward")
                    trainable_variable.update(
                        {
                            "nb_visit": nb_visit + 1,
                            "cumulative_reward": cumulative_reward + reward,
                        }
                    )
            trainable_variable.update(
                {
                    "predictions": predictions + current_predictions,
                    "current_predictions": [],
                }
            )

    async def assign_candidate(
        self,
        trainable_variable,
        new_candidate=None,
        examples=None,
    ):
        """Assign a new candidate configuration to a trainable variable.

        This method updates a variable with either a complete new candidate
        or just new examples for few-shot learning.

        Args:
            trainable_variable (Variable): The variable to update
            new_candidate (JsonDataModel): New candidate (optional)
            examples (list): New examples for few-shot learning (optional)
        """
        if new_candidate:
            if examples:
                # Update with both new candidate and examples
                trainable_variable.update(
                    {
                        **new_candidate.get_json(),
                        "examples": examples,
                    },
                )
            else:
                # Update with just new candidate
                trainable_variable.update(
                    {
                        **new_candidate.get_json(),
                    },
                )
        elif examples:
            # Update with just new examples
            trainable_variable.update(
                {
                    "examples": examples,
                },
            )

    async def maybe_add_candidate(
        self,
        step,
        trainable_variable,
        new_candidate=None,
        examples=None,
        reward=None,
    ):
        """Maybe add new candidate to candidates.

        Args:
            step (int): The training step.
            trainable_variable (Variable): The variable to add candidate to.
            new_candidate (dict): New candidate configuration (optional).
            examples (list): New examples for few-shot learning (optional).
            reward (float): The candidate reward.
        """
        if not reward:
            reward = 0.0
        mask = list(Trainable.keys())
        mask.append("reward")
        if new_candidate:
            new_candidate = out_mask_json(
                new_candidate.get_json(),
                mask=mask,
            )
        else:
            new_candidate = out_mask_json(
                trainable_variable.get_json(),
                mask=mask,
            )
        if not examples:
            examples = trainable_variable.get("examples")

        candidates = trainable_variable.get("candidates")
        best_candidates = trainable_variable.get("best_candidates")
        all_candidates = best_candidates + candidates
        is_present = False
        for candidate in all_candidates:
            if out_mask_json(candidate, mask=mask) == new_candidate:
                is_present = True
                break
        if not is_present:
            candidates.append(
                {
                    **new_candidate,
                    "examples": examples,
                    "reward": reward,
                }
            )

    def get_config(self):
        return {
            "merging_rate": self.merging_rate,
            "population_size": self.population_size,
            "name": self.name,
            "description": self.description,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __repr__(self):
        return f"<Optimizer name={self.name} description={self.description}>"
