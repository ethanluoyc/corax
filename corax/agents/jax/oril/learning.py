"""Learner for ORIL rewarder."""
import functools
import itertools
import time
from typing import Any, Callable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import jax
import optax

import corax
from corax import types
from corax.jax import networks as networks_lib
from corax.jax import types as jax_types
from corax.jax import utils
from corax.utils import counting
from corax.utils import loggers


class RewarderTrainingState(NamedTuple):
    params: networks_lib.Params
    optimizer_state: optax.OptState
    key: jax_types.PRNGKey
    steps: int


class TrainingState(NamedTuple):
    rewarder_state: RewarderTrainingState
    learner_state: Any


class ORILSample(NamedTuple):
    expert_sample: types.Transition
    unlabeled_sample: types.Transition
    learner_sample: types.Transition


class ORILLearner(corax.Learner):
    def __init__(
        self,
        iterator: Iterator[ORILSample],
        offline_learner_factory: Callable[[Iterator[types.Transition]], corax.Learner],
        network: networks_lib.FeedForwardNetwork,
        loss_fn,
        random_key: jax_types.PRNGKey,
        optimizer: optax.GradientTransformation,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):
        # self._iterator = iterator
        iterator, offline_rl_iterator = itertools.tee(iterator)
        self._iterator = iterator
        offline_rl_iterator = (
            self._process_sample(sample.learner_sample)
            for sample in offline_rl_iterator
        )
        self._offline_learner = offline_learner_factory(offline_rl_iterator)

        def update_step(
            state: RewarderTrainingState,
            data: Tuple[types.Transition, types.Transition],
        ):
            key, _ = jax.random.split(state.key)
            compute_loss = functools.partial(loss_fn, network.apply)
            expert_transitions, unlabaled_transitions = data
            grads, metrics = jax.grad(compute_loss, has_aux=True)(
                state.params, expert_transitions, unlabaled_transitions
            )

            (updates, opt_state) = optimizer.update(grads, state.optimizer_state)

            # Apply optimizer updates to parameters.
            params = optax.apply_updates(state.params, updates)
            new_state = RewarderTrainingState(
                params=params, optimizer_state=opt_state, steps=state.steps + 1, key=key
            )
            return new_state, metrics

        self._update_step = jax.jit(update_step)
        init_key, state_key = jax.random.split(random_key)
        params = network.init(init_key)

        self._state = RewarderTrainingState(
            params=params,
            optimizer_state=optimizer.init(params),
            key=state_key,
            steps=0,
        )

        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            "learner",
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
            steps_key=self._counter.get_steps_key(),
        )

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

        def compute_reward(params, transition):
            logits = network.apply(params, transition.observation)
            return jax.nn.sigmoid(logits)

        self._get_reward = jax.jit(compute_reward)

    def _process_sample(self, transitions):
        rewards = self._get_reward(self._state.params, transitions)
        return transitions._replace(reward=rewards)

    def step(self):
        sample = next(self._iterator)
        expert_sample = sample.expert_sample
        unlabeled_sample = sample.unlabeled_sample
        self._state, metrics = self._update_step(
            self._state, (expert_sample, unlabeled_sample)
        )
        # Update the offline learner
        self._offline_learner.step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Increment counts and record the current time.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)

        # Attempts to write the logs.
        self._logger.write({**metrics, **counts})

    def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
        rewarder_dict = {"rewarder": self._state.params}

        learner_names = [name for name in names if name not in rewarder_dict]
        learner_dict = {}
        if learner_names:
            learner_dict = dict(
                zip(learner_names, self._offline_learner.get_variables(learner_names))
            )

        variables = [
            rewarder_dict.get(name, learner_dict.get(name, None)) for name in names
        ]
        return variables

    def save(self) -> TrainingState:
        return TrainingState(
            rewarder_state=self._state, learner_state=self._offline_learner.save()
        )

    def restore(self, state: TrainingState):
        self._state = state.rewarder_state
        self._offline_learner.restore(state.learner_state)
