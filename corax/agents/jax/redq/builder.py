"""REDQ Builder."""
from typing import Callable, Iterator, List, Optional

import jax
import numpy as np
import optax
import reverb

from corax import adders
from corax import core
from corax import specs
from corax import types
from corax.adders import reverb as adders_reverb
from corax.agents.jax import actor_core as actor_core_lib
from corax.agents.jax import actors
from corax.agents.jax import builders
from corax.agents.jax import learners
from corax.agents.jax.redq import config as red_config
from corax.agents.jax.redq import learning
from corax.agents.jax.redq import networks as redq_networks
from corax.datasets import reverb as datasets
from corax.jax import networks as networks_lib
from corax.jax import utils
from corax.jax import variable_utils
from corax.utils import counting
from corax.utils import loggers

REDQNetworks = redq_networks.REDQNetworks
DemonstrationFactory = Callable[[int], Iterator[types.Transition]]


def _generate_rlpd_samples(
    demonstration_iterator: Iterator[types.Transition],
    replay_iterator: Iterator[types.Transition],
) -> Iterator[reverb.ReplaySample]:
    """Generator which creates the sample iterator for RLPD.

    Args:
      demonstration_iterator: Iterator of demonstrations.
      replay_iterator: Replay buffer sample iterator.

    Yields:
      Samples having a mix of offline and online data.
    """

    def combine(one, other):
        def combine_leaf(a, b):
            # No interleave is performed here but it is important to shuffle
            # the batch. See comments in `learning.py`.
            return np.concatenate([a, b], axis=0)

        return jax.tree_map(combine_leaf, one, other)

    while True:
        replay_transitions = types.Transition(*next(replay_iterator))
        offline_transitions = next(demonstration_iterator)
        combined = combine(offline_transitions, replay_transitions)
        yield combined


class REDQBuilder(
    builders.ActorLearnerBuilder[
        REDQNetworks,
        actor_core_lib.FeedForwardPolicy,
        reverb.ReplaySample,
    ]
):
    """REDQ Builder."""

    def __init__(
        self,
        config: red_config.REDQConfig,
        make_demonstrations: Optional[DemonstrationFactory] = None,
    ):
        self._config = config
        self._make_demonstrations = make_demonstrations

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: REDQNetworks,
        dataset: Iterator[reverb.ReplaySample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
    ) -> learners.DefaultJaxLearner:
        del replay_client
        policy_optimizer = optax.adam(self._config.actor_learning_rate)
        critic_optimizer = optax.adam(self._config.critic_learning_rate)
        temperature_optimizer = optax.adam(self._config.temperature_learning_rate)

        target_entropy = self._config.target_entropy
        if target_entropy is None:
            target_entropy = redq_networks.target_entropy_from_spec(
                environment_spec.actions
            )

        learner_core = learning.REDQLearnerCore(
            networks,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            temperature_optimizer=temperature_optimizer,
            target_entropy=target_entropy,
            discount=self._config.discount,
            tau=self._config.tau,
            init_temperature=self._config.init_temperature,
            backup_entropy=self._config.backup_entropy,
            utd_ratio=self._config.num_sgd_steps_per_step,
            reward_scale=self._config.reward_scale,
            reward_bias=self._config.reward_bias,
        )

        return learners.DefaultJaxLearner(
            learner_core, random_key, dataset, logger_fn("learner"), counter
        )

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy: actor_core_lib.FeedForwardPolicy,
        environment_spec: specs.EnvironmentSpec,
        variable_source: Optional[core.VariableSource] = None,
        adder: Optional[adders.Adder] = None,
    ) -> actors.GenericActor:
        del environment_spec
        assert variable_source is not None
        actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
        variable_client = variable_utils.VariableClient(
            variable_source, "policy", device="cpu"
        )
        return actors.GenericActor(
            actor_core, random_key, variable_client, adder, backend="cpu"
        )

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy: actor_core_lib.FeedForwardPolicy,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""
        del policy
        samples_per_insert_tolerance = (
            self._config.samples_per_insert_tolerance_rate
            * self._config.samples_per_insert
        )
        error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
        limiter = reverb.rate_limiters.SampleToInsertRatio(
            min_size_to_sample=self._config.min_replay_size,
            samples_per_insert=self._config.samples_per_insert,
            error_buffer=max(error_buffer, 2 * self._config.samples_per_insert),
        )
        return [
            reverb.Table(
                name=self._config.replay_table_name,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=self._config.max_replay_size,
                rate_limiter=limiter,
                signature=adders_reverb.NStepTransitionAdder.signature(
                    environment_spec
                ),
            )
        ]

    def make_dataset_iterator(
        self, replay_client: reverb.Client
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        sgd_steps_per_step = self._config.num_sgd_steps_per_step
        batch_size = self._config.batch_size

        if self._make_demonstrations is not None:
            offline_batch_size = int(batch_size * self._config.offline_fraction)
            online_batch_size = batch_size - offline_batch_size

            replay_iterator = datasets.make_reverb_dataset(
                table=self._config.replay_table_name,
                server_address=replay_client.server_address,
                batch_size=online_batch_size * sgd_steps_per_step,
                num_parallel_calls=max(16, 4 * jax.local_device_count()),
            ).as_numpy_iterator()
            replay_iterator = (sample.data for sample in replay_iterator)  # type: ignore

            if offline_batch_size > 0:
                offline_iterator = self._make_demonstrations(
                    offline_batch_size * sgd_steps_per_step
                )
                iterator = _generate_rlpd_samples(offline_iterator, replay_iterator)  # type: ignore
            else:
                iterator = replay_iterator

        else:
            online_batch_size = batch_size
            iterator = datasets.make_reverb_dataset(
                table=self._config.replay_table_name,
                server_address=replay_client.server_address,
                batch_size=online_batch_size * sgd_steps_per_step,
                num_parallel_calls=max(16, 4 * jax.local_device_count()),
            ).as_numpy_iterator()
            iterator = (sample.data for sample in iterator)  # type: ignore

        return utils.device_put(iterator, jax.local_devices()[0])

    def make_adder(
        self,
        replay_client: reverb.Client,
        environment_spec: Optional[specs.EnvironmentSpec],
        policy: Optional[actor_core_lib.FeedForwardPolicy],
    ) -> adders_reverb.NStepTransitionAdder:
        """Create an adder which records data generated by the actor/environment."""
        del environment_spec, policy
        return adders_reverb.NStepTransitionAdder(
            priority_fns={self._config.replay_table_name: None},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount,
        )

    def make_policy(
        self,
        networks: REDQNetworks,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool = False,
    ) -> actor_core_lib.FeedForwardPolicy:
        """Construct the policy."""
        del environment_spec

        return redq_networks.apply_policy_and_sample(networks, evaluation)
