"""DrQ-v2 builder"""
from typing import Iterator, List, Optional

import jax
import optax
import reverb
from reverb import rate_limiters

from corax import adders
from corax import core
from corax import specs
from corax.adders import reverb as adders_reverb
from corax.agents.jax import actor_core as actor_core_lib
from corax.agents.jax import actors
from corax.agents.jax import builders
from corax.agents.jax.drq_v2 import config as drq_v2_config
from corax.agents.jax.drq_v2 import learning as learning_lib
from corax.agents.jax.drq_v2 import networks as drq_v2_networks
from corax.datasets import reverb as reverb_datasets
from corax.jax import networks as networks_lib
from corax.jax import utils
from corax.jax import variable_utils
from corax.utils import counting
from corax.utils import loggers


class DrQV2Builder(builders.ActorLearnerBuilder):
    """DrQ-v2 Builder."""

    def __init__(self, config: drq_v2_config.DrQV2Config):
        self._config = config

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
        limiter = rate_limiters.SampleToInsertRatio(
            min_size_to_sample=self._config.min_replay_size,
            samples_per_insert=self._config.samples_per_insert,
            error_buffer=error_buffer,
        )
        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=adders_reverb.NStepTransitionAdder.signature(
                environment_spec=environment_spec
            ),
        )
        return [replay_table]

    def make_dataset_iterator(
        self, replay_client: reverb.Client
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        dataset = reverb_datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=replay_client.server_address,
            batch_size=(self._config.batch_size * self._config.num_sgd_steps_per_step),
            num_parallel_calls=max(12, 4 * jax.local_device_count()),
        )
        iterator = utils.device_put(dataset.as_numpy_iterator(), jax.devices()[0])
        return iterator

    def make_adder(
        self,
        replay_client: reverb.Client,
        environment_spec: specs.EnvironmentSpec,
        policy: actor_core_lib.FeedForwardPolicy,
    ) -> Optional[adders.Adder]:
        del environment_spec, policy
        return adders_reverb.NStepTransitionAdder(
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount,
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
        device = "cpu"
        variable_client = variable_utils.VariableClient(
            variable_source,
            "policy",
            device=device,
            update_period=self._config.variable_update_period,
        )

        actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
        return actors.GenericActor(
            actor_core, random_key, variable_client, adder, backend=device
        )

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: drq_v2_networks.DrQV2Networks,
        dataset: Iterator[reverb.ReplaySample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
    ) -> learning_lib.DrQV2Learner:
        del replay_client, environment_spec
        config = self._config
        critic_optimizer = optax.adam(config.learning_rate)
        policy_optimizer = optax.adam(config.learning_rate)
        encoder_optimizer = optax.adam(config.learning_rate)

        return learning_lib.DrQV2Learner(
            random_key=random_key,
            dataset=dataset,
            networks=networks,
            sigma=config.sigma,
            augmentation=config.augmentation,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            encoder_optimizer=encoder_optimizer,
            noise_clip=config.noise_clip,
            critic_soft_update_rate=config.critic_q_soft_update_rate,
            discount=config.discount,
            bc_alpha=config.bc_alpha,
            num_sgd_steps_per_step=config.num_sgd_steps_per_step,
            logger=logger_fn("learner"),
            counter=counter,
        )

    def make_policy(
        self,
        networks,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool = False,
    ) -> actor_core_lib.FeedForwardPolicy:
        """Construct the policy."""
        sigma = 0 if evaluation else self._config.sigma
        return drq_v2_networks.apply_policy_and_sample(
            networks, environment_spec.actions, sigma
        )
