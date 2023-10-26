# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TD3 Builder."""
from typing import Iterator, List, Optional

import jax
import optax
import reverb
from reverb import rate_limiters

from corax import adders
from corax import core
from corax import specs
from corax import types
from corax.adders import reverb as adders_reverb
from corax.agents.jax import actor_core as actor_core_lib
from corax.agents.jax import actors
from corax.agents.jax import builders
from corax.agents.jax.td3 import config as td3_config
from corax.agents.jax.td3 import learning
from corax.agents.jax.td3 import networks as td3_networks
from corax.datasets import reverb as datasets
from corax.jax import networks as networks_lib
from corax.jax import utils
from corax.jax import variable_utils
from corax.utils import counting
from corax.utils import loggers


class TD3Builder(
    builders.ActorLearnerBuilder[
        td3_networks.TD3Networks,
        actor_core_lib.FeedForwardPolicy,
        "reverb.ReplaySample",
    ]
):
    """TD3 Builder."""

    def __init__(
        self,
        config: td3_config.TD3Config,
    ):
        """Creates a TD3 learner, a behavior policy and an eval actor.

        Args:
          config: a config with TD3 hps
        """
        self._config = config

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: td3_networks.TD3Networks,
        dataset: Iterator[reverb.ReplaySample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: Optional["reverb.Client"] = None,
        counter: Optional[counting.Counter] = None,
    ) -> core.Learner:
        del environment_spec, replay_client

        critic_optimizer = optax.adam(self._config.critic_learning_rate)
        twin_critic_optimizer = optax.adam(self._config.critic_learning_rate)
        policy_optimizer = optax.adam(self._config.policy_learning_rate)

        if self._config.policy_gradient_clipping is not None:
            policy_optimizer = optax.chain(
                optax.clip_by_global_norm(self._config.policy_gradient_clipping),
                policy_optimizer,
            )

        iterator = (types.Transition(*sample.data) for sample in dataset)
        return learning.TD3Learner(
            networks=networks,
            random_key=random_key,
            discount=self._config.discount,
            target_sigma=self._config.target_sigma,
            noise_clip=self._config.noise_clip,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            twin_critic_optimizer=twin_critic_optimizer,
            bc_alpha=self._config.bc_alpha,
            iterator=iterator,
            logger=logger_fn("learner"),
            counter=counter,
        )

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy: actor_core_lib.FeedForwardPolicy,
        environment_spec: specs.EnvironmentSpec,
        variable_source: Optional[core.VariableSource] = None,
        adder: Optional[adders.Adder] = None,
    ) -> core.Actor:
        del environment_spec
        assert variable_source is not None
        actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
        # Inference happens on CPU, so it's better to move variables there too.
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
        """Creates reverb tables for the algorithm."""
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
        """Creates a dataset iterator to use for learning."""
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=replay_client.server_address,
            batch_size=(self._config.batch_size),
            prefetch_size=self._config.prefetch_size,
        )
        return utils.device_put(dataset.as_numpy_iterator(), jax.devices()[0])

    def make_adder(
        self,
        replay_client: reverb.Client,
        environment_spec: Optional[specs.EnvironmentSpec],
        policy: Optional[actor_core_lib.FeedForwardPolicy],
    ) -> Optional[adders.Adder]:
        """Creates an adder which handles observations."""
        del environment_spec, policy
        return adders_reverb.NStepTransitionAdder(
            priority_fns={self._config.replay_table_name: None},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount,
        )

    def make_policy(
        self,
        networks: td3_networks.TD3Networks,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool = False,
    ) -> actor_core_lib.FeedForwardPolicy:
        """Creates a policy."""
        sigma = 0 if evaluation else self._config.sigma
        return td3_networks.get_default_behavior_policy(
            networks=networks, action_specs=environment_spec.actions, sigma=sigma
        )
