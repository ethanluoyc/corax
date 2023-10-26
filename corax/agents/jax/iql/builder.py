"""IQL Builder."""
from typing import TYPE_CHECKING, Iterator, Optional

import optax

from corax import adders
from corax import core
from corax import specs
from corax import types
from corax.agents.jax import actor_core as actor_core_lib
from corax.agents.jax import actors
from corax.agents.jax import builders
from corax.agents.jax.iql import config as iql_config
from corax.agents.jax.iql import learning as iql_learning
from corax.agents.jax.iql import networks as iql_networks
from corax.jax import networks as networks_lib
from corax.jax import variable_utils
from corax.utils import counting
from corax.utils import loggers

if TYPE_CHECKING:
    import reverb


class IQLBuilder(
    builders.OfflineBuilder[
        iql_networks.IQLNetworks, actor_core_lib.FeedForwardPolicy, types.Transition
    ]
):
    """IQL Builder."""

    def __init__(self, config: iql_config.IQLConfig):
        """Creates a IQL learner, a behavior policy and an eval actor.

        Args:
          config: a config with IQL hps
        """
        self._config = config

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: iql_networks.IQLNetworks,
        dataset: Iterator[types.Transition],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: Optional["reverb.Client"] = None,
        counter: Optional[counting.Counter] = None,
    ) -> core.Learner:
        del environment_spec, replay_client

        policy_optimizer = optax.adam(self._config.learning_rate)
        critic_optimizer = optax.adam(self._config.learning_rate)
        value_optimizer = optax.adam(self._config.learning_rate)

        return iql_learning.IQLLearner(
            networks=networks,
            random_key=random_key,
            dataset=dataset,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            value_optimizer=value_optimizer,
            discount=self._config.discount,
            tau=self._config.tau,
            expectile=self._config.expectile,
            temperature=self._config.temperature,
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
            variable_source, key="policy", device="cpu"
        )
        return actors.GenericActor(
            actor_core, random_key, variable_client, adder, backend="cpu"
        )

    def make_policy(
        self,
        networks: iql_networks.IQLNetworks,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool = False,
    ) -> actor_core_lib.FeedForwardPolicy:
        """Creates a policy."""
        return iql_networks.apply_policy_and_sample(
            networks,
            environment_spec.actions,
            eval_mode=evaluation,
        )
