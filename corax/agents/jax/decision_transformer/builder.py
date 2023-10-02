from typing import Any, Iterator, Optional

import optax

from corax import core
from corax import specs
from corax.agents.jax import builders
from corax.agents.jax.decision_transformer import acting
from corax.agents.jax.decision_transformer import config as dt_config
from corax.agents.jax.decision_transformer import learning
from corax.jax import networks as networks_lib
from corax.jax import running_statistics
from corax.jax import variable_utils
from corax.utils import counting
from corax.utils import loggers

DecisionTransformerNetwork = networks_lib.FeedForwardNetwork
DecisionTransformerPolicy = Any
Sample = Any


class DecisionTransformerBuilder(builders.OfflineBuilder):
    def __init__(
        self,
        config: dt_config.DecisionTransformerConfig,
        observation_mean_std: Optional[running_statistics.NestedMeanStd] = None,
        max_num_learner_steps: Optional[int] = None,
        actor_device=None,
    ):
        self._config = config
        self._max_num_learner_steps = max_num_learner_steps
        self._actor_device = actor_device
        self._observation_mean_std = observation_mean_std

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: DecisionTransformerNetwork,
        dataset: Iterator[Sample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        *,
        counter: Optional[counting.Counter] = None,
    ) -> learning.DecisionTransformerLearner:
        del environment_spec
        learning_rate = optax.warmup_cosine_decay_schedule(
            0,
            self._config.learning_rate,
            warmup_steps=self._config.warmup_steps,
            decay_steps=self._max_num_learner_steps,  # type: ignore
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(self._config.grad_norm_clipping),
            optax.adamw(
                learning_rate=learning_rate,
                weight_decay=self._config.weight_decay,
                mask=self._config.weight_decay_mask,
            ),
        )
        return learning.DecisionTransformerLearner(
            model=networks,
            key=random_key,
            dataset=dataset,
            optimizer=optimizer,
            logger=logger_fn("learner"),
            counter=counter,
        )

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy: DecisionTransformerPolicy,
        environment_spec: specs.EnvironmentSpec,
        variable_source: core.VariableSource,
    ) -> acting.DecisionTransformerActor:
        variable_client = variable_utils.VariableClient(
            variable_source, "model", device=self._actor_device
        )
        return acting.DecisionTransformerActor(
            environment_spec,
            random_key=random_key,
            forward_fn=policy,
            context_length=self._config.context_length,
            target_return=self._config.target_return,
            return_scale=self._config.return_scale,
            variable_client=variable_client,
            observation_mean_std=self._observation_mean_std,
            mode=self._config.mode,
        )

    def make_policy(
        self,
        networks: DecisionTransformerNetwork,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool,
    ) -> DecisionTransformerPolicy:
        del environment_spec, evaluation
        return networks.apply
