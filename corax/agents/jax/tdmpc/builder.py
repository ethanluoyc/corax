from typing import Iterator, List, Optional, Tuple

import jax
import reverb

from corax import adders
from corax import specs
from corax.adders import reverb as adders_reverb
from corax.agents.jax import builders
from corax.agents.jax.tdmpc import acting
from corax.agents.jax.tdmpc import config as tdmpc_config
from corax.agents.jax.tdmpc import learning
from corax.agents.jax.tdmpc import networks as tdmpc_networks
from corax.agents.jax.tdmpc import types as tdmpc_types
from corax.datasets import reverb as reverb_datasets
from corax.jax import networks as networks_lib
from corax.jax import utils as jax_utils
from corax.jax import variable_utils
from corax.utils import counting
from corax.utils import loggers

TDMPCNetworks = tdmpc_networks.TDMPCNetworks
TDMPCPolicy = Tuple[tdmpc_networks.TDMPCNetworks, bool]


class TDMPCBuilder(builders.ActorLearnerBuilder):
    def __init__(self, config: tdmpc_config.TDMPCConfig):
        self._config = config

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy: Optional[TDMPCPolicy] = None,
    ) -> List[reverb.Table]:
        del policy
        min_replay_size = self._config.min_replay_size
        samples_per_insert_tolerance = (
            self._config.samples_per_insert_tolerance_rate
            * self._config.samples_per_insert
        )
        error_buffer = min_replay_size * samples_per_insert_tolerance
        limiter = reverb.rate_limiters.SampleToInsertRatio(
            min_size_to_sample=min_replay_size,
            samples_per_insert=self._config.samples_per_insert,
            error_buffer=error_buffer,
        )

        return [
            reverb.Table(
                name=adders_reverb.DEFAULT_PRIORITY_TABLE,
                sampler=reverb.selectors.Prioritized(
                    self._config.importance_sampling_exponent
                ),
                remover=reverb.selectors.Fifo(),
                rate_limiter=limiter,
                max_size=self._config.max_replay_size,
                # For horizon of H, we insert sequences of size H+1 for bootstrapping
                signature=adders_reverb.SequenceAdder.signature(
                    environment_spec,
                    sequence_length=self._config.horizon + 1,
                ),
            )
        ]

    def make_adder(
        self,
        replay_client: reverb.Client,
        environment_spec: Optional[specs.EnvironmentSpec] = None,
        policy: Optional[TDMPCPolicy] = None,
    ) -> Optional[adders.Adder]:
        del policy, environment_spec
        return adders_reverb.SequenceAdder(
            replay_client,
            sequence_length=self._config.horizon + 1,
            period=1,
            end_of_episode_behavior=adders_reverb.EndBehavior.WRITE,
            max_in_flight_items=1,
        )

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy: TDMPCPolicy,
        environment_spec: specs.EnvironmentSpec,
        variable_source: learning.TDMPCLearner,
        adder: Optional[adders.Adder] = None,
    ) -> acting.TDMPCActor:
        networks, evaluation = policy
        variable_client = variable_utils.VariableClient(
            variable_source,
            "policy",
            update_period=self._config.variable_update_period,
        )
        return acting.TDMPCActor(
            variable_client,
            environment_spec,
            networks,
            random_key,
            std_schedule=self._config.std_schedule,
            horizon_schedule=self._config.horizon_schedule,
            discount=self._config.discount,
            num_samples=self._config.num_trajectories,
            min_std=self._config.min_std,
            temperature=self._config.temperature,
            momentum=self._config.momentum,
            num_elites=self._config.num_elites,
            iterations=self._config.num_iterations,
            seed_steps=self._config.min_replay_size,
            mixture_coef=self._config.policy_trajectory_fraction,
            value_tx_pair=self._config.value_tx_pair,
            horizon=self._config.horizon,
            adder=adder,
            evaluation=evaluation,
        )

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: TDMPCNetworks,
        dataset: Iterator[learning.TDMPCReplaySample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: reverb.Client,
        counter: Optional[counting.Counter] = None,
    ) -> learning.TDMPCLearner:
        loss_scale = tdmpc_types.LossScalesConfig(
            consistency=self._config.consistency_loss_scale,
            reward=self._config.reward_loss_scale,
            value=self._config.value_loss_scale,
        )

        return learning.TDMPCLearner(
            spec=environment_spec,
            networks=networks,
            random_key=random_key,
            iterator=dataset,
            replay_client=replay_client,
            optimizer=self._config.optimizer,
            importance_sampling_exponent=self._config.priority_exponent,
            discount=self._config.discount,
            min_std=self._config.min_std,
            target_update_rate=self._config.critic_update_rate,
            value_tx_pair=self._config.value_tx_pair,
            loss_scale=loss_scale,
            rho=self._config.rho,
            counter=counter,
            logger=logger_fn("learner"),
        )

    def make_dataset_iterator(
        self, replay_client: reverb.Client
    ) -> Iterator[learning.TDMPCReplaySample]:
        dataset = reverb_datasets.make_reverb_dataset(
            server_address=replay_client.server_address,
            batch_size=self._config.batch_size,
            num_parallel_calls=4,
        )
        return jax_utils.device_put(
            dataset.as_numpy_iterator(),
            split_fn=jax_utils.keep_key_on_host,
            device=jax.devices()[0],
        )

    def make_policy(
        self,
        networks: TDMPCNetworks,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool = False,
    ) -> TDMPCPolicy:
        del environment_spec
        return networks, evaluation
