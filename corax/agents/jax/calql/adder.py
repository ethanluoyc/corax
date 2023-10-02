import dataclasses
from typing import Union

import dm_env
import numpy as np

from corax.adders import Adder


@dataclasses.dataclass
class SparseReward:
    reward_scale: float
    reward_bias: float

    positive_reward: float
    negative_reward: float


@dataclasses.dataclass
class DenseReward:
    reward_scale: float
    reward_bias: float


def compute_return_to_go(
    rewards,
    terminals,
    gamma,
    reward_scale,
    reward_bias,
    is_sparse_reward,
    negative_reward=0.0,
):
    """
    A config dict for getting the default high/low rewrd values for each envs
    This is used in calc_return_to_go func in sampler.py and replay_buffer.py
    """
    if len(rewards) == 0:
        return np.array([])

    reward_neg = negative_reward * reward_scale + reward_bias

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        # assuming failure reward is negative
        # use r / (1-gamma) for negative trajctory
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (
                1 - terminals[-i - 1]
            )
            prev_return = return_to_go[-i - 1]

    return np.array(return_to_go, dtype=np.float32)


class CalQLAdder(Adder):
    def __init__(
        self,
        base_adder: Adder,
        discount,
        reward_config: Union[SparseReward, DenseReward],
    ):
        self._adder = base_adder
        self._discount = discount
        self._reward_config = reward_config
        self._steps = []

    def add_first(self, timestep: dm_env.TimeStep):
        assert timestep.first()
        self._steps = [(None, timestep)]
        # return self._adder.add_first(timestep)

    def add(self, action, next_timestep: dm_env.TimeStep, extras=()):
        del extras
        self._steps.append((action, next_timestep))
        if next_timestep.last():
            dones = [step.last() for (_, step) in self._steps[1:]]
            rewards = [
                (
                    step.reward * self._reward_config.reward_scale
                    + self._reward_config.reward_bias
                )
                for (_, step) in self._steps[1:]
            ]
            if isinstance(self._reward_config, SparseReward):
                mc_returns = compute_return_to_go(
                    rewards,
                    dones,
                    self._discount,
                    self._reward_config.reward_scale,
                    self._reward_config.reward_bias,
                    is_sparse_reward=True,
                    negative_reward=self._reward_config.negative_reward,
                )
            else:
                mc_returns = compute_return_to_go(
                    rewards,
                    dones,
                    self._discount,
                    self._reward_config.reward_scale,
                    self._reward_config.reward_bias,
                    is_sparse_reward=False,
                )

            initial_step = self._steps[0][1]
            self._adder.add_first(initial_step)
            for (action, step), mc_return, reward in zip(
                self._steps[1:], mc_returns, rewards
            ):
                step = step._replace(reward=reward.astype(np.float32))
                self._adder.add(action, step, extras={"mc_return": mc_return})
            self._steps = []

    def reset(self):
        self._steps = []
        self._adder.reset()
