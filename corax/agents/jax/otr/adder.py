import dm_env

from corax import adders
from corax import types
from corax.agents.jax.otr import rewarder


class OTAdder(adders.Adder):
    """Adder wrapper substituting OT rewards."""

    def __init__(
        self,
        direct_rl_adder: adders.Adder,
        otil_rewarder: rewarder.OTRewarder,
    ):
        self._adder = direct_rl_adder
        self._rewarder = otil_rewarder
        self._steps = []
        self._timesteps = []

    def add_first(self, timestep: dm_env.TimeStep):
        self._steps = []
        self._timesteps = []
        self._timesteps.append(timestep)

    def add(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
        extras: types.NestedArray = (),
    ):
        # TODO(yl): Handle extras
        del extras
        self._steps.append(
            types.Transition(self._timesteps[-1].observation, action, (), (), ())
        )
        self._timesteps.append(next_timestep)
        if next_timestep.last():
            self._add_episode()

    def _add_episode(self):
        # Compute pseudo-rewards and add
        psuedo_rewards = self._rewarder.compute_offline_rewards(self._steps[:])
        first_timestep = self._timesteps[0]
        self._adder.add_first(first_timestep._replace(reward=psuedo_rewards[0]))
        actions = [step.action for step in self._steps]
        assert len(actions) == len(self._timesteps) - 1 == len(psuedo_rewards)
        for action, next_ts, pr in zip(actions, self._timesteps[1:], psuedo_rewards):
            self._adder.add(action, next_ts._replace(reward=pr))
        self._adder.reset()

    def reset(self):
        self._adder.reset()
        self._steps = []
        self._timesteps = []
