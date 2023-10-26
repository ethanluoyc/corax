"""Twin-Delayed DDPG agent."""

from corax.agents.jax.td3.learning import TD3LearnerCore
from corax.agents.jax.td3.networks import TD3Networks
from corax.agents.jax.td3.networks import get_default_behavior_policy
from corax.agents.jax.td3.networks import make_networks
from corax.utils import lazy_loader

with lazy_loader.LazyImports(__name__, False):
    from corax.agents.jax.td3.builder import TD3Builder
    from corax.agents.jax.td3.config import TD3Config

del lazy_loader
