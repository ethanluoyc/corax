from corax.agents.jax.redq.learning import REDQLearnerCore
from corax.agents.jax.redq.networks import apply_policy_and_sample
from corax.agents.jax.redq.networks import make_networks
from corax.agents.jax.redq.networks import target_entropy_from_spec
from corax.utils import lazy_loader

with lazy_loader.LazyImports(__name__, False):
    from corax.agents.jax.redq.builder import REDQBuilder
    from corax.agents.jax.redq.config import REDQConfig

del lazy_loader
