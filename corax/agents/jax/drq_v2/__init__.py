"""DrQ-v2 agent implementation."""
from corax.agents.jax.drq_v2 import augmentations
from corax.agents.jax.drq_v2.builder import DrQV2Builder
from corax.agents.jax.drq_v2.config import DrQV2Config
from corax.agents.jax.drq_v2.learning import DrQV2Learner
from corax.agents.jax.drq_v2.networks import DrQV2Networks
from corax.agents.jax.drq_v2.networks import apply_policy_and_sample
from corax.agents.jax.drq_v2.networks import make_networks
from corax.agents.jax.drq_v2.networks import make_state_networks
