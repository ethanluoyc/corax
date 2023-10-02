"""TDMPC agent module."""
from corax.agents.jax.tdmpc.acting import TDMPCActor
from corax.agents.jax.tdmpc.builder import TDMPCBuilder
from corax.agents.jax.tdmpc.config import TDMPCConfig
from corax.agents.jax.tdmpc.learning import TDMPCLearner
from corax.agents.jax.tdmpc.networks import TDMPCNetworks
from corax.agents.jax.tdmpc.networks import TDMPCParams
from corax.agents.jax.tdmpc.networks import make_networks
