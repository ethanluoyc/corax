"""Corax is a framework for reinforcement learning in JAX."""

# Expose specs and types modules.
from corax import specs
from corax import types

# Expose core interfaces.
from corax.core import Actor
from corax.core import Learner
from corax.core import Saveable
from corax.core import VariableSource
from corax.core import Worker

# Expose the environment loop.
from corax.specs import make_environment_spec
