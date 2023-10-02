from typing import Callable

import jax
import jax.numpy as jnp

from corax import types
from corax.jax import networks as networks_lib

RewarderFn = Callable[
    [networks_lib.Params, networks_lib.Observation], networks_lib.Logits
]


def oril_loss(
    rewarder_fn: RewarderFn,
    params: networks_lib.Params,
    expert_transitions: types.Transition,
    unlabeled_transitions: types.Transition,
):
    """Loss function for learning reward function.

    See https://arxiv.org/pdf/2011.13885.pdf Eqn 1
    """
    expert_logits = rewarder_fn(params, expert_transitions.observation)
    unlabeled_logits = rewarder_fn(params, unlabeled_transitions.observation)
    # R(s) = signmoid(NN(s))
    # -log R(expert) + -log(1-R(unlabeled))
    # = -log(sigmoid(NN(expert))) + -log(1-sigmoid(NN(unlabeled)))
    # = -log_sigmoid(NN(expert)) + -log(-sigmoid(NN(unlabeled)))
    expert_loss = -jax.nn.log_sigmoid(expert_logits)
    unlabeled_loss = jax.nn.softplus(unlabeled_logits)
    total_loss = jnp.mean(expert_loss + unlabeled_loss)
    metrics = {
        "total_loss": total_loss,
        "expert_loss": jnp.mean(expert_loss),
        "unlabeled_loss": jnp.mean(unlabeled_loss),
    }
    return total_loss, metrics


# Positive-unlabeled learning (PU-learning)
def oril_pu_loss(
    rewarder_fn: RewarderFn,
    params: networks_lib.Params,
    expert_transitions: types.Transition,
    unlabeled_transitions: types.Transition,
    gamma: float = 0.5,
):
    """ORIL Positive-Unlabeled (PU) Learning loss.

    See https://arxiv.org/pdf/2011.13885.pdf Eqn 5
    """
    expert_logits = rewarder_fn(params, expert_transitions.observation)
    unlabeled_logits = rewarder_fn(params, unlabeled_transitions.observation)
    # https://arxiv.org/pdf/2011.13885.pdf Eqn (5)
    #   gamma * -log(R(expert)) + (1 - gamma) * -log(1-R(false))
    # = gamma * -log(R(expert)) + -log(1-R(unlabaled)) - gamma * -log(1-R(expert))

    # gamma * -log(R(expert))
    positive_loss = gamma * -jax.nn.log_sigmoid(expert_logits)
    # -log(1-R(unlabaled)) - gamma * -log(1-R(expert))
    negative_loss = jax.nn.softplus(unlabeled_logits) - gamma * jax.nn.softplus(
        expert_logits
    )
    total_loss = jnp.mean(positive_loss + negative_loss)
    metrics = {
        "total_loss": total_loss,
        "positive_loss": jnp.mean(positive_loss),
        "negative_loss": jnp.mean(negative_loss),
        "expert_logits": jnp.mean(expert_logits),
        "unlabeled_logits": jnp.mean(unlabeled_logits),
    }
    return total_loss, metrics
