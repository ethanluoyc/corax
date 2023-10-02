# type: ignore
"""Decision Transformer model definition."""
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from corax import specs
from corax.jax import networks as networks_lib
from corax.jax import utils


class CausalSelfAttention(hk.MultiHeadAttention):
    """Self attention with a causal mask applied."""

    def __call__(
        self,
        query: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        value: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        key = key if key is not None else query
        value = value if value is not None else query

        if query.ndim != 3:
            raise ValueError("Expect queries of shape [B, T, D].")

        seq_len = query.shape[1]
        causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))
        mask = mask * causal_mask if mask is not None else causal_mask

        return super().__call__(query, key, value, mask)


class DenseBlock(hk.Module):
    """A 2-layer MLP which widens then narrows the input."""

    def __init__(
        self, init_scale: float, widening_factor: int = 4, name: Optional[str] = None
    ):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(hiddens, w_init=initializer)(x)


class Transformer(hk.Module):
    """A transformer stack."""

    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

    def __call__(
        self, h: jnp.ndarray, mask: Optional[jnp.ndarray], is_training: bool
    ) -> jnp.ndarray:
        """Connects the transformer.

        Args:
          h: Inputs, [B, T, D].
          mask: Padding mask, [B, T].
          is_training: Whether we're training or not.

        Returns:
          Array of shape [B, T, D].
        """

        init_scale = 2.0 / self._num_layers
        dropout_rate = self._dropout_rate if is_training else 0.0
        if mask is not None:
            mask = mask[:, None, None, :]

        # Note: names chosen to approximately match those used in the GPT-2 code;
        # see https://github.com/openai/gpt-2/blob/master/src/model.py.
        for i in range(self._num_layers):
            h_norm = layer_norm(h, name=f"h{i}_ln_1")
            h_attn = CausalSelfAttention(
                num_heads=self._num_heads,
                key_size=32,
                model_size=h.shape[-1],
                w_init=hk.initializers.VarianceScaling(init_scale),
                name=f"h{i}_attn",
            )(h_norm, mask=mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn
            h_norm = layer_norm(h, name=f"h{i}_ln_2")
            h_dense = DenseBlock(init_scale, name=f"h{i}_mlp")(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense
        h = layer_norm(h, name="ln_f")

        return h


class DecisionTransformer(hk.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: int,
        action_tanh=True,
        max_ep_len=4096,
        name: str = "decision_transformer",
    ):
        super().__init__(name=name)
        self._state_size = state_size
        self._action_size = action_size
        self._hidden_size = hidden_size

        self.transformer = Transformer(
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        w_init = embed_init

        self.embed_timestep = hk.Embed(max_ep_len, hidden_size, w_init=embed_init)
        self.embed_return = hk.Linear(hidden_size, w_init=w_init)
        self.embed_state = hk.Linear(hidden_size, w_init=w_init)
        self.embed_action = hk.Linear(hidden_size, w_init=w_init)
        self.embed_ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        # note: we don't predict states or returns for the paper
        self.predict_state = hk.Linear(self._state_size, w_init=w_init)
        self.predict_action = hk.Sequential(
            [hk.Linear(self._action_size, w_init=w_init)]
            + ([jnp.tanh] if action_tanh else [])
        )
        self.predict_return = hk.Linear(1, w_init=w_init)

    def __call__(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
        is_training=True,
    ):
        del rewards
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = jnp.transpose(
            jnp.stack(
                (returns_embeddings, state_embeddings, action_embeddings), axis=1
            ),
            (0, 2, 1, 3),
        ).reshape((batch_size, 3 * seq_length, self._hidden_size))
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = jnp.transpose(
            jnp.stack((attention_mask, attention_mask, attention_mask), axis=1),
            (0, 2, 1),
        ).reshape((batch_size, 3 * seq_length))

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            stacked_inputs,
            mask=stacked_attention_mask,
            is_training=is_training,
        )
        x = transformer_outputs

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = jnp.transpose(
            x.reshape(batch_size, seq_length, 3, self._hidden_size), (0, 2, 1, 3)
        )

        # get predictions
        return_preds = self.predict_return(
            x[:, 2]
        )  # predict next return given state and action
        state_preds = self.predict_state(
            x[:, 2]
        )  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return state_preds, action_preds, return_preds


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


def make_gym_networks(
    spec: specs.EnvironmentSpec,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    dropout_rate: float,
    episode_length: int,
):
    state_dim = spec.observations.shape[-1]
    act_dim = spec.actions.shape[-1]

    def _model_fn(
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask,
        is_training=True,
    ):
        model = DecisionTransformer(
            state_size=state_dim,
            action_size=act_dim,
            max_ep_len=episode_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        return model(
            states,
            actions,
            rewards,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
            is_training=is_training,
        )

    dummy_obs = utils.zeros_like(spec.observations)
    dummy_act = utils.zeros_like(spec.actions)

    dummy_obs = utils.add_batch_dim(jnp.stack([dummy_obs]))
    dummy_act = utils.add_batch_dim(jnp.stack([dummy_act]))
    dummy_reward = jnp.zeros((1, 1, 1))
    dummy_rtg = jnp.zeros((1, 1, 1))
    dummy_timestep = jnp.zeros(
        (
            1,
            1,
        ),
        dtype=jnp.int32,
    )
    dummy_mask = jnp.zeros((1, 1), dtype=jnp.bool_)

    model = hk.transform(_model_fn)
    init_fn = lambda key: model.init(
        key,
        dummy_obs,
        dummy_act,
        dummy_reward,
        dummy_rtg,
        dummy_timestep,
        dummy_mask,
        is_training=True,
    )
    return networks_lib.FeedForwardNetwork(init=init_fn, apply=model.apply)
