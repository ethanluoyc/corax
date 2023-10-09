import os

os.environ["MUJOCO_GL"] = "egl"

# ruff: noqa: E402
import dm_env_wrappers
import jax
import numpy as np
import optax
import reverb
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from dm_control import suite

import corax
from baselines.drq_bc import drq_frame_stacking
from baselines.drq_bc import vd4rl_preprocessor
from corax import types
from corax.agents.jax import actor_core as actor_core_lib
from corax.agents.jax import actors
from corax.agents.jax import drq_v2
from corax.agents.jax.drq_v2 import augmentations
from corax.jax import utils
from corax.jax import variable_utils
from corax.utils import counting
from corax.utils import loggers
from corax.wrappers import mujoco


def make_environment(domain: str, task: str):
    env = suite.load(domain, task, task_kwargs={"random": 0})
    env = mujoco.MujocoPixelWrapper(env, height=84, width=84, camera_id=0)
    env = dm_env_wrappers.ActionRepeatWrapper(env, 2)
    env = drq_frame_stacking.FrameStackingWrapper(env, 3, flatten=True)
    env = dm_env_wrappers.SinglePrecisionWrapper(env)
    return env


def get_dataset_iterator(domain, task):
    name = "medium_expert"
    dataset = tfds.load(f"vd4rl/main_{domain}_{task}_{name}_84px")["train"]  # type: ignore
    dataset = dataset.map(
        lambda episode: {"steps": vd4rl_preprocessor.process_data(episode["steps"], 3)},
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # TODO: use shuffling
    dataset = dataset.flat_map(
        lambda episode: vd4rl_preprocessor.tfds_get_n_step_transitions(
            episode["steps"], 3, 0.99
        )
    )

    iterator = dataset.cache().batch(256).repeat().prefetch(4).as_numpy_iterator()
    while True:
        batch = next(iterator)
        transitions = types.Transition(**batch)
        yield reverb.ReplaySample(data=transitions, info=None)  # type: ignore


def main(_):
    tf.config.set_visible_devices([], "GPU")
    np.random.seed(0)
    domain = "walker"
    task = "walk"
    dataset = utils.device_put(get_dataset_iterator(domain, task), jax.devices()[0])
    dataset = utils.prefetch(dataset, 2)
    environment = make_environment(domain, task)
    env_spec = corax.make_environment_spec(environment)
    key = jax.random.PRNGKey(0)
    learner_key, actor_key = jax.random.split(key)
    networks = drq_v2.make_networks(env_spec)
    parent_counter = counting.Counter(time_delta=0.0)

    learner = drq_v2.DrQV2Learner(
        learner_key,
        dataset,
        networks,
        optax.linear_schedule(0.5, 0.1, 250000),
        augmentation=augmentations.batched_random_shift_aug,
        policy_optimizer=optax.adam(1e-4),
        critic_optimizer=optax.adam(1e-4),
        encoder_optimizer=optax.adam(1e-4),
        critic_soft_update_rate=0.01,
        discount=0.99,
        bc_alpha=2.5,
        logger=loggers.make_default_logger("learner", asynchronous=True),
        counter=counting.Counter(parent_counter, "learner", time_delta=0.0),
    )

    device = "gpu"
    variable_client = variable_utils.VariableClient(learner, "policy", device=device)

    evaluator = actors.GenericActor(
        actor_core_lib.batched_feed_forward_to_actor_core(
            drq_v2.apply_policy_and_sample(networks, env_spec.actions, 0.0)
        ),
        actor_key,
        variable_client,
        backend=device,
    )

    eval_loop = corax.EnvironmentLoop(
        environment=environment,
        actor=evaluator,
        logger=loggers.make_default_logger("evaluator"),
        counter=counting.Counter(parent_counter, "evaluator", time_delta=0.0),
    )

    # Run the environment loop.
    max_steps = int(1e6)
    eval_every = 5000
    eval_episodes = 10
    steps = 0
    while steps < max_steps:
        for _ in range(eval_every):
            learner.step()
        steps += eval_every
        eval_loop.run(eval_episodes)


if __name__ == "__main__":
    app.run(main)
