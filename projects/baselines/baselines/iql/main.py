from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags

from baselines.iql import base_config
from baselines.iql import train

_CONFIG = config_flags.DEFINE_config_file("config", None)
flags.mark_flag_as_required("config")

FLAGS = flags.FLAGS


def main(_):
    config: base_config.Config = _CONFIG.value

    logging.info("Using config: %s", config_flags.get_config_filename(FLAGS["config"]))
    logging.info("Overrides: %s", config_flags.get_override_values(FLAGS["config"]))

    train.train_and_evaluate(config, workdir=None)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
