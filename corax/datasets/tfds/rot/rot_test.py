"""rot dataset."""

import tensorflow_datasets as tfds

from corax.datasets.tfds.rot import rot_dmc


class RotDmcTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for rot dataset."""

    DATASET_CLASS = rot_dmc.RotDmc

    BUILDER_CONFIG_NAMES_TO_TEST = ["cheetah_run"]

    SPLITS = {
        "train": 1,  # Number of fake train example
    }

    SKIP_CHECKSUMS = True
    SKIP_TF1_GRAPH_MODE = True
    DL_EXTRACT_RESULT = {"file_path": "."}


if __name__ == "__main__":
    tfds.testing.test_main()
