"""exorl dataset."""

import tensorflow_datasets as tfds

from corax.datasets.tfds.exorl import exorl


class ExorlTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for exorl dataset."""

    DATASET_CLASS = exorl.Exorl
    SPLITS = {
        "train": 1,  # Number of fake train example
    }

    SKIP_CHECKSUMS = True
    SKIP_TF1_GRAPH_MODE = True

    BUILDER_CONFIG_NAMES_TO_TEST = ["walker_proto"]

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    DL_EXTRACT_RESULT = {"file_path": "walker_proto"}
    DL_DOWNLOAD_RESULT = {"file_path": "walker_proto"}


if __name__ == "__main__":
    tfds.testing.test_main()
