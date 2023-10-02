"""Testing utilities."""

import sys
from typing import Optional

from absl import flags
from absl.testing import parameterized


class TestCase(parameterized.TestCase):
    """A custom TestCase which handles FLAG parsing for pytest compatibility."""

    def get_tempdir(self, name: Optional[str] = None) -> str:
        try:
            flags.FLAGS.test_tmpdir
        except flags.UnparsedFlagAccessError:
            # Need to initialize flags when running `pytest`.
            flags.FLAGS(sys.argv, known_only=True)
        return self.create_tempdir(name).full_path
