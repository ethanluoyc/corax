# type: ignore
# ruff: noqa: F401
import sys

from absl.testing import absltest


class ImportTest(absltest.TestCase):
    def test_import_does_not_require_reverb(self):
        if "reverb" in sys.modules:
            del sys.modules["reverb"]

        self.assertNotIn("reverb", sys.modules)

        from corax.jax import utils  # noqa

        self.assertNotIn("reverb", sys.modules)

    def test_import_jax_agents(self):
        """Test that some agents can be used withour reverb"""
        if "reverb" in sys.modules:
            del sys.modules["reverb"]
        if "tensorflow" in sys.modules:
            del sys.modules["tensorflow"]

        self.assertNotIn("reverb", sys.modules)
        from corax.agents.jax import calql
        from corax.agents.jax import decision_transformer
        from corax.agents.jax import iql
        from corax.agents.jax import oril
        from corax.agents.jax import otr
        from corax.agents.jax import redq
        from corax.agents.jax import td3

        self.assertNotIn("reverb", sys.modules)

    def test_import_mujoco_py(self):
        pass
        # import mujoco_py


if __name__ == "__main__":
    absltest.main()
