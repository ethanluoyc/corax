# type: ignore
import numpy as np
import ray
import reverb
from absl.testing import absltest

from corax.raypad import actors


class Service:
    def ping(self):
        return "pong"

    def __call__(self):
        return "called"


class ActorsTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(num_cpus=1)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_client_basic(self):
        server = actors.PyActor.remote(Service)
        client = actors.PyActorClient(server)
        self.assertEqual(client.ping(), "pong")
        self.assertEqual(client(), "called")

    def test_client_async(self):
        server = actors.PyActor.remote(Service)
        client = actors.PyActorClient(server)
        self.assertEqual(client.futures.ping().result(), "pong")
        self.assertEqual(client.futures().result(), "called")


_TABLE_NAME = "dist"


def priority_tables_fn():
    return [
        reverb.Table(
            name=_TABLE_NAME,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=100,
            rate_limiter=reverb.rate_limiters.MinSize(100),
        )
    ]


class ReverbActorTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(num_cpus=1)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_reverb_actor(self):
        replay_server_actor = actors.ReverbActor.remote(priority_tables_fn)
        reverb_address = ray.get(replay_server_actor.get_reverb_address.remote())
        client = reverb.Client(reverb_address)
        client.insert([np.zeros((81, 81))], {_TABLE_NAME: 1})


if __name__ == "__main__":
    absltest.main()
