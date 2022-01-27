import unittest
import jax
import jax.numpy as jnp
from models.generator import Generator
from models.discriminator import Discriminator
from models.recognition import Recognition


from utils.create_latents_with_codes import create_latents_with_codes


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = jax.random.PRNGKey(0)

    def test_create_latents(self):

        z = create_latents_with_codes(
            num_noise=62, num_cts=2, num_cat=10, rng_key=self.rng, num_samples=128
        )
        self.assertEqual((128, 74), z.shape)

        z = create_latents_with_codes(
            num_noise=62, num_cts=2, num_cat=11, rng_key=self.rng, num_samples=1
        )
        self.assertEqual((1, 75), z.shape)


class TestGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = jax.random.PRNGKey(0)
        self.model = Generator()

        self.latent_size = 74

        input_shape = (1, self.latent_size)

        self.variables = self.model.init(
            self.rng, jnp.ones(input_shape, dtype=jnp.float32), train=True
        )

        self.z = create_latents_with_codes(
            num_noise=62, num_cts=2, num_cat=10, rng_key=self.rng, num_samples=128
        )

    def test_generator_apply(self):

        output, _ = self.model.apply(
            {
                "params": self.variables["params"],
                "batch_stats": self.variables["batch_stats"],
            },
            self.z,
            mutable=["batch_stats"],
            train=False,
        )

        self.assertEqual((128, 28, 28, 1), output.shape)


class TestDiscriminator(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = jax.random.PRNGKey(0)
        self.model = Discriminator(filter_list=[64, 128, 1024])

        self.image_size = 28

        input_shape = (128, self.image_size, self.image_size, 1)

        self.variables = self.model.init(
            self.rng, jnp.ones(input_shape, dtype=jnp.float32), train=True
        )

        self.x = jnp.ones(input_shape, dtype=jnp.float32)

    def test_discriminator_apply(self):

        output, _ = self.model.apply(
            {
                "params": self.variables["params"],
                "batch_stats": self.variables["batch_stats"],
            },
            self.x,
            mutable=["batch_stats"],
            train=False,
        )

        self.assertEqual((128, 1), output.shape)


class TestQ(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = jax.random.PRNGKey(0)

        self.num_cts_codes = 50

        self.num_cat = 100

        self.model = Recognition(filter_list=[64, 128, 1024, 128], 
            num_cts_codes=self.num_cts_codes, num_cat=self.num_cat 
            )

        self.image_size = 28

        input_shape = (128, self.image_size, self.image_size, 1)

        self.variables = self.model.init(
            self.rng, jnp.ones(input_shape, dtype=jnp.float32), train=True
        )

        self.x = jnp.ones(input_shape, dtype=jnp.float32)

    def test_discriminator_apply(self):

        q_out,  _ = self.model.apply(
            {
                "params": self.variables["params"],
                "batch_stats": self.variables["batch_stats"],
            },
            self.x,
            mutable=["batch_stats"],
            train=False,
        )

        q_logits, q_mean, q_var, = q_out[0], q_out[1], q_out[2]
        self.assertEqual((128, self.num_cat), q_logits.shape)

        self.assertEqual((128, self.num_cts_codes), q_mean.shape)

        self.assertEqual((128, self.num_cts_codes), q_var.shape)