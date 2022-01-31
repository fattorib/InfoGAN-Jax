import unittest
import jax
import jax.numpy as jnp
import numpy as np

from models.generator import Generator
from models.discriminator import Discriminator
from models.recognition import Q_head

from utils.create_latents_with_codes import create_latents_with_codes
from loss import binary_cross_entropy_loss, cross_entropy_loss, q_cts_loss

import torch.nn as nn
import torch


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
        self.x = jnp.ones(input_shape, dtype=jnp.float32)

        self.variables = self.model.init(
            self.rng,
            jnp.ones(input_shape, dtype=jnp.float32),
            train=True,
            with_head=True,
        )

    def test_discriminator_apply_with_head(self):

        output, _ = self.model.apply(
            {
                "params": self.variables["params"],
                "batch_stats": self.variables["batch_stats"],
            },
            self.x,
            mutable=["batch_stats"],
            train=True,
            with_head=True,
        )

        self.assertEqual((128, 1), output.shape)

    def test_discriminator_apply_no_head(self):

        output, _ = self.model.apply(
            {
                "params": self.variables["params"],
                "batch_stats": self.variables["batch_stats"],
            },
            self.x,
            mutable=["batch_stats"],
            train=True,
            with_head=False,
        )

        self.assertEqual((128, 1, 1, 1024), output.shape)


class TestQ(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = jax.random.PRNGKey(0)
        self.model = Discriminator(filter_list=[64, 128, 1024])

        self.image_size = 28
        self.num_cts_codes = 2
        self.num_cat = 10

        input_shape = (128, self.image_size, self.image_size, 1)

        self.variables = self.model.init(
            self.rng,
            jnp.ones(input_shape, dtype=jnp.float32),
            train=True,
            with_head=True,
        )

        self.x = jnp.ones(input_shape, dtype=jnp.float32)

        self.Q_head = Q_head(filter_size=128)

        input_shape = (128, 1, 1, 1024)

        self.q_variables = self.Q_head.init(
            self.rng,
            jnp.ones(input_shape, dtype=jnp.float32),
            train=True,
        )

    def test_Q_head(self):

        disc_output, _ = self.model.apply(
            {
                "params": self.variables["params"],
                "batch_stats": self.variables["batch_stats"],
            },
            self.x,
            mutable=["batch_stats"],
            train=True,
            with_head=False,
        )

        q_out, _ = self.Q_head.apply(
            {
                "params": self.q_variables["params"],
                "batch_stats": self.q_variables["batch_stats"],
            },
            disc_output,
            mutable=["batch_stats"],
            train=True,
        )

        q_logits, q_mean, q_var, = (
            q_out[0],
            q_out[1],
            q_out[2],
        )
        self.assertEqual((128, self.num_cat), q_logits.shape)

        self.assertEqual((128, self.num_cts_codes), q_mean.shape)

        self.assertEqual((128, self.num_cts_codes), q_var.shape)


class TestLosses(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 128
        self.num_classes = 25

    def test_BCE_loss(self):
        scores = np.random.rand(self.batch_size)
        labels = np.random.binomial(1, 0.2, size=self.batch_size)

        jnp_scores, jnp_labels = jnp.array(scores), jnp.array(labels)

        torch_scores, torch_labels = torch.Tensor(scores), torch.Tensor(labels)

        jnp_loss = jax.device_get(
            binary_cross_entropy_loss(logit=jnp_scores, label=jnp_labels)
        ).item()

        torch_loss = nn.BCEWithLogitsLoss()(torch_scores, torch_labels).item()

        np.testing.assert_allclose(jnp_loss, torch_loss, rtol=1e-5, atol=0)

    def test_CE_loss(self):
        scores = np.random.rand(self.batch_size, self.num_classes)
        labels = np.random.randint(
            low=0, high=self.num_classes - 1, size=self.batch_size
        )

        jnp_scores, jnp_labels = jnp.array(scores), jnp.array(labels)

        jnp_labels_OH = jax.nn.one_hot(jnp_labels, num_classes=self.num_classes)

        torch_scores, torch_labels = torch.tensor(scores), torch.tensor(
            labels, dtype=torch.long
        )

        jnp_loss = jax.device_get(
            cross_entropy_loss(q_logits=jnp_scores, q_codes=jnp_labels_OH)
        ).item()

        torch_loss = nn.CrossEntropyLoss()(torch_scores, torch_labels).item()

        np.testing.assert_allclose(jnp_loss, torch_loss, rtol=1e-5, atol=0)

    def gaussian_nll(self):
        pass
