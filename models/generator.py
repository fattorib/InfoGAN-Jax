from ast import Str
import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import copy
from functools import partial
import numpy as np

ModuleDef = Any
dtypedef = Any


class Generator(nn.Module):

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv layers
    kernel_init: Callable = nn.initializers.normal(stddev=0.02, dtype=dtype)

    @nn.compact
    def __call__(self, x, train):

        x = x.reshape((x.shape[0], 1, 1, x.shape[1]))

        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.1,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        x = nn.ConvTranspose(
            features=1024,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=False,
            kernel_init=self.kernel_init,
        )(x)

        x = nn.relu(x)

        x = norm()(x)

        x = nn.ConvTranspose(
            features=128,
            kernel_size=(6, 6),
            strides=(1, 1),
            padding="VALID",
            use_bias=False,
            kernel_init=self.kernel_init,
        )(x)

        x = nn.relu(x)

        x = norm()(x)

        x = nn.ConvTranspose(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            use_bias=False,
            kernel_init=self.kernel_init,
        )(x)

        x = nn.relu(x)

        x = norm()(x)

        x = nn.ConvTranspose(
            features=1,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            use_bias=False,
            kernel_init=self.kernel_init,
        )(x)

        return nn.sigmoid(x)


if __name__ == "__main__":

    model = Generator()

    def initialized(key, latent_size, model):
        input_shape = (1, latent_size)

        @jax.jit
        def init(rng, shape):
            return model.init(rng, shape, train=True)

        variables = init(rng=key, shape=jnp.ones(input_shape, dtype=model.dtype))
        return variables["params"], variables["batch_stats"]

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    params, batch_stats = initialized(rng, 74, model)
