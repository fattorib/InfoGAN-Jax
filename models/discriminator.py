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


class Discriminator(nn.Module):

    filter_list: Sequence[int]

    num_channels: int = 1

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    @nn.compact
    def __call__(self, x, train):

        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.1,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        x = nn.Conv(
            kernel_size=(4, 4),
            features=self.filter_list[0],
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            use_bias=True,
            dtype=self.dtype,
        )(x)

        x = nn.leaky_relu(x, negative_slope=0.1)

        x = nn.Conv(
            kernel_size=(4, 4),
            features=self.filter_list[1],
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            dtype=self.dtype,
        )(x)

        x = norm()(x)

        x = nn.leaky_relu(x, negative_slope=0.1)

        x = nn.Conv(
            kernel_size=(7, 7),
            features=self.filter_list[2],
            strides=(1, 1),
            padding=((0, 0), (0, 0)),
            use_bias=False,
            dtype=self.dtype,
        )(x)

        x = norm()(x)

        x = nn.leaky_relu(x, negative_slope=0.1)

        x = nn.Conv(
            kernel_size=(1, 1),
            features=1,
            strides=(1, 1),
            padding=((0, 0), (0, 0)),
            use_bias=False,
            dtype=self.dtype,
        )(x)

        return nn.sigmoid(x)


if __name__ == "__main__":

    model = Discriminator(filter_list=[64, 128, 1024])

    def initialized(key, image_size, model):
        input_shape = (1, image_size, image_size, 1)

        @jax.jit
        def init(rng, shape):
            return model.init(rng, shape, train=True)

        variables = init(rng=key, shape=jnp.ones(input_shape, dtype=model.dtype))
        return variables["params"], variables["batch_stats"]

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    params, batch_stats = initialized(rng, 28, model)
