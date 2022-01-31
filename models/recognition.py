from ast import Str
import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import copy
from functools import partial
from matplotlib.pyplot import axes
import numpy as np

ModuleDef = Any
dtypedef = Any


class Q_head(nn.Module):
    """Basic Q network for MNIST

    Args:
        filter_size (int): 

        num_channels (int):

        dtype (dtypedef): 

        num_cts_codes (int):

        num_cat (int):

        num_channels (int): 
    
    References:
        Chen et al, 2016: https://arxiv.org/abs/1606.03657
    
    """

    filter_size: int

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv layers
    kernel_init: Callable = nn.initializers.normal(stddev=0.02, dtype=dtype)

    num_cts_codes: int = 2

    num_cat: int = 10

    num_channels: int = 1

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
            kernel_size=(1, 1),
            features=self.filter_size,
            strides=(1, 1),
            padding=((0, 0), (0, 0)),
            use_bias=False,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
        )(x)

        x = norm()(x)

        x = nn.leaky_relu(x, negative_slope=0.1)

        x_logits = nn.Conv(
            kernel_size=(1, 1),
            features=self.num_cat,
            strides=(1, 1),
            padding=((0, 0), (0, 0)),
            use_bias=False,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
        )(x)

        x_mean = nn.Conv(
            kernel_size=(1, 1),
            features=self.num_cts_codes,
            strides=(1, 1),
            padding=((0, 0), (0, 0)),
            use_bias=False,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
        )(x)

        x_var = nn.Conv(
            kernel_size=(1, 1),
            features=self.num_cts_codes,
            strides=(1, 1),
            padding=((0, 0), (0, 0)),
            use_bias=False,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
        )(x)

        return (
            x_logits.squeeze(axis=[1, 2]),
            x_mean.squeeze(axis=[1, 2]),
            jnp.exp(x_var).squeeze(axis=[1, 2]),
        )


if __name__ == "__main__":

    model = Q_head(filter_size=128)

    def initialized(key, image_size, model):
        input_shape = (1, image_size, image_size, 1)

        @jax.jit
        def init(rng, shape):
            return model.init(rng, shape, train=True)

        variables = init(rng=key, shape=jnp.ones(input_shape, dtype=model.dtype))
        return variables["params"], variables["batch_stats"]

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
