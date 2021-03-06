from typing import Any, Callable, Sequence
from jax import numpy as jnp
from flax import linen as nn
from functools import partial

ModuleDef = Any
dtypedef = Any


class Discriminator(nn.Module):
    """Basic discriminator network for MNIST

    Args:
        filter_list (Sequence[int]): Sequence of filters to use.

        num_channels (int): Number of input channels for image. Defaults to 1.

        dtype (dtypedef): Model dtype. Defaults to float32

    References:
        Chen et al, 2016: https://arxiv.org/abs/1606.03657

    """

    filter_list: Sequence[int]

    num_channels: int = 1

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv layers
    kernel_init: Callable = nn.initializers.normal(stddev=0.02, dtype=dtype)

    @nn.compact
    def __call__(self, x, train, with_head):

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
            kernel_init=self.kernel_init,
        )(x)

        x = nn.leaky_relu(x, negative_slope=0.1)

        x = nn.Conv(
            kernel_size=(4, 4),
            features=self.filter_list[1],
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
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
            kernel_init=self.kernel_init,
        )(x)

        x = norm()(x)

        x = nn.leaky_relu(x, negative_slope=0.1)

        if with_head:

            x = nn.Conv(
                kernel_size=(1, 1),
                features=1,
                strides=(1, 1),
                padding=((0, 0), (0, 0)),
                use_bias=False,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
            )(x)

            return x.squeeze(axis=(2, 3))
        else:
            return x
