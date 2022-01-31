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
from utils.zero_init import zeros_

ModuleDef = Any
dtypedef = Any


class Generator(nn.Module):
    """Basic generator network for MNIST

    Args:
        dtype (dtypedef): Model dtype. Defaults to float32

    References:
        Chen et al, 2016: https://arxiv.org/abs/1606.03657

    """

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
            scale_init=self.kernel_init,
            bias_init=zeros_(),
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
