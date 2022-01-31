from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import lax
from jax import random
from jax import core
from jax._src.util import prod
from jax import dtypes


def zeros_(dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return jnp.zeros(shape, dtypes.canonicalize_dtype(dtype))

    return init
