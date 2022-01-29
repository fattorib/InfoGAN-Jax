import jax.numpy as jnp
import jax
import flax.linen as nn


@jax.jit
def binary_cross_entropy_loss(*, logit, label):
    """
    Standard BCELoss. We assume that scores are unscaled.

    Used for discriminator loss.

    """

    return jnp.mean(
        jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(-jnp.abs(logit)))
    )


@jax.jit
def cross_entropy_loss(*, q_logits, q_codes):
    """
    Softmax + Standard CE Loss. Used for the categorical code loss.
    """

    return -jnp.mean(jnp.sum(q_codes * nn.log_softmax(q_logits, axis=-1), axis=-1))


@jax.jit
def q_cts_loss(*, q_mu, q_var, y):
    """
    NLL for Gaussian. Used for the continuous code loss.

    TODO: Still don't completely understand why we use this loss function
    """

    # LL for single example
    loglikelihood = (1 / (2 * (q_var ** 2 + 1e-6))) * (
        (y - q_mu) ** (2) + 0.5 * jnp.log(2 * jnp.pi * q_var + 1e-6)
    )

    return -jnp.mean(jnp.sum(-1 * loglikelihood, axis=-1))
