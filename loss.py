import jax.numpy as jnp
import jax
import flax.linen as nn


@jax.jit
def binary_cross_entropy_loss(*, scores, labels):
    """
    Standard BCELoss. We assume that scores are unscaled.

    Used for discriminator loss.

    TODO: Might need to update implementation for numerical stability.
          See: https://github.com/pytorch/pytorch/issues/751
    """

    scaled_scores = nn.sigmoid(scores)

    # Clipping log values
    clipped_score_pos = jnp.clip(jnp.log(scaled_scores), -100)
    clipped_score_neg = jnp.clip(jnp.log(1 - scaled_scores), -100)

    return -jnp.mean(
        jnp.sum(
            labels * clipped_score_pos + (1 - labels) * clipped_score_neg,
            axis=-1,
        )
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
