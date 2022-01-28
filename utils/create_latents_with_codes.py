import jax
import jax.numpy as jnp
import operator


def create_latents_with_codes(num_noise, num_cts, num_cat, rng_key, num_samples):
    """Create a latent variable to feed into the generator

    Args:
        num_noise (int): number of 'noise' variables to use. For InfoGAN MNIST example, defaults to 62
        num_cts (int): number of continuous 'codes' to use. For InfoGAN MNIST example, defaults to 2
        num_cat (int): number of category 'codes' to use. For InfoGAN MNIST example, defaults to 10 categories
        rnk_key (*): jax prng key
        num_samples (int): number of latent samples to generate

    """

    # Noise

    z = jax.random.normal(rng_key, shape=(num_samples, num_noise))

    # Continuous

    for _ in range(num_cts):
        c = jax.random.uniform(key=rng_key, shape=(num_samples, 1), minval=-1, maxval=1)
        z = jnp.concatenate([z, c], axis=1)

    # Categorical
    logit_probs = [1 / num_cat for _ in range(num_cat)]
    c_idx = jax.random.categorical(
        key=rng_key, logits=jnp.array(logit_probs), shape=(num_samples,)
    )

    c = jax.nn.one_hot(c_idx, num_classes=num_cat)

    z = jnp.concatenate([z, c], axis=1)

    return z
